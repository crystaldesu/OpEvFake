import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入跨模态Transformer和胶囊序列到图的转换模块
from src.CrossmodalTransformer import MULTModel
from src.StoG import CapsuleSequenceToGraph
# 导入多模态噪声预测器
from Diffusion.Multimodal_Model import Text_Noise_Pre, Audio_Noise_Pre, Visual_Noise_Pre

def extract(v, t, x_shape):
    """
    从系数向量v中提取指定时间步t的系数，并重塑为适合广播的形状
    
    参数:
        v: 系数向量 [T]
        t: 时间步张量 [batch_size]
        x_shape: 输入张量的形状
        
    返回:
        out: 重塑后的系数 [batch_size, 1, 1, ...] (与x_shape相同维度)
    """
    device = t.device  # 获取t所在的设备
    # 从v中收集对应时间步t的系数
    out = torch.gather(v, index=t, dim=0).float().to(device)
    # 将输出重塑为与输入x相同的形状（添加必要的维度）
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, modelConfig, beta_1, beta_T, T, t_in, a_in, v_in, d_m, dropout, label_dim,
                 unified_size, vertex_num, routing, T_t, T_a, T_v, batch_size):
        """
        高斯扩散模型训练器 - 多模态扩散模型的核心框架
        
        参数:
            modelConfig: 模型配置字典
            beta_1: 起始beta值
            beta_T: 终止beta值
            T: 扩散时间步数
            t_in: 文本输入维度
            a_in: 音频输入维度
            v_in: 视觉输入维度
            d_m: 模态内部维度
            dropout: dropout概率
            label_dim: 标签维度
            unified_size: 统一表示维度
            vertex_num: 胶囊图顶点数量
            routing: 胶囊路由迭代次数
            T_t: 文本序列长度
            T_a: 音频序列长度
            T_v: 视觉序列长度
            batch_size: 批大小
        """
        super().__init__()

        self.T = T  # 扩散总步数
        self.batch_size = batch_size
        self.mult_dropout = dropout  # 多模态dropout

        # 注册beta调度参数为缓冲区
        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas  # 计算alpha
        alphas_bar = torch.cumprod(alphas, dim=0)  # alpha的累积乘积

        # 注册扩散过程相关参数为缓冲区
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))  # sqrt(alpha_bar)
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))  # sqrt(1-alpha_bar)

        # ========== 特征提取层 ==========
        # 文本特征提取（LSTM）
        self.fc_pre_t_1 = nn.LSTM(100, modelConfig["t_in_pre"], bidirectional=True)
        self.fc_pre_t_2 = nn.Linear(modelConfig["t_in_pre"] * 2, modelConfig["t_in_pre"])
        # 视觉特征提取
        self.fc_pre_v = torch.nn.Linear(v_in, modelConfig["v_in_pre"])
        # 评论特征提取
        self.fc_pre_com = nn.Sequential(
            torch.nn.Linear(modelConfig["t_in"], unified_size), 
            torch.nn.ReLU(),
            nn.Dropout(p=modelConfig["comments_dropout"]))
        # 用户介绍特征提取
        self.fc_pre_user = nn.Sequential(
            torch.nn.Linear(modelConfig["t_in"], unified_size), 
            torch.nn.ReLU(),
            nn.Dropout(p=modelConfig["comments_dropout"]))
        # C3D特征提取
        self.fc_pre_c3d = torch.nn.Linear(modelConfig["c3d_in"], unified_size)
        # GPT描述特征提取
        self.fc_pre_gpt_1 = nn.LSTM(768, modelConfig["t_in_pre"], bidirectional=True)
        self.fc_pre_gpt_2 = nn.Linear(modelConfig["t_in_pre"] * 2, modelConfig["t_in_pre"])
        # 音频特征提取
        self.fc_pre_a = nn.Linear(a_in, modelConfig["a_in_pre"])

        # ========== 模态内增强层 ==========
        self.fc_g_t = nn.Linear(d_m * 6, d_m)  # 文本特征融合
        self.fc_a_MTout = nn.Linear(d_m * 3, d_m)  # 音频特征融合
        self.fc_v_MTout = nn.Linear(d_m * 3, d_m)  # 视觉特征融合
        # 跨模态Transformer
        self.CrossmodalTransformer = MULTModel(
            modelConfig["t_in_pre"], 
            modelConfig["a_in_pre"],
            modelConfig["v_in_pre"], 
            d_m, 
            self.mult_dropout)
        # 胶囊序列到图转换器
        self.StoG = CapsuleSequenceToGraph(
            d_m, unified_size, vertex_num, routing, T_t, T_a, T_v)

        # ========== 跨模态交互层 ==========
        # 文本噪声预测器
        self.model_t = Text_Noise_Pre(
            T=modelConfig["T"], 
            ch=modelConfig["vertex_num"],
            dropout=modelConfig["Text_Pre_dropout"],
            in_ch=unified_size)
        # 音频噪声预测器
        self.model_a = Audio_Noise_Pre(
            T=modelConfig["T"], 
            ch=modelConfig["vertex_num"],
            dropout=modelConfig["Img_Pre_dropout"],
            in_ch=unified_size)
        # 视觉噪声预测器
        self.model_v = Visual_Noise_Pre(
            T=modelConfig["T"], 
            ch=modelConfig["vertex_num"],
            dropout=modelConfig["Img_Pre_dropout"],
            in_ch=unified_size)

        # ========== 特征转换层 ==========
        self.fc_t = nn.Linear(in_features=vertex_num, out_features=1)  # 文本特征转换
        self.fc_a = nn.Linear(in_features=vertex_num, out_features=1)  # 音频特征转换
        self.fc_v = nn.Linear(in_features=vertex_num, out_features=1)  # 视觉特征转换
        self.fc_m = nn.Linear(in_features=unified_size * 3, out_features=unified_size)  # 多模态融合

        # ========== 预测层 ==========
        self.fc_pre = nn.Linear(in_features=unified_size, out_features=label_dim)  # 最终预测
        self.trm = nn.TransformerEncoderLayer(
            d_model=unified_size, 
            nhead=2, 
            batch_first=True)  # Transformer编码层

    def forward(self, texts, audios, videos, comments, c3d, user_intro, gpt_description):
        """
        前向传播流程
        
        参数:
            texts: 文本特征 [batch_size, seq_len, t_in]
            audios: 音频特征 [batch_size, seq_len, a_in]
            videos: 视觉特征 [batch_size, seq_len, v_in]
            comments: 评论特征 [batch_size, seq_len, t_in]
            c3d: C3D特征 [batch_size, seq_len, c3d_in]
            user_intro: 用户介绍 [batch_size, seq_len, t_in]
            gpt_description: GPT描述 [batch_size, seq_len, 768]
            
        返回:
            loss: 扩散损失
            output_m: 预测结果
        """
        # ========== 特征提取 ==========
        # 文本特征提取（LSTM）
        texts_local, _ = self.fc_pre_t_1(texts)
        texts_local = self.fc_pre_t_2(texts_local)

        # 音频特征提取
        audios_local = self.fc_pre_a(audios)
        # C3D特征提取
        c3d_local = self.fc_pre_c3d(c3d)
        # GPT描述特征提取
        gpt_local, _ = self.fc_pre_gpt_1(gpt_description)
        gpt_local = self.fc_pre_gpt_2(gpt_local)
        # 评论特征提取
        comments_global = self.fc_pre_com(comments)
        # 用户介绍特征提取
        user_intro_global = self.fc_pre_user(user_intro.squeeze())
        # 视觉特征提取
        videos = self.fc_pre_v(videos)
        videos_global = torch.mean(videos, -2)  # 全局平均池化

        # ========== 模态内增强 ==========
        # 跨模态Transformer处理
        z_t, z_g, z_a, z_v = self.CrossmodalTransformer(
            texts_local, gpt_local, audios_local, c3d_local)
        # 特征融合
        z_t = self.fc_g_t(torch.cat([z_t, z_g], dim=2))
        z_a = self.fc_a_MTout(z_a)
        z_v = self.fc_v_MTout(z_v)
        # 胶囊序列到图转换
        x_t, x_a, x_v = self.StoG(z_t, z_a, z_v, self.batch_size)

        # ========== 跨模态交互 ==========
        # 多模态特征融合
        x_m = torch.concat([x_t.squeeze(), x_a.squeeze(), x_v.squeeze()], dim=2)
        x_m = self.fc_m(x_m)

        # 文本扩散过程
        t_t = torch.randint(self.T, size=(x_t.shape[0],), device=x_t.device)  # 随机时间步
        noise_t = torch.randn_like(x_t)  # 生成高斯噪声
        # 应用噪声 (前向扩散)
        x_tmp_t = (
            extract(self.sqrt_alphas_bar, t_t, x_t.shape) * x_t +
            extract(self.sqrt_one_minus_alphas_bar, t_t, x_t.shape) * noise_t)

        # 音频扩散过程
        t_a = torch.randint(self.T, size=(x_a.shape[0],), device=x_a.device)
        noise_a = torch.randn_like(x_a)
        x_tmp_a = (
            extract(self.sqrt_alphas_bar, t_a, x_a.shape) * x_a +
            extract(self.sqrt_one_minus_alphas_bar, t_a, x_a.shape) * noise_a)

        # 视觉扩散过程
        t_v = torch.randint(self.T, size=(x_v.shape[0],), device=x_v.device)
        noise_v = torch.randn_like(x_v)
        x_tmp_v = (
            extract(self.sqrt_alphas_bar, t_v, x_v.shape) * x_v +
            extract(self.sqrt_one_minus_alphas_bar, t_v, x_v.shape) * noise_v)

        # 噪声预测 (反向去噪)
        x_a_pre = self.model_a(x_tmp_a, t_a, x_m)  # 音频去噪
        x_v_pre = self.model_v(x_tmp_v, t_v, x_m)  # 视觉去噪
        x_t_pre = self.model_t(x_tmp_t, t_t, x_m)  # 文本去噪
        
        # 计算各模态损失
        loss_a = F.mse_loss(x_a_pre.squeeze(), x_a, reduction='none')
        loss_t = F.mse_loss(x_t_pre.squeeze(), x_t, reduction='none')
        loss_v = F.mse_loss(x_v_pre.squeeze(), x_v, reduction='none')
        loss = loss_t + loss_a + loss_v  # 总损失

        # ========== 特征转换 ==========
        output_a = self.fc_a(x_a_pre.transpose(2, 1))
        output_t = self.fc_t(x_t_pre.transpose(2, 1))
        output_v = self.fc_v(x_v_pre.transpose(2, 1))
        output_a = output_a.transpose(2, 1)
        output_t = output_t.transpose(2, 1)
        output_v = output_v.transpose(2, 1)

        # 调整特征维度
        comments_global = comments_global.unsqueeze(1)
        videos_global = videos_global.unsqueeze(1)
        user_intro_global = user_intro_global.unsqueeze(1)

        # ========== 预测层 ==========
        # 多模态特征融合
        output_m = torch.concat([
            output_t, output_a, videos_global, 
            user_intro_global, output_v, comments_global], dim=1)
        # Transformer编码
        output_m = self.trm(output_m)
        # 全局平均池化
        output_m = torch.mean(output_m, -2)
        # 最终预测
        output_m = self.fc_pre(output_m.squeeze())

        return loss, output_m