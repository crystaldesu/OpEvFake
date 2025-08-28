import torch.nn as nn
import torch.nn.functional as F
import torch

class CapsuleSequenceToGraph(nn.Module):
    """胶囊序列到图转换模块"""
    
    def __init__(self, MULT_d, dim_capsule, vertex_num, routing,
                 T_t, T_a, T_v):
        """
        初始化胶囊转换模块
        
        参数:
            MULT_d: 多模态特征维度
            dim_capsule: 胶囊维度
            vertex_num: 顶点数量
            routing: 路由迭代次数
            T_t: 文本序列长度
            T_a: 音频序列长度
            T_v: 视觉序列长度
        """
        super(CapsuleSequenceToGraph, self).__init__()
        self.d_c = dim_capsule  # 胶囊维度
        self.n = vertex_num  # 顶点数量
        self.routing = routing  # 路由迭代次数
        
        # 创建主胶囊权重
        self.W_tpc = nn.Parameter(torch.Tensor(T_t, self.n, MULT_d, self.d_c))  # 文本
        self.W_apc = nn.Parameter(torch.Tensor(T_a, self.n, MULT_d, self.d_c))  # 音频
        self.W_vpc = nn.Parameter(torch.Tensor(T_v, self.n, MULT_d, self.d_c))  # 视觉
        
        # 参数初始化
        nn.init.xavier_normal_(self.W_tpc)
        nn.init.xavier_normal_(self.W_apc)
        nn.init.xavier_normal_(self.W_vpc)

    def forward(self, text, audio, video, batch_size):
        """
        前向传播
        
        参数:
            text: 文本特征 [T_t, batch_size, MULT_d]
            audio: 音频特征 [T_a, batch_size, MULT_d]
            video: 视觉特征 [T_v, batch_size, MULT_d]
            batch_size: 批大小
            
        返回:
            text_vertex: 文本顶点特征 [batch_size, n, d_c]
            audio_vertex: 音频顶点特征 [batch_size, n, d_c]
            video_vertex: 视觉顶点特征 [batch_size, n, d_c]
        """
        T_t = text.shape[0]  # 文本序列长度
        T_a = audio.shape[0]  # 音频序列长度
        T_v = video.shape[0]  # 视觉序列长度
        batch_size = text.shape[1]  # 批大小
        
        # ========== 创建主胶囊 ==========
        # 文本主胶囊: [batch_size, T_t, n, d_c]
        text_pri_caps = (torch.einsum('tbj, tnjd->tbnd', text, self.W_tpc)).permute(1, 0, 2, 3)
        # 音频主胶囊: [batch_size, T_a, n, d_c]
        audio_pri_caps = (torch.einsum('tbj, tnjd->tbnd', audio, self.W_apc)).permute(1, 0, 2, 3)
        # 视觉主胶囊: [batch_size, T_v, n, d_c]
        video_pri_caps = (torch.einsum('tbj, tnjd->tbnd', video, self.W_vpc)).permute(1, 0, 2, 3)

        # 分离计算图 (路由不参与梯度计算)
        text_pri_caps_temp = text_pri_caps.detach()
        audio_pri_caps_temp = audio_pri_caps.detach()
        video_pri_caps_temp = video_pri_caps.detach()

        # ========== 路由机制 ==========
        for r in range(self.routing + 1):
            # 初始化路由系数
            if r == 0:
                b_t = torch.zeros(batch_size, T_t, self.n).cuda()  # 文本路由系数
                b_a = torch.zeros(batch_size, T_a, self.n).cuda()  # 音频路由系数
                b_v = torch.zeros(batch_size, T_v, self.n).cuda()  # 视觉路由系数
            
            # 路由系数归一化
            rc_t = F.softmax(b_t, 2).cuda()  # 文本
            rc_a = F.softmax(b_a, 2).cuda()  # 音频
            rc_v = F.softmax(b_v, 2).cuda()  # 视觉

            # 计算顶点表示
            text_vertex = torch.tanh(torch.sum(text_pri_caps_temp * rc_t.unsqueeze(-1), 1))
            audio_vertex = torch.tanh(torch.sum(audio_pri_caps_temp * rc_a.unsqueeze(-1), 1))
            video_vertex = torch.tanh(torch.sum(video_pri_caps_temp * rc_v.unsqueeze(-1), 1))

            # 更新路由系数
            if r < self.routing:
                # 文本路由更新
                last = b_t
                new = ((text_vertex.unsqueeze(1)) * text_pri_caps_temp).sum(3)
                b_t = last + new
                
                # 音频路由更新
                last = b_a
                new = (audio_vertex.unsqueeze(1) * audio_pri_caps_temp).sum(3)
                b_a = last + new
                
                # 视觉路由更新
                last = b_v
                new = (video_vertex.unsqueeze(1) * video_pri_caps_temp).sum(3)
                b_v = last + new

        # 使用最终路由系数创建顶点
        text_vertex = torch.tanh(torch.sum(text_pri_caps * rc_t.unsqueeze(-1), 1))
        audio_vertex = torch.tanh(torch.sum(audio_pri_caps * rc_a.unsqueeze(-1), 1))
        video_vertex = torch.tanh(torch.sum(video_pri_caps * rc_v.unsqueeze(-1), 1))
        
        return text_vertex, audio_vertex, video_vertex