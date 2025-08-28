import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

class Swish(nn.Module):
    """Swish激活函数: x * sigmoid(x)"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    """时间步嵌入层 - 将离散时间步转换为连续向量表示"""
    def __init__(self, T, d_model, dim):
        """
        参数:
            T: 最大时间步数
            d_model: 嵌入维度
            dim: 输出维度
        """
        assert d_model % 2 == 0  # 确保嵌入维度为偶数
        super().__init__()
        # 计算位置编码的频率因子
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)  # 指数衰减
        
        # 创建位置序列
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]  # 外积
        
        # 构造正弦/余弦位置编码
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        emb = emb.view(T, d_model)  # 重塑形状

        # 嵌入层序列
        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),  # 从预训练张量创建嵌入
            nn.Linear(d_model, dim),  # 线性变换
            Swish(),  # Swish激活
            nn.Linear(dim, dim),  # 再次线性变换
        )
        self.initialize()  # 初始化参数

    def initialize(self):
        """参数初始化 - Xavier均匀初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)  # 权重初始化
                init.zeros_(module.bias)  # 偏置归零

    def forward(self, t):
        """
        前向传播
        
        参数:
            t: 时间步张量 [batch_size]
            
        返回:
            emb: 时间嵌入向量 [batch_size, dim]
        """
        emb = self.timembedding(t)  # 嵌入处理
        return emb


class Text_Noise_Pre(nn.Module):
    """文本噪声预测器 - 使用LSTM处理时序文本特征"""
    def __init__(self, T, ch, dropout, in_ch):
        """
        参数:
            T: 最大时间步数
            ch: 通道数（未直接使用）
            dropout: dropout概率
            in_ch: 输入维度
        """
        super().__init__()
        tdim = int(in_ch / 2)  # 时间嵌入维度
        self.time_embedding = TimeEmbedding(T, ch, tdim)  # 时间嵌入层

        # 文本特征处理LSTM
        self.fc1 = nn.LSTM(
            input_size=in_ch, 
            hidden_size=int(in_ch/2), 
            num_layers=2)  # 双层LSTM
        # 上下文特征处理LSTM
        self.fc1_1 = nn.LSTM(
            input_size=in_ch, 
            hidden_size=int(in_ch/2), 
            num_layers=2)  # 双层LSTM
        # 输出LSTM
        self.fc2 = nn.LSTM(
            input_size=int(in_ch/2), 
            hidden_size=in_ch, 
            num_layers=2)  # 双层LSTM

        self.dropout = dropout  # dropout概率
        self.swish = Swish()  # Swish激活函数

    def forward(self, x, t, y):
        """
        文本噪声预测前向传播
        
        参数:
            x: 带噪文本特征 [batch_size, seq_len, in_ch]
            t: 时间步 [batch_size]
            y: 上下文特征 [batch_size, seq_len, in_ch]
            
        返回:
            h: 预测的干净文本特征 [batch_size, seq_len, in_ch]
        """
        # 时间嵌入
        temb = self.time_embedding(t)[:, None, :]  # [batch_size, 1, tdim]
        
        # 文本特征提取
        h, _ = self.fc1(x)
        # 上下文特征提取
        h_y, _ = self.fc1_1(y)
        
        # 特征融合 (带噪特征 + 上下文 + 时间嵌入)
        h = h + h_y + temb

        # 激活与正则化
        h = self.swish(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # 输出层
        h, _ = self.fc2(h)
        return h


class Audio_Noise_Pre(nn.Module):
    """音频噪声预测器 - 使用全连接网络处理音频特征"""
    def __init__(self, T, ch, dropout, in_ch):
        """
        参数:
            T: 最大时间步数
            ch: 通道数（未直接使用）
            dropout: dropout概率
            in_ch: 输入维度
        """
        super().__init__()
        tdim = int(in_ch/2)  # 时间嵌入维度
        self.time_embedding = TimeEmbedding(T, in_ch, tdim)  # 时间嵌入层

        # 全连接层
        self.fc1 = nn.Linear(in_ch, int(in_ch/2))  # 特征压缩
        self.fc1_1 = nn.Linear(in_ch, int(in_ch/2))  # 上下文处理
        self.fc2 = nn.Linear(int(in_ch/2), in_ch)  # 特征扩展
        
        self.dropout = dropout  # dropout概率
        self.swish = Swish()  # Swish激活函数

    def forward(self, x, t, y):
        """
        音频噪声预测前向传播
        
        参数:
            x: 带噪音频特征 [batch_size, seq_len, in_ch]
            t: 时间步 [batch_size]
            y: 上下文特征 [batch_size, seq_len, in_ch]
            
        返回:
            h: 预测的干净音频特征 [batch_size, seq_len, in_ch]
        """
        # 时间嵌入
        temb = self.time_embedding(t)[:, None, :]  # [batch_size, 1, tdim]
        
        # 特征提取
        h = self.fc1(x)  # 带噪特征
        h_y = self.fc1_1(y)  # 上下文特征
        
        # 特征融合
        h = h + temb + h_y
        
        # 激活与正则化
        h = self.swish(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # 输出层
        h = self.fc2(h)
        return h


class Visual_Noise_Pre(nn.Module):
    """视觉噪声预测器 - 使用全连接网络处理视觉特征"""
    def __init__(self, T, ch, dropout, in_ch):
        """
        参数:
            T: 最大时间步数
            ch: 通道数（未直接使用）
            dropout: dropout概率
            in_ch: 输入维度
        """
        super().__init__()
        tdim = int(in_ch/2)  # 时间嵌入维度
        self.time_embedding = TimeEmbedding(T, ch, tdim)  # 时间嵌入层

        # 全连接层
        self.fc1 = nn.Linear(in_ch, int(in_ch/2))  # 特征压缩
        self.fc1_1 = nn.Linear(in_ch, int(in_ch/2))  # 上下文处理
        self.fc2 = nn.Linear(int(in_ch/2), in_ch)  # 特征扩展
        
        self.dropout = dropout  # dropout概率
        self.swish = Swish()  # Swish激活函数

    def forward(self, x, t, y):
        """
        视觉噪声预测前向传播
        
        参数:
            x: 带噪视觉特征 [batch_size, seq_len, in_ch]
            t: 时间步 [batch_size]
            y: 上下文特征 [batch_size, seq_len, in_ch]
            
        返回:
            h: 预测的干净视觉特征 [batch_size, seq_len, in_ch]
        """
        # 时间嵌入
        temb = self.time_embedding(t)[:, None, :]  # [batch_size, 1, tdim]
        
        # 特征提取
        h = self.fc1(x)  # 带噪特征
        h_y = self.fc1_1(y)  # 上下文特征
        
        # 特征融合
        h = h + temb + h_y
        
        # 激活与正则化
        h = self.swish(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # 输出层
        h = self.fc2(h)
        return h