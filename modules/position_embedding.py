import math
import torch
import torch.nn as nn

def make_positions(tensor, padding_idx, left_pad):
    """
    用位置编号替换非填充符号
    
    参数:
        tensor: 输入张量 [batch_size, seq_len]
        padding_idx: 填充索引
        left_pad: 是否在左侧填充
        
    返回:
        new_tensor: 包含位置信息的新张量
    """
    max_pos = padding_idx + 1 + tensor.size(1)  # 最大位置
    device = tensor.get_device()  # 获取设备
    
    # 创建设备特定的缓冲区
    buf_name = f'range_buf_{device}'
    if not hasattr(make_positions, buf_name):
        setattr(make_positions, buf_name, tensor.new())
    setattr(make_positions, buf_name, getattr(make_positions, buf_name).type_as(tensor))
    
    # 扩展缓冲区
    if getattr(make_positions, buf_name).numel() < max_pos:
        torch.arange(padding_idx + 1, max_pos, out=getattr(make_positions, buf_name))
    
    # 创建掩码
    mask = tensor.ne(padding_idx)
    positions = getattr(make_positions, buf_name)[:tensor.size(1)].expand_as(tensor)
    
    # 处理左侧填充
    if left_pad:
        positions = positions - mask.size(1) + mask.long().sum(dim=1).unsqueeze(1)
    
    # 替换非填充位置
    new_tensor = tensor.clone()
    return new_tensor.masked_scatter_(mask, positions[mask]).long()


class SinusoidalPositionalEmbedding(nn.Module):
    """正弦位置嵌入实现"""
    
    def __init__(self, embedding_dim, padding_idx=0, left_pad=0, init_size=128):
        """
        初始化位置嵌入层
        
        参数:
            embedding_dim: 嵌入维度
            padding_idx: 填充索引
            left_pad: 是否在左侧填充
            init_size: 初始大小
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.left_pad = left_pad
        self.weights = dict()   # 设备到权重的映射
        # 注册浮点张量缓冲区
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """
        构建正弦嵌入
        
        参数:
            num_embeddings: 嵌入数量
            embedding_dim: 嵌入维度
            padding_idx: 填充索引
            
        返回:
            emb: 正弦嵌入矩阵
        """
        half_dim = embedding_dim // 2  # 半维度
        emb = math.log(10000) / (half_dim - 1)  # 缩放因子
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)  # 指数衰减
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)  # 外积
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)  # 正弦+余弦
        
        # 处理奇数维度
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)  # 零填充
        
        # 设置填充索引
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        
        return emb

    def forward(self, input):
        """
        前向传播
        
        参数:
            input: 输入张量 [bsz, seq_len]
            
        返回:
            位置嵌入张量 [bsz, seq_len, embedding_dim]
        """
        bsz, seq_len = input.size()  # 批次大小和序列长度
        max_pos = self.padding_idx + 1 + seq_len  # 最大位置
        device = input.get_device()  # 获取设备
        
        # 设备特定的权重管理
        if device not in self.weights or max_pos > self.weights[device].size(0):
            # 重新计算或扩展嵌入
            self.weights[device] = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        
        # 确保数据类型一致
        self.weights[device] = self.weights[device].type_as(self._float_tensor)
        # 创建位置张量
        positions = make_positions(input, self.padding_idx, self.left_pad)
        # 返回位置嵌入 (分离计算图)
        return self.weights[device].index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

    def max_positions(self):
        """支持的最大位置数 (任意大数)"""
        return int(1e5)