import torch
from torch import nn
import torch.nn.functional as F
from modules.position_embedding import SinusoidalPositionalEmbedding
from modules.multihead_attention import MultiheadAttention
import math

def fill_with_neg_inf(t):
    """用负无穷填充张量 (FP16兼容)"""
    return t.float().fill_(float('-inf')).type_as(t)

def buffered_future_mask(tensor, tensor2=None):
    """创建未来掩码 (上三角矩阵)"""
    dim1 = dim2 = tensor.size(0)  # 目标长度
    if tensor2 is not None:
        dim2 = tensor2.size(0)  # 源长度
    # 创建上三角矩阵
    future_mask = torch.triu(fill_with_neg_inf(torch.ones(dim1, dim2)), 1+abs(dim2-dim1))
    # 设备处理
    if tensor.is_cuda:
        future_mask = future_mask.cuda()
    return future_mask[:dim1, :dim2]

def Linear(in_features, out_features, bias=True):
    """创建线性层 (带Xavier初始化)"""
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)  # Xavier初始化
    if bias:
        nn.init.constant_(m.bias, 0.)  # 偏置归零
    return m

def LayerNorm(embedding_dim):
    """创建层归一化层"""
    m = nn.LayerNorm(embedding_dim)
    return m

class TransformerEncoder(nn.Module):
    """Transformer编码器 (多层堆叠)"""
    
    def __init__(self, embed_dim, num_heads, layers, attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0,
                 embed_dropout=0.0, attn_mask=False, position_emb=True):
        """
        初始化Transformer编码器
        
        参数:
            embed_dim: 嵌入维度
            num_heads: 注意力头数量
            layers: 编码器层数
            attn_dropout: 注意力dropout
            relu_dropout: ReLU dropout
            res_dropout: 残差连接dropout
            embed_dropout: 嵌入dropout
            attn_mask: 是否使用注意力掩码
            position_emb: 是否使用位置嵌入
        """
        super().__init__()
        self.dropout = embed_dropout      # 嵌入dropout
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)  # 嵌入缩放因子
        
        # 位置嵌入
        if position_emb:
            self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)
        else:
            self.embed_positions = None
        
        self.attn_mask = attn_mask  # 注意力掩码标志

        # 创建编码器层
        self.layers = nn.ModuleList([])
        for layer in range(layers):
            new_layer = TransformerEncoderLayer(
                embed_dim,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                relu_dropout=relu_dropout,
                res_dropout=res_dropout,
                attn_mask=attn_mask)
            self.layers.append(new_layer)

        self.register_buffer('version', torch.Tensor([2]))  # 版本号
        self.normalize = True
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)  # 层归一化

    def forward(self, x_in, x_in_k=None, x_in_v=None):
        """
        前向传播
        
        参数:
            x_in: 输入嵌入 [src_len, batch, embed_dim]
            x_in_k: 键嵌入 [src_len, batch, embed_dim]
            x_in_v: 值嵌入 [src_len, batch, embed_dim]
            
        返回:
            编码器输出 [src_len, batch, embed_dim]
        """
        # 嵌入缩放
        x = self.embed_scale * x_in
        # 添加位置嵌入
        if self.embed_positions is not None:
            x += self.embed_positions(x_in.transpose(0, 1)[:, :, 0]).transpose(0, 1)
        # 应用嵌入dropout
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 处理键值对
        if x_in_k is not None and x_in_v is not None:
            # 键处理
            x_k = self.embed_scale * x_in_k
            # 值处理
            x_v = self.embed_scale * x_in_v
            # 添加位置嵌入
            if self.embed_positions is not None:
                x_k += self.embed_positions(x_in_k.transpose(0, 1)[:, :, 0]).transpose(0, 1)
                x_v += self.embed_positions(x_in_v.transpose(0, 1)[:, :, 0]).transpose(0, 1)
            # 应用dropout
            x_k = F.dropout(x_k, p=self.dropout, training=self.training)
            x_v = F.dropout(x_v, p=self.dropout, training=self.training)
        
        # 编码器层处理
        intermediates = [x]  # 中间结果
        for layer in self.layers:
            if x_in_k is not None and x_in_v is not None:
                x = layer(x, x_k, x_v)  # 使用键值对
            else:
                x = layer(x)  # 自注意力
            intermediates.append(x)

        # 层归一化
        if self.normalize:
            x = self.layer_norm(x)

        return x

    def max_positions(self):
        """支持的最大输入长度"""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""
    
    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1,
                 attn_mask=False):
        """
        初始化编码器层
        
        参数:
            embed_dim: 嵌入维度
            num_heads: 注意力头数量
            attn_dropout: 注意力dropout
            relu_dropout: ReLU dropout
            res_dropout: 残差连接dropout
            attn_mask: 是否使用注意力掩码
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # 自注意力层
        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout
        )
        self.attn_mask = attn_mask  # 注意力掩码标志

        self.relu_dropout = relu_dropout  # ReLU dropout
        self.res_dropout = res_dropout  # 残差连接dropout
        self.normalize_before = True  # 是否在层前归一化

        # 前馈网络
        self.fc1 = Linear(self.embed_dim, 4*self.embed_dim)  # 扩展层
        self.fc2 = Linear(4*self.embed_dim, self.embed_dim)  # 压缩层
        # 层归一化 (两个位置)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(2)])

    def forward(self, x, x_k=None, x_v=None):
        """
        前向传播
        
        参数:
            x: 输入 [seq_len, batch, embed_dim]
            x_k: 键输入 [src_len, batch, embed_dim]
            x_v: 值输入 [src_len, batch, embed_dim]
            
        返回:
            编码器层输出 [batch, src_len, embed_dim]
        """
        residual = x  # 残差连接
        
        # 第一个归一化 (可选)
        x = self.maybe_layer_norm(0, x, before=True)
        
        # 创建注意力掩码
        mask = buffered_future_mask(x, x_k) if self.attn_mask else None
        
        # 注意力计算
        if x_k is None and x_v is None:
            # 自注意力
            x, _ = self.self_attn(query=x, key=x, value=x, attn_mask=mask)
        else:
            # 键值对处理
            x_k = self.maybe_layer_norm(0, x_k, before=True)
            x_v = self.maybe_layer_norm(0, x_v, before=True) 
            # 交叉注意力
            x, _ = self.self_attn(query=x, key=x_k, value=x_v, attn_mask=mask)
        
        # 残差连接和dropout
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        # 归一化后处理
        x = self.maybe_layer_norm(0, x, after=True)

        # 前馈网络处理
        residual = x
        x = self.maybe_layer_norm(1, x, before=True)  # 第二个归一化
        x = F.relu(self.fc1(x))  # ReLU激活
        x = F.dropout(x, p=self.relu_dropout, training=self.training)  # ReLU dropout
        x = self.fc2(x)  # 前馈网络第二部分
        x = F.dropout(x, p=self.res_dropout, training=self.training)  # 残差连接dropout
        x = residual + x  # 残差连接
        x = self.maybe_layer_norm(1, x, after=True)  # 归一化后处理
        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        """条件层归一化"""
        assert before ^ after  # 确保只在一个阶段应用
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)  # 应用层归一化
        else:
            return x  # 直接返回

def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)


def buffered_future_mask(tensor, tensor2=None):
    dim1 = dim2 = tensor.size(0)
    if tensor2 is not None:
        dim2 = tensor2.size(0)
    future_mask = torch.triu(fill_with_neg_inf(torch.ones(dim1, dim2)), 1+abs(dim2-dim1))
    if tensor.is_cuda:
        future_mask = future_mask.cuda()
    return future_mask[:dim1, :dim2]


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)
    return m


if __name__ == '__main__':
    encoder = TransformerEncoder(300, 4, 2)
    x = torch.tensor(torch.rand(20, 2, 300))
    print(encoder(x).shape)
