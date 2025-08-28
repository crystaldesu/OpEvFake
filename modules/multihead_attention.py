import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

class MultiheadAttention(nn.Module):
    """多头注意力机制实现 (基于Attention Is All You Need论文)"""
    
    def __init__(self, embed_dim, num_heads, attn_dropout=0.,
                 bias=True, add_bias_kv=False, add_zero_attn=False):
        """
        初始化多头注意力层
        
        参数:
            embed_dim: 嵌入维度
            num_heads: 注意力头数量
            attn_dropout: 注意力dropout概率
            bias: 是否使用偏置
            add_bias_kv: 是否为key/value添加偏置
            add_zero_attn: 是否添加零注意力
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        # 计算每个头的维度
        self.head_dim = embed_dim // num_heads
        # 确保嵌入维度能被头数整除
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5  # 缩放因子

        # 投影权重矩阵 (Q,K,V组合)
        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        # 注册偏置参数
        self.register_parameter('in_proj_bias', None)
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        # 输出投影层
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # 可选偏置
        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn  # 是否添加零注意力标志

        self.reset_parameters()  # 参数初始化

    def reset_parameters(self):
        """参数初始化 - Xavier均匀初始化"""
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, attn_mask=None):
        """
        前向传播
        
        参数:
            query: 查询向量 [tgt_len, batch, embed_dim]
            key: 键向量 [src_len, batch, embed_dim]
            value: 值向量 [src_len, batch, embed_dim]
            attn_mask: 注意力掩码 [tgt_len, src_len]
            
        返回:
            attn: 注意力输出 [tgt_len, batch, embed_dim]
            attn_weights: 注意力权重 [batch, num_heads, tgt_len, src_len]
        """
        # 检查输入是否共享数据指针
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()  # 获取目标长度、批次大小和嵌入维度
        # 维度验证
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        aved_state = None  # 未使用

        # 投影处理
        if qkv_same:
            # 自注意力: 共享QKV
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # 编码器-解码器注意力
            q = self.in_proj_q(query)
            if key is None:
                # 如果key为空则value也应为空
                assert value is None
                k = v = None
            else:
                k, v = self.in_proj_kv(key)
        else:
            # 独立投影
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q *= self.scaling  # 应用缩放

        # 添加可选的偏置
        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                # 扩展注意力掩码
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        # 重塑形状以进行多头计算
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)  # 源长度

        # 添加零注意力
        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
        
        # 计算注意力权重
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        # 验证注意力权重形状
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        # 应用注意力掩码
        if attn_mask is not None:
            try:
                attn_weights += attn_mask.unsqueeze(0)
            except:
                # 形状不匹配时打印错误信息
                print(attn_weights.shape)
                print(attn_mask.unsqueeze(0).shape)
                assert False
                
        # 注意力权重归一化
        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        # 可选的其他归一化方法
        # attn_weights = F.relu(attn_weights)
        # attn_weights = attn_weights / torch.max(attn_weights)
        # 应用注意力dropout
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)

        # 注意力输出计算
        attn = torch.bmm(attn_weights, v)
        # 验证输出形状
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        # 重塑输出形状
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)  # 输出投影

        # 平均注意力权重
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights.sum(dim=1) / self.num_heads
        return attn, attn_weights

    # 以下为投影方法
    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query, **kwargs):
        return self._in_proj(query, end=self.embed_dim, **kwargs)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None, **kwargs):
        """通用投影方法"""
        weight = kwargs.get('weight', self.in_proj_weight)
        bias = kwargs.get('bias', self.in_proj_bias)
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)