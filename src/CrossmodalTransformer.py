import torch
import sys
from torch import nn
import torch.nn.functional as F

# 导入Transformer编码器模块
from modules.transformer import TransformerEncoder

class MULTModel(nn.Module):
    """多模态Transformer模型 (MulT架构)"""
    
    def __init__(self, orig_d_l, orig_d_a, orig_d_v, MULT_d, mult_dropout):
        """
        初始化MulT模型
        
        参数:
            orig_d_l: 原始文本维度
            orig_d_a: 原始音频维度
            orig_d_v: 原始视觉维度
            MULT_d: 统一特征维度
            mult_dropout: dropout概率
        """
        super(MULTModel, self).__init__()
        # 存储原始维度
        self.orig_d_l, self.orig_d_a, self.orig_d_v = orig_d_l, orig_d_a, orig_d_v
        # 统一特征维度
        self.d_l, self.d_a, self.d_v = MULT_d, MULT_d, MULT_d
        # 模型配置
        self.num_heads = 2  # 注意力头数
        self.layers = 5  # Transformer层数
        self.attn_dropout = 0.1  # 注意力dropout
        self.attn_dropout_a = 0.0  # 音频注意力dropout
        self.attn_dropout_v = 0.0  # 视觉注意力dropout
        self.relu_dropout = 0.1  # ReLU dropout
        self.res_dropout = 0.1  # 残差dropout
        self.out_dropout = mult_dropout  # 输出dropout
        self.embed_dropout = 0.25  # 嵌入dropout
        self.attn_mask = True  # 使用注意力掩码

        # ========== 1. 时间卷积层 ==========
        # 文本投影
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        # 音频投影
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        # 视觉投影
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)
        # GPT描述投影
        self.proj_g = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)

        # ========== 2. 跨模态注意力层 ==========
        # 文本-音频注意力
        self.trans_l_with_a = self.get_network(self_type='la')
        # 音频-文本注意力
        self.trans_a_with_l = self.get_network(self_type='al')
        # 文本-视觉注意力
        self.trans_l_with_v = self.get_network(self_type='lv')
        # 视觉-文本注意力
        self.trans_v_with_l = self.get_network(self_type='vl')
        # 音频-视觉注意力
        self.trans_a_with_v = self.get_network(self_type='av')
        # 视觉-音频注意力
        self.trans_v_with_a = self.get_network(self_type='va')
        # GPT-文本注意力
        self.trans_g_with_l = self.get_network(self_type='l')
        # GPT-音频注意力
        self.trans_g_with_a = self.get_network(self_type='la')
        # GPT-视觉注意力
        self.trans_g_with_v = self.get_network(self_type='lv')
        # 文本-GPT注意力
        self.trans_l_with_g = self.get_network(self_type='l')
        # 音频-GPT注意力
        self.trans_a_with_g = self.get_network(self_type='la')
        # 视觉-GPT注意力
        self.trans_v_with_g = self.get_network(self_type='lv')

    def get_network(self, self_type='l', layers=-1):
        """
        获取Transformer编码器网络
        
        参数:
            self_type: 模态类型标识
            layers: 层数
            
        返回:
            TransformerEncoder实例
        """
        # 确定嵌入维度和dropout
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v

        # 创建Transformer编码器
        return TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=self.num_heads,
            layers=max(self.layers, layers),
            attn_dropout=attn_dropout,
            relu_dropout=self.relu_dropout,
            res_dropout=self.res_dropout,
            embed_dropout=self.embed_dropout,
            attn_mask=self.attn_mask,
            position_emb=True)  # 使用位置嵌入

    def forward(self, x_l, x_g, x_a, x_v):
        """
        前向传播
        
        参数:
            x_l: 文本特征 [batch_size, seq_len, orig_d_l]
            x_g: GPT特征 [batch_size, seq_len, orig_d_l]
            x_a: 音频特征 [batch_size, seq_len, orig_d_a]
            x_v: 视觉特征 [batch_size, seq_len, orig_d_v]
            
        返回:
            h_ls: 增强文本特征 [seq_len, batch_size, d_l*3]
            h_gs: 增强GPT特征 [seq_len, batch_size, d_l*3]
            h_as: 增强音频特征 [seq_len, batch_size, d_a*3]
            h_vs: 增强视觉特征 [seq_len, batch_size, d_v*3]
        """
        # 应用嵌入dropout并调整维度
        x_l = F.dropout(x_l.transpose(2, 1), p=self.embed_dropout, training=self.training)
        x_g = x_g.transpose(2, 1)
        x_a = x_a.transpose(2, 1)
        x_v = x_v.transpose(2, 1)

        # 特征投影
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_g = x_g if self.orig_d_l == self.d_l else self.proj_g(x_g)
        
        # 调整维度顺序: [seq_len, batch_size, dim]
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_g = proj_x_g.permute(2, 0, 1)

        # ========== 跨模态注意力处理 ==========
        # (A,V,G) -> L
        h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)  # 文本-音频
        h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)  # 文本-视觉
        h_l_with_gs = self.trans_l_with_g(proj_x_l, proj_x_g, proj_x_g)  # 文本-GPT
        # 连接结果并应用dropout
        h_ls = F.dropout(torch.cat([h_l_with_as, h_l_with_vs, h_l_with_gs], dim=2), 
                         p=self.out_dropout, training=self.training)

        # (L,V,G) -> A
        h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)  # 音频-文本
        h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)  # 音频-视觉
        h_a_with_gs = self.trans_a_with_g(proj_x_a, proj_x_g, proj_x_g)  # 音频-GPT
        h_as = F.dropout(torch.cat([h_a_with_ls, h_a_with_vs, h_a_with_gs], dim=2), 
                        p=self.out_dropout, training=self.training)

        # (L,A,G) -> V
        h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)  # 视觉-文本
        h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)  # 视觉-音频
        h_v_with_gs = self.trans_v_with_g(proj_x_v, proj_x_g, proj_x_g)  # 视觉-GPT
        h_vs = F.dropout(torch.cat([h_v_with_ls, h_v_with_as, h_v_with_gs], dim=2), 
                        p=self.out_dropout, training=self.training)

        # (L,A,V) -> G
        h_g_with_ls = self.trans_g_with_l(proj_x_g, proj_x_l, proj_x_l)  # GPT-文本
        h_g_with_as = self.trans_g_with_a(proj_x_g, proj_x_a, proj_x_a)  # GPT-音频
        h_g_with_vs = self.trans_g_with_v(proj_x_g, proj_x_g, proj_x_g)  # GPT-视觉
        h_gs = F.dropout(torch.cat([h_g_with_ls, h_g_with_as, h_g_with_vs], dim=2), 
                        p=self.out_dropout, training=self.training)

        return h_ls, h_gs, h_as, h_vs