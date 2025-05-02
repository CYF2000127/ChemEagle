import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, pre_kv=None, attn_mask=None):
        N, B, C = x.shape
        qkv = self.qkv(x).reshape(N, B, 3, self.num_heads, C // self.num_heads).permute(2, 1, 3, 0, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        if not self.training:
            k = torch.cat([pre_kv[0], k], dim=2)
            v = torch.cat([pre_kv[1], v], dim=2)
            pre_kv = torch.stack([k, v], dim=0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            attn.masked_fill_(attn_mask, float('-inf'))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(2, 0, 1, 3).reshape(N, B, C)
        x = self.proj(x)
        return x, pre_kv
