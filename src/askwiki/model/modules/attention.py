import torch
import numpy as np
import torch.nn as nn

class MultiHeadAttentionDecoder(nn.Module):
    def __init__(self, n_dim: int, n_heads: int):
        super().__init__()
        if n_dim <= 0 or n_heads <= 0:
            raise ValueError('MultiHeadAttention: one or more parameters passed in is negative.')
        elif n_dim % n_heads != 0:
            raise ValueError('MultiHeadAttention: number of heads is not divisible by the embedding dimension')

        # linear projection layers to split later
        self.attn_proj = nn.Linear(n_dim, n_dim * 3)
        self.n_heads = n_heads

        # post-concat mixing of multi-head attention outputs
        self.out_proj = nn.Linear(n_dim, n_dim)

    def forward(self, x: torch.Tensor):
        x_proj: torch.Tensor = self.attn_proj(x)
        q, k, v = x_proj.chunk(3, dim=-1)

        # batches, tokens and embedding dimension
        B, T, C = q.shape
        H = self.n_heads
        D = C // H

        # split the output dimension into heads; and then transpose such that heads is the second dimension
        # this means that matrix multiplications will treat the (batch, head) as the batch dimension
        q = q.view(B, T, H, D).transpose(1, 2)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)

        # get attention scores
        attention = q @ k.transpose(-2, -1)
        attention = attention / np.sqrt(D)

        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)

        # causal mask, then soft max on keys
        attention = attention.masked_fill(mask, float('-inf'))
        attention = attention.softmax(dim=-1)

        out = attention @ v
        # remerge output, then pass through linear output layer
        out = out.transpose(1, 2).reshape(B, T, C)

        return self.out_proj(out)