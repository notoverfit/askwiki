import torch
import torch.nn as nn
from src.askwiki.model.modules.attention import MultiHeadAttentionDecoder

class TransformerBlock(nn.Module):
    def __init__(self, n_dim: int, n_heads: int, mlp_dim: int):
        super().__init__()
        if n_dim <= 0 or n_heads <= 0: raise ValueError('MultiHeadAttention: one or more parameters passed in is negative.')
        elif n_dim % n_heads != 0: raise ValueError('MultiHeadAttention: number of heads is not divisible by the embedding dimension')

        self.attn = MultiHeadAttentionDecoder(n_dim, n_heads)
        self.ln1 = nn.LayerNorm(n_dim)
        self.mlp = nn.Sequential(nn.Linear(n_dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, n_dim))
        self.ln2 = nn.LayerNorm(n_dim)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x