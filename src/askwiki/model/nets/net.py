import torch
import torch.nn as nn
from typing import List
from src.askwiki.model.modules.block import TransformerBlock

class TransformerNet(nn.Module):
    def __init__(self, n_dim: int, n_heads: List[int], mlp_dim: List[int]):
        super().__init__()
        if len(n_heads) != len(mlp_dim): raise ValueError('TransformerNet: the number of values given in n_heads does not equal the number of values in mlp_dim.')
        self.blocks = [ TransformerBlock(n_dim, n_heads[i], mlp_dim[i]) for i in range(len(n_heads)) ]

    def forward(self, x: torch.Tensor):
        for block in self.blocks: x = block(x)
        return x