import torch
import torch.nn as nn
from typing import List
from askwiki.model.modules.block import TransformerBlock

class TransformerNet(nn.Module):
    def __init__(self, vocab_size: int, n_dim: int, seq_len: int, n_heads: List[int], mlp_dim: List[int]):
        super().__init__()
        if len(n_heads) != len(mlp_dim): raise ValueError('TransformerNet: the number of values given in n_heads does not equal the number of values in mlp_dim.')

        self.seq_len    = seq_len
        # token and position embeddings. absolute positional embedding
        self.tok_emb    = nn.Embedding(vocab_size, n_dim)
        self.pos_emb    = nn.Embedding(seq_len, n_dim)
        self.blocks     = nn.Sequential(*[ TransformerBlock(n_dim, n_heads[i], mlp_dim[i]) for i in range(len(n_heads)) ])
        # final layer normalisation
        self.ln         = nn.LayerNorm(n_dim)
        # cast vector embedding dimension to vocab size for token prediction
        self.head       = nn.Linear(n_dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x will batches x tokens passed in
        _, T = x.shape
        if T > self.seq_len: raise ValueError(f'Sequence length {T} exceeds max_seq_len {self.seq_len}')
        # arange creates a range vector [0, T)
        pos = torch.arange(T, device=x.device)
        x = self.tok_emb(x) + self.pos_emb(pos)
        x = self.blocks(x)
        x = self.ln(x)
        return self.head(x)