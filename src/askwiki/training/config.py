from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Config:
    # data paths
    name: str
    train_data: str
    # empty value ('' or blank) skips validation
    val_data: Optional[str]
    # model config
    seq_len: int
    vocab_size: int
    num_blocks: int
    n_heads: int
    emb_dim: int
    mlp_dim: int
    # training config
    device: str
    epochs: int
    optimiser: str
    betas: List[float, float]
    batch_size: int
    learning_rate: float
    weight_decay: float
    accum_steps: int