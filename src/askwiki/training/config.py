from dataclasses import dataclass

@dataclass
class Config:
    # data paths
    name: str
    train_data: str
    val_data: str
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
    batch_size: int
    learning_rate: float
    weight_decay: float