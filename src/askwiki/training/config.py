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
    # lora config
    lora_r: Optional[int]
    lora_alpha: Optional[int]
    lora_targets: Optional[List[str]]
    # optional path to a checkpoint (.pt) whose model weights initialise the network.
    # only the weights are taken: optimiser, scaler and epoch/batch progress start fresh,
    # so the run can use a different dataset (e.g. sft on top of a pretrained net).
    # empty value ('' or blank) trains from scratch.
    base_model: Optional[str] = None