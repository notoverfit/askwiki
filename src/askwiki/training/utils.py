import torch
import numpy as np
import torch.nn as nn

from askwiki.training.config import Config
from askwiki.training.state import State
from askwiki.model.nets.net import TransformerNet
from askwiki.data.classes.token import TokenSequence

def generate_state(config: Config, optim_map: dict) -> State:
    device = torch.device(config.device)
    net = TransformerNet(vocab_size=config.vocab_size, n_dim=config.emb_dim, seq_len=config.seq_len, n_heads=[config.n_heads] * config.num_blocks, mlp_dim=[config.mlp_dim] * config.num_blocks).to(device)
    Optim = optim_map[config.optimiser]
    return State(
        net=net,
        optimiser=Optim(net.parameters(), lr=config.learning_rate, betas=config.betas, weight_decay=config.weight_decay),
        scaler=torch.amp.GradScaler(config.device),
        loss_fn=nn.CrossEntropyLoss(),
        train_data=TokenSequence(config.train_data, dtype=np.uint16, seq_len=config.seq_len),
        val_data=TokenSequence(config.val_data, dtype=np.uint16, seq_len=config.seq_len),
        config=config,
        device=device
    )