"""
an ideally robust training loop for the transformer network.
supports quitting at any time with a checkpoint.
"""

import os
import yaml
import torch
import numpy as np
import torch.nn as nn
import pandas as pd

from datetime import datetime
from collections import deque
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from askwiki.model.nets.net import TransformerNet
from askwiki.training.state import State
from askwiki.training.config import Config
from askwiki.data.classes.token import TokenSequence
from dotenv import load_dotenv
from torch.optim import AdamW

load_dotenv()

# global variables
CONFIG_PATH = os.environ['CONFIGPATH']

REQUIRED_CONFIG_FIELDS= [
    'train_data', 'val_data', 'seq_len', 
    'vocab_size', 'num_blocks', 'n_heads',
    'emb_dim', 'mlp_dim', 'epochs', 'scale',
    'optimiser', 'learning_rate', 'weight_decay'
]

OPTIMISER_MAP = {
    'AdamW': AdamW
}

CHECKPOINT_PATH = os.environ['CHECKPOINTPATH']
CHECKPOINT_DIR = Path(CHECKPOINT_PATH)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# load in config and config params

config_path = f'{CONFIG_PATH}/base.yaml'
from_checkpoint=False

try:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    if 'config' not in config:
        raise Exception('config must have `config` parent field.')

    config: Config = Config(**config['config'])
except Exception as e:
    raise ValueError(f'config path was not read in properly. does config path exist? is config set up properly? {e}')

# set up network and training params
if config.optimiser not in OPTIMISER_MAP.keys():
    raise ValueError(f'{config.optimiser} is not a valid optimiser.')

# util functions

def generate_state(config: Config) -> State:
    device = torch.device(config.device)
    net = TransformerNet(vocab_size=config.vocab_size, n_dim=config.emb_dim, seq_len=config.seq_len, n_heads=[config.n_heads] * config.num_blocks, mlp_dim=[config.mlp_dim] * config.num_blocks).to(device)
    Optim = OPTIMISER_MAP[config.optimiser]
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

def save_checkpoint(state: State) -> None:
    state_dict = state.state_dict()
    torch.save(state_dict, f'{CHECKPOINT_DIR}/{state.config.name}_latest.pt')

def load_checkpoint(state: State, config: Config) -> State:
    path = f'{CHECKPOINT_DIR}/{config.name}_latest.pt'
    if not os.path.exists(path):
        raise ValueError(f'checkpoint for config {config.name} does not exist at {path}.')

    state_dict = torch.load(path)
    state.load_state_dict(state_dict)

# train loop

state = generate_state(config)
if from_checkpoint: load_checkpoint(state, config)

ValLoader = DataLoader(state.val_data, config.batch_size)

state.net.train()
print_every = 10
accum_steps = config.accum_steps
running_count = deque(maxlen=100)

try:
    for epoch in range(state.epoch, config.epochs):
        train_subset = DataLoader(Subset(state.train_data, range(state.batch_idx, len(state.train_data))))
        for i, (X_tr, y_tr) in enumerate(train_subset, start=1):
            state.optimiser.zero_grad()
            
    pass
except KeyboardInterrupt:
    pass
finally:
    pass
