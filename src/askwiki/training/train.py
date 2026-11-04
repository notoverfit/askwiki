"""
an ideally robust training loop for the transformer network.
supports quitting at any time with a checkpoint.
"""

import os
import yaml
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from askwiki.model.nets.net import TransformerNet
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

# load in config and config params

config_path = f'{CONFIG_PATH}/base.yaml'

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

device = torch.device(config.device)

Optim = OPTIMISER_MAP[config.optimiser]
Net = TransformerNet(vocab_size=config.vocab_size, n_dim=config.emb_dim, seq_len=config.seq_len, n_heads=[config.n_heads] * config.num_blocks, mlp_dim=[config.mlp_dim] * config.num_blocks).to(device)
Loss = nn.CrossEntropyLoss()
Optimiser = Optim(Net.parameters(), lr=config.learning_rate, weight_decay=config.learning_rate)
TrainData = TokenSequence(config.train_data, dtype=np.uint16, seq_len=config.seq_len)
ValData = TokenSequence(config.val_data, dtype=np.uint16, seq_len=config.seq_len)
TrainLoader = 

Net.train()
loss_hist = []

try:
    for epoch in range(config.epochs):
        pass
except Exception as e:
    raise e
finally:
    pass