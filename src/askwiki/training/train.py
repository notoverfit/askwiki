"""
an ideally robust training loop for the transformer network.
supports quitting at any time with a checkpoint.
"""

import os
import yaml
import torch
import numpy as np
import torch.nn as nn

from datetime import datetime
from collections import deque
from pathlib import Path
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
CHECKPOINT_DIR = Path(CHECKPOINT_PATH)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

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

# all the stuff you need for training
Optim = OPTIMISER_MAP[config.optimiser]
Net = TransformerNet(vocab_size=config.vocab_size, n_dim=config.emb_dim, seq_len=config.seq_len, n_heads=[config.n_heads] * config.num_blocks, mlp_dim=[config.mlp_dim] * config.num_blocks).to(device)
Optimiser = Optim(Net.parameters(), lr=config.learning_rate, betas=config.betas, weight_decay=config.weight_decay)
Scaler = torch.amp.GradScaler(config.device)
Loss = nn.CrossEntropyLoss()

TrainData = TokenSequence(config.train_data, dtype=np.uint16, seq_len=config.seq_len)
ValData = TokenSequence(config.val_data, dtype=np.uint16, seq_len=config.seq_len)
TrainLoader = DataLoader(TrainData, config.batch_size)
ValLoader = DataLoader(ValData, config.batch_size)

Net.train()
loss_hist = []
best_val_loss = float('inf')
today = datetime.now()
print_every = 10
accum_steps = config.accum_steps
running_count = deque(maxlen=100)

def save_checkpoint(epoch):
    dt_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    torch.save({
        'epoch': epoch,
        'model': Net.state_dict(),
        'optimiser': Optimiser.state_dict(),
        'config': vars(config)
    }, CHECKPOINT_DIR / f'{config.name}_{dt_str}.pt')

@torch.no_grad()
def validate():
    Net.eval()
    total_loss, total_tokens = 0.0, 0
    for X_val, y_val in ValLoader:
        X_val, y_val = X_val.to(device), y_val.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            logits = Net(X_val)
            loss = Loss(logits.reshape(-1, config.vocab_size), y_val.reshape(-1))
        total_loss += loss.item() * y_val.numel()
        total_tokens += y_val.numel()
    Net.train()
    return total_loss / total_tokens

epoch = 0
try:
    for epoch in range(config.epochs):
        for i, (X_tr, y_tr) in enumerate(TrainLoader, start=1):
            X_tr, y_tr = X_tr.to(device), y_tr.to(device)

            with torch.autocast(device_type=config.device, dtype=torch.float16):
                logits = Net(X_tr)
                loss = Loss(logits.reshape(-1, config.vocab_size), y_tr.reshape(-1))

            # average the loss by the accumulation steps
            Scaler.scale(loss / accum_steps).backward()

            if i % accum_steps == 0 or i == len(TrainLoader):
                Scaler.unscale_(Optimiser)
                grad_norm = nn.utils.clip_grad_norm_(Net.parameters(), max_norm=1)
                Scaler.step(Optimiser)
                Scaler.update()

                # now reset gradients
                Optimiser.zero_grad(set_to_none=True)
                step = i // accum_steps
                loss_hist.append({'epoch': i, 'step': step, 'loss': loss.item(), 'grad_norm': grad_norm.item()})

                running_count.append(loss.item())

                if step % print_every == 0:
                    print(f'epoch {epoch} | step {step} | loss {(np.mean(running_count)):.4f} | grad {grad_norm.item():.4f}')

        val_loss = validate()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(epoch)

except KeyboardInterrupt as e:
    print('Exiting training...')
    save_checkpoint(epoch)
except Exception as e:
    save_checkpoint(epoch)
    raise e