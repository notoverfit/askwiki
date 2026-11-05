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
from_checkpoint = True

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

def save_checkpoint(state: State, kind: str = 'latest') -> None:
    path = CHECKPOINT_DIR / f'{state.config.name}_{kind}.pt'
    tmp_path = path.with_name(path.name + '.tmp')

    torch.save(state.state_dict(), tmp_path)
    os.replace(tmp_path, path)

def load_checkpoint(state: State, config: Config) -> State:
    path = f'{CHECKPOINT_DIR}/{config.name}_latest.pt'
    if not os.path.exists(path):
        raise ValueError(f'checkpoint for config {config.name} does not exist at {path}.')

    state_dict = torch.load(path, map_location=state.device)
    state.load_state_dict(state_dict)

@torch.no_grad()
def validate(state: State, loader: DataLoader) -> float:
    state.net.eval()
    total_loss, total_tokens = 0.0, 0
    for X_val, y_val in loader:
        X_val, y_val = X_val.to(state.device), y_val.to(state.device)
        with torch.autocast(device_type=state.device.type, dtype=torch.float16):
            logits = state.net(X_val)
            loss = state.loss_fn(logits.reshape(-1, logits.size(-1)), y_val.reshape(-1))
        total_loss += loss.item() * y_val.numel()
        total_tokens += y_val.numel()
    state.net.train()
    return total_loss / total_tokens

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
        # sampling which allows the training step to safely skip samples for the initial epoch of a checkpoint that has been continued
        # sample takes a subsample of a given dataset from the given range. batch_idx should be reset to 0 after every epoch so is safe for runs
        start_sample = state.batch_idx * config.batch_size
        remaining = Subset(state.train_data, range(start_sample, len(state.train_data)))
        loader = DataLoader(remaining, config.batch_size)

        # training with scaling works by; scaled(logits + loss) to unscale grads to normalise grad to backprop
        for i, (X_tr, y_tr) in enumerate(loader, start=state.batch_idx + 1):
            X_tr, y_tr = X_tr.to(state.device), y_tr.to(state.device)

            # automatic mixed precision. so notes for my learning sake
            # by default, tensors are stored in fp32 and thus operations on tensors are done in fp32.
            # fp16 by nature is half the size (bits) of fp32. gpus can compute fp16 operations at
            # much high throughput and speed than fp16; more than the 2x drop in precision
            # wrapping the loss in autocast does not store the tensors in fp32, rather, the tensors
            # are copied into fp16 temporarily for matrix operations.
            with torch.autocast(device_type=state.device.type, dtype=torch.float16):
                logits = state.net(X_tr)
                loss = state.loss_fn(logits.reshape(-1, logits.size(-1)), y_tr.reshape(-1))

            # average the loss by the accumulation steps
            state.scaler.scale(loss / accum_steps).backward()

            if i % accum_steps == 0 or i == len(loader):
                # scalers scale up the tensor magnitudes as the loss in precision can cause tensors
                # to become 0, that have information in them. before we propagate the weight updates,
                # we need to unscale them to their original precision.
                state.scaler.unscale_(state.optimiser)

                # gradient normalisation tries to solve the exploding gradient problem. it ensures that
                # the sum of the l1 norms of the gradients sum to 1
                grad_norm = nn.utils.clip_grad_norm_(state.net.parameters(), max_norm=1)

                # update the actual weights of the network
                state.scaler.step(state.optimiser)

                # the amp scaler doesn't scale at a fixed factor, but rather starts at a base level and then
                # increases and decreases as required. since fp16 can only hold a set range of values, 
                # if the float overflows, it becomes represented as infinity. if infinity is detected, then
                # the scaler "backs off" by 0.5x on default. otherwise, the scaler attempts to keep growing
                # the scale to ensure that it preserves as many digits of precision as possible.
                state.scaler.update()
                state.optimiser.zero_grad(set_to_none=True)

                # grad_accum steps done, so we can checkpoint from here
                # step is defined by the number of accum steps we have gone through
                state.batch_idx = i
                step = i // accum_steps

                running_count.append(loss.item())
                running_avg = float(np.mean(running_count))
                state.loss_hist.append({'epoch': epoch, 'step': step, 'loss': running_avg, 'grad_norm': grad_norm.item()})

                if step % print_every == 0: print(f'epoch {epoch}\t| step {step}\t| loss {running_avg:.4f}\t| grad {grad_norm.item():.4f}')

        val_loss = validate(state, ValLoader)
        print(f'epoch {epoch} done\t|\tval loss {val_loss:.4f}')

        state.epoch += 1
        state.batch_idx = 0

        if val_loss < state.best_val_loss:
            state.best_val_loss = val_loss
            save_checkpoint(state, 'best')
        save_checkpoint(state)

except KeyboardInterrupt:
    print('Exiting training, saving checkpoint...')
    save_checkpoint(state)
except Exception:
    save_checkpoint(state, 'crash')
    raise
