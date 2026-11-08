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
from dotenv import load_dotenv
from torch.optim import AdamW

from askwiki.training.utils import generate_state, validate
from askwiki.training.state import State
from askwiki.training.config import Config


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

config_path = f'{CONFIG_PATH}/sft.yaml'
from_checkpoint = False

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

def load_base_model(state: State, path: str) -> None:
    # weights only; optimiser, scaler and progress stay fresh for the new dataset
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise ValueError(f'base model checkpoint does not exist at {path}.')

    state_dict = torch.load(path, map_location=state.device)
    state.net.load_state_dict(state_dict['model'])
    print(f'initialised network weights from base model at {path}')

# train loop
state = generate_state(config, OPTIMISER_MAP)
# resuming a run takes precedence: its checkpoint already holds the fine-tuned weights
if from_checkpoint: load_checkpoint(state, config)
elif config.base_model: load_base_model(state, config.base_model)

ValLoader = DataLoader(state.val_data, config.batch_size) if state.val_data is not None else None
if ValLoader is None: print('no val_data given, skipping validation')

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

        # for training only runs
        if ValLoader is not None:
            val_loss = validate(state, ValLoader)
            print(f'epoch {epoch} done\t|\tval loss {val_loss:.4f}')
        else:
            val_loss = None
            print(f'epoch {epoch} done')

        state.epoch += 1
        state.batch_idx = 0

        if val_loss is not None and val_loss < state.best_val_loss:
            state.best_val_loss = val_loss
            save_checkpoint(state, 'best')
        save_checkpoint(state)

except KeyboardInterrupt:
    print('Exiting training, saving checkpoint...')
    save_checkpoint(state)
except Exception:
    save_checkpoint(state, 'crash')
    raise
