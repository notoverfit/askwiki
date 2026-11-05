import torch

from typing import List, Any, Dict
from dataclasses import dataclass, field, asdict
from askwiki.data.classes.token import TokenSequence
from askwiki.training.config import Config

@dataclass
class State:
    net: torch.nn.Module
    optimiser: torch.optim.Optimizer
    scaler: torch.amp.GradScaler
    loss_fun: torch.nn.CrossEntropyLoss
    device: torch.device
    train_data: TokenSequence
    val_data: TokenSequence
    config: Config
    epoch: int = 0
    # next batch to run, aligned to the last optimiser step
    batch_idx: int = 0
    best_val_loss: float = float("inf")
    loss_hist: List[Any] = field(default_factory=list)

    def state_dict(self) -> Dict[str, Any]:
        return {
            'config': asdict(self.config),
            'model': self.net.state_dict(),
            'optimiser': self.optimiser.state_dict(),
            'scaler': self.scaler.state_dict(),
            'progress': {
                'epoch': self.epoch,
                'batch_idx': self.batch_idx,
                'best_val_loss': self.best_val_loss,
            },
            'loss_hist': list(self.loss_hist),
        }

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        # net before optimiser: optimiser state is matched to parameters by position
        self.net.load_state_dict(d['model'])
        self.optimiser.load_state_dict(d['optimiser'])
        self.scaler.load_state_dict(d['scaler'])

        progress = d['progress']
        self.epoch = progress['epoch']
        self.batch_idx = progress['batch_idx']
        self.best_val_loss = progress['best_val_loss']
        self.loss_hist = list(d['loss_hist'])