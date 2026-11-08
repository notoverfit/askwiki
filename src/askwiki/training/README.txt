-- how to use --
to run the training loop; ensure that you have uv installed and have run `uv sync` to instally any required packages for the virtual environment.
given that you have set all appropiate virtual environment paths, and have downloaded and cleaned the relevant data:

    uv run python -m src/askwiki/training/train

will start your training process. there are a few nice features about this training loop:
1. it has pretty robust checkpointing. lots of dataclass shenanigans means that the loop runs off of one `State` class, which gives a single source-of-truth for where the training is at.
2. it's highly customisable. the basic config is written in `.yaml`; i detail what you need in the `yaml` below

-- required config params --
a basic setup for the testing config is below

```
config:
  # data paths
  name: 'base'
  train_data: '~/path/to/training/data'
  val_data: '~/path/to/val/data'

  # model config
  seq_len: 512
  vocab_size: 20000
  num_blocks: 12
  n_heads: 12
  emb_dim: 768
  mlp_dim: 3072

  # train config
  device: 'cuda'
  epochs: 5
  optimiser: 'AdamW'
  betas: [0.9, 0.95]
  batch_size: 8
  learning_rate: 0.0003
  weight_decay: 0.01
  accum_steps: 8

  # optional: start from another run's weights (e.g. sft on a pretrained net).
  # only the model weights are loaded; optimiser and progress start fresh.
  # model config above must match the base model. leave blank to train from scratch.
  base_model: '~/path/to/checkpoints/base_best.pt'
```