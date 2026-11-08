# askwiki

> A ~116M parameter GPT-2-style model pre-trained from scratch on English Wikipedia (test loss 3.07), then full-model SFT for 3 epochs on 25k synthetic Q&A pairs.

askwiki is a ~116M parameter decoder-only transformer that I pre-trained from scratch on English Wikipedia and then fine-tuned to answer short questions. I built it to work through the whole pipeline myself on one consumer GPU with 8GB of VRAM: cleaning the data, training a tokeniser, writing the model, pre-training, generating synthetic SFT data, and fine-tuning.

The trained weights, tokeniser and datasets are on Hugging Face: **[notoverfit/askwiki](https://huggingface.co/notoverfit/askwiki)**.


---

## Repository layout

```
src/askwiki/
├── data/            # text cleaning, tokeniser training, binary token dataset
├── model/           # attention, transformer block, full net
├── training/        # config-driven training loop with checkpointing (see training/README.txt)
├── inference/       # sampling helpers: chat (base model) and chat_v2 (SFT format)
└── research/
    ├── pre_training/notebooks/
    │   ├── 01_model_and_training_decisions.ipynb
    │   └── 02_pre_training_results.ipynb
    └── post_training/notebooks/
        ├── 01_generating_q_and_a_responses.ipynb
        └── 02_post_training_results.ipynb
outputs/             # figures referenced below
references/          # papers cited in the notebooks
```

---

## 1. Data and tokeniser

The corpus is an English Wikipedia dump, distributed as parquet shards. Raw articles contain a lot of noise: empty parentheses left behind by stripped pronunciation guides (`Albedo (; ) is ...`), trailing *See also* / *References* / *External links* sections, leftover template and HTML fragments, and odd Unicode whitespace. `src/askwiki/data/utils/clean.py` removes these with a set of regex passes before tokenisation.

I trained a BPE tokeniser with a **20,000-token vocabulary** on a hash-based 20% sample of the training split and pushed it to the Hugging Face repo under `tokenizer/`. Each split is then tokenised and written to a flat `uint16` `.bin` file. The `TokenSequence` dataset reads that file in fixed-length windows, so the data loader never has to hold the corpus in memory.

## 2. Model and training decisions

*Notebook: [`pre_training/01_model_and_training_decisions.ipynb`](src/askwiki/research/pre_training/notebooks/01_model_and_training_decisions.ipynb)*

To save compute on hyperparameter search I took most architectural choices from GPT-2 small:

| | |
|---|---|
| Embedding dim | 768 |
| Blocks | 12 |
| Heads | 12 (head dim 64) |
| MLP dim | 3072 (4×) |
| Context length | 512 |
| Positional encoding | learned, absolute |
| Norm placement | pre-norm |
| Parameters | ~116M (≈85M in blocks, the rest in untied input/output embeddings) |

I didn't tune the number of heads. Michel & Levy (2019) found that many heads can be pruned at test time with little loss in quality, so I just used the standard 64-dim head. Each block adds about $12d^2 \approx 7.1\text{M}$ parameters ($4d^2$ for attention, $8d^2$ for the MLP), which is how I ended up at 12 blocks for a ~100M target.

The main empirical question was how large a batch fits in 8GB. I profiled one forward and backward pass under bf16 autocast:

| Batch | Time | Peak VRAM |
|---:|---:|---:|
| 2 | 46 ms | 1.74 GB |
| 4 | 88 ms | 2.78 GB |
| 8 | 202 ms | 4.87 GB |
| 16 | 1,661 ms | 9.05 GB |
| 32 | 16,046 ms | 17.39 GB |

From 4 to 8 the time roughly doubles, which is what you'd expect. From 8 to 16 it grows ~8×, which shows memory spilling off the card. I settled on a micro-batch of 8 with 8 gradient accumulation steps, for an effective batch of 64 sequences (~32k tokens). The optimiser is AdamW with lr 3e-4, betas (0.9, 0.95) and weight decay 0.01 (`training/configs/base.yaml`).

## 3. Pre-training results

*Notebook: [`pre_training/02_pre_training_results.ipynb`](src/askwiki/research/pre_training/notebooks/02_pre_training_results.ipynb)*

Pre-training ran for about three days (~96k steps). I stopped it once the loss stopped improving.

![Pre-training loss](outputs/pre_loss_hist.png)

![Pre-training gradient norm](outputs/pre_grad_norm.png)

Loss falls quickly from ~10 to ~3.5 in the first 10k steps, then slowly flattens to just above 3. Over the run the rolling gradient norm drifts up from ~0.55 to ~0.7, but it stays within a narrow range.

The odd part is around step 60k. Loss briefly drops below 2 while the gradient norm first spikes and then collapses. My best guess is a run of duplicated or degenerate text in the data that the model could briefly memorise. The run recovers immediately afterwards, but I'd like to find the exact shard.

| | Loss |
|---|---:|
| Train (mean of last 1,000 steps) | 3.10 |
| Test | **3.07** |

Train and test loss are almost the same, so the model isn't overfitting. At this size it is, if anything, capacity-limited. Qualitatively, samples are grammatical and clearly *sound* like Wikipedia: declarative, encyclopaedic, full of dates and proper nouns. But they drift off-topic within a sentence or two and mix up facts freely. At lower temperatures the model also tends to get stuck repeating a phrase. None of this is surprising for a model with ~100M parameters.

## 4. Generating SFT data

*Notebook: [`post_training/01_generating_q_and_a_responses.ipynb`](src/askwiki/research/post_training/notebooks/01_generating_q_and_a_responses.ipynb)*

The base model can only continue text. To make it answer questions, I generated a synthetic Q&A set with a larger model:

- I sampled **25,000 articles** from the training split, truncated each to ~6,000 characters at a sentence or paragraph boundary, and asked **Qwen3.5 4B** (via Ollama, thinking disabled for throughput) to write one question and answer pair grounded in the passage.
- Responses were constrained to a Pydantic schema (`{user, assistant}`) through Ollama's structured outputs, so every successful call parses.
- Each pair is formatted as `user: <question> assistant: <answer><eos>` and tokenised into `sft_tr.bin`.

The source passage is intentionally **not** included in the training text. With the passage in context this would turn into RAG training. The aim here is to teach the model the *format* of answering a question, and to have it rely on whatever it learned during pre-training.

## 5. Post-training results

*Notebook: [`post_training/02_post_training_results.ipynb`](src/askwiki/research/post_training/notebooks/02_post_training_results.ipynb)*

I fine-tuned the full model from the pre-trained checkpoint for 3 epochs on the SFT set (43 optimiser steps per epoch), using the same optimiser settings as pre-training (`training/configs/sft.yaml`). There was no held-out validation split for this stage.

![SFT loss](outputs/sft_loss_hist.png)

![SFT gradient norm](outputs/sft_grad_norm.png)

Loss goes from ~3.6 to ~2.0 and is still falling at the end. There are visible steps at the epoch boundaries (~43 and ~86). Combined with how small the dataset is, I read this as the model starting to memorise the SFT set by epoch 3. Gradient norms settle quickly at ~0.5, with small bumps at each new epoch.

Here is a comparison on prompts the model never saw. The base model gets the same `user: ... assistant:` prefix but has never seen that format. I sampled the base model at T=1.0 because it falls into repetition loops at lower temperatures; the SFT model is at T=0.5.

| Prompt | Base model (truncated) | SFT model |
|---|---|---|
| What is the capital of France? | *René Corot, not a minister. In 1998 in Legio XII, he shouted to his press…* | Paris. |
| Who was Isaac Newton? | *Anosyntaph. Toddrai Watts: Sext fighting: Dartfield thanAR…* | Isaac Newton (also spelled Isaac Newton) was an English priest who was born in 1653. |
| What is the largest ocean on Earth? | *So it is when we change things. Keyboards Medical Navck Scientists use…* | The largest ocean on Earth is the North Atlantic Ocean. |
| Which country is the Great Wall located in? | *Love a soul. The journey, sat near, is unmistakably composed of…* | Great Wall is located in the Czech Republic. |
| Who painted the Mona Lisa? | *ANCY! "PDistribution Return Painting" by Krenew…* | The painting was created by artist Mary Beth Williams. |
| Where is Mount Everest? | *Res1989. Caintance of the Person.Withnote: George VI…* | Mount Everest is a mountain in southern California, United States… |

The change in *behaviour* is large given how little data it took: about 130 optimiser steps moved the model from free-associating text to giving short, direct, well-formed answers that stop at EOS. The *content* is a different story. Apart from "Paris", the answers are confidently wrong. Several are close in some sense ("English", "ocean", the 1600s for Newton), which fits a model that has picked up topical associations but not reliable facts. This is the familiar LLM failure of sounding confident while being wrong, just very obvious at this scale.

## Reproducing

```bash
uv sync
# set DATAPATH, CHECKPOINTPATH and CONFIGPATH in .env
uv run python -m askwiki.training.train   # config file is selected in train.py
```

See `src/askwiki/training/README.txt` for the config format. Pre-built token files, the SFT data and both checkpoints can be downloaded from [notoverfit/askwiki](https://huggingface.co/notoverfit/askwiki) if you'd rather skip straight to the results notebooks.

## References

- Vaswani et al. (2017). *Attention Is All You Need.*
- Michel, Levy & Neubig (2019). *Are Sixteen Heads Really Better than One?*
- Radford et al. (2019). *Language Models are Unsupervised Multitask Learners* (GPT-2).
- Hoffmann et al. (2022). *Training Compute-Optimal Large Language Models* (Chinchilla).
