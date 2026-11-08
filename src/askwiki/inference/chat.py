import torch
import torch.nn as nn

from askwiki.model.nets.net import TransformerNet
from transformers import AutoTokenizer

@torch.no_grad()
def chat(
    net: nn.Module,
    tokeniser: AutoTokenizer,
    device: torch.device,
    prompt: str,
    temperature: float = 1.0,
    stream: bool = False,
    max_new_tokens: int = 1024,
) -> str:
    net.eval()
    tks = torch.tensor(tokeniser.encode(prompt)).unsqueeze(0).to(device)
    new_tks = []
    printed = ""

    if stream: print(prompt, end="", flush=True)

    for _ in range(max_new_tokens):
        lgts = net(tks[:, -512:])[0, -1] / temperature
        prediction = torch.multinomial(torch.softmax(lgts, dim=-1), 1)
        if prediction.item() == tokeniser.eos_token_id: break
        new_tks.append(prediction.item())
        tks = torch.cat([tks, prediction.view(1, 1)], dim=1)

        if stream:
            text = tokeniser.decode(new_tks)
            print(text[len(printed):], end="", flush=True)
            printed = text

    return tokeniser.decode(new_tks)

@torch.no_grad()
def chat_v2(
    net: nn.Module,
    tokeniser: AutoTokenizer,
    device: torch.device,
    prompt: str,
    temperature: float = 1.0,
    stream: bool = False,
    max_new_tokens: int = 1024,
) -> str:
    # chat version for the sft model: wraps the prompt in the sft training format
    # ('user: {q} assistant: {a}' + eos). no trailing space after 'assistant:' so the
    # model produces the ' answer' token itself, matching how training text was tokenised.
    net.eval()
    chat_prompt = f'user: {prompt} assistant:'
    tks = torch.tensor(tokeniser.encode(chat_prompt)).unsqueeze(0).to(device)
    new_tks = []
    printed = ""

    if stream: print(chat_prompt, end="", flush=True)

    for _ in range(max_new_tokens):
        lgts = net(tks[:, -512:])[0, -1] / temperature
        prediction = torch.multinomial(torch.softmax(lgts, dim=-1), 1)
        # sft samples end with eos, so this is where the answer stops
        if prediction.item() == tokeniser.eos_token_id: break
        new_tks.append(prediction.item())
        tks = torch.cat([tks, prediction.view(1, 1)], dim=1)

        if stream:
            text = tokeniser.decode(new_tks)
            print(text[len(printed):], end="", flush=True)
            printed = text

    if stream: print()
    return tokeniser.decode(new_tks).strip()
