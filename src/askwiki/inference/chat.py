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

# device = torch.device('cuda')
# model_data = torch.load('C:/temp/base_20261106.pt')
# tksr = AutoTokenizer.from_pretrained("notoverfit/askwiki", subfolder="tokenizer")
# net = TransformerNet(vocab_size=20_000, n_dim=768, seq_len=512, n_heads=[12] * 12, mlp_dim=[3072] * 12)
# net.load_state_dict(model_data['model'])
# net.to(device)
# print(chat(net, tksr, device, 'The French Revolution began in', stream=True))