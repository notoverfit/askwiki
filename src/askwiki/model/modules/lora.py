import math, torch, torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, r: int, alpha: int):
        super().__init__()

        self.linear = linear
        # we do not want to update the weights of the existing linear layer
        for p in self.linear.parameters(): p.requires_grad = False
        self.r, self.scaler = r, alpha / r
        self.A = nn.Parameter(torch.empty(r, self.linear.in_features, device=self.linear.weight.device, dtype=self.linear.weight.dtype))
        # B being full of zeroes is important as it means the training starts from the original model
        self.B = nn.Parameter(torch.zeros(self.linear.out_features, r, device=self.linear.weight.device, dtype=self.linear.weight.dtype))

        # i couldc all kaiming uniform here, but for calrity
        bound = 1 / math.sqrt(self.linear.in_features)
        nn.init.uniform_(self.A, -bound, bound)

    def forward(self, x: torch.Tensor):
        linear_out = self.linear(x)
        delta_out = (x @ self.A.T @ self.B.T) * self.scaler
        return linear_out + delta_out
