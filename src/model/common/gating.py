import torch
from torch import nn
class Gating(nn.Module):
    def __init__(self, shape: tuple, additive: bool = True, init_value: float = 0.0):
        super().__init__()
        self.additive = additive
        self.gate = nn.Parameter(torch.full(shape, init_value))
        if not additive:
            self.gate.data.clamp_(min=0) # Equivalent to NonNeg constraint

    def forward(self, x):
        if self.additive:
            return x + self.gate
        else:
            # Equivalent to NonNeg constraint during training
            if self.training:
                # Use non-inplace operation to avoid breaking gradients
                gate = torch.clamp(self.gate, min=0)
            else:
                gate = self.gate
            return x * gate

def ma_gating(seq_len: int, hidden_size: int):#
    shape = (seq_len, hidden_size)
    return nn.Sequential(
        Gating(shape, additive=False, init_value=1.0),
        Gating(shape, additive=True, init_value=0.0)
    )