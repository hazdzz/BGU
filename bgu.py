import torch
import torch.nn as nn
from torch import Tensor


class BinaryGatingUnit(nn.Module):
    def __init__(self) -> None:
        super(BinaryGatingUnit, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        # p = torch.sigmoid(input)
        # sigmoid(x) = 0.5 * (1 + tanh(0.5 * x))
        p = 0.5 * (1 + torch.tanh(0.5 * input))
        bern = torch.bernoulli(p)
        eps = torch.where(bern == 1, 1 - p, -p)
        gate = p + eps
        bgu = input * gate

        return bgu
