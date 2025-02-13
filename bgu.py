import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BinaryGatingUnit(nn.Module):
    def __init__(self) -> None:
        super(BinaryGatingUnit, self).__init__()
        self.tau = nn.Parameter(torch.ones(1))

    def forward(self, x: Tensor) -> Tensor:
        prob = torch.sigmoid(x)
        logits = torch.stack([prob, 1 - prob], dim=-1)
        gumbel_mask = F.gumbel_softmax(logits, tau=self.tau, hard=True)
        bern = gumbel_mask[:, 0]
        bias = torch.where(bern == 1, 1 - prob, -prob)
        mask = prob + bias
        bgu = x * mask

        return bgu
