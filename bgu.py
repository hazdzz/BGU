import torch
import torch.nn as nn
from torch import Tensor


class BinaryGatingUnit(nn.Module):
    def __init__(self, act_func: str = 'sigmoid') -> None:
        super(BinaryGatingUnit, self).__init__()
        assert act_func in ['sigmoid', 'tanh', 'erf']
        self.act_func = act_func

    def forward(self, input: Tensor) -> Tensor:
        if self.act_func == 'sigmoid':
            p = torch.sigmoid(input)
        elif self.act_func == 'tanh':
            p = 0.5 * (1.0 + torch.tanh(0.5 * input))
        elif self.act_func == 'erf':
            p = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        else:
            raise NotImplementedError(f'ERROR: The {self.act_func} based bgu is undefined.')
        bern = torch.bernoulli(p)
        eps = torch.where(bern == 1, 1 - p, -p)
        gate = p + eps
        bgu = input * gate

        return bgu
