import torch
from torch import nn
import time


class MCDropoutWrapper(nn.Module):
    """MC-Dropout inference wrapper. During forward, runs T stochastic
    forward passes with dropout active and returns mean and variance.
    """

    def __init__(self, base_model: nn.Module, T: int = 30):
        super().__init__()
        self.base_model = base_model
        self.T = int(T)

    def forward(self, x):
        # enable dropout during eval
        training_states = {}
        for m in self.base_model.modules():
            if isinstance(m, nn.Dropout):
                training_states[m] = m.training
                m.train()

        outputs = []
        for _ in range(self.T):
            out = self.base_model(x)
            if isinstance(out, (list, tuple)):
                out = out[0]
            outputs.append(out.unsqueeze(0))
        out_tensor = torch.cat(outputs, dim=0)

        # restore dropout states
        for m, st in training_states.items():
            m.train(st)

        mean = out_tensor.mean(dim=0)
        var = out_tensor.var(dim=0, unbiased=False)
        return mean, var
