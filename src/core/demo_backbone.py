import torch
from torch import nn


class DemoBackbone(nn.Module):
    """Tiny feed-forward backbone used for self-contained demos.

    Simple MLP: input_dim -> hidden -> out_dim. Saved as a small checkpoint
    so the demo runs without external models.
    """

    def __init__(self, input_dim: int = 16, hidden: int = 32, out_dim: int = 8):
        super().__init__()
        self.input_dim = int(input_dim)
        self.out_dim = int(out_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, int(hidden)),
            nn.ReLU(),
            nn.Linear(int(hidden), self.out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect (batch, input_dim)
        if x.dim() != 2:
            x = x.view(x.size(0), -1)
        return self.net(x)
