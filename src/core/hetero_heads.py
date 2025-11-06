import torch
from torch import nn


class LowRankHeteroHead(nn.Module):
    """Low-rank heteroscedastic head: predicts mean and a low-rank factorization
    that is used to produce a positive variance estimate per output dimension.

    Design (prototype):
    - mean = Linear(in_dim -> out_dim)
    - factors = Linear(in_dim -> out_dim * rank) -> reshape (batch, out_dim, rank)
    - variance = sum(factors**2, dim=-1) + softplus(offset)

    Returns (mean, var) with shapes (batch, out_dim).
    """

    def __init__(self, in_dim: int, out_dim: int = 1, rank: int = 4):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.rank = int(rank)

        self.mean_head = nn.Linear(self.in_dim, self.out_dim)
        self.factor_head = nn.Linear(self.in_dim, self.out_dim * self.rank)
        # small positive offsets per output to avoid zero variance
        self.register_parameter("offset", nn.Parameter(torch.ones(self.out_dim) * 1e-3))

        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor):
        """Compute mean and variance.

        Args:
            x: (batch, in_dim)

        Returns:
            mean: (batch, out_dim)
            var:  (batch, out_dim) (non-negative)
        """
        if x.dim() != 2:
            # support (batch, seq, in_dim) by pooling the seq dimension
            if x.dim() == 3:
                x = x.mean(dim=1)
            else:
                raise ValueError("Input x must be (batch, in_dim) or (batch,seq,in_dim)")

        mean = self.mean_head(x)

        factors = self.factor_head(x)  # (batch, out_dim * rank)
        factors = factors.view(-1, self.out_dim, self.rank)
        var = torch.sum(factors * factors, dim=-1)  # (batch, out_dim)
        var = var + self.softplus(self.offset)
        return mean, var
