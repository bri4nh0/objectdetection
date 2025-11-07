import torch
from torch import nn
from typing import List, Optional


class DiscountedFusion(nn.Module):
    """Simple learnable discounted fusion.

    Usage:
      - Provide modality sizes (list of ints) for how to slice the input feature vector.
      - The module predicts a scalar discount weight per modality in [0,1]
        and applies it to each modality's features before concatenation.

    This is intentionally small: a two-layer MLP that consumes per-modal
    pooled features and outputs weights.
    """

    def __init__(self, modality_sizes: List[int], hidden: int = 32):
        super().__init__()
        self.modality_sizes = list(modality_sizes)
        self.n_modalities = len(modality_sizes)

        # per-modality pooling -> scalar; use a shared MLP to predict weights
        # Input to MLP will be concatenation of per-modality pooled scalars
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(self.n_modalities, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.n_modalities),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor, modality_sizes: Optional[List[int]] = None):
        """Forward.

        Args:
            features: (batch, total_dim) concatenated modality features
            modality_sizes: optional override for modality sizes

        Returns:
            fused: (batch, total_dim) fused (discounted & concatenated) features
        """
        if modality_sizes is None:
            modality_sizes = self.modality_sizes
        assert sum(modality_sizes) == features.shape[1], "modality sizes don't match feature dim"

        # slice per-modality and compute pooled scalar per batch
        parts = []
        start = 0
        pooled = []
        for s in modality_sizes:
            part = features[:, start:start + s]  # (batch, s)
            parts.append(part)
            # pool to scalar per modality
            if s == 0:
                pooled.append(features.new_zeros(features.shape[0], 1))
            else:
                p = part.unsqueeze(1) if part.dim() == 2 else part
                # AdaptiveAvgPool1d expects (batch, channels, L); treat dim as channels
                p = part.unsqueeze(1)
                pooled_val = self.pool(p).squeeze(2)  # (batch, s) -> (batch, s) pooled to (batch,1)?
                # reduce to scalar per modality by mean over channels
                pooled.append(pooled_val.mean(dim=1, keepdim=True))
            start += s

        pooled_cat = torch.cat(pooled, dim=1)  # (batch, n_modalities)
        weights = self.mlp(pooled_cat)  # (batch, n_modalities) in (0,1)

        # apply weights per modality
        fused_parts = []
        for i, part in enumerate(parts):
            w = weights[:, i:i+1]
            fused_parts.append(part * w)

        fused = torch.cat(fused_parts, dim=1)
        return fused
