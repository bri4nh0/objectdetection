import torch
from torch import nn
from typing import List, Tuple, Optional


class DiscountedFusion(nn.Module):
    """A small learnable module that predicts per-modality discount weights
    and applies them to concatenated modality features before a fusion MLP.

    Usage (prototype):
    - Build concatenated features = [mod1_feats | mod2_feats | ...]
    - Provide `modality_slices` as a list of (start, end) indices for each modality
      inside the concatenated vector. If omitted, a single global discount is used.
    - The module predicts a scalar weight in (0,1) per modality using a tiny MLP
      (global-average pooling followed by two-layer net + sigmoid).

    This is a non-invasive prototype: it can be placed before an existing
    FusionMLP by calling `discounted = DiscountedFusion(...)(concat_feats)`
    and then passing `discounted` into the fusion network.
    """

    def __init__(self, input_dim: int, modality_slices: Optional[List[Tuple[int, int]]] = None, hidden: int = 32):
        super().__init__()
        self.input_dim = int(input_dim)
        self.modality_slices = modality_slices
        self.hidden = int(hidden)

        # if modality_slices is None we predict a single global discount
        self.num_modalities = len(modality_slices) if modality_slices is not None else 1

        # small MLP to produce a scalar logit per modality from that modality's features
        self._mlps = nn.ModuleList()
        for _ in range(self.num_modalities):
            self._mlps.append(nn.Sequential(nn.Linear( max(1, input_dim if modality_slices is None else (1)), hidden),
                                             nn.ReLU(),
                                             nn.Linear(hidden, 1)))

        # numeric stability
        self.eps = 1e-6

    def forward(self, concat_features: torch.Tensor) -> torch.Tensor:
        """Apply predicted discounts to `concat_features` and return discounted tensor.

        Args:
            concat_features: (batch, input_dim) concatenated modality features

        Returns:
            discounted: (batch, input_dim)
        """
        if concat_features.dim() != 2:
            raise ValueError("concat_features must be (batch, input_dim)")
        batch = concat_features.shape[0]
        if self.modality_slices is None:
            # global discount scalar
            x = concat_features.mean(dim=1, keepdim=True)  # (batch,1)
            logit = self._mlps[0](x)
            weight = torch.sigmoid(logit)  # (batch,1)
            return concat_features * weight

        # else compute per-modality weights
        out = concat_features.clone()
        for i, (start, end) in enumerate(self.modality_slices):
            # slice features
            sl = concat_features[:, start:end]
            if sl.numel() == 0:
                continue
            # global pooling to produce small vector
            pooled = sl.mean(dim=1, keepdim=True)
            logit = self._mlps[i](pooled)
            weight = torch.sigmoid(logit)  # (batch,1)
            out[:, start:end] = sl * weight

        return out
