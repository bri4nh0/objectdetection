import torch
from torch import nn


class TinyDeepEnsemble(nn.Module):
    """A tiny deep ensemble wrapper that shares a single base model forward
    and learns per-member affine heads (scale + bias) to cheaply approximate
    ensemble diversity.

    Behavior:
    - Forward runs the shared base_model once.
    - Lazily initializes per-member scale and bias parameters matching the
      base output dimensionality on first forward.
    - Returns (mean, variance) across ensemble members for the batch.

    This is designed as a low-memory proxy for full ensembles on edge GPUs.
    """

    def __init__(self, base_model: nn.Module, num_members: int = 4, eps: float = 1e-6):
        super().__init__()
        self.base_model = base_model
        self.num_members = int(num_members)
        self.eps = float(eps)
        # parameters initialized lazily when first forward determines output dim
        self._initialized = False

    def _lazy_init(self, out_dim: int, device=None, dtype=None):
        """Create per-member scale and bias parameters for given output dim."""
        if self._initialized:
            return
        # create parameters of shape (num_members, out_dim)
        scales = torch.ones((self.num_members, out_dim), device=device, dtype=dtype)
        biases = torch.zeros((self.num_members, out_dim), device=device, dtype=dtype)
        self.scales = nn.Parameter(scales)
        self.biases = nn.Parameter(biases)
        self._initialized = True

    def forward(self, *args, **kwargs):
        """Forward through base_model and apply per-member affine heads.

        Returns:
            mean: Tensor of shape (batch, out_dim)
            var:  Tensor of shape (batch, out_dim) (non-negative)
        """
        base_out = self.base_model(*args, **kwargs)
        if not isinstance(base_out, torch.Tensor):
            # Try to handle single-element tuples/lists returned by some models
            if isinstance(base_out, (list, tuple)) and len(base_out) == 1 and isinstance(base_out[0], torch.Tensor):
                base_out = base_out[0]
            else:
                raise TypeError("TinyDeepEnsemble expects the base model to return a torch.Tensor (or single-element tuple/list)")

        # ensure shape is (batch, out_dim)
        if base_out.dim() == 1:
            base_out = base_out.unsqueeze(1)

        batch = base_out.shape[0]
        out_dim = base_out.shape[1]

        # lazy init parameters with matching device/dtype
        if not self._initialized:
            self._lazy_init(out_dim, device=base_out.device, dtype=base_out.dtype)

        # prepare base_out for per-member affine transforms: (num_members, batch, out_dim)
        base_expanded = base_out.unsqueeze(0).expand(self.num_members, batch, out_dim)

        # scales: (num_members, out_dim) -> (num_members, 1, out_dim) for broadcasting
        scales = self.scales.unsqueeze(1)
        biases = self.biases.unsqueeze(1)

        member_outs = base_expanded * scales + biases

        mean = member_outs.mean(dim=0)
        var = member_outs.var(dim=0, unbiased=False) + self.eps

        return mean, var
