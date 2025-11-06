import torch
from torch import nn

from .ensembles import TinyDeepEnsemble


class AdaptiveTinyEnsemble(nn.Module):
    """A lightweight runtime-adaptive controller for TinyDeepEnsemble.

    Strategy (simple, video-friendly prototype):
    - Always run the shared base forward once (same as TinyDeepEnsemble).
    - Choose how many affine heads to evaluate using a small temporal
      smoothing heuristic on the previously-observed predictive variance.
    - If recent variance is below `tau`, evaluate only `min_heads` (fast path).
      Otherwise evaluate all ensemble members.

    This is intentionally conservative and cheap: it avoids re-running the
    heavy base forward (the main cost) while reducing per-frame head work
    when uncertainty is low. It's a prototype for studies of frame-adaptive
    inference budgets; a research-grade controller can replace the heuristic.
    """

    def __init__(self, tiny_de_model: TinyDeepEnsemble, tau: float = 1e-3, min_heads: int = 1, smoothing: float = 0.9):
        super().__init__()
        if not isinstance(tiny_de_model, TinyDeepEnsemble):
            raise TypeError("AdaptiveTinyEnsemble expects a TinyDeepEnsemble instance")
        self.tde = tiny_de_model
        self.tau = float(tau)
        self.min_heads = int(min_heads)
        self.smoothing = float(smoothing)

        # smoothed scalar uncertainty proxy (initialized high so first frame uses full ensemble)
        self.register_buffer("_smoothed_var", torch.tensor(1.0))

    def _compute_heads(self, base_out: torch.Tensor, k: int):
        """Compute outputs from the first k affine heads without touching others.

        Args:
            base_out: (batch, out_dim)
            k: number of heads to evaluate (1..num_members)

        Returns:
            mean: (batch, out_dim)
            var:  (batch, out_dim)
        """
        assert self.tde._initialized, "TinyDeepEnsemble parameters not initialized"
        num_members = self.tde.num_members
        k = max(1, min(k, num_members))

        batch = base_out.shape[0]
        out_dim = base_out.shape[1]

        # expand base_out to (k, batch, out_dim)
        base_expanded = base_out.unsqueeze(0).expand(k, batch, out_dim)

        scales = self.tde.scales[:k].unsqueeze(1)  # (k,1,out_dim)
        biases = self.tde.biases[:k].unsqueeze(1)

        member_outs = base_expanded * scales + biases

        mean = member_outs.mean(dim=0)
        var = member_outs.var(dim=0, unbiased=False) + self.tde.eps
        return mean, var

    def forward(self, *args, **kwargs):
        """Adaptive forward: run base once, decide how many heads to evaluate.

        Returns the same (mean,var) tuple as TinyDeepEnsemble.
        """
        # run the heavy shared base forward directly to control head evaluation
        base_out = self.tde.base_model(*args, **kwargs)
        if isinstance(base_out, (list, tuple)) and len(base_out) == 1 and isinstance(base_out[0], torch.Tensor):
            base_out = base_out[0]
        if not isinstance(base_out, torch.Tensor):
            raise TypeError("base_model must return a torch.Tensor")
        if base_out.dim() == 1:
            base_out = base_out.unsqueeze(1)

        # ensure tde params are initialized
        if not self.tde._initialized:
            self.tde._lazy_init(base_out.shape[1], device=base_out.device, dtype=base_out.dtype)

        # cheap decision: if smoothed variance below tau use fast path
        # note: smoothed var is a scalar buffer
        use_fast = float(self._smoothed_var) < float(self.tau)

        if use_fast:
            k = self.min_heads
        else:
            k = self.tde.num_members

        mean, var = self._compute_heads(base_out, k)

        # update smoothed proxy using mean variance (scalar)
        with torch.no_grad():
            cur_proxy = float(var.mean().detach().cpu())
            sm = float(self.smoothing)
            new = sm * float(self._smoothed_var) + (1.0 - sm) * cur_proxy
            self._smoothed_var.fill_(new)

        return mean, var
