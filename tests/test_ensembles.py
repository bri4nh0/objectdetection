import torch
import pytest

from src.core.ensembles import TinyDeepEnsemble
from torch import nn


def test_tiny_ensemble_shapes_and_nonnegative_var():
	base = nn.Linear(3, 2)
	td = TinyDeepEnsemble(base, num_members=4)
	td.eval()

	inp = torch.randn(5, 3)
	mean, var = td(inp)

	assert isinstance(mean, torch.Tensor)
	assert isinstance(var, torch.Tensor)
	assert mean.shape == (5, 2)
	assert var.shape == (5, 2)
	assert torch.all(var >= 0)


def test_tiny_ensemble_handles_single_dim_output():
	base = nn.Linear(3, 1)
	td = TinyDeepEnsemble(base, num_members=3)
	inp = torch.randn(2, 3)
	mean, var = td(inp)
	assert mean.shape == (2, 1)
	assert var.shape == (2, 1)
