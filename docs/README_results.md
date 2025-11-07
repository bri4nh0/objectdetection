README — Reproducing results and mapping claims

This README provides a minimal set of steps for a committee member (or reviewer) to run the project's core demonstrations locally with minimal dependencies.

1) Set up environment (recommended)

PowerShell (recommended):

```powershell
conda create -n trt_env python=3.10 -y; conda activate trt_env
pip install -r requirements_enhanced.txt
```

2) Produce demo artifacts (self-contained)

```powershell
conda activate trt_env; python scripts/demo_prepare.py
```

This script will create `data/processed/demo_synth.npz`, `results/demo_backbone.pth` and `results/demo_tde.pth`. It is CPU-friendly and deterministic.

3) Run the demo inference

```powershell
conda activate trt_env; python scripts/run_demo.py
```

This will load the demo checkpoints and write `results/demo_outputs.npz` and `results/demo_run_metadata.json`.

4) Reproduce other artifacts

- Unit tests: `python -m pytest -q` (requires `pytest` in environment)
- Smoke comparisons: `python scripts/smoke_compare_tiny_mc.py` (requires GPU for timings; will run on CPU too)
- Multi-seed reliability: `python scripts/multiseed_reliability_tiny_mc.py`

5) If you want a single-file provenance bundle for submission

I can create a ZIP containing the demo artifacts, `results/` JSONs, and a short `run_me.ps1` script for the committee. Say "bundle results" and I'll create `results/demo_bundle.zip`.

If you want me to wire the demo outputs into a single PDF figure or add explicit script-to-figure mapping for the paper's figures, say which figure(s) you'd like and I'll generate them.
