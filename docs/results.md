## Results mapping — what to run to reproduce each claim

This file maps each paper claim to the exact code and script that implements it, plus the saved artifacts under `results/` that you can inspect. These are self-contained: they do not require external YOLO weights or a database connection (use `scripts/demo_prepare.py` and `scripts/run_demo.py`).

- Claim: TinyDeepEnsemble (low-cost ensemble via shared base + per-member affine heads)
  - Implementation: `src/core/ensembles.py::TinyDeepEnsemble`
  - Quick reproduce: run the demo (creates small backbone + tde checkpoint and runs one batch)
    - `python scripts/demo_prepare.py`
    - `python scripts/run_demo.py`
  - Saved artifacts:
    - `results/demo_backbone.pth` — tiny backbone state dict
    - `results/demo_tde.pth` — TinyDeepEnsemble wrapper state dict
    - `results/demo_outputs.npz` — mean/var arrays from a sample batch

- Claim: MC-dropout baseline and smoke comparison
  - Implementation: `src/core/uq_baselines.py` and `scripts/smoke_compare_tiny_mc.py`
  - Reproduce (requires `trt_env` with torch): `python scripts/smoke_compare_tiny_mc.py`
  - Saved artifacts (examples): `results/smoke_*.json` or `results/demo_outputs.npz` for demo-run comparisons

- Claim: Runtime-adaptive TinyDE (prototype)
  - Implementation (prototype): `src/core/adaptive_tiny_ensemble.py::AdaptiveTinyEnsemble`
  - How to test: wrap an existing `TinyDeepEnsemble` instance with `AdaptiveTinyEnsemble(tde, tau=...)` and call it; see `scripts/test_multimodal_smoke.py` for examples of runtime wiring.

- Claim: Learnable discounted fusion (prototype)
  - Implementation (prototype): `src/core/discounted_fusion.py::DiscountedFusion`
  - How to test: create concatenated modality features (or use a small synthetic concat) and call `DiscountedFusion(modality_slices=...)(concat_features)`; see `scripts/` for wiring examples.

- Claim: Low-rank heteroscedastic head (prototype)
  - Implementation (prototype): `src/core/hetero_heads.py::LowRankHeteroHead`
  - How to test: plug into your fusion head (replace simple Linear head) and run head-only finetune stub `scripts/train_tiny_ensemble.py` after adjusting the runner to use the hetero head.

### Where to find plots and multi-seed results

- Multi-seed reliability runs and their JSON/PNG artifacts are saved under `results/`, examples:
  - `results/multiseed_*.json`
  - `results/*_multiseed_results.json`

If you need a single command to reproduce the demo outputs (quick):

```powershell
conda activate trt_env; python scripts/demo_prepare.py; python scripts/run_demo.py
```

If you want me to add a short script that produces a single PDF with the main demo figures, say "make demo figures" and I'll add it under `scripts/` and run it.
