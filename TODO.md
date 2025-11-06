# Project TODO (near‑doctoral roadmap)

This file lists prioritized tasks to bring the TRD‑UQ project to a near‑doctoral level: rigorous baselines, reproducible experiments, statistical validation, and deliverable artifacts (paper + release).

Each entry includes an acceptance criterion and a rough time estimate targeted to a dev machine with an RTX 3050 Ti.

---

## 1. Define novelty claims & paper skeleton (1–2 days)
- Write 1–2 crisp novelty claims and map each planned experiment to the claim it validates.
- Create `docs/paper/draft.md` with TOC, short method descriptions, and an experiment→figure mapping.
- Acceptance: clear claims and an experiment plan that maps directly to figures and tables.

## 2. Reproducible experiment harness & metadata (0.5–1 day) ✅
- Implement `src/utils/metadata.py` — captures git SHA, python version, `pip freeze`, model checksums.
- Acceptance: every experiment writes `results/<run_id>_metadata.json` with repo SHA and env details.

## 3. Robust baselines (3–7 days)
- Implement MC‑dropout wrapper, full DeepEnsemble training harness, and a lightweight VI prototype.
- Files: `src/core/uq_baselines.py`, `src/training/*` scripts.
- Acceptance: baselines runnable and producing mean+variance outputs on smoke datasets.

## 4. TinyDeepEnsemble refinement & training stub (1–2 days)
- Add training/fine‑tuning stubs to initialize per‑member affine heads with diversity (randomized starts, small LR head-only fine-tune).
- Acceptance: training produces measurable non-zero ensemble variance on a small dataset and increases meaningful UQ metrics.

## 5. Evaluation & benchmarking suite (2–4 days)
- Implement evaluation CLI to compute NLL, RMSE/Brier, ECE, reliability diagrams, MC convergence plots and E‑QR separation metrics.
- Files: `src/evaluation/trd_uq_eval.py`, `scripts/eval_cli.py`.
- Acceptance: single command produces CSV metrics and figures for a checkpoint.

## 6. Ablations & automated sweeps (2–4 days)
- Implement `scripts/ablation_runner.py` to sweep seeds, td_members, mc_T, hetero/legacy heads and collect structured CSV outputs and metadata.
- Acceptance: reproducible sweeps producing CSV with metrics and latency for each config.

## 7. Statistical validation & reporting (1–2 days)
- Add bootstrap CIs, paired permutation tests, and effect-size reporting; integrate into evaluation CLI.
- Acceptance: output CSVs include CI and p-values for primary comparisons.

## 8. Curated datasets & OOD bench (1–3 days)
- Create stable train/val/test splits and synthetic OOD protocols (occlusion, temporal jitter, noise) and produce small labelled validation sets for calibration metrics.
- Acceptance: documentation and scripts under `scripts/prepare_datasets.py` and `data/ood/`.

## 9. Paper-quality experiments & figures (1–3 weeks)
- Produce main experiment table, reliability diagrams, NLL vs latency curves, ablation figures, and living figures in `docs/paper/figs`.
- Acceptance: figures and tables sufficient for a paper/thesis Results draft.

## 10. Unit tests & continuous integration (1–2 days)
- Expand pytest tests for critical modules; add GitHub Actions workflow for smoke tests and linters.
- Acceptance: CI runs smoke tests and unit tests on PRs and reports status.

## 11. Reproducible packaging & containerization (2–4 days)
- Add `Dockerfile` (optional CUDA), `requirements_repro.txt` and `Makefile` targets for reproducible smoke runs.
- Acceptance: docker build reproduces env and runs smoke experiment (documented fallback for local dev). 

## 12. Runtime optimizations (optional) (2–5 days)
- Add AMP/torch.compile flags and provide ONNX export and optional TensorRT/torch‑tensorrt scripts. Document fallbacks and measured speedups.
- Acceptance: documented improvements and safe fallback path.

## 13. Ethics & deployment risk assessment (1–2 days)
- Write `docs/ethics.md` detailing privacy, bias, surveillance, and mitigation strategies.

## 14. Release artifacts & reproducibility checklist (2–3 days)
- Package top models, final configs, metadata and create `release/README.md` and a reproducibility checklist.

## 15. Low‑rank heteroscedastic head (research prototype) (2–4 days)
- Prototype a low-rank covariance head in `src/core/hetero_heads.py` and add a small demo training script.

## 16. Discounted belief fusion (research prototype) (2–4 days)
- Implement conflict discounting fusion in `src/core/discounted_fusion.py` and provide an integration demo comparing to `FusionMLP`.

## 17. Final experimental checklist & milestone plan (0.5–1 day)
- Produce an 8–12 week milestone plan (tasks, compute budget, figure list) aligned to RTX 3050 Ti and submission schedule.

---

How to use this file
- Mark items completed by adding a ✅ after the heading.
- Focus on items 1–6 first to achieve reproducibility, baselines, and rigorous evaluation.

If you want, I can convert each item into individual tracked issues or generate GitHub Actions/Makefile stubs for the most critical tasks.
