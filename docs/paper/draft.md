# Draft: Paper skeleton and experiment plan

This document captures two concise novelty claims, the experimental plan required to validate them, an experiment→figure mapping, and acceptance criteria.  Use this as the single-source plan to run experiments and produce paper‑quality figures.

## Working title
Efficient Uncertainty Quantification for Multimodal Threat Detection: Tiny Deep Ensembles for Low-latency Fusion

## 1–2 Novelty claims

1. Lightweight ensemble proxy: We introduce TinyDeepEnsemble, a low-cost ensemble proxy that computes a single shared forward pass through a heavy fusion backbone and instantiates per-member affine heads to estimate epistemic uncertainty with substantially lower latency and memory than standard deep ensembles or MC‑dropout, enabling near‑real-time uncertainty-aware multimodal fusion on edge GPUs.

2. Practical TRD‑UQ pipeline: We show that combining TinyDeepEnsemble with a runtime risk‑dynamics module (EMA + online CUSUM) and heteroscedastic fusion produces uncertainty estimates that are (a) more computationally efficient and (b) comparably calibrated to heavier baselines on held‑out labelled validation sets and OOD conditions relevant to dangerous event detection.

Notes on claims: these are empirical claims. The acceptance criteria (below) map the exact experiments required to validate or refute each claim.

## Experiments to validate claims (overview)

Common setup across experiments:
- Hardware: RTX 3050 Ti (report CPU/GPU timings separately). Include metadata for every run (git sha, pip freeze, model checksums).
- Models/Backbone: FusionMLP (or hetero fusion when available) as fusion backbone. Behavior model and object detector are fixed between methods to ensure fairness.
- Baselines: MC‑dropout (T=30), Full Deep Ensembles (M=4, independently trained), Single deterministic FusionMLP (no UQ), and a lightweight VI prototype (optional).
- Metrics: Negative Log-Likelihood (NLL), Brier score, calibration (ECE; reliability diagrams), sharpness, latency (ms per sample), memory footprint, and runtime per-frame throughput (FPS). Statistical reporting: bootstrap 95% CIs and paired permutation tests for main comparisons.

Experiment A — Latency vs probabilistic performance (main engineering claim)
- Goal: quantify latency savings of TinyDeepEnsemble vs MC‑dropout and Deep Ensembles while measuring probabilistic metrics.
- Procedure: run inference on a held‑out labelled validation set (N ≥ 500 samples), measure per-sample latency on GPU, compute NLL/Brier/ECE for each method. Repeat over 5 seeds for Deep Ensembles; use bootstrap to estimate CIs for metrics and latency.
- Figure/table mapping:
  - Table 1: metrics (NLL, Brier, ECE) ±95% CI and latency (ms) per method. Include memory footprint.
  - Figure 1: Latency vs NLL (scatter/line) per method, log-scaled latency axis if needed.

Experiment B — Calibration & reliability on held‑out & OOD (main scientific claim)
- Goal: show that TinyDeepEnsemble yields calibration comparable to baselines in both in-distribution (ID) and OOD settings (occlusion, noise, temporal jitter).
- Procedure: compute reliability diagrams and ECE on held‑out labelled data and on a curated OOD set; report ECE, Brier, and visualization.
- Figure/table mapping:
  - Figure 2: Reliability diagrams (ID) for TinyDeepEnsemble vs MC‑dropout vs Ensemble vs Deterministic.
  - Figure 3: Reliability diagrams (OOD transforms) and ECE table for each OOD transform.

Experiment C — Ablation: members, T, head-finetune vs full training
- Goal: understand sensitivity to TinyDeepEnsemble members (M), MC‑dropout T, and head-only fine‑tuning vs full member training.
- Procedure: sweep M ∈ {2,4,8} and T ∈ {5,10,30}, run small-scale experiments (N=200) and report NLL/ECE vs latency.
- Figure mapping:
  - Figure 4: Ablation curves (NLL and ECE vs latency) and table of optimal operating points for RTX 3050 Ti.

Experiment D — Runtime & reliability at scale (multi-seed)
- Goal: reproducibility & statistical robustness: run multi-seed experiments (≥5 seeds) for primary comparisons, compute paired permutation tests, bootstrap CIs, and report effect sizes.
- Outputs:
  - CSV with per-seed metrics and timing.
  - Results JSON with aggregated means, CIs and p-values.

## Data & splits
- Identify or create a held‑out labelled validation set: `data/val/` (N ≥ 500 ideally) with ground-truth labels aligned to model outputs. If unavailable, create a small curated validation set (`data/val_small/`) with at least 200 labeled frames for early experiments.
- OOD transformations: implement `scripts/prepare_datasets.py` to generate occlusion, gaussian noise, and temporal jitter variants.

## Methods & implementation notes
- TinyDeepEnsemble: share a frozen base fusion forward; instantiate M per-member affine heads (scale + bias) initialized with small random offsets. Provide a head-only fine‑tune routine in `scripts/train_tiny_ensemble.py` (already present as stub) that randomizes head inits and trains small LR for a few epochs.
- MC‑dropout: inference wrapper `src/core/uq_baselines.py` that runs T stochastic forward passes with dropout enabled at test time.
- Deep Ensembles: train M independent copies of the fusion backbone (or support head-only independent random seeds for a cheap variant).
- Heteroscedastic fusion: optionally load `trd_uq_fusion_hetero.pth` for experiments that require aleatoric modeling. Compare heteroscedastic vs legacy FusionMLP.

## Acceptance criteria (what constitutes success)
- For the engineering claim: TinyDeepEnsemble reduces per-sample latency by ≥3× compared to MC‑dropout (T=30) on RTX 3050 Ti while producing similar NLL/Brier (difference within bootstrap CI) on held‑out validation.
- For the scientific claim: Calibration (ECE) of TinyDeepEnsemble is not statistically worse than MC‑dropout and Deep Ensembles on ID data (paired test p > 0.05 for non-inferiority) and shows graceful degradation under OOD transforms.
- Reproducibility: experiments include metadata JSON (git SHA, pip freeze, model checksums). All scripts produce deterministic outputs when seeds are fixed.

## Experiment-to-figure/table mapping (concise)
- Fig 0 (supp): System diagram and latency profiling table per module.
- Table 1: NLL / Brier / ECE ±95% CI and latency (ms) per method (TinyDE, MC‑dropout, Ensembles, Deterministic).
- Fig 1: Latency vs NLL (main engineering tradeoff).
- Fig 2: Reliability diagrams (ID) for all methods.
- Fig 3: Reliability diagrams (OOD transforms) and ECE table for each transform.
- Fig 4: Ablation curves (members M and T) and operating-point table.
- Table 2: Statistical tests (paired permutation p-values) for main comparisons.

## Minimal timeline & compute budget (RTX 3050 Ti)
- Initial verification (today): run smoke & multiseed (done). Produce reproducible metadata. (0.5 day)
- Small-scale experiments (ID, N=200, multiple seeds): ~1 day per method to run and collect metrics. (1–2 days)
- Full held‑out run (N≥500) + OOD transforms + multi-seed (5 seeds): 3–5 days depending on training needs for Deep Ensembles.

## Next steps (immediate)
1. Confirm these two claims and acceptance criteria with me.
2. Point me to or create a held‑out labelled validation split (path under `data/val/`) or ask me to create `data/val_small/` from available raw frames.
3. I will implement `src/evaluation/trd_uq_eval.py` (metrics + reliability diagrams + CSV) and run Experiment A on the held-out split.

---
If you approve the claims and the plan, I'll commit this file and start implementing the evaluation CLI and the small labelled validation split (unless you prefer I begin with another item).
