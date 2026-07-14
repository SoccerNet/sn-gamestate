# Phase 3A — Trajectory Forecaster (TactPipeline)

**Objective (proposal Obj. 3):** establish the *first* ADE/FDE trajectory-forecasting
benchmark on broadcast-extracted, noise-contaminated tracks, and test whether a learned
model beats physics baselines under that noise.

**Status:** ✅ complete on the valid (train) / test (eval) splits. Runs on Mac CPU.
Spec: `docs/superpowers/specs/2026-07-14-phase3a-trajectory-forecaster-design.md`.
Plan: `docs/superpowers/plans/2026-07-14-phase3a-trajectory-forecaster.md`.

## Method

Per-track, per-player (players + goalkeepers) forecasting on ego-centric windows built from
the Phase-2 smoothed feature tables.
- **Input:** 50 frames (2 s @ 25 fps) of `[Δx, Δy, vx, vy]` (position relative to the last
  observed point + velocity).
- **Model:** small **GRU encoder → 3 sparse displacement points** at 1 s / 2 s / 3 s (25/50/75
  frames), per the proposal + Victor et al.'s sparse-output regime.
- **Reconstruction:** constant-acceleration interpolation of the 3 points into a dense
  75-frame trajectory; ADE/FDE computed on the dense reconstruction, per horizon.
- **Baselines:** constant-velocity (linear extrapolation) and constant-acceleration.
- Trained on valid minus 6 held-out sequences; those 6 = validation (early stop on ADE@3s);
  final evaluation on the test split.
- Package: `phase3a/{windows,physics,metrics,models,train,eval,viz}.py` + pytest tests.

## Outlier guard (calibration blowups)

Some frames have physically impossible pitch coordinates from calibration failure (position
jumps implying hundreds of m/s). `windows.py` drops any window whose position steps imply
**> 12 m/s**. Impact on the test split: **1595 of 4426 windows (~36%) dropped as calibration
garbage.** This both (a) exposes the calibration failure rate as a first-class finding and
(b) removes a ~4× inflation of the naive ADE/FDE means.

## Results (test split, after guard, metres)

| model | ADE@1s | FDE@1s | ADE@2s | FDE@2s | ADE@3s | FDE@3s |
|-------|--------|--------|--------|--------|--------|--------|
| constant-velocity | 0.66 | 1.32 | 1.40 | 2.97 | 2.25 | 4.89 |
| constant-acceleration | 2.24 | 5.87 | 7.72 | 21.55 | 16.48 | 47.04 |
| **GRU (ours)** | **0.47** | **0.85** | **0.94** | **1.90** | **1.47** | **3.12** |

## Findings

1. **The GRU beats constant-velocity at every horizon (~35%)** — ADE@3s 1.47 vs 2.25 m,
   FDE@3s 3.12 vs 4.89 m. Notably this *contrasts* Victor et al., who found LSTMs often fail
   to beat linear extrapolation on clean data; here, on noisy broadcast tracks, the learned
   model adds clear value.
2. **~36% of windows are calibration failures** (>12 m/s implied). Reported alongside the
   forecaster result: performance is measured on plausible-calibration windows, and the
   failure rate itself quantifies part of the error cascade (proposal Obj. 5).
3. **Constant-acceleration explodes** (FDE@3s 47 m) — the 2nd derivative amplifies calibration
   noise. Concrete evidence of how upstream noise cascades downstream.
4. Qualitatively (see `phase3a/forecast_viz.png`): the GRU tracks the true future closely on
   the pitch, while constant-velocity overshoots on curved runs.

## Reproducibility & gotchas

- Train + eval: `python -m phase3a.train` then `python -m phase3a.eval`; visual:
  `python -m phase3a.viz` → `phase3a/forecast_viz.png`. Tests: `python -m pytest phase3a/tests`.
- Env: sn-gamestate conda (torch 1.13.1, matplotlib). **Train on CPU** — torch 1.13.1's MPS
  path saves NaN weights (`torch.save` off an MPS model), even though the live model computes
  fine; `device="cpu"` avoids it. Model is tiny, so CPU speed is a non-issue.
- Training is slightly unstable (val ADE swings); early-stopping on val ADE@3s selects the
  best epoch. Lowering the learning rate would stabilise it (optional).

## Next

- Phase 3B: GCN/GAT tactical classifier (formation + pressing) with geometry-derived labels.
- Later: feed 3A into the error-propagation study (Obj. 5) — inject noise upstream, measure
  ADE/FDE degradation.
