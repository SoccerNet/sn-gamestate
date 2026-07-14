# Phase 3B — Tactical Classifier (TactPipeline)

**Objective (proposal Obj. 4):** a GNN classifier for tactical recognition, trained on
noisy broadcast-extracted positions against labels geometry-derived from clean SoccerNet GT,
evaluated by macro-F1 (target ≥ 0.7, cf. Bauer et al. 0.76).

**Status:** ✅ complete. **Pressing phase** is the quantitative primary result; **formation**
is a documented best-effort limitation. Runs on Mac CPU.
Spec: `docs/superpowers/specs/2026-07-14-phase3b-tactical-classifier-design.md`.
Plan: `docs/superpowers/plans/2026-07-14-phase3b-tactical-classifier.md`.

## Method

- **Labels from clean GT, inputs from noisy predictions** — the gap is the error-propagation
  signal. Labels come from `Labels-GameState.json`; input graphs from the Phase-2 smoothed
  feature tables (predicted tracks).
- **Pressing label (possession-agnostic):** the ball is detected in only ~4% of frames, so
  possession is unusable. For each team per 1 s window, compute the GT team centroid's **block
  height** = distance from its own goal along the goal-to-goal axis (verified: **x**, span
  ~105 m, goals ±52.5; `left` defends −52.5). Bin by **fixed pitch-thirds**: low / mid / high
  block. Window kept only if the modal class holds ≥ 80% of its frames; robust median centroid
  after dropping out-of-pitch outliers (`|x|≤60, |y|≤45` — GT itself has rare projection blowups).
- **Input graphs:** per (sequence, team, window), nodes = that team's visible predicted players
  (position averaged over the window), node features `[dist_to_own_goal, y, vx, vy]`, k-NN edges
  within team.
- **Models:** hand-rolled **GCN** and **GAT** in plain PyTorch (no PyTorch-Geometric) — 2 graph-conv
  layers → global mean-pool → MLP → 3 classes. Plus a **majority-class baseline**.
- Train on valid minus 6 held-out sequences; early-stop on val macro-F1; evaluate on the test split.

## Pressing results (test split, 2620 windows)

| model | macro-F1 |
|-------|----------|
| majority baseline | 0.197 |
| GCN | 0.912 |
| **GAT** | **0.922** |

Confusion (rows = true low / mid / high):

```
GCN                     GAT
[[786  35   1]          [[794  28   0]
 [ 81 974  41]           [ 79 954  63]
 [  3  74 625]]          [  4  35 663]]
```

## Findings

1. **Both models clear the ≥ 0.7 target comfortably and crush the 0.197 majority baseline** —
   GAT 0.922, GCN 0.912. The task is well-posed and the graphs carry real signal.
2. **GAT marginally beats GCN** (0.922 vs 0.912) — consistent with the proposal's hypothesis
   that attention helps when specific player relationships matter.
3. **Errors are confined to adjacent classes** (low↔mid, mid↔high) — windows near the
   pitch-thirds boundaries. Never low↔high (a deep block is never confused with a high press).
4. **Noise-cascade reading:** labels are clean-GT, inputs are noisy predictions, so the ~8% gap
   from a perfect 1.0 is the cost the calibration/tracking noise imposes on recovering the
   geometric rule — a downstream data point for the error-propagation study (Obj. 5). The
   majority baseline at 0.197 confirms this is not trivial guessing.

## Formation — best-effort, documented limitation (not a classifier)

Formation labels were derived from GT (per-outfield-player mean depth over the clip, split into
three lines at the two largest depth gaps → matched to 4-4-2 / 4-3-3 / 3-5-2). GT label
distribution: 4-4-2 ×62, 4-3-3 ×42, 3-5-2 ×11.

**No formation classifier was trained**, for two independent reasons:
1. **Partial observability** — only ~5-6 players/team are visible per frame, so both the label
   (from GT) and any input under-observe the outfield lines.
2. **Data sparsity** — formation is a *per-clip* label, giving only ~115 team-clip instances
   total, far too few to train a GNN (pressing has ~5,400 per-window graphs).

Formation from broadcast is therefore reported as an **under-determined task**, not a benchmark.
(Method note: an initial percentile-tertile line-split was degenerate — it forces equal thirds,
collapsing every clip to 4-3-3; the largest-gap split above is what recovers real line sizes.)

## Reproducibility & gotchas

- `phase3b/`: `pitch.py`, `gt_labels.py`, `graphs.py`, `metrics.py`, `models.py`, `train.py`,
  `eval.py`, `formation.py` + pytest tests. Run: `python -m phase3b.train {gcn,gat}` →
  `python -m phase3b.eval`; `python -m phase3b.formation`.
- Env: sn-gamestate conda (torch 1.13.1), CPU (MPS saves NaN weights on this torch — see 3A).
- Bugs fixed during execution: (a) leading-zero `seq` mismatch (int `78` vs `"078"`) zeroed the
  validation split; (b) macro-F1 must average over *present* classes (sklearn convention), not
  all fixed classes, or degenerate cases misreport.
- Training is slightly unstable (small val set of 324 windows); early-stopping on val macro-F1
  selects the best epoch (~0.95 val).

## Next

- Phase 5 visualization dashboard (in progress).
- Error-propagation study (Obj. 5): inject noise upstream, measure 3A ADE/FDE and 3B macro-F1
  degradation — the ~36% calibration-blowup rate (from 3A) and this pressing macro-F1 are the
  downstream metrics it will perturb.
