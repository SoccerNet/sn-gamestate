# Phase 1 — SoccerNet Game State Reconstruction Baseline (HEX reproduction)

**Objective:** reproduce the official SoccerNet-GSR baseline end-to-end on the Bath HEX
GPU cluster and record a clean, citable GS-HOTA number on the full valid and test splits.

**Status:** ✅ complete. Baseline reproduced and (on test) exceeded, both splits run clean.

## Results

Metric: **GS-HOTA** (official config — `EVAL_SPACE=pitch`, `USE_ROLES=USE_TEAMS=USE_JERSEY_NUMBERS=True`, `EVAL_DIST_TOL=5 m`). Values are `cls_comb_det_av`.

| Split | Seqs | GS-HOTA | DetA | AssA | DetRe | DetPr | AssRe | AssPr | LocA | IDF1 | MOTA | IDSW | Frag |
|-------|------|---------|------|------|-------|-------|-------|-------|------|------|------|------|------|
| valid | 58/58 | **19.96%** | 8.44 | 47.24 | 13.69 | 17.95 | 50.21 | 86.69 | 91.08 | 13.30 | −47.82 | 894 | 5849 |
| test  | 49/49 | **29.22%** | 15.86 | 53.86 | 25.24 | 29.79 | 57.55 | 87.75 | 92.78 | 23.71 | −32.58 | 1071 | 6032 |

Detection counts — valid: 515,145 pred / 675,711 GT, 5,160 pred IDs / 1,348 GT IDs. test: 418,038 pred / 493,374 GT, 4,059 pred IDs / 1,145 GT IDs.

### Comparison to the paper
Somers et al., *SoccerNet Game State Reconstruction* (CVPRW 2024, arXiv:2404.11335) report a
full-baseline **GS-HOTA = 22.26% on test**. This run scores **29.22% on test** — higher because the
current codebase uses newer modules than the paper baseline (see config note). Reproduced and exceeded.

### Analysis
- **Association is solid** (AssA 47–54, AssPr ~87): once a detection is matched, tracking holds it well.
- **Detection accuracy is the bottleneck** (DetA 8–16, DetRe 14–25): under strict pitch-space matching
  with all attributes, most predictions fail to match GT. Driven by calibration error (pitch registration)
  + identification (jersey/team) — not by the tracker.
- **Heavy track over-fragmentation**: predicted IDs are ~3–4× the GT count (valid 5160 vs 1348), high Frag.
- Consistent with the paper's own ablation: *"severe performance impact of enabling calibration and pitch
  localization."* → **calibration + detection/identification are the Phase-2 improvement targets.**

## Environment
- **Cluster:** Bath HEX, node `cheery`, NVIDIA RTX A5000 24 GB (GPU 5), via `hare`/Docker
  (`nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04`).
- **Software:** tracklab **1.3.24**, sn-gamestate (repo default config), Python 3.9, mmcv 2.0.1.
- **Dataset:** SoccerNet-GSR **v1.3** (`gamestate-2024`), splits extracted under scratch `/data/SoccerNetGS/`.
- **Data loading:** `num_cores=0` (single-process DataLoader — avoids the 64 MB container `/dev/shm` Bus-error crash).

## Pipeline (repo default)
| Subtask | Module |
|---|---|
| Detection | YOLOv11 (Ultralytics) |
| Re-ID | PRTReID |
| Tracking | BPBReID-StrongSORT |
| Pitch + Calibration | **NBJW_Calib** *(repo default; paper baseline used TVCalib)* |
| Jersey number | MMOCR |
| Tracklet aggregation | majority vote |
| Team affiliation | K-means on ReID embeddings |
| Team side | mean position |

## Exact commands
Full runs (inside a `hare` GPU container, in `screen`, venv activated):
```bash
tracklab -cn soccernet num_cores=0 dataset.nvid=-1 dataset.eval_set=valid \
  'hydra.run.dir=/data/outputs/valid_full/${now:%Y-%m-%d_%H-%M-%S}'
tracklab -cn soccernet num_cores=0 dataset.nvid=-1 dataset.eval_set=test \
  'hydra.run.dir=/data/outputs/test_full/${now:%Y-%m-%d_%H-%M-%S}'
```
`data_dir` set to `/data` (config appends `/SoccerNetGS`). Outputs directed to scratch (`/data`, mounted
from `/mnt/faster2/ga757`) to avoid the 25 GB home quota.

## Fix applied to complete the runs
NBJW calibration crashed the full run on degenerate frames: `cv2.calibrateCamera()` threw `cv2.error`
(too few / collinear point correspondences) → uncaught → killed the job (originally died on valid seq 58).
Patch (`patch_nbjw_calib.py`, idempotent):
1. `plugins/calibration/nbjw_calib/utils/utils_calib.py` — wrap `cv2.calibrateCamera` in `try/except cv2.error` → return `(None, None)` (matches the existing failure contract).
2. `sn_gamestate/calibration/nbjw_calib.py` — guard both `heuristic_voting()['cam_params']` sites against a `None` return.
Behaviour change: unrecoverable-calibration frames yield no camera params (handled by existing None /
previous-homography paths) instead of crashing. Only affects frames that previously aborted the run.

## Artifacts (on HEX scratch — back up, not persisted by HEX)
- valid: `/mnt/faster2/ga757/outputs/valid_full/2026-07-13_03-02-41/`
- test:  `/mnt/faster2/ga757/outputs/test_full/2026-07-13_17-52-11/`
- Each contains: `states/sn-gamestate.pklz` (tracker state), `visualization/videos/*.mp4`, `eval/results/tracklab/*_summary.txt`, `main.log`, `configs/`.

## Reproducibility notes
- Re-score any saved state without re-tracking: `tracklab -cn soccernet num_cores=0 'pipeline=[]' state.load_file=<run>/states/sn-gamestate.pklz visualization.cfg.save_videos=False dataset.nvid=-1 'hydra.run.dir=/data/outputs/eval_x/${now:%H-%M-%S}'`.
- GS-HOTA has large per-clip variance (paper reports single clips from 0.23% to 49.69%); only full-split averages are meaningful.
- Runtime ≈ 14 min/sequence on the A5000 (paper: ~11 min on A100).
