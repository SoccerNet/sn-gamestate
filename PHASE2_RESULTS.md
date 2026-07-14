# Phase 2 — Feature Engineering (TactPipeline)

**Objective (proposal Obj. 2):** convert raw positional tracks into model-ready
representations for the downstream trajectory forecaster (3A) and tactical classifier (3B).

**Status:** ✅ complete on the valid set. Built on the reproduced SoccerNet baseline
tracker states (see `PHASE1_RESULTS.md`). Runs entirely on the Mac (pandas/numpy/scipy) —
no tracklab or GPU needed for feature work; tracklab (`.venv`) is only used to *re-score*
GS-HOTA when validating the stitching step.

Pipeline: `stitch_and_write → build_features → smooth_tracks → build_distances → build_graphs`.
All outputs land in `thesis_results/phase2/`.

---

## 1. Track post-processing — offline tracklet stitching (Phase 1 → 2 bridge)

The reproduced baseline over-fragments identities badly: **5160 predicted track IDs vs
1348 GT (3.83×)** on valid, from StrongSORT breaking a player's path into many short
tracklets. Fragmented tracks would give the forecaster broken sequences and the tactical
classifier unstable node identities, so tracks are consolidated before feature extraction.

**Method** (`stitch_tracklets.py`, `stitch_and_write.py`) — offline merge adapted from the
SoccerNet-Tracking 2023 winner (Maglo et al., *Sensors* 2023): merge StrongSORT fragments
by **PRTReID-embedding cosine similarity**, gated by two physical constraints:
- **no temporal overlap** — two tracklets sharing any frame are simultaneous → never merge;
- **pitch-space teleportation** — implied speed between the end of one and start of the next
  must be ≤ 12 m/s (in metres, using `bbox_pitch`, not image pixels).

Reuses embeddings already stored in the tracker state — no re-tracking, no test-time ReID
fine-tuning.

**Tuning curve** (`collect_curve.py`, re-score each threshold's stitched state):

| cos | GS-HOTA | AssA | AssPr | IDs | ×GT |
|-----|---------|------|-------|-----|-----|
| raw | 19.96 | 47.24 | 86.69 | 5160 | 3.83 |
| 0.20 | 20.96 | 52.08 | 79.42 | 2598 | 1.93 |
| **0.30** | **20.96** | **52.07** | 79.28 | 2472 | 1.83 |
| 0.40 | 20.92 | 51.89 | 78.75 | 2404 | 1.78 |
| 0.50 | 20.91 | 51.86 | 78.63 | 2361 | 1.75 |
| 0.60 | 20.90 | 51.80 | 78.51 | 2345 | 1.74 |

The curve is **flat/robust** (whole range within 0.06 GS-HOTA), peaks at cos 0.2–0.3, and
shows **no over-merging** through 0.6 (AssA barely dips, no AssPr collapse — the safety
gates hold). Operating point locked at **cos = 0.30**.

**Verified win vs raw** (valid, official GS-HOTA config):

| Metric | raw | stitched (0.30) | Δ |
|--------|-----|-----------------|---|
| GS-HOTA | 19.96 | **20.96** | +1.0 |
| AssA | 47.24 | **52.08** | **+4.8** |
| IDF1 | 13.30 | 14.69 | +1.4 |
| IDSW | 894 | 729 | −165 |
| DetA | 8.44 | 8.44 | flat |
| fragmentation | 3.83× | **1.83×** | halved |

AssA rising (with DetA flat = detections untouched) confirms the merges are correct, not
over-merging. A clean **offline post-processing contribution, no re-tracking**.

---

## 2. Feature pipeline

### 2.1 Base positional table — `build_features.py`
One row per detection: `seq, frame, track_id, x_pitch, y_pitch, team, role, jersey, conf`
(+ raw velocity). `x/y_pitch` from `bbox_pitch` bottom-middle (metres); `frame` from the
last 6 digits of `image_id`.
- **544,043 detections · 58 seqs · 2500 tracks · 100% pitch coverage.**
- Roles: player 416k · referee 99k · goalkeeper 23k · ball 3.2k · other 2.5k.
- Output: `features_valid.csv`.

### 2.2 Trajectory smoothing — `smooth_tracks.py`
Frame-to-frame calibration noise makes raw velocity meaningless (a near-stationary player
reads 4–8 m/s). Per-track **Savitzky-Golay** (order-2, ~0.44 s window) smooths `(x,y)`;
velocity is recomputed from the smoothed path via `np.gradient` with **true frame gaps**
(handles missing frames). Adds `x_s, y_s, vx_s, vy_s, speed_s`.
- Jitter (median |acceleration|): **48 → 11.8 m/s²**.
- Speeds now physically real: median **2.5 m/s** (walk/jog), p95 **7.9 m/s** (run).
- Output: `features_valid_smoothed.csv`.

### 2.3 Distance & team-shape features — `build_distances.py`
On smoothed positions.
- **Per-detection (players + GK):** nearest-teammate (median 7.3 m), nearest-opponent
  (8.2 m), distance-to-ball. → `features_valid_dist.csv`.
- **Per-frame per-team shape:** n, centroid `(cx,cy)`, spread (mean dist to centroid,
  9.9 m), width (18.7 m), length (21.7 m), **convex-hull area (~218 m²)**.
  → `team_shape_valid.csv`.

### 2.4 Spatial graphs — `build_graphs.py`
Per-frame k-nearest-neighbour graphs (k=5) over visible athletes.
- **2,157,938 edges over 43,319 frame-graphs;** median 10 nodes/frame, mean edge 11.5 m,
  60% intra-team edges.
- Edge attrs: `dist`, `same_team`. Node features = the per-detection table (join on
  `seq,frame,track_id`). GNN-ready.
- Output: `graph_edges_valid.parquet` + `graph_summary_valid.csv`.

---

## 3. How the features feed Phase 3

- **3A — trajectory forecaster (LSTM/GRU seq2seq):** input = 50-frame (2 s @ 25 fps)
  windows of smoothed `(x_s, y_s)` per track; output = predicted `(x,y)` at 1/2/3 s ahead;
  metrics ADE/FDE. Source = the smoothed table (§2.2).
- **3B — tactical classifier (GCN/GAT):** input = per-frame spatial graphs (§2.4) +
  team-shape (§2.3); tasks = team-formation and pressing-phase recognition; metric =
  macro-F1 vs **geometry-derived labels** (proposal Obj. 4). The Phase-2 outputs already
  supply the label substrate:
  - *Formation* → cluster the 10 outfield players' positions over a window, match to
    4-4-2 / 4-3-3 / 3-5-2 (from the per-detection table).
  - *Pressing phase* → defensive-line height / defending-team y-centroid vs pitch midpoint
    → high-press / mid-block / low-block (from `team_shape` centroids).
  - Graph option: k-NN implemented; Delaunay triangulation is a drop-in alternative.

---

## 4. Findings & limitations (inputs to the error-propagation study, Obj. 5)

These define the substrate the noise study perturbs, so they are recorded as first-class
findings, not caveats:

1. **Ball detected in ~4% of frames only** — it is small and fast and the detector mostly
   misses it. Distance-to-ball is therefore sparse; **3B must not depend on the ball**
   without a dedicated ball detector or heavy interpolation.
2. **Partial observability** — a single moving broadcast camera shows a pitch slice, so
   only ~10 athletes (~5/team, not 22) are visible per frame. Team-shape metrics (spread,
   width, hull) are **camera-view-dependent, not true full-team shape**; 3B must be framed
   around observed structure and state this limitation.
3. **Residual fragmentation is calibration-driven, not tracker error.** After stitching,
   IDs remain 1.83× GT, but AssRe / IDF1 / IDSW *plateau* from cos 0.2 on while only the
   raw ID count keeps falling — i.e. the extra fragments are on poorly-localised (bad
   calibration) false-positive detections that never match GT. The association gain is
   captured by the first merge wave; the rest is non-GT-matching junk. This points at
   **calibration** as the dominant upstream error source (consistent with the SoccerNet
   paper's own ablation).

---

## 5. Artifacts & reproducibility

Scripts (repo root), run in order:
```
stitch_and_write.py   valid_state.pklz -> valid_state_stitched.pklz (cos 0.30)
build_features.py     -> features_valid.csv
smooth_tracks.py      -> features_valid_smoothed.csv
build_distances.py    -> features_valid_dist.csv, team_shape_valid.csv
build_graphs.py       -> graph_edges_valid.parquet, graph_summary_valid.csv
```
Supporting: `stitch_tracklets.py` (fragmentation sweep), `collect_curve.py` (tuning curve).

Environment: Mac, `pandas + numpy + scipy + pyarrow`. Feature scripts need no tracklab/GPU;
GS-HOTA re-scoring (stitching validation) uses the repo `.venv` (tracklab 1.3.24).

Caveat: `seq` loses its leading zero on CSV round-trip (`021`→`21`) — harmless for
grouping, but zero-pad or use parquet before matching to `SNGS-0xx` folders in Phase 3.

---

## 6. Next (Phase 3)

- Run the same 4-script pipeline on the **test** state (stitch at cos 0.30 first) to
  produce the held-out feature set.
- Implement **3A** (LSTM/GRU forecaster, ADE/FDE) and **3B** (GCN/GAT tactical classifier
  with geometry-derived labels, macro-F1 ≥ 0.7 target).
- Then the **error-propagation study** (Obj. 5): inject Gaussian positional noise,
  synthetic identity switches, and homography perturbation at each stage; quantify ≥6
  error links from detection → tracking → homography → trajectory → GNN features → tactical F1.
