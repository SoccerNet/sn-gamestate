#!/usr/bin/env python3
"""Build the stitching tuning curve from the per-threshold eval summaries."""
import os

GT_IDS = 1348
BASE = "thesis_results/phase1/valid_eval/results/tracklab/cls_comb_det_av_summary.txt"
THRESHOLDS = ["0.20", "0.30", "0.40", "0.50", "0.60"]
COLS = ["HOTA", "DetA", "AssA", "AssRe", "AssPr", "IDF1", "IDSW", "IDs"]


def read_summary(p):
    lines = open(p).read().splitlines()
    return dict(zip(lines[0].split(), lines[1].split()))


def fmt(k, s):
    x = float(s)
    if k in ("IDSW", "IDs"):
        return "%9d" % int(x)
    return "%9.2f" % x


rows = [("raw", read_summary(BASE))]
for c in THRESHOLDS:
    p = "sweep/eval_c%s/eval/results/tracklab/cls_comb_det_av_summary.txt" % c
    if os.path.exists(p):
        rows.append((c, read_summary(p)))
    else:
        print("(missing, skipped: %s)" % p)

hdr = ("%-7s" % "cos") + "".join("%9s" % c for c in COLS) + ("%9s" % "IDs/GT")
print()
print(hdr)
print("-" * len(hdr))
for name, d in rows:
    line = ("%-7s" % name) + "".join(fmt(c, d[c]) for c in COLS)
    line += "%9.2f" % (float(d["IDs"]) / GT_IDS)
    print(line)
print()
print("HOTA col = GS-HOTA. Pick the cos that maximises AssA (or GS-HOTA).")
print("If AssPr collapses while AssA plateaus -> over-merging; back off.")
