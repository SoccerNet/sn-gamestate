#%%
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd

# variant 1) multiple initializations, groupby image_id select minimum loss; global tau
# --> see argmin_from_individual_results.py

# variant 2) per-camera optim (subset); stack results; global tau
args = ArgumentParser()
args.add_argument("--result_dir_base", type=Path, default=Path("./experiments"))
args.add_argument("--subset_glob", type=str, default="val_?_gt_optimal_extrem_pred")
args.add_argument(
    "--subsets",
    nargs="+",
    default=["center", "left", "right"],
    help="Replace '?' with provided values",
)
args = args.parse_args()

subset_dirs = [args.subset_glob.replace("?", subset) for subset in args.subsets]

dfs = []
for subset_dir in subset_dirs:
    file_per_sample_output = args.result_dir_base / subset_dir / "per_sample_output.json"
    if not file_per_sample_output.exists():
        raise FileNotFoundError(file_per_sample_output)

    df_subset = pd.read_json(file_per_sample_output, orient="records", lines=True)
    df_subset["subset"] = file_per_sample_output.parent.name
    df_subset.drop(
        columns=[
            "batch_idx",
            "time_s",
            "mask_lines",
            "mask_circles",
            "loss_ndc_lines_distances_raw",
            "loss_ndc_circles_distances_raw",
            "loss_ndc_lines_distances_max",
            "loss_ndc_circles_distances_max",
        ],
        inplace=True,
        errors="ignore",
    )
    dfs.append(df_subset)
df = pd.concat(dfs)
df["image_id"] = df["image_ids"].apply(lambda s: s.split(".jpg")[0])
df.set_index("image_id", inplace=True)
print(df.subset.value_counts())
#%%
fout = args.result_dir_base / args.subset_glob.replace("?", "stacked") / "per_sample_output.json"
print(fout)
fout.parent.mkdir(exist_ok=True, parents=True)
df.to_json(fout, orient="records", lines=True)
