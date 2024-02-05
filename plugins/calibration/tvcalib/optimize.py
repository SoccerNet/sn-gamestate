import json
from argparse import ArgumentParser
from pathlib import Path
from time import time
from datetime import datetime

import torch
import matplotlib.pyplot as plt
import pandas as pd
from pytorch_lightning import seed_everything

seed_everything(seed=10, workers=True)

from tvcalib.cam_modules import SNProjectiveCamera
from tvcalib.sncalib_dataset import FixedInputSizeDataset, custom_list_collate
from tvcalib.module import TVCalibModule
from tvcalib.utils.objects_3d import (
    SoccerPitchLineCircleSegments,
    SoccerPitchSN,
    SoccerPitchSNCircleCentralSplit,
)
import tvcalib.utils.io as io
from tvcalib.utils.visualization_mpl import (
    plot_loss_dataset,
    plot_per_stadium_loss,
    plot_per_step_loss,
    plot_per_step_lr,
)


args = ArgumentParser()
args.add_argument("--hparams", type=Path)
args.add_argument("--log_per_step", action="store_true")
args.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
args.add_argument("--output_dir", type=Path, default="experiments")
args.add_argument("--exp_timestmap", action="store_true")
args.add_argument("--overwrite_init_cam_distr", type=str)
args = args.parse_args()

with open(args.hparams) as fw:
    hparams = json.load(fw)

init_cam_distr = hparams["dataset"]["filter_cam_type"]
# TODO generic import
if args.overwrite_init_cam_distr is not None:
    init_cam_distr = args.overwrite_init_cam_distr
if init_cam_distr == "Main camera left":
    from tvcalib.cam_distr.tv_main_left import get_cam_distr, get_dist_distr
elif init_cam_distr == "Main camera right":
    from tvcalib.cam_distr.tv_main_right import get_cam_distr, get_dist_distr
elif init_cam_distr == "Main behind the goal":
    from tvcalib.cam_distr.tv_main_behind import get_cam_distr, get_dist_distr
elif init_cam_distr == "Main camera center":
    from tvcalib.cam_distr.tv_main_center import get_cam_distr, get_dist_distr
elif init_cam_distr == "Main tribune":
    from tvcalib.cam_distr.tv_main_tribune import get_cam_distr, get_dist_distr

    hparams["dataset"]["filter_cam_type"] = False
else:
    from tvcalib.cam_distr.tv_main_center import get_cam_distr, get_dist_distr


distr_lens_disto = None
distr_cam = get_cam_distr(hparams["sigma_scale"], hparams["batch_dim"], hparams["temporal_dim"])
if hparams["lens_distortion"] == True:
    distr_lens_disto = get_dist_distr(hparams["batch_dim"], hparams["temporal_dim"])
hparams["distr_cam"] = distr_cam
hparams["distr_lens_disto"] = distr_lens_disto


output_dir = args.output_dir / args.hparams.stem
if args.exp_timestmap:
    output_dir = output_dir / datetime.now().strftime("%y%m%d-%H%M")
output_dir.mkdir(exist_ok=True, parents=True)
print("output directory", output_dir)

if (
    "split_circle_central" in hparams["dataset"]
    and hparams["dataset"]["split_circle_central"] == True
):
    base_field = SoccerPitchSNCircleCentralSplit()
else:
    base_field = SoccerPitchSN()
object3d = SoccerPitchLineCircleSegments(device=args.device, base_field=base_field)
print(base_field.__class__, object3d.__class__)

print("Init Dataset")
dataset = FixedInputSizeDataset(
    model3d=object3d,
    image_width=hparams["image_width"],
    image_height=hparams["image_height"],
    constant_cam_position=hparams["temporal_dim"],
    **hparams["dataset"],
)
print(dataset.df_match_info.head(5))
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=hparams["batch_dim"],
    num_workers=4,
    shuffle=False,
    collate_fn=custom_list_collate,
)

print("Init TVCalibModule")
model = TVCalibModule(
    object3d,
    distr_cam,
    distr_lens_disto,
    (hparams["image_height"], hparams["image_width"]),
    hparams["optim_steps"],
    args.device,
    log_per_step=args.log_per_step,
    tqdm_kwqargs={"ncols": 100},
)
hparams["TVCalibModule"] = model.hparams
print(output_dir / "hparams.yml")
io.write_yaml(hparams, output_dir / "hparams.yml")

dataset_dict_stacked = {}
dataset_dict_stacked["batch_idx"] = []
for batch_idx, x_dict in enumerate(dataloader):

    print(f"{batch_idx}/{len(dataloader)-1}")
    points_line = x_dict["lines__px_projected_selection_shuffled"].clone().detach()
    points_circle = x_dict["circles__px_projected_selection_shuffled"].clone().detach()
    batch_size = points_line.shape[0]

    fout_prefix = f"batch_{batch_idx}"

    start = time()
    per_sample_loss, cam, per_step_info = model.self_optim_batch(x_dict)

    output_dict = {
        "image_ids": x_dict["image_id"],
        "camera": cam.get_parameters(batch_size),
        # "time_s": time() - start,
        **per_sample_loss,
        "meta": x_dict["meta"],
    }
    if args.log_per_step:
        output_dict["per_step_lr"] = per_step_info["lr"].squeeze(-1)  # (optim_steps,)
        output_dict["per_step_loss"] = per_step_info["loss"]  # (B, T, optim_steps)

    print(output_dir / f"{fout_prefix}.pt")
    torch.save(io.detach_dict(output_dict), output_dir / f"{fout_prefix}.pt")

    if args.log_per_step:
        _ = plot_per_step_loss(per_step_info["loss"].cpu(), output_dict["image_ids"])
        print(output_dir / f"{fout_prefix}_loss.pdf")
        plt.savefig(output_dir / f"{fout_prefix}_loss.pdf")
        _ = plot_per_step_lr(per_step_info["lr"].cpu())
        print(output_dir / f"{fout_prefix}_lr.pdf")
        plt.savefig(output_dir / f"{fout_prefix}_lr.pdf")
        plt.close("all")

    # format for per_sample_output.json
    if "per_step_lr" in output_dict:
        del output_dict["per_step_lr"]
    # max distance over all given points
    output_dict["loss_ndc_lines_distances_max"] = (
        output_dict["loss_ndc_lines_distances_raw"].amax(dim=[-2, -1]).squeeze(-1)
    )
    output_dict["loss_ndc_circles_distances_max"] = (
        output_dict["loss_ndc_circles_distances_raw"].amax(dim=[-2, -1]).squeeze(-1)
    )
    output_dict["loss_ndc_total_max"] = torch.stack(
        [
            output_dict["loss_ndc_lines_distances_max"],
            output_dict["loss_ndc_circles_distances_max"],
        ],
        dim=-1,
    ).max(dim=-1)[0]
    output_dict = io.tensor2list(output_dict)

    output_dict["batch_idx"] = [[str(batch_idx)]] * batch_size

    if "time_s" in output_dict:
        output_dict["time_s"] /= batch_size

    if "camera" in output_dict["meta"]:
        del output_dict["meta"]["camera"]
    output_dict.update(output_dict["meta"])
    output_dict.update(output_dict["camera"])
    del output_dict["meta"]
    del output_dict["camera"]

    for k in output_dict.keys():
        if k not in dataset_dict_stacked:
            dataset_dict_stacked[k] = output_dict[k]
        elif isinstance(dataset_dict_stacked[k], list):
            dataset_dict_stacked[k].extend(output_dict[k])
        else:
            dataset_dict_stacked[k] = output_dict[k]

df = pd.DataFrame.from_dict(dataset_dict_stacked)
print(df)
explode_cols = [k for k, v in dataset_dict_stacked.items() if isinstance(v, list)]
df = df.explode(column=explode_cols)  # explode over t
df["image_id"] = df["image_ids"].apply(lambda l: l.split(".jpg")[0])
df.set_index("image_id", inplace=True)

if "match" in df.columns:
    df["stadium"] = df["match"].apply(lambda s: s.split(" - ")[0].strip())
    number_of_images_per_stadium = df.groupby("stadium")["stadium"].agg(len).to_dict()
    df["stadium (number of images)"] = df["stadium"].apply(
        lambda stadium: f"{stadium} ({number_of_images_per_stadium[stadium]})"
    )
fout = output_dir / "per_sample_output.json"
df.to_json(fout, orient="records", lines=True)

if "match" in df.columns:
    plot_per_stadium_loss(df, output_dir)

plot_loss_dataset(df, output_dir)
