import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


sns.set_style("whitegrid", {"axes.grid": False})


def _get_ax(b, t, batch_size, temporal_dim, axs):
    if batch_size == 1 and temporal_dim == 1:
        return axs
    elif batch_size > 1 and temporal_dim == 1:
        return axs[b]
    elif batch_size == 1 and temporal_dim > 1:
        return axs[t]
    else:
        return axs[b, t]


def plot_per_step_loss(per_step_losses, image_ids=None, ylim=[0.0, 0.2]):

    batch_dim, temporal_dim, max_iter = per_step_losses.shape
    _, axs = plt.subplots(
        nrows=batch_dim,
        ncols=temporal_dim,
        sharex=True,
        sharey=True,
        figsize=(10, 2 * batch_dim),
    )

    for b in range(batch_dim):
        for t in range(temporal_dim):
            ax = _get_ax(b, t, batch_dim, temporal_dim, axs)
            ax.plot(per_step_losses[b, t])
            ax.grid(visible=True)
            ax.set_ylim(ylim)
            # ax.set_yscale("log")
            ax.set_xlim([0, max_iter])
            if image_ids:
                ax.set_title(str(image_ids[b][t]))

    plt.tight_layout()
    return ax


def plot_per_step_lr(per_step_lr):
    _, ax = plt.subplots(figsize=(8, 2))
    ax.plot(per_step_lr)
    ax.grid(visible=True)
    plt.xlabel("step")
    plt.ylabel("learning rate")
    plt.tight_layout()
    return ax


def plot_per_stadium_loss(
    df,
    output_dir,
    tau_palette=["gray", "k", "r", "g", "b", "purple", "yellow", "orange"],
    taus=[0.02, 0.017, 0.015, 0.01],
    loss_keys=["loss_ndc_total"],
    axline_style_kwargs={"alpha": 0.5, "linestyle": "--"},
):

    for loss_key in loss_keys:
        fig, ax = plt.subplots(figsize=(20, 14))
        sns.boxplot(x=loss_key, y="stadium (number of images)", data=df)
        for i, (tau, color) in enumerate(zip(taus, tau_palette)):
            if i == 0:
                continue
            ax.axvline(x=tau, color=color, **axline_style_kwargs)
        plt.xlim([0.0, 0.15])
        fout = output_dir / f"box_per_stadium__{loss_key}.pdf"
        plt.savefig(fout, bbox_inches="tight")


def plot_loss_dataset(
    df,
    output_dir,
    tau_palette=["gray", "k", "r", "g", "b", "purple", "yellow", "orange"],
    taus=[0.02, 0.017, 0.015, 0.01],
    col_subset=["loss_ndc_total", "loss_ndc_total_max"],
    rename_col_subset=["mean", "max"],
    axline_style_kwargs={"alpha": 0.5, "linestyle": "--"},
):
    sns.set()
    df_plot = df.sort_values(by=col_subset[0], ascending=True)[col_subset]
    df_plot = df_plot.rename(columns={k: v for k, v in zip(col_subset, rename_col_subset)})
    df_plot["dataset fraction"] = list(range(0, len(df_plot.index)))
    df_plot["dataset fraction"] /= len(df_plot.index)
    df_plot = df_plot.melt(
        value_vars=rename_col_subset,
        value_name="projection error",
        var_name="per-point aggregation",
        id_vars=["dataset fraction"],
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(
        x="dataset fraction",
        y="projection error",
        hue="per-point aggregation",
        data=df_plot,
        ax=ax,
    )
    plt.xticks(np.arange(0, 1.05, 0.05))
    for i, (tau, color) in enumerate(zip(taus, tau_palette)):
        if i == 0:
            continue
        ax.axhline(y=tau, color=color, **axline_style_kwargs)

    plt.ylim([0.0, 0.1])
    plt.yticks(np.arange(0, 0.105, 0.01))
    fout = output_dir / f"projection_error_{col_subset[0]}.pdf"
    plt.savefig(fout, bbox_inches="tight")
    print(fout)
