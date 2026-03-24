import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.typing import CapStyleType

from source.config import locations
from source.config.model_register import ModelRegister

mpl.rcParams["figure.dpi"] = 600
plt.style.use("seaborn-v0_8")


def get_ap05_result(model_state: int, nms_thresh: str = "") -> pd.Series:
    file = (
        f"model_state_{model_state}.json"
        if nms_thresh == ""
        else f"model_state_{model_state}_nms_{nms_thresh}.json"
    )
    with open(
        locations.Results.ap05_jsons / file,
        mode="r",
        encoding="utf-8",
    ) as f:
        return pd.Series(json.load(fp=f)).sort_values()


def print_averages(df: pd.DataFrame):
    print("Averages:")
    for name in df.columns:
        print(f"{name}: {df[name].mean()}")


def plot_results(df: pd.DataFrame, file_name: str, legend_title: str = ""):

    fig, ax = plt.subplots()
    for name in df.columns:
        ax.plot(
            df.index,
            df[name],
            label=name,
        )
    ax.tick_params("x", rotation=90, labelsize=6)
    ax.legend(title=legend_title, alignment="left")
    ax.set_title("Drone Detection Performance on Drone vs. Bird Videos")
    ax.set_ylabel("AP @ IoU=0.5")
    ax.set_xmargin(m=0)
    fig.tight_layout()
    plt.savefig(locations.Results.ap05_images / file_name)


def baseline_plot():

    data = get_ap05_result(ModelRegister.baseline, "0.5").to_frame(name="Baseline")
    data["Starting Point"] = get_ap05_result(
        ModelRegister.starting_point,
        "0.5",
    )

    fig, ax = plt.subplots()

    ax.plot(data.index, data["Baseline"], label="Baseline", zorder=3)
    change = data["Baseline"] - data["Starting Point"]
    better = change > 0
    ax.errorbar(
        data.index[better],
        data.loc[better, "Starting Point"],
        yerr=abs(change.loc[better]),
        uplims=(data.loc[better, "Baseline"] - data.loc[better, "Starting Point"]) <= 0,
        lolims=(data.loc[better, "Baseline"] - data.loc[better, "Starting Point"]) > 0,
        label="Baseline Better Than Starting Point",
        linestyle="",
        marker=".",
        elinewidth=1,
        capsize=0.1,
    )
    worse = change <= 0
    ax.errorbar(
        data.index[worse],
        data.loc[worse, "Starting Point"],
        yerr=abs(change.loc[worse]),
        uplims=(data.loc[worse, "Baseline"] - data.loc[worse, "Starting Point"]) <= 0,
        lolims=(data.loc[worse, "Baseline"] - data.loc[worse, "Starting Point"]) > 0,
        label="Baseline Worse Than Starting Point",
        linestyle="",
        marker=".",
        elinewidth=1,
        capsize=0.1,
    )

    ap_thresholds = [0.2, 0.4, 0.6, 0.8]
    masks = [data.iloc[:, 0] < treshold for treshold in ap_thresholds]
    x_thresholds = [len(data[mask]) - 0.5 for mask in masks]

    for x_threshold in x_thresholds:
        ax.vlines(
            x=x_threshold,
            ymin=0,
            ymax=1.2,
            color="C3",
            zorder=2,
            linewidth=1.2,
            linestyle="--",
        )

    ap_range_names = ["Lowest", "Low", "Mid", "Good", "Best"]
    x_text = [
        (0 + x_thresholds[0]) / 2,
        (x_thresholds[0] + x_thresholds[1]) / 2,
        (x_thresholds[1] + x_thresholds[2]) / 2,
        (x_thresholds[2] + x_thresholds[3]) / 2,
        (x_thresholds[3] + len(data.index)) / 2,
    ]
    y_text = [1.1] * 5  # [0.9, 0.9, 0.9, 0.9, 0.7]

    for x_pos, y_pos, ap_range_name in zip(x_text, y_text, ap_range_names):
        ax.text(
            x=x_pos,
            y=y_pos,
            s=ap_range_name,
            horizontalalignment="center",
            verticalalignment="center",
        )

    ax.tick_params("x", rotation=90, labelsize=6)
    ax.set_title("Drone Detection Performance on Drone vs. Bird Videos")
    ax.set_ylabel("AP @ IoU=0.5")
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xmargin(m=0)
    ax.legend(frameon=True, facecolor="white", framealpha=1, edgecolor="black")
    fig.tight_layout()
    plt.savefig(locations.Results.ap05_images / "ap05_baseline.svg")


def new_data_plot():

    data_new = get_ap05_result(ModelRegister.baseline, "0.5").to_frame(name="Baseline")
    data_new["New Data"] = get_ap05_result(ModelRegister.additional_data, "0.5")

    with open(
        locations.Results.ap05_jsons / "target.json",
        mode="r",
        encoding="utf-8",
    ) as f:
        target = pd.Series(json.load(fp=f))
    data_new["Target"] = target

    fig, ax = plt.subplots()
    ax.plot(
        data_new.index,
        data_new["Baseline"],
        label="Baseline",
    )
    change = data_new["New Data"] - data_new["Baseline"]
    better = change > 0
    ax.errorbar(
        data_new.index[better],
        data_new.loc[better, "New Data"],
        yerr=abs(change.loc[better]),
        uplims=(data_new.loc[better, "New Data"] - data_new.loc[better, "Baseline"])
        >= 0,
        lolims=(data_new.loc[better, "New Data"] - data_new.loc[better, "Baseline"])
        < 0,
        label="Additional Data Better Than Baseline",
        linestyle="",
        marker=".",
        elinewidth=1,
        capsize=0.1,
    )
    worse = change <= 0
    ax.errorbar(
        data_new.index[worse],
        data_new.loc[worse, "New Data"],
        yerr=abs(change.loc[worse]),
        uplims=(data_new.loc[worse, "New Data"] - data_new.loc[worse, "Baseline"]) >= 0,
        lolims=(data_new.loc[worse, "New Data"] - data_new.loc[worse, "Baseline"]) < 0,
        label="Additional Data Worse Than Baseline",
        linestyle="",
        marker=".",
        elinewidth=1,
        capsize=0.1,
    )

    ax.plot(
        data_new.index,
        data_new["Target"],
        label="Improvement Potential for $\\Delta Score_{TP} = +0.05$",
        linestyle="--",
        linewidth=1,
    )
    parot = data_new.index.str.contains("parot")
    parrot = data_new.index.str.contains("parrot")
    fixed_wing = data_new.index[parot | parrot]
    ax.vlines(
        fixed_wing,
        ymin=0,
        ymax=data_new.loc[fixed_wing, "Target"],
        label="Parrot Drone in Video",
        color="C5",
        zorder=1,
        linewidth=3,
        alpha=0.5,
    )

    ax.tick_params("x", rotation=90, labelsize=6)
    ax.set_title("Drone Detection Performance on Drone vs. Bird Videos")
    ax.set_ylabel("AP @ IoU=0.5")
    ax.set_xmargin(m=0)
    ax.legend(frameon=True, facecolor="white", framealpha=1, edgecolor="black")
    fig.tight_layout()
    plt.savefig(locations.Results.ap05_images / "data_new_data.svg")


def extra_data_plot():

    data = get_ap05_result(ModelRegister.baseline, "0.5").to_frame(name="Baseline")
    data["New Data"] = get_ap05_result(ModelRegister.additional_data, "0.5")
    data["Extra Data"] = get_ap05_result(ModelRegister.parrot_data, "0.5")

    fig, ax = plt.subplots()
    ax.plot(
        data.index,
        data["New Data"],
        label="Additional Data",
    )
    change = data["Extra Data"] - data["New Data"]
    better = change > 0
    ax.errorbar(
        data.index[better],
        data.loc[better, "Extra Data"],
        yerr=abs(change.loc[better]),
        uplims=(data.loc[better, "Extra Data"] - data.loc[better, "New Data"]) >= 0,
        lolims=(data.loc[better, "Extra Data"] - data.loc[better, "New Data"]) < 0,
        label="With Parrot Data Better Than Without",
        linestyle="",
        marker=".",
        elinewidth=1,
        capsize=0.1,
    )
    worse = change <= 0
    ax.errorbar(
        data.index[worse],
        data.loc[worse, "Extra Data"],
        yerr=abs(change.loc[worse]),
        uplims=(data.loc[worse, "Extra Data"] - data.loc[worse, "New Data"]) >= 0,
        lolims=(data.loc[worse, "Extra Data"] - data.loc[worse, "New Data"]) < 0,
        label="With Parrot Data Worse Than Without",
        linestyle="",
        marker=".",
        elinewidth=1,
        capsize=0.1,
    )

    with open(
        locations.Results.ap05_jsons / "target.json",
        mode="r",
        encoding="utf-8",
    ) as f:
        target = pd.Series(json.load(fp=f))
    data["Target"] = target
    ax.plot(
        data.index,
        data["Target"],
        label="Improvement Potential for $\\Delta Score_{TP} = +0.05$",
        linestyle="--",
        linewidth=1,
    )
    parot = data.index.str.contains("parot")
    parrot = data.index.str.contains("parrot")
    fixed_wing = data.index[parot | parrot]
    ax.vlines(
        fixed_wing,
        ymin=0,
        ymax=data.loc[fixed_wing, "Target"],
        label="Parrot Drone in Video",
        color="C5",
        zorder=1,
        linewidth=3,
        alpha=0.5,
    )

    ax.tick_params("x", rotation=90, labelsize=6)
    ax.set_title("Drone Detection Performance on Drone vs. Bird Videos")
    ax.set_ylabel("AP @ IoU=0.5")
    ax.set_xmargin(m=0)
    ax.legend(frameon=True, facecolor="white", framealpha=1, edgecolor="black")
    fig.tight_layout()
    plt.savefig(locations.Results.ap05_images / "extra_data.svg")


if __name__ == "__main__":

    baseline_plot()
    new_data_plot()
    extra_data_plot()
