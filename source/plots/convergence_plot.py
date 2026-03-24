import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from IPython.core.pylabtools import figsize
from matplotlib.lines import lineStyles
from torch.nn.modules import conv

from source.config import locations

mpl.rcParams["figure.dpi"] = 600
plt.style.use("seaborn-v0_8")


LOSS_NAME_MAPPING = {
    "training_box_reg.json": "Head: Box Regression",
    "training_classifier.json": "Head: Classification",
    "training_objectness.json": "RPN: Objectness",
    "training_rpn_box_reg.json": "RPN: Box Regression",
}

PERFORMANCE_FILES = (
    "00_02_45_to_00_03_10_cut.json",
    "2019_09_02_GOPR5871_1058_solo.json",
    "gopro_001.json",
    "gopro_004.json",
)

NAME_ABBREVIATION = {
    "00_02_45_to_00_03_10_cut": "Easy",
    "2019_09_02_GOPR5871_1058_solo": "Hard",
    "gopro_001": "Medium 1",
    "gopro_004": "Medium 2",
}

FIG_SIZE = (8, 4)

TARGET_FOLDER = locations.RESULTS_DIR / "convergence" / "final"


def get_convergence_data(convergence_file: Path) -> pd.DataFrame:

    with open(
        convergence_file,
        mode="r",
        encoding="utf-8",
    ) as f:
        raw_data_list = json.load(fp=f)

    result = []
    for i, row in enumerate(raw_data_list):
        result.append({"epoch": i + 1, "images": row[1], "value": row[2]})

    return pd.DataFrame(result)


def add_ax_content(
    axis,
    data_original,
    ylabel,
    legend_title,
    show_comparison,
    data_comp,
    legend_title_comp,
):

    line_handles_1 = []
    line_labels_1 = []

    for entry in data_original:
        for name, df in entry.items():
            if name in NAME_ABBREVIATION:
                name = NAME_ABBREVIATION[name]
            (line,) = axis.plot(
                df["images"],
                df["value"],
                label=name,
                marker=".",
            )
            line_handles_1.append(line)
            line_labels_1.append(name)

    legend_1 = axis.legend(
        handles=line_handles_1,
        labels=line_labels_1,
        title=legend_title,
        loc="upper left",
        bbox_to_anchor=(1.01, 0.95),
        borderaxespad=0.0,
        alignment="left",
    )

    # after plotting baseline
    color_by_name = {line.get_label(): line.get_color() for line in line_handles_1}

    if show_comparison:
        line_handles_2 = []
        line_labels_2 = []
        for entry in data_comp:
            for name, df in entry.items():
                if name in NAME_ABBREVIATION:
                    name = NAME_ABBREVIATION[name]
                (line,) = axis.plot(
                    df["images"],
                    df["value"],
                    label=name,
                    marker=".",
                    linestyle="--",
                    color=color_by_name.get(name, None),
                )
                line_handles_2.append(line)
                line_labels_2.append(name)

        legend_2 = axis.legend(
            handles=line_handles_2,
            labels=line_labels_2,
            title=legend_title_comp,
            loc="upper left",
            bbox_to_anchor=(1.01, 0.45),
            borderaxespad=0.0,
            alignment="left",
        )
        axis.add_artist(legend_1)

    axis.set_ylabel(ylabel)


def plot_convergence(
    title: str,
    data_loss: list[dict[str, pd.DataFrame]],
    data_perf: list[dict[str, pd.DataFrame]],
    target_file: Path,
    legend_title: str = "",
    add_comparison: bool = False,
    comparison_legend_title: str = "",
    data_loss_comp: list[dict[str, pd.DataFrame]] = [{"": pd.DataFrame([])}],
    data_perf_comp: list[dict[str, pd.DataFrame]] = [{"": pd.DataFrame([])}],
):

    # DvB Performance
    # ---------------

    if len(data_perf) > 0:

        fig = plt.figure(figsize=FIG_SIZE, layout="constrained")
        ax = fig.add_axes((0.10, 0.12, 0.65, 0.85))
        add_ax_content(
            axis=ax,
            data_original=data_perf,
            ylabel="AP @ IoU=0.5",
            legend_title=legend_title,
            show_comparison=add_comparison,
            data_comp=data_perf_comp,
            legend_title_comp=f"{comparison_legend_title}",
        )

        ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
        ax.set_xlabel("Images Seen in Training")
        # fig.suptitle(title)
        plt.savefig(TARGET_FOLDER / f"{target_file.stem}_dvb.svg")

    # Loss
    # ----

    if len(data_loss) > 0:

        fig = plt.figure(figsize=FIG_SIZE, layout="constrained")
        ax = fig.add_axes((0.10, 0.12, 0.65, 0.85))
        # fig, ax = plt.subplots(
        #     figsize=FIG_SIZE,
        #     layout="constrained",
        # )

        add_ax_content(
            axis=ax,
            data_original=data_loss,
            ylabel="Loss",
            legend_title=legend_title,
            show_comparison=add_comparison,
            data_comp=data_loss_comp,
            legend_title_comp=comparison_legend_title,
        )

        ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
        ax.set_xlabel("Images Seen in Training")
        # fig.suptitle(title)  # , x=(rect[0] + rect[2]) / 2)
        plt.savefig(TARGET_FOLDER / f"{target_file.stem}_loss.svg")


def produce_convergence_plot_from_data(
    convergence_folder: Path,
    target_file_name: str,
    plot_title: str,
    legend_title: str = "",
    show_dvb: bool = True,
    show_loss: bool = True,
    show_comparison: bool = False,
    comparison_folder: Path = Path("."),
    comparison_name: str = "",
):

    loss_folder = convergence_folder / "loss"
    performance_folder = convergence_folder / "dvb"

    dvb_performance = []
    if show_dvb:
        for file_name in PERFORMANCE_FILES:
            data_file = performance_folder / file_name
            dvb_performance.append({data_file.stem: get_convergence_data(data_file)})

    loss = []
    if show_loss:
        for file_name in LOSS_NAME_MAPPING.keys():
            data_file = loss_folder / file_name
            loss.append({LOSS_NAME_MAPPING[file_name]: get_convergence_data(data_file)})

    comparison_kwargs = {}

    if show_comparison:

        comparison_kwargs["add_comparison"] = True
        comparison_kwargs["comparison_legend_title"] = comparison_name

        comp_loss_folder = comparison_folder / "loss"
        comp_perf_folder = comparison_folder / "dvb"

        if show_dvb:
            comp_dvb = []
            for file_name in PERFORMANCE_FILES:
                data_file = comp_perf_folder / file_name
                comp_dvb.append({data_file.stem: get_convergence_data(data_file)})
            comparison_kwargs["data_perf_comp"] = comp_dvb

        if show_loss:
            comp_loss = []
            for file_name in LOSS_NAME_MAPPING.keys():
                data_file = comp_loss_folder / file_name
                comp_loss.append(
                    {LOSS_NAME_MAPPING[file_name]: get_convergence_data(data_file)}
                )
            comparison_kwargs["data_loss_comp"] = comp_loss

    plot_convergence(
        title=plot_title,
        data_loss=loss,
        data_perf=dvb_performance,
        target_file=convergence_folder.parent / target_file_name,
        legend_title=legend_title,
        **comparison_kwargs,
    )


if __name__ == "__main__":

    # Starting Point
    produce_convergence_plot_from_data(
        locations.RESULTS_DIR / "convergence" / "final" / "baseline",
        plot_title="Starting Point Convergence",
        target_file_name="baseline_convergence.svg",
        legend_title="Starting Point",
        show_loss=True,
    )

    # Size and anchors
    produce_convergence_plot_from_data(
        locations.RESULTS_DIR / "convergence" / "final" / "resolution_anchor",
        plot_title="Convergence with Smaller Anchors and Higher Resolution",
        target_file_name="anchor_convergence.svg",
        show_comparison=True,
        show_loss=True,
        legend_title="Anchors and Resolution",
        comparison_folder=locations.RESULTS_DIR / "convergence" / "final" / "baseline",
        comparison_name="Starting Point",
    )

    # Backbone
    produce_convergence_plot_from_data(
        locations.RESULTS_DIR / "convergence" / "final" / "backbone",
        plot_title="Convergence with Frozen Backbone",
        target_file_name="backbone_convergence.svg",
        legend_title="Frozen Backbone",
        show_comparison=True,
        show_loss=True,
        comparison_folder=locations.RESULTS_DIR
        / "convergence"
        / "final"
        / "resolution_anchor",
        comparison_name="Anchors and Resolution",
    )

    # Learning Rate
    produce_convergence_plot_from_data(
        locations.RESULTS_DIR / "convergence" / "final" / "learning_rate",
        plot_title="Convergence with Increased Learning Rate",
        target_file_name="lr_convergence.svg",
        legend_title="Learning Rate Increase",
        show_comparison=True,
        comparison_folder=locations.RESULTS_DIR / "convergence" / "final" / "backbone",
        comparison_name="Frozen Backbone",
    )

    # Bouning Box Min Size Variation
    produce_convergence_plot_from_data(
        locations.RESULTS_DIR
        / "convergence"
        / "final"
        / "min-box-size-variation"
        / "min-box-size-4.0",
        plot_title="Convergence with Bounding Box Minimum Size Variation",
        target_file_name="bbox_min_4.0_vs_2.0.svg",
        legend_title="min_size = 4",
        show_dvb=True,
        show_loss=False,
        show_comparison=True,
        comparison_folder=locations.RESULTS_DIR
        / "convergence"
        / "final"
        / "min-box-size-variation"
        / "min-box-size-2.0",
        comparison_name="min_size = 2",
    )
    produce_convergence_plot_from_data(
        locations.RESULTS_DIR
        / "convergence"
        / "final"
        / "min-box-size-variation"
        / "min-box-size-4.0",
        plot_title="Convergence with Bounding Box Minimum Size Variation",
        target_file_name="bbox_min_4.0_vs_8.0.svg",
        legend_title="min_size = 4",
        show_dvb=True,
        show_loss=False,
        show_comparison=True,
        comparison_folder=locations.RESULTS_DIR
        / "convergence"
        / "final"
        / "min-box-size-variation"
        / "min-box-size-8.0",
        comparison_name="min_size = 8",
    )

    # Versus Baseline
    produce_convergence_plot_from_data(
        locations.RESULTS_DIR
        / "convergence"
        / "final"
        / "min-box-size-variation"
        / "min-box-size-4.0",
        plot_title="Convergence with All Model Modifications",
        target_file_name="overall_convergence.svg",
        legend_title="New Baseline",
        show_comparison=True,
        show_loss=True,
        comparison_folder=locations.RESULTS_DIR / "convergence" / "final" / "baseline",
        comparison_name="Starting Point",
    )
