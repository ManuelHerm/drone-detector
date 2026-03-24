import json
from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from cycler import cycler
from IPython.core.pylabtools import figsize
from matplotlib import rcParams

from source.config import locations, names
from source.config.model_register import ModelRegister

mpl.rcParams["figure.dpi"] = 600
plt.style.use("seaborn-v0_8")
plt.rcParams.update(
    {
        "legend.frameon": True,
        "legend.facecolor": "white",
        "legend.edgecolor": "black",
        "legend.framealpha": 1.0,
    }
)

FIG_SIZE = (6, 6)

colors = rcParams["axes.prop_cycle"].by_key()["color"]
linestyles = ["-", "--", ":", "-."]  # add more if you have lots of columns

# build a combined cycle:
# first len(colors) lines are solid, next len(colors) dashed, etc.
prop_cycle = cycler(
    color=colors * len(linestyles),
    linestyle=[ls for ls in linestyles for _ in range(len(colors))],
)


def get_prec_recall_file(model_state: int, dataset_name: str, nms: float = 0.5) -> Path:
    return (
        locations.Results.coco_jsons
        / "dvb"
        / f"{model_state}"
        / f"box_nms_{nms:0.1f}"
        / f"{dataset_name}_rec_prec_conf_0.05_nms_{nms:0.2f}.json"
    )


def get_recall_precision_results(
    chart_files: Sequence[Path],
    column_variable: str,
) -> pd.DataFrame:

    initialized = False
    for chart_file in chart_files:
        with open(
            chart_file,
            mode="r",
            encoding="utf-8",
        ) as f:
            chart_dict = json.load(fp=f)
            if isinstance(chart_dict[column_variable], str):
                column_name = chart_dict[column_variable]
            else:
                column_name = f"{chart_dict[column_variable]:0.2f}"
            if not initialized:
                initialized = True
                data = pd.Series(
                    chart_dict["precision"], index=chart_dict["recall"]
                ).to_frame(name=column_name)
            else:
                data[column_name] = pd.Series(
                    chart_dict["precision"], index=chart_dict["recall"]
                )

    return data


def plot(data: pd.DataFrame, target: Path, fig_size: tuple[int, int]):
    fig, ax = plt.subplots(figsize=fig_size)

    # ax.set_prop_cycle(prop_cycle)
    for column in data.columns:
        ax.plot(
            data.index, data[column], label=f"{column} ({data[column].mean():0.2f})"
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(
        title="Video Name (AP05)",
        alignment="left",
        title_fontproperties={"weight": "bold"},
    )
    ax.set_xlim(left=-0.05, right=1.05)
    ax.set_ylim(bottom=-0.05, top=1.05)
    fig.tight_layout()
    plt.savefig(target)
    plt.close()


def plot_nms_conf_experiment():
    fig_size = (5, 5)
    dir = locations.RESULTS_DIR / "mns-conf-experiment"
    nms_dir = dir / "nms-variation"
    conf_dir = dir / "conf-variation"

    nms = get_recall_precision_results(
        [folder for folder in sorted(nms_dir.iterdir())],
        column_variable="nms_threshold",
    )

    fig, ax = plt.subplots(figsize=fig_size)

    for column in nms.columns:
        ax.plot(nms.index, nms[column], label=f"{column} ({nms[column].mean():0.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(
        title="box_nms_thres Value",
        alignment="left",
        title_fontproperties={"weight": "bold"},
    )
    ax.set_xlim(left=-0.05, right=1.05)
    ax.set_ylim(bottom=-0.05, top=1.05)
    fig.tight_layout()
    plt.savefig(dir / "nms.svg")
    plt.close()

    conf = get_recall_precision_results(
        [folder for folder in sorted(conf_dir.iterdir())],
        column_variable="score_threshold",
    )
    fig, ax = plt.subplots(figsize=fig_size)

    for column in conf.columns:
        ax.plot(
            conf.index, conf[column], label=f"{column} ({conf[column].mean():0.2f})"
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(
        title="box_score_thresh Value",
        alignment="left",
        title_fontproperties={"weight": "bold"},
    )
    ax.set_xlim(left=-0.05, right=1.05)
    ax.set_ylim(bottom=-0.05, top=1.05)
    fig.tight_layout()
    plt.savefig(dir / "conf.svg")
    plt.close()


ap05_401 = pd.Series(
    {
        "dji_mavick_hillside_off_focus": 0.003601,
        "gopro_007": 0.004645,
        "2019_10_16_C0003_3633_inspire": 0.006453,
        "2019_08_19_GP015869_1520_inspire": 0.017156,
        "parrot_clear_birds": 0.037927,
        "two_distant_phantom": 0.081223,
        "gopro_006": 0.114787,
        "dji_phantom_4_hillside_cross": 0.145979,
        "GOPR5848_002": 0.249776,
        "dji_phantom_4_long_takeoff": 0.250340,
        "distant_parrot_with_birds": 0.257625,
        "gopro_002": 0.273307,
        "GOPR5843_002": 0.276458,
        "dji_phantom_4_swarm_noon": 0.278093,
        "00_01_52_to_00_01_58": 0.278199,
        "parrot_disco_distant_cross": 0.286589,
        "GOPR5846_002": 0.294613,
        "swarm_dji_phantom": 0.296319,
        "GOPR5844_004": 0.324874,
        "dji_mavick_distant_hillside": 0.326453,
        "gopro_005": 0.338451,
        "parrot_disco_distant_cross_3": 0.372270,
        "parrot_clear_birds_med_range": 0.394487,
        "2019_09_02_GOPR5871_1058_solo": 0.428203,
        "off_focus_parrot_birds": 0.450331,
        "gopro_004": 0.451091,
        "2019_08_19_GOPR5869_1530_phantom": 0.484971,
        "gopro_000": 0.489522,
        "gopro_001": 0.528298,
        "GOPR5842_005": 0.573157,
        "GOPR5844_002": 0.576830,
        "gopro_008": 0.586565,
        "parrot_disco_long_session": 0.594941,
        "GOPR5842_002": 0.617226,
        "two_parrot_disco_1": 0.621605,
        "2019_08_19_C0001_5319_phantom": 0.625696,
        "matrice_600_2": 0.635946,
        "2019_10_16_C0003_4613_mavic": 0.648552,
        "dji_matrice_210_hillside": 0.656371,
        "two_uavs_plus_airplane": 0.663090,
        "GOPR5848_004": 0.699731,
        "GOPR5846_005": 0.718117,
        "custom_fixed_wing_1": 0.718294,
        "parrot_disco_zoomin_zoomout": 0.737750,
        "00_02_45_to_00_03_10_cut": 0.760187,
        "dji_pantom_landing_custom_fixed_takeoff": 0.774655,
        "gopro_003": 0.778647,
        "GOPR5845_001": 0.784809,
        "GOPR5847_004": 0.785880,
        "dji_mavick_mountain": 0.801411,
        "dji_phantom_4_mountain_hover": 0.809510,
        "dji_mavick_close_buildings": 0.845758,
        "2019_11_14_C0001_3922_matrice": 0.849641,
        "dji_matrice_210_off_focus": 0.850053,
        "custom_fixed_wing_2": 0.854476,
        "distant_parrot_2": 0.860818,
        "GOPR5847_003": 0.862039,
        "GOPR5845_004": 0.914381,
        "swarm_dji_phantom4_2": 0.914887,
        "GOPR5843_005": 0.915939,
        "dji_matrice_210_mountain": 0.916409,
        "2019_09_02_C0002_3700_mavic": 0.917317,
        "fixed_wing_over_hill_1": 0.919683,
        "dji_phantom_mountain_cross": 0.923034,
        "parrot_disco_midrange_cross": 0.941304,
        "00_10_09_to_00_10_40": 0.943189,
        "dji_mavick_mountain_cruise": 0.947187,
        "fixed_wing_over_hill_2": 0.950558,
        "2019_09_02_C0002_2527_inspire": 0.967049,
        "00_06_10_to_00_06_27": 0.971953,
        "GOPR5842_007": 0.972834,
        "00_09_30_to_00_10_09": 0.975506,
        "2019_10_16_C0003_1700_matrice": 0.983794,
        "parot_disco_takeoff": 0.988288,
        "matrice_600_3": 0.989482,
        "dji_matrice_210_sky": 0.998194,
        "2019_10_16_C0003_5043_mavic": 0.999838,
    }
)


def plot_recall_precision_scenarios():

    # ---------------------------
    # Precision Recall for all videos grouped by performance.
    # 6 Videos in one plot
    prec_rec_files_bl = [
        get_prec_recall_file(ModelRegister.baseline, name) for name in ap05_401.index
    ]
    prec_rec_files_new_data = [
        get_prec_recall_file(ModelRegister.additional_data, name)
        for name in ap05_401.index
    ]
    files_per_file = 6

    for i in range(0, len(prec_rec_files_bl), files_per_file):
        data = get_recall_precision_results(
            chart_files=prec_rec_files_bl[i : i + files_per_file],
            column_variable="name",
        )
        plot(
            data,
            locations.RESULTS_DIR / "precision-recall" / f"{i}-{i+files_per_file}.svg",
            FIG_SIZE,
        )
    # -----------------------------
    # Showing the best three videos in one plot
    good_videos = [
        get_prec_recall_file(ModelRegister.baseline, name)
        for name in [
            "dji_matrice_210_sky",
            "00_06_10_to_00_06_27",
            "parot_disco_takeoff",
        ]
    ]
    good_video_data = get_recall_precision_results(good_videos, "name")
    plot(
        good_video_data,
        locations.RESULTS_DIR / "precision-recall" / f"good_videos.svg",
        (4, 4),
    )

    # -----------------------------
    # Showing the before and after for every video individually
    for bl_file, new_data_file in zip(prec_rec_files_bl, prec_rec_files_new_data):

        with open(
            bl_file,
            mode="r",
            encoding="utf-8",
        ) as f:
            chart_dict = json.load(fp=f)
            data = pd.Series(
                chart_dict["precision"], index=chart_dict["recall"]
            ).to_frame(name="Baseline")

        with open(
            new_data_file,
            mode="r",
            encoding="utf-8",
        ) as f:
            chart_dict = json.load(fp=f)
            data["Additional Data"] = pd.Series(
                chart_dict["precision"], index=chart_dict["recall"]
            )

        fig, ax = plt.subplots(figsize=FIG_SIZE)

        for column in data.columns:
            ax.plot(
                data.index, data[column], label=f"{column} ({data[column].mean():0.2f})"
            )
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"{chart_dict["name"]}")
        ax.legend(
            title="Model Version (AP05)",
            alignment="left",
            title_fontproperties={"weight": "bold"},
        )
        ax.set_xlim(left=-0.05, right=1.05)
        ax.set_ylim(bottom=-0.05, top=1.05)
        fig.tight_layout()
        plt.savefig(
            locations.RESULTS_DIR
            / "precision-recall"
            / "new-data-effect"
            / f"{chart_dict["name"]}.svg"
        )
        plt.close()

    # -----------------------------
    # Showing the plots grouped in 0.2 ranges
    # The plots do not show individual video but average / max and min.

    prec_rec_files = [
        get_prec_recall_file(ModelRegister.baseline, name)
        for name in names.DroneVsBirdVideos.video_names
    ]

    data = get_recall_precision_results(prec_rec_files, "name")

    ap = data.mean()
    ap_ranges = [[i * 0.2, (i + 1) * 0.2] for i in range(0, 5)]
    masks = [(ap >= low) & (ap < high) for low, high in ap_ranges]

    for i, mask_set in enumerate(masks):
        fig, ax = plt.subplots(figsize=(4, 4))
        lower = f"{ap_ranges[i][0]:0.2f}"
        upper = f"{ap_ranges[i][1]:0.2f}"
        mask_data = data.loc[:, mask_set]
        mask_data.loc[:, "min"] = mask_data.min(axis=1)
        mask_data.loc[:, "max"] = mask_data.max(axis=1)
        mask_data.loc[:, "mean"] = mask_data.mean(axis=1)
        ax.plot(
            mask_data.index,
            mask_data["mean"],
            label=f"Average ({mask_data["mean"].mean():0.2f})",
        )
        ax.plot(
            mask_data.index,
            mask_data["min"],
            label=f"Minimum ({mask_data["min"].mean():0.2f})",
        )
        ax.plot(
            mask_data.index,
            mask_data["max"],
            label=f"Maximum ({mask_data["max"].mean():0.2f})",
        )
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Summary Across AP05 Band {lower} ... {upper}")
        ax.legend(
            title="Model Version (AP05)",
            alignment="left",
            title_fontproperties={"weight": "bold"},
        )
        ax.set_xlim(left=-0.05, right=1.05)
        ax.set_ylim(bottom=-0.05, top=1.05)
        fig.tight_layout()
        plt.savefig(
            locations.RESULTS_DIR
            / "precision-recall"
            / f"prec_rec_summary_{lower}_{upper}.svg"
        )
        plt.close()


if __name__ == "__main__":

    plot_nms_conf_experiment()
