from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import tqdm
from IPython.core.pylabtools import figsize

from source.config import locations, names
from source.db import database_manager as dbm

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


def get_dataset_drone_sizes(dataset_name: str) -> list[float]:
    image_ids = dbm.get_image_ids_for_data_origin_name(data_origin_name=dataset_name)
    annotation_short_sides = []
    pytorch_min_size = 1080
    pytorch_max_size = 1920
    for image_id in tqdm.tqdm(image_ids):
        # Getting the image scaling factor
        width, height = dbm.get_image_width_and_height_for_image_id(image_id=image_id)
        shortest_side = min(width, height)
        longest_side = max(width, height)
        scale = pytorch_min_size / shortest_side
        if scale * longest_side > pytorch_max_size:
            scale = pytorch_max_size / longest_side
        # Getting the annotations
        try:
            image_annotations = dbm.get_annotations_for_image_id(image_id=image_id)
            for image_annotation in image_annotations:
                annotation_width = (
                    image_annotation.x_max - image_annotation.x_min
                ) * scale
                annotation_height = (
                    image_annotation.y_max - image_annotation.y_min
                ) * scale
                annotation_short_sides.append(min(annotation_height, annotation_width))
        except IndexError:
            continue
    return annotation_short_sides


def plot_dataset_drone_sizes(
    title: str,
    dataset_name: str,
    anchor_sizes: Sequence[float],
    result_json: Path,
    plot_file: Path,
):
    if result_json.exists():
        annotations = pd.read_json(result_json, orient="index")
    else:
        annotation_list = get_dataset_drone_sizes(dataset_name=dataset_name)
        annotations = pd.Series(annotation_list)
        annotations.to_json(result_json)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(annotations.to_numpy(), bins=100)
    ax.vlines(
        anchor_sizes,
        ymin=0,
        ymax=1,
        transform=ax.get_xaxis_transform(),  # y in axes-fraction (0..1)
        colors="C1",
        linestyles="--",
        label="Anchors",
    )
    median = annotations.iloc[:, 0].median()
    ax.axvline(median, ymin=0, ymax=1, color="C2", label=f"Size Median = {median:0.1f}")
    q_10 = annotations.iloc[:, 0].quantile(q=0.1)
    ax.axvline(q_10, ymin=0, ymax=1, color="C3", label=f"10th Percentile = {q_10:0.1f}")
    ax.set_xlabel("Drone Size [Pixel]")
    ax.set_xlim(left=0, right=140)
    ax.set_ylabel("Drone Count")
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(0.7, 1))
    fig.tight_layout()

    plt.savefig(plot_file)


if __name__ == "__main__":
    baseline_anchors = (32, 64, 128, 256, 512)
    smaller_anchors = (8, 16, 32, 64, 128)
    plot_dataset_drone_sizes(
        title="",  # "Drone versus Bird Drone Sizes - Baseline Configuration",
        dataset_name=names.DatasetNames.drone_vs_bird,
        anchor_sizes=baseline_anchors,
        result_json=locations.RESULTS_DIR / "drone-sizes" / "drone_size.json",
        plot_file=locations.RESULTS_DIR / "drone-sizes" / "drone_size_baseline.svg",
    )
    plot_dataset_drone_sizes(
        title="",  # "Drone versus Bird Drone Sizes - Adaptation",
        dataset_name=names.DatasetNames.drone_vs_bird,
        anchor_sizes=smaller_anchors,
        result_json=locations.RESULTS_DIR / "drone-sizes" / "drone_size_larger.json",
        plot_file=locations.RESULTS_DIR / "drone-sizes" / "drone_size_adapted.svg",
    )
    plot_dataset_drone_sizes(
        title="",  # "Drone Sizes: Cranfield Default Dataset",
        dataset_name=names.DatasetNames.cranfield_default,
        anchor_sizes=smaller_anchors,
        result_json=locations.RESULTS_DIR
        / "drone-sizes"
        / "cranfield_default_larger.json",
        plot_file=locations.RESULTS_DIR
        / "drone-sizes"
        / "cranfield_default_larger.svg",
    )
