import json
from itertools import cycle
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from source.config import locations
from source.db import database_manager as dbm

mpl.rcParams["figure.dpi"] = 800
plt.style.use("seaborn-v0_8")

default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_cycle = cycle(default_colors)


def get_results_from_json(json_file: str) -> pd.Series:
    with open(json_file, mode="r", encoding="utf-8") as f:
        ap05_results = json.load(f)
    return pd.Series(ap05_results["results"])


def get_hard_videos(videos_to_ap05: dict[str, float]) -> tuple[str]:
    series = pd.Series(videos_to_ap05)
    mean = series.mean()
    hard_video_mask = series < mean
    hard_videos = series[hard_video_mask]
    return tuple(hard_videos.index.to_list())


def create_result_plot(
    ap_05_results_dict: dict[str, float], plot_label: str, target_file: Path, title: str
):
    ap05_results = pd.Series(ap_05_results_dict)
    average = ap05_results.mean()

    fig, ax = plt.subplots()
    # Trim the axis to the data extents (adds a tiny margin)
    ax.autoscale(enable=True, axis="x", tight=True)
    ax.plot(ap05_results.index, ap05_results.values, label=plot_label)
    ax.set_title(title)
    ax.tick_params(axis="x", labelsize=6, rotation=90)
    ax.set_ylabel("AP @ IoU=0.5")
    ax.plot(
        (0, len(ap05_results)), (average, average), label=f"Average = {average:.2f}"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(target_file)


def get_shape_of_videos(video_names: list[str]) -> dict[str, tuple[int, int]]:
    video_geometry = {}
    for video_name in video_names:
        # pylint: disable=no-value-for-parameter
        # cursor parameter is provided by decorator
        video_id = dbm.get_video_id_for_video_name_stem(video_name_stem=video_name)
        width, height = dbm.get_video_shape_for_video_id(video_id)
        video_geometry[video_name] = (width, height)
    return video_geometry


if __name__ == "__main__":
    result_dir = locations.ROOT_DIR / "documentation" / "images" / "results"
    result_json = result_dir / "ap05_model_state_80.json"
    results = get_results_from_json(str(result_json)).sort_values(ascending=True)

    training_shape = (1920, 1080)
    video_shapes = pd.Series(get_shape_of_videos(video_names=results.index.to_list()))

    # Checking for impact of input video size
    # in comparison with the training data image size
    results_smaller_videos = results[video_shapes < training_shape]
    results_same_size = results[video_shapes == training_shape]
    results_larger_size = results[video_shapes > training_shape]

    create_result_plot(
        ap_05_results_dict=results.to_dict(),
        title="Model Performance on Drone vs. Bird Videos",
        plot_label="Faster R-CNN Trained on Cranfield Dataset",
        target_file=result_dir / "ap05_model_state_80.png",
    )

    create_result_plot(
        ap_05_results_dict=results_smaller_videos.to_dict(),
        title=(
            "Model Performance on Drone vs. Bird Videos | "
            "Test Image Size < Training Image Size"
        ),
        plot_label="Faster R-CNN Trained on Cranfield Dataset",
        target_file=result_dir / "ap05_model_state_80_smaller_size.png",
    )

    create_result_plot(
        ap_05_results_dict=results_same_size.to_dict(),
        title=(
            "Model Performance on Drone vs. Bird Videos | "
            "Test Image Size = Training Image Size"
        ),
        plot_label="Faster R-CNN Trained on Cranfield Dataset",
        target_file=result_dir / "ap05_model_state_80_same_size.png",
    )

    create_result_plot(
        ap_05_results_dict=results_larger_size.to_dict(),
        title=(
            "Model Performance on Drone vs. Bird Videos | "
            "Test Image Size > Training Image Size"
        ),
        plot_label="Faster R-CNN Trained on Cranfield Dataset",
        target_file=result_dir / "ap05_model_state_80_larger_size.png",
    )
