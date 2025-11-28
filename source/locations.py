"""This module allows setting the directories across the project."""

from dataclasses import dataclass
from pathlib import Path

# pylint: disable=missing-class-docstring
#         The classes are trivial

ROOT_DIR = Path(__file__).resolve().parents[1]
DB_PATH = ROOT_DIR / "data" / "data.db"
CACHE_DIR = ROOT_DIR / "cache"
RESULTS_DIR = ROOT_DIR / "results"
TEMP_DIR = ROOT_DIR / "temp"

if not DB_PATH.parent.exists():
    DB_PATH.parent.mkdir(parents=True)

if not CACHE_DIR.exists():
    CACHE_DIR.mkdir(parents=True)

if not RESULTS_DIR.exists():
    RESULTS_DIR.mkdir(parents=True)

if not TEMP_DIR.exists():
    TEMP_DIR.mkdir(parents=True)


@dataclass
class DroneVsBird:
    root: Path = ROOT_DIR / "data" / "drone-vs-bird"
    videos: Path = ROOT_DIR / "data" / "drone-vs-bird" / "video"
    annotations: Path = ROOT_DIR / "data" / "drone-vs-bird" / "annotations"


@dataclass
class AntiUAV:
    root: Path = ROOT_DIR / "data" / "anti-uav" / "Anti-UAV-RGBT"
    training: Path = ROOT_DIR / "data" / "anti-uav" / "Anti-UAV-RGBT" / "train"
    validation: Path = ROOT_DIR / "data" / "anti-uav" / "Anti-UAV-RGBT" / "val"
    testing: Path = ROOT_DIR / "data" / "anti-uav" / "Anti-UAV-RGBT" / "test"


@dataclass
class Cranfield:
    root: Path = ROOT_DIR / "data" / "cranfield" / "multi_drone_no_birds_40m"
    images: Path = (
        ROOT_DIR / "data" / "cranfield" / "multi_drone_no_birds_40m" / "Images"
    )
    masks: Path = ROOT_DIR / "data" / "cranfield" / "multi_drone_no_birds_40m" / "Masks"


@dataclass
class CranfieldDistractors:
    images: Path = (
        ROOT_DIR / "data" / "cranfield" / "drones_generic_distractors_40m" / "Images"
    )
    masks: Path = (
        ROOT_DIR / "data" / "cranfield" / "drones_generic_distractors_40m" / "Masks"
    )


@dataclass
class Cache:
    cranfield: Path = ROOT_DIR / "cache" / "cranfield"
    coco_annotations: Path = ROOT_DIR / "cache" / "coco_annotations"
    drone_vs_bird: Path = ROOT_DIR / "cache" / "drone_vs_bird"


@dataclass
class Logs:
    training: Path = ROOT_DIR / "logs" / "training"
    validation: Path = ROOT_DIR / "logs" / "validation"


@dataclass
class Results:
    coco_jsons: Path = RESULTS_DIR / "coco_jsons"
    coco_videos: Path = RESULTS_DIR / "coco_videos"
