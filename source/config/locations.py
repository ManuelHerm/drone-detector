"""This module allows setting the directories across the project."""

from dataclasses import dataclass
from pathlib import Path

# pylint: disable=missing-class-docstring
#         The classes are trivial

ROOT_DIR = Path(__file__).resolve().parents[2]
DB_PATH = ROOT_DIR / "data" / "data.db"
CACHE_DIR = ROOT_DIR / "cache"
RESULTS_DIR = ROOT_DIR / "results"
TEMP_DIR = ROOT_DIR / "temp"
DATA_DIR = ROOT_DIR / "data"

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
    root: Path = DATA_DIR / "drone-vs-bird"
    videos: Path = DATA_DIR / "drone-vs-bird" / "video"
    annotations: Path = DATA_DIR / "drone-vs-bird" / "annotations"
    coco_ground_truths: Path = DATA_DIR / "drone-vs-bird" / "coco_ground_truth_jsons"


@dataclass
class CurrentWorkRealWorldFootage:
    root: Path = DATA_DIR / "current" / "real_world"
    images: Path = DATA_DIR / "current" / "real_world" / "images"
    videos: Path = DATA_DIR / "current" / "real_world" / "videos"


@dataclass
class AntiUAV:
    root: Path = DATA_DIR / "anti-uav" / "Anti-UAV-RGBT"
    training: Path = DATA_DIR / "anti-uav" / "Anti-UAV-RGBT" / "train"
    validation: Path = DATA_DIR / "anti-uav" / "Anti-UAV-RGBT" / "val"
    testing: Path = DATA_DIR / "anti-uav" / "Anti-UAV-RGBT" / "test"


@dataclass
class Cranfield:
    root: Path = DATA_DIR / "cranfield" / "multi_drone_no_birds_40m"
    images: Path = DATA_DIR / "cranfield" / "multi_drone_no_birds_40m" / "Images"
    masks: Path = DATA_DIR / "cranfield" / "multi_drone_no_birds_40m" / "Masks"


@dataclass
class CranfieldDistractors:
    images: Path = DATA_DIR / "cranfield" / "drones_generic_distractors_40m" / "Images"
    masks: Path = DATA_DIR / "cranfield" / "drones_generic_distractors_40m" / "Masks"


@dataclass
class ForrestHutPhantom:
    images: Path = DATA_DIR / "current" / "forrest_hut" / "phantom" / "images"
    masks: Path = DATA_DIR / "current" / "forrest_hut" / "phantom" / "masks"
    gt: Path = (
        DATA_DIR
        / "current"
        / "forrest_hut"
        / "phantom"
        / "ground-truth"
        / "ground_truth.json"
    )


@dataclass
class ForrestHutMini:
    images: Path = DATA_DIR / "current" / "forrest_hut" / "mini3" / "images"
    masks: Path = DATA_DIR / "current" / "forrest_hut" / "mini3" / "masks"
    gt: Path = (
        DATA_DIR
        / "current"
        / "forrest_hut"
        / "mini3"
        / "ground-truth"
        / "ground_truth.json"
    )


@dataclass
class ForrestHutInspire:
    images: Path = DATA_DIR / "current" / "forrest_hut" / "inspire" / "images"
    masks: Path = DATA_DIR / "current" / "forrest_hut" / "inspire" / "masks"
    gt: Path = (
        DATA_DIR
        / "current"
        / "forrest_hut"
        / "inspire"
        / "ground-truth"
        / "ground_truth.json"
    )


@dataclass
class DroneSchoolInspire:
    images: Path = DATA_DIR / "current" / "drone-school" / "inspire-close" / "images"
    masks: Path = DATA_DIR / "current" / "drone-school" / "inspire-close" / "masks"


@dataclass
class DroneSchoolMini:
    images: Path = DATA_DIR / "current" / "drone-school" / "mini-close" / "images"
    masks: Path = DATA_DIR / "current" / "drone-school" / "mini-close" / "masks"


@dataclass
class DroneSchoolPhantom:
    images: Path = DATA_DIR / "current" / "drone-school" / "phantom-close" / "images"
    masks: Path = DATA_DIR / "current" / "drone-school" / "phantom-close" / "masks"


@dataclass
class DroneSchoolMid:
    images: Path = DATA_DIR / "current" / "drone-school" / "mid" / "images"
    masks: Path = DATA_DIR / "current" / "drone-school" / "mid" / "masks"


@dataclass
class DroneSchoolFar:
    images: Path = DATA_DIR / "current" / "drone-school" / "far" / "images"
    masks: Path = DATA_DIR / "current" / "drone-school" / "far" / "masks"


@dataclass
class Campus:
    images: Path = DATA_DIR / "current" / "campus" / "images"
    masks: Path = DATA_DIR / "current" / "campus" / "masks"
    gt: Path = DATA_DIR / "current" / "campus" / "ground-truth" / "ground_truth.json"


@dataclass
class Hill:
    images: Path = DATA_DIR / "current" / "hill" / "images"
    masks: Path = DATA_DIR / "current" / "hill" / "masks"
    gt: Path = DATA_DIR / "current" / "hill" / "ground-truth" / "ground_truth.json"


@dataclass
class Meadow:
    images: Path = DATA_DIR / "current" / "meadow" / "images"
    masks: Path = DATA_DIR / "current" / "meadow" / "masks"
    gt: Path = DATA_DIR / "current" / "meadow" / "ground-truth" / "ground_truth.json"


@dataclass
class Container:
    images: Path = DATA_DIR / "current" / "container" / "section"
    images_full: Path = DATA_DIR / "current" / "container" / "full"
    gt: Path = (
        DATA_DIR
        / "current"
        / "container"
        / "full"
        / "ground-truth"
        / "ground_truth.json"
    )


@dataclass
class City:
    images: Path = DATA_DIR / "current" / "city" / "images"
    masks: Path = DATA_DIR / "current" / "city" / "masks"
    gt: Path = DATA_DIR / "current" / "city" / "ground-truth" / "ground_truth.json"


@dataclass
class Lake:
    images: Path = DATA_DIR / "current" / "lake" / "images"
    masks: Path = DATA_DIR / "current" / "lake" / "masks"
    gt: Path = DATA_DIR / "current" / "lake" / "ground-truth" / "ground_truth.json"


@dataclass
class Disorder:
    images: Path = DATA_DIR / "current" / "disorder" / "section"
    images_full: Path = DATA_DIR / "current" / "disorder" / "full"
    gt: Path = (
        DATA_DIR
        / "current"
        / "disorder"
        / "full"
        / "ground-truth"
        / "ground_truth.json"
    )


@dataclass
class Lanterns:
    images: Path = DATA_DIR / "current" / "lanterns" / "section"
    images_full: Path = DATA_DIR / "current" / "lanterns" / "full"
    gt: Path = (
        DATA_DIR
        / "current"
        / "lanterns"
        / "full"
        / "ground-truth"
        / "ground_truth.json"
    )


@dataclass
class Trees:
    images_pine: Path = DATA_DIR / "current" / "tree" / "pine"
    images_normal: Path = DATA_DIR / "current" / "tree" / "normal"
    gt_pine: Path = (
        DATA_DIR / "current" / "tree" / "pine" / "ground-truth" / "ground_truth.json"
    )
    gt_tree: Path = (
        DATA_DIR / "current" / "tree" / "normal" / "ground-truth" / "ground_truth.json"
    )


@dataclass
class ParrotOne:
    images: Path = DATA_DIR / "current" / "parrot-straight-flight" / "images"
    masks: Path = DATA_DIR / "current" / "parrot-straight-flight" / "masks"
    gt: Path = (
        DATA_DIR
        / "current"
        / "parrot-straight-flight"
        / "ground-truth"
        / "ground_truth.json"
    )


@dataclass
class ParrotTwo:
    images: Path = DATA_DIR / "current" / "parrot-curved-flight" / "images"
    masks: Path = DATA_DIR / "current" / "parrot-curved-flight" / "masks"
    gt: Path = (
        DATA_DIR
        / "current"
        / "parrot-curved-flight"
        / "ground-truth"
        / "ground_truth.json"
    )


@dataclass
class Moon:
    images: Path = DATA_DIR / "current" / "moon"


@dataclass
class Cache:
    cranfield: Path = CACHE_DIR / "cranfield"
    coco_annotations: Path = CACHE_DIR / "coco_annotations"
    drone_vs_bird: Path = CACHE_DIR / "drone_vs_bird"


@dataclass
class Logs:
    logs: Path = ROOT_DIR / "logs"
    training: Path = ROOT_DIR / "logs" / "training"
    validation: Path = ROOT_DIR / "logs" / "validation"


@dataclass
class Results:
    coco_jsons: Path = RESULTS_DIR / "coco_jsons"
    coco_videos: Path = RESULTS_DIR / "coco_videos"
    ap05_jsons: Path = RESULTS_DIR / "ap05_jsons"
    ap05_images: Path = RESULTS_DIR / "ap05_images"
    coco_images: Path = RESULTS_DIR / "coco_images"
    real_world_videos: Path = RESULTS_DIR / "coco_videos" / "real_world"
