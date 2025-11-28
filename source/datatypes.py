"""This module provides all dataclasses."""

from __future__ import annotations

import datetime
import hashlib
from dataclasses import dataclass
from pathlib import Path

import configuration
from logger import logging

log = logging.getLogger(__name__)
log.setLevel(configuration.LOG_LEVEL)

# pylint: disable=missing-class-docstring
#         Disabled because the classes are trivial


@dataclass
class VideoProperties:
    width: int
    height: int
    frame_rate: float
    frame_count: int
    convert_rgb: bool


@dataclass
class Video:
    name: str
    path_abs: str
    sha_256: str
    data_origin_id: int
    properties: VideoProperties
    id: int = -1  # For when there is no database id yet.

    @staticmethod
    def get_sha_256(video_path: Path):
        # Read the file in chunks to avoid loading large videos into memory
        hash_sha256 = hashlib.sha256()
        with video_path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()


@dataclass
class Label:
    id: int
    name: str
    number: int


@dataclass
class Image:
    name: str
    data_origin_id: int
    width: int
    height: int
    image: bytes


@dataclass
class Annotation:
    # pylint: disable=too-many-instance-attributes
    #         Prefer the flat hierarchy over nested classes.
    image_id: int
    label_id: int
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    id: int = -1  # Placeholder for annotations without database entry.
    score: float = None

    def get_area(self) -> float:
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)


@dataclass
class NormalizationData:
    dataset_id: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


@dataclass
class FasterRCNNLoss:
    box_reg: float
    classifier: float
    objectness: float
    rpn_box_reg: float


@dataclass
class CocoAnnotation:
    # pylint: disable=too-many-instance-attributes
    #         Prefer the flat hierarchy over nested classes.
    id: int
    image_id: int
    area: float
    score: float
    bbox: tuple[float, float, float, float]  # [x_min, y_min, width, height]
    category_id: int = 1
    segmentation: tuple[()] = tuple()
    iscrowd: int = 0

    def __init__(self, annotation: Annotation, score=1.0):
        self.id = annotation.id
        self.image_id = annotation.image_id
        self.area = annotation.get_area()
        if self.area <= 1:
            raise ValueError(
                f"Invalid annotation area for annotation with id = {annotation.id}\n"
                f"x_min = {annotation.x_min} x_max = {annotation.x_max}\n"
                f"y_min = {annotation.y_min} y_max = {annotation.y_max}\n"
            )
        self.bbox = (
            annotation.x_min,
            annotation.y_min,
            (annotation.x_max - annotation.x_min),
            (annotation.y_max - annotation.y_min),
        )
        self.category_id = annotation.label_id
        self.score = score


@dataclass
class CocoImage:
    id: int
    width: int
    height: int
    license: int = 1
    # File names are captured in the database
    # and can be retrieved by the image's id:
    file_name: str = ""


@dataclass
class CocoCategory:
    id: int
    name: str
    supercategory: str


@dataclass
class CocoLicense:
    id: int = 1
    name: str = ""
    url: str = ""


@dataclass
class CocoInfo:
    description: str
    date_created: str = str(datetime.time())
    year: int = 0
    version: str = 1
    contributor: str = ""
    url: str = ""


@dataclass
class CocoPredictionEntry:
    image_id: int
    category_id: int
    bbox: tuple[float, float, float, float]  # x, y, w, h
    score: float


@dataclass
class CocoDataset:
    info: CocoInfo
    images: tuple[CocoImage, ...]
    annotations: tuple[CocoAnnotation, ...]
    categories: tuple[CocoCategory, ...]
    licenses: tuple[CocoLicense, ...]


@dataclass
class ImageLegendEntry:
    xy: tuple[float, float]
    text: str
