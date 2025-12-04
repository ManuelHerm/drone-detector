"""This module provides functions around bounding box annotations."""

from pathlib import Path

import skimage
import torch
import tqdm
from torchvision.io import decode_image
from torchvision.ops import masks_to_boxes

from source.config import locations, settings
from source.data import datatypes as dt
from source.db import database_manager as dbm
from source.utils import image_utilities as img_util
from source.utils.logger import logging

log = logging.getLogger(__name__)
log.setLevel(settings.LOG_LEVEL)

# pylint: disable=no-value-for-parameter
#         Disabled, because the dbm-function receive the
#         cursor parameter from the decorator.


def parse_dvb_annotations(file: Path) -> list[dt.Annotation]:
    """
    Parse the drone versus bird annotation file

    The file layout is::

        <frame_#> <drone count> <x1> <y1> <w1> <h1> drone <x2> <y2> <w2> <h2> drone  ...
        <frame_#> <drone count> <x1> <y1> <w1> <h1> drone <x2> <y2> <w2> <h2> drone  ...

    All values are space separated.

    The database setup is so, that if there are no drones detected,
    then no annotation is needed.

    Hence, when the <number of drones> is zero, the line will be skipped.

    Args:
        file: The location of the annotation file

    Returns:
        The annotations in the file
    """
    log.debug("Reading annotation file from %s.", file)
    video_id = dbm.get_video_id_for_video_name_stem(file.stem)

    lines = []
    with open(file, mode="r", encoding="utf-8") as f:
        lines = f.readlines()

    annotations = []
    for line in lines:
        line = line.strip()
        elements = line.split(" ")
        # This captures frames, where the number of drones is zero and empty lines.
        if len(elements) < 3:
            continue
        # The annotation info block for each drone on the line has 5 elements.
        # The - 2 is for the frame number and the number of detected drones.
        if (len(elements) - 2) % 5 != 0:
            raise ValueError(f"Line is incorrectly formatted:\n{line}")
        number_of_drones = int(elements[1])
        if number_of_drones != (len(elements) - 2) / 5:
            raise ValueError(
                f"Number of detected drones does not match the annotation info:\n{line}"
            )

        frame_number = int(elements[0])
        try:
            image_id = dbm.get_image_id_for_video_id_and_frame_number(
                video_id, frame_number
            )
        # If annotation file contains frames that are not in the video,
        # then just continue.
        except ValueError:
            continue

        for i in range(number_of_drones):
            x_min = int(elements[2 + i * 5])
            y_min = int(elements[3 + i * 5])
            width = int(elements[4 + i * 5])
            height = int(elements[5 + i * 5])
            annotation = dt.Annotation(
                image_id=image_id,
                label_id=1,  # drone is always 1.
                x_min=x_min,
                y_min=y_min,
                x_max=x_min + width,
                y_max=y_min + height,
            )
            annotations.append(annotation)
    return annotations


def insert_all_dvb_annotations_into_db():
    """
    Read all drone vs. bird annotation files and insert them in the database.

    Their location is defined in `locations.py`.
    """
    files = list(sorted(locations.DroneVsBird.annotations.iterdir()))
    for file in tqdm.tqdm(files):
        annotations = parse_dvb_annotations(file)
        dbm.insert_annotations(annotations)


def cranfield_seg_mask_to_bbox(mask_path: Path) -> torch.Tensor:
    # Inspired by:
    # https://docs.pytorch.org/vision/stable/auto_examples/others/plot_repurposing_annotations.html
    mask_raw = decode_image(str(mask_path), mode="GRAY")
    # Mask cleaning: What should be zero, is between 0 and 1.
    # What should be 1 is between 206 and 207.
    # Let's draw the line at 100.
    mask_raw[mask_raw <= 100] = 0
    mask_raw[mask_raw > 100] = 255
    # Give connected areas a unique value:
    mask = torch.Tensor(skimage.morphology.label(mask_raw))
    object_ids = torch.unique(mask)
    # Rejecting background id
    object_ids = object_ids[1:]
    # Move objects into different dimensions
    masks = mask == object_ids[:, None, None]
    # Returning the bounding boxes
    bboxes = masks_to_boxes(masks)
    return bboxes


def get_image_name_for_cranfield_mask_name(mask_name: str) -> str:
    """
    The naming pattern for the Cranfield masks is:

    - Mask name: Mask<4-digit image_number>.png_<4-digit mask_counter>.png
    - Image name: Image<4-digit image_number>.jpg

    Hence, the 4-digit image number is enough to extract from the maks name
    to create the image name.

    Args:
        mask_name: String following the mask name pattern above

    Returns:
        Image name following the mask name pattern above.
    """
    image_number = mask_name[4:8]
    return f"Image{image_number}.jpg"


def get_annotations_from_cranfield_segmentation_mask(
    mask_path: Path,
) -> list[dt.Annotation]:
    boxes = cranfield_seg_mask_to_bbox(mask_path)
    image_name = get_image_name_for_cranfield_mask_name(str(mask_path.name))
    image_id = dbm.get_image_id_for_image_name(image_name)
    annotations = [
        dt.Annotation(
            image_id=image_id,
            label_id=1,
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
        )
        for (x_min, y_min, x_max, y_max) in boxes.tolist()
    ]
    return [
        annotation
        for annotation in annotations
        if annotation.get_area() >= settings.BBOX_MIN_AREA
    ]


def find_broken_annotations() -> set[int]:
    """
    Find all annotations that do not meet the quality criteria.

    Returns:
        The ids of all bad annotations
    """
    annotations = dbm.get_annotations()
    area_too_small = set(
        annotation.id
        for annotation in annotations
        if annotation.get_area() < settings.BBOX_MIN_AREA
    )
    return area_too_small


def fix_broken_annotations():
    dbm.remove_annotations(find_broken_annotations())


def write_images_with_broken_annotation_to_disk(target_folder: Path):
    if not target_folder.exists():
        log.info("Creating folder %s", target_folder)
        target_folder.mkdir(parents=True)
    broken_annotation_ids = find_broken_annotations()
    broken_annotations = [
        dbm.get_annotation_for_annotation_id(anno_id)
        for anno_id in broken_annotation_ids
    ]
    images_with_broken_annotations = {}
    for annotation in broken_annotations:
        if annotation.image_id not in images_with_broken_annotations:
            images_with_broken_annotations[annotation.image_id] = [annotation]
        else:
            images_with_broken_annotations[annotation.image_id].append(annotation)
    for (
        image_id,
        broken_annotations_for_image,
    ) in images_with_broken_annotations.items():
        broken_annotation_ids_for_image = set(
            annotation.id for annotation in broken_annotations_for_image
        )
        intact_annotations = [
            annotation
            for annotation in dbm.get_annotations_for_image_id(image_id)
            if annotation.id not in broken_annotation_ids_for_image
        ]
        image = img_util.image_blob_to_image(dbm.get_image_for_id(image_id))
        image = img_util.add_broken_bounding_boxes_to_image(
            img=image, broken_bboxes=broken_annotations_for_image
        )
        image = img_util.add_bounding_boxes_to_image(
            img=image, bboxes=intact_annotations
        )
        image.save(target_folder / f"{image_id}.png")


if __name__ == "__main__":
    broken = find_broken_annotations()
