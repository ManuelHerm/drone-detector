"""This module provides functions around bounding box annotations."""

from pathlib import Path

import skimage
import torch
import torchvision
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


def fix_one_inspire_annotation_set():

    original_file = (
        locations.DroneVsBird.annotations / "2019_10_16_C0003_3633_inspire.txt"
    )
    new_file = (
        locations.DroneVsBird.annotations
        / "own-correction"
        / "2019_10_16_C0003_3633_inspire.txt"
    )

    # Read original
    with open(original_file, mode="r", encoding="utf-8") as f:
        content = f.readlines()

    # Reformat
    new_content = []
    for row in content:
        row = row.strip()
        new_content.append(row.split(" "))

    # Shift frame number by offset
    offset = 15
    for row_no, row in enumerate(new_content):
        new_frame = int(row[0]) - offset
        new_content[row_no][0] = str(new_frame)

    # Remove negative frames
    del new_content[0:offset]

    # Write new content to disk
    with open(new_file, mode="w", encoding="utf-8") as nf:
        nf.writelines([" ".join(row) + "\n" for row in new_content])


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


def seg_mask_to_bbox(mask_path: Path, black_value: int = 100) -> torch.Tensor:
    """
    Turn an segmentation mask image into an annotation.

    Inspired by:
    https://docs.pytorch.org/vision/stable/auto_examples/others/plot_repurposing_annotations.html

    Args:
        mask_path: Path of the segmentation image.
        black_value: Every value equal and below `black_value` get turned to 0.
            All Above becomes 255. This ensures a clear separation between
            the background and the object to detect.
            Blender segmentation masks have for background 0 and 1.
            The object is typically much brighter.
            For Cranfield datasets, a `black_value` of 100 was used.

    Returns:
        The bounding boxes as torch.Tensor[N, 4]: bounding boxes
    """
    mask_raw = torchvision.io.decode_image(str(mask_path), mode="GRAY")
    mask_raw[mask_raw <= black_value] = 0
    mask_raw[mask_raw > black_value] = 255
    # Give connected areas a unique value:
    mask = torch.Tensor(skimage.morphology.label(mask_raw))
    object_ids = torch.unique(mask)
    # Rejecting background id
    object_ids = object_ids[1:]
    # Move objects into different dimensions
    masks = mask == object_ids[:, None, None]
    # Returning the bounding boxes
    bboxes = torchvision.ops.masks_to_boxes(masks)
    return bboxes


def seg_mask_to_bbox_color_based(
    mask_path: Path, black_value: int = 100
) -> torch.Tensor:
    """
    Turn an segmentation mask image into an annotation.

    Each drone has its own color channel.

    Args:
        mask_path: Path of the segmentation image.
        black_value: Every value equal and below `black_value` get turned to 0.
            All Above becomes 255. This ensures a clear separation between
            the background and the object to detect.
            Blender segmentation masks have for background 0 and 1.
            The object is typically much brighter.
            For Cranfield datasets, a `black_value` of 100 was used.

    Returns:
        The bounding boxes as torch.Tensor[N, 4]: bounding boxes
    """
    mask_image = decode_image(str(mask_path), mode="RGBA")

    # Separate into channels
    red = mask_image[0]
    green = mask_image[1]
    blue = mask_image[2]

    # Turn all black noise into zero values
    red[red <= black_value] = 0
    green[green <= black_value] = 0
    blue[blue <= black_value] = 0

    # Initialize clean mask tensor,
    # where every item has its own unique value
    h = int(mask_image.shape[1])
    w = int(mask_image.shape[2])
    item_labels = torch.zeros(h, w, dtype=torch.uint8)
    # Use only the maximum value per color
    item_labels[(red > green) & (red > blue)] = 1
    item_labels[(green > red) & (green > blue)] = 2
    item_labels[(blue > red) & (blue > green)] = 3

    # Getting all numeric labels
    object_ids = torch.unique(item_labels)
    # Rejecting background id
    object_ids = object_ids[1:]

    # Move objects into different dimensions
    masks = item_labels == object_ids[:, None, None]

    # Getting the bounding boxes
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


def get_annotations_from_segmentation_mask(
    image_id: int, mask_path: Path, mask_mode: str, black_value: int
) -> list[dt.Annotation]:
    """
    Turn segmentation mask image into bounding box definitions.

    Args:
        image_id: The id of the image that belongs to the annotations.
        mask_path: The location of the mask image.
        mask_mode: Options are:
            - "cluster": All drones are shown in white and clusters of
                white pixels are grouped to form a bounding box.
            - "color_based": Each drone is shown in either green, red,
                or blue. Hence, there can a maximum of three drones
                in the image.
        black_value: Everything below this value is treated as background.

    """
    if mask_mode == "cluster":
        boxes = seg_mask_to_bbox(mask_path=mask_path, black_value=black_value)
    elif mask_mode == "color_based":
        boxes = seg_mask_to_bbox_color_based(
            mask_path=mask_path, black_value=black_value
        )
    else:
        raise ValueError(
            f"Unknown mode provided for mask conversion to bounding box: {mask_mode}"
        )
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
    seg_mask_to_bbox_color_based(locations.TEMP_DIR / "problem.png", black_value=100)
