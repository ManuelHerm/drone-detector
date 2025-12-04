"""This module provides functionality to help with the COCO evaluation."""

import json
from dataclasses import asdict
from pathlib import Path

import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from source.config import locations, settings
from source.data import datatypes as dt
from source.data.cache import get_cache
from source.db import database_manager as dbm
from source.utils.logger import logging

coco_annotations_cache = get_cache(locations.Cache.coco_annotations)

log = logging.getLogger(__name__)
log.setLevel(settings.LOG_LEVEL)

# pylint: disable=no-value-for-parameter
#         Disabled, because the dbm-function receive the
#         cursor parameter from the decorator.


def get_category_data() -> tuple[dt.CocoCategory, ...]:
    """
    Get the available labels from the database in COCO category format.

    Returns:
        All labels in the database.
    """
    log.debug("Creating the CocoCategories ...")
    label_list = []
    for label in dbm.get_labels():
        label_list.append(
            dt.CocoCategory(id=label.number, name=label.name, supercategory="UAV")
        )
    return tuple(label_list)


def get_annotation_data(image_id_list: list[int]) -> tuple[dt.CocoAnnotation, ...]:
    """
    Get all annotations from the database for the given image ids.

    Args:
        image_id_list: The ids of the images to get annotations for.

    Returns:
        All annotations for the images in `image_id_list` in CocoAnnotation format
    """
    log.debug("Creating the CocoAnnotations ...")
    annotation_list: list[dt.CocoAnnotation] = []
    for image_id in tqdm.tqdm(image_id_list):
        try:
            annotations_for_image = dbm.get_annotations_for_image_id(image_id=image_id)
        except IndexError:
            # No annotations for a given image just mean, there are none.
            continue
        for annotation_for_image in annotations_for_image:
            try:
                annotation_list.append(
                    dt.CocoAnnotation(annotation=annotation_for_image)
                )
            except ValueError:
                log.warning(
                    "Annotation with id %s has a zero area.",
                    annotation_for_image.id,
                )
                # Just skipping over zero area annotations.
                continue
    return tuple(annotation_list)


def get_image_data(image_id_list: list[int]) -> tuple[dt.CocoImage, ...]:
    """
    Get all image meta-information from the database for the given
    image ids in CocoImage format.

    Args:
        image_id_list: The ids of the images to get meta-information for.

    Returns:
        All meta-data for the images in `image_id_list`
    """
    log.debug("Creating the CocoImages ...")
    image_list: list[dt.CocoImage] = []
    for image_id in tqdm.tqdm(image_id_list):
        width, height = dbm.get_image_width_and_height_for_image_id(image_id=image_id)
        image = dt.CocoImage(id=image_id, width=width, height=height)
        image_list.append(image)
    return tuple(image_list)


@coco_annotations_cache.memoize(typed=True)
def get_coco_dataset(dataset_name: str, data_category: str) -> dt.CocoDataset:
    """
    Generate a CocoDataset object.

    It can be used to create the json file for the evaluation with
    the pycocotools library.

    Args:
        dataset_name: The name of the dataset in the database
        data_category: The name of the data category like "validation" or "testing"

    Returns:
        The CocoDataset
    """
    log.info("Creating '%s' %s CocoDataset", dataset_name, data_category)

    dataset_id = dbm.get_dataset_id_for_dataset_name(dataset_name=dataset_name)
    data_category_id = dbm.get_data_category_id_for_name(name=data_category)

    log.debug("Getting the image id list ...")
    image_ids = dbm.get_image_ids_for_data_category_and_dataset_id(
        dataset_id=dataset_id, data_category_id=data_category_id
    )

    return dt.CocoDataset(
        info=dt.CocoInfo(description=dataset_name),
        images=get_image_data(image_id_list=image_ids),
        annotations=get_annotation_data(image_id_list=image_ids),
        categories=get_category_data(),
        licenses=(dt.CocoLicense(),),
    )


def create_coco_ground_truth_json(
    dataset_name: str, data_category: str, ground_truth_json: Path
):
    """
    Get ground truth data for `dataset_name` and `data_category`
    from database and write COCO json.

    Args:
        dataset_name: Dataset name in database to create the COCO ground truth for
        data_category: The data category (training, validation, or testing)
            from the dataset
        ground_truth_json: The target file
    """
    coco_ground_truth = get_coco_dataset(
        dataset_name=dataset_name, data_category=data_category
    )
    with open(ground_truth_json, "w", encoding="utf-8") as f:
        json.dump(asdict(coco_ground_truth), f)


def evaluate_model_on_ground_truth(
    detection_json: Path, ground_truth_json: Path
) -> float:
    """
    Runs the COCO evaluation.

    Args:
        detection_json: The detection results in COCO json result format.
        ground_truth_json: The ground truth in COCO json format.

    Returns:
        The Average Precision @ Intersection over Union of 0.5 (AP@IoU=0.5)
    """
    coco_ground_truth = COCO(ground_truth_json)
    with open(detection_json, "r", encoding="utf-8") as f:
        detection_results = json.load(f)
    if not detection_results:
        print("No detection results found.")
        return 0.0
    coco_detections = coco_ground_truth.loadRes(str(detection_json))
    coco_eval = COCOeval(
        cocoGt=coco_ground_truth, cocoDt=coco_detections, iouType="bbox"
    )
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats.tolist()[1]
