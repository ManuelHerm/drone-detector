"""Module provides support functionality for datasets."""

from typing import TypeVar

import torch
import torchvision
import torchvision.tv_tensors
import tqdm

from source.config import locations, names, settings
from source.data import datatypes as dt
from source.data.cache import get_cache
from source.db import database_manager as dbm
from source.utils import image_utilities
from source.utils.logger import logging

T = TypeVar("T")
cranfield_cache = get_cache(locations.Cache.cranfield)

log = logging.getLogger(__name__)
log.setLevel(settings.LOG_LEVEL)

# pylint: disable=no-value-for-parameter
#         Disabled, because the dbm-function receive the
#         cursor parameter from the decorator.


def split_cranfields_dataset_into_train_and_val_and_insert_to_db(
    dataset_name: str,
    data_origin: str,
    validation_stride: int,
    has_limit: bool = False,
    limit: int = None,
):
    """
    Divide Cranfield's dataset with no birds and 40m box size
    into training and validation sets.

    Divide the set considering the number of drones in each picture.
    Each dataset shall have a comparable distribution of number of drones per picture.

    Args:
        dataset_name: The name of the dataset for Cranfield's default dataset.
        data_origin: The origin of the data origin for Cranfield's default dataset.
        validation_stride: Every `validation_stride` element in the dataset
            will become validation data.
        has_limit: Set to True, if you want to limit the dataset size.
        limit: Sum of images in training and validation set.
    """
    validation_flag_id = dbm.get_data_category_id_for_name(
        name=names.DataCategoryNames.validation
    )
    training_flag_id = dbm.get_data_category_id_for_name(
        name=names.DataCategoryNames.training
    )
    dataset_id = dbm.get_dataset_id_for_dataset_name(dataset_name=dataset_name)
    image_ids = dbm.get_image_ids_for_data_origin_name(
        data_origin_name=data_origin, has_limit=has_limit, limit=limit
    )
    image_ids_and_drone_numbers = dbm.get_number_of_drones_for_images(image_ids)
    # Sort considering number of drones - the second column.
    image_ids_and_drone_numbers = sorted(
        image_ids_and_drone_numbers, key=lambda row: row[1]
    )
    validation_indices = list(
        reversed(
            range(
                0,
                len(image_ids_and_drone_numbers),
                validation_stride,
            )
        )
    )
    validation_image_ids = [
        image_ids_and_drone_numbers[index][0] for index in validation_indices
    ]
    # Sort considering image_id - the first column.
    image_ids_and_drone_numbers = sorted(
        image_ids_and_drone_numbers, key=lambda row: row[0]
    )
    data_subsets = [{"image_id": row[0]} for row in image_ids_and_drone_numbers]
    for i, element in enumerate(data_subsets):
        if element["image_id"] in validation_image_ids:
            data_subsets[i]["data_category_id"] = validation_flag_id
        else:
            data_subsets[i]["data_category_id"] = training_flag_id
        data_subsets[i]["dataset_id"] = dataset_id
    dbm.insert_data_subset(data_subset_content=data_subsets)


def split_dataset_into_train_and_val_and_insert_to_db(
    dataset_name: str,
    data_origin: str,
    validation_stride: int,
):
    """
    Divide dataset into training and validation sets and insert into database.

    Every `validation_stride`-th image will be in the validation set.

    Args:
        dataset_name: The name of the dataset.
        data_origin: The data origin name of the data.
        validation_stride: Every `validation_stride`-th element in the dataset
            will become validation data.
    """
    validation_flag_id = dbm.get_data_category_id_for_name(
        name=names.DataCategoryNames.validation
    )
    training_flag_id = dbm.get_data_category_id_for_name(
        name=names.DataCategoryNames.training
    )
    dataset_id = dbm.get_dataset_id_for_dataset_name(dataset_name=dataset_name)
    image_ids = dbm.get_image_ids_for_data_origin_name(data_origin_name=data_origin)
    validation_indices = list(
        range(
            0,
            len(image_ids),
            validation_stride,
        )
    )
    validation_image_ids = [image_ids[index] for index in validation_indices]
    data_subsets = [{"image_id": image_id} for image_id in image_ids]
    for i, element in enumerate(data_subsets):
        if element["image_id"] in validation_image_ids:
            data_subsets[i]["data_category_id"] = validation_flag_id
        else:
            data_subsets[i]["data_category_id"] = training_flag_id
        data_subsets[i]["dataset_id"] = dataset_id
    dbm.insert_data_subset(data_subset_content=data_subsets)


def create_drone_vs_bird_testing_data_subset():
    """
    Create the Drone versus Bird testing data subset.

    It contains all previously ingested images from the Drone vs. Bird data origin.

    The names are defined by the variables names:
        - names.DatasetNames.drone_vs_bird
        - names.DataOriginNames.drone_vs_bird
    """
    log.info("Creating the Drone versus Bird data subset in the database ...")
    testing_flag_id = dbm.get_data_category_id_for_name(
        name=names.DataCategoryNames.testing
    )
    dataset_id = dbm.get_dataset_id_for_dataset_name(
        dataset_name=names.DatasetNames.drone_vs_bird
    )
    image_ids = dbm.get_image_ids_for_data_origin_name(
        data_origin_name=names.DataOriginNames.drone_vs_bird
    )
    data_subset = [
        {
            "data_category_id": testing_flag_id,
            "image_id": image_id,
            "dataset_id": dataset_id,
        }
        for image_id in tqdm.tqdm(image_ids)
    ]
    dbm.insert_data_subset(data_subset_content=data_subset)


if __name__ == "__main__":
    pass
