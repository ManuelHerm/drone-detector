"""This module captures all actions needed, to train and run the AI model."""

import sqlite3
from pathlib import Path

import tqdm

import annotation_utilites as annot_util
import configuration
import database_manager as dbm
import dataset_utilities as dataset_utils
import image_utilities as img_util
import locations
import names
import video_utilities as video_utils
from logger import logging
from source.annotation_utilites import fix_broken_annotations

log = logging.getLogger(__name__)
log.setLevel(configuration.LOG_LEVEL)


# pylint: disable=no-value-for-parameter
#         Disabled, because the dbm-function receive the
#         cursor parameter from the decorator.


def ingest_cranfield_data(
    data_origin_name: str,
    dataset_name: str,
    image_folder: Path,
    mask_folder: Path,
    validation_stride: int,
):
    # Insert Cranfield's data origin
    log.info("Ingesting Cranfield data with origin = {%s}", data_origin_name)
    dbm.insert_data_origin(data_origin_name)

    log.info("Inserting images ...")
    cranfield_images_paths = list(sorted(image_folder.iterdir()))
    for image_path in tqdm.tqdm(cranfield_images_paths):
        dbm.insert_image(
            img_util.create_image_from_path(
                path=image_path, data_origin=data_origin_name
            )
        )

    log.info("Inserting annotations ...")
    cranfield_masks_paths = list(sorted(mask_folder.iterdir()))
    for cranfield_masks_path in tqdm.tqdm(cranfield_masks_paths):
        annotations_per_image = (
            annot_util.get_annotations_from_cranfield_segmentation_mask(
                cranfield_masks_path
            )
        )
        dbm.insert_annotations(annotations_per_image)

    # Insert Cranfield's default dataset
    dbm.insert_dataset(name=dataset_name)

    log.info("Splitting dataset into training and validation ...")
    # Split into training and validation and record the split in the database
    dataset_utils.split_cranfields_dataset_into_training_and_validation_and_insert_to_db(
        dataset_name=dataset_name,
        data_origin=data_origin_name,
        validation_stride=validation_stride,
    )

    log.info("Computing and capturing the normalization data ...", data_origin_name)
    # Compute the normalization data for the training set and store the
    # result in the database
    dataset_utils.compute_normalization_data_and_insert_to_db(dataset_name=dataset_name)


def ingest_drone_vs_bird_data():
    # Insert Drone versus Bird data origin
    # dbm.insert_data_origin(names.DataOriginNames.drone_vs_bird)
    # Insert video meta data to database
    log.info("Ingesting drone vs. bird video meta data into database.")
    videos = []
    for video_path in tqdm.tqdm(sorted(locations.DroneVsBird.videos.iterdir())):
        videos.append(
            video_utils.create_video_from_path(
                path=video_path, data_origin_name=names.DataOriginNames.drone_vs_bird
            )
        )
    dbm.insert_videos(videos)
    log.info(
        "Turning the images on the disk into images in the database.\n"
        "Also recording the relationship between video, video frame and image."
    )
    video_utils.video_to_images(data_origin=names.DataOriginNames.drone_vs_bird)
    # Ingest the annotation data
    log.info("Ingesting drone vs. bird annotations into database.")
    annot_util.insert_all_dvb_annotations_into_db()


def create_drone_vs_bird_video_datasets():
    for name in tqdm.tqdm(names.DroneVsBirdVideos.video_names):
        video_id = dbm.get_video_id_for_video_name_stem(video_name_stem=name)
        image_ids = dbm.get_image_ids_for_video_id(video_id=video_id)
        data_category_id = dbm.get_data_category_id_for_name(
            name=names.DataCategoryNames.testing
        )
        dataset_id = dbm.insert_dataset(name=name)
        data_subset_content = list(
            {
                "dataset_id": dataset_id,
                "image_id": image_id,
                "data_category_id": data_category_id,
            }
            for image_id in image_ids
        )
        dbm.insert_data_subset(data_subset_content=data_subset_content)


def combining_datasets(
    dataset_name: str, first_dataset_name: str, second_dataset_name: str
):
    """
    Combining two datasets into a single dataset.

    Merging the training subsets with each other, the validation subsets with each other,
    and the testing subsets with each other.

    Args:
        dataset_name: The name of the combined dataset.
        first_dataset_name: Name of the first dataset to merge
        second_dataset_name: Name of the second dataset to merge
    """
    log.info(
        "Combining the datasets %s and %s.", first_dataset_name, second_dataset_name
    )
    dataset_id_1 = dbm.get_dataset_id_for_dataset_name(dataset_name=first_dataset_name)
    dataset_id_2 = dbm.get_dataset_id_for_dataset_name(dataset_name=second_dataset_name)
    try:
        dataset_id = dbm.insert_dataset(name=dataset_name)
    except sqlite3.Error:
        # If the dataset with the name is already inserted:
        dataset_id = dbm.get_dataset_id_for_dataset_name(dataset_name=dataset_name)
    data_categories = (
        dbm.get_data_category_id_for_name(name=names.DataCategoryNames.training),
        dbm.get_data_category_id_for_name(name=names.DataCategoryNames.validation),
        dbm.get_data_category_id_for_name(name=names.DataCategoryNames.testing),
    )
    for category in data_categories:
        try:
            image_ids = dbm.get_image_ids_for_data_category_and_dataset_id(
                dataset_id=dataset_id_1, data_category_id=category
            )
        except IndexError:
            # No images found for the combination of dataset_id and data_category_id
            image_ids = []
        try:
            image_ids_2 = dbm.get_image_ids_for_data_category_and_dataset_id(
                dataset_id=dataset_id_2, data_category_id=category
            )
        except IndexError:
            # No images found for the combination of dataset_id and data_category_id
            image_ids_2 = []
        image_ids.extend(image_ids_2)
        combined_image_ids = set(image_ids)
        data_subset_content = list(
            {
                "image_id": image_id,
                "dataset_id": dataset_id,
                "data_category_id": category,
            }
            for image_id in combined_image_ids
        )
        dbm.insert_data_subset(data_subset_content=data_subset_content)
    log.info("Computing the normalization data and inserting it into the database")
    dataset_utils.compute_normalization_data_and_insert_to_db(dataset_name=dataset_name)


if __name__ == "__main__":
    # Create the database tables
    dbm.initialize_database()
    # Insert the default dataset categories
    dbm.insert_data_categories()
    # Insert the Cranfield data
    ingest_cranfield_data(
        data_origin_name=names.DataOriginNames.cranfield_default,
        dataset_name=names.DatasetNames.cranfield_default,
        image_folder=locations.Cranfield.images,
        mask_folder=locations.Cranfield.masks,
        validation_stride=10,
    )
    ingest_cranfield_data(
        data_origin_name=names.DataOriginNames.cranfield_distractors,
        dataset_name=names.DatasetNames.cranfield_distractors,
        image_folder=locations.CranfieldDistractors.images,
        mask_folder=locations.CranfieldDistractors.masks,
        validation_stride=20,
    )
    combining_datasets(
        dataset_name=names.DatasetNames.cranfield_combined,
        first_dataset_name=names.DatasetNames.cranfield_default,
        second_dataset_name=names.DatasetNames.cranfield_distractors,
    )
    fix_broken_annotations()
    # Insert the Drone versus Bird data
    ingest_drone_vs_bird_data()
    # Create Drone versus Bird dataset
    dbm.insert_dataset(name=names.DatasetNames.drone_vs_bird)
    # Create the testing Drone versus Bird data subset
    dataset_utils.create_drone_vs_bird_testing_data_subset()
    # Insert a dataset for each drone vs. bird video
    create_drone_vs_bird_video_datasets()
