"""This module captures all actions needed, to train and run the AI model."""

import sqlite3
from pathlib import Path

import tqdm

from source.config import locations, names, settings
from source.db import database_initializer
from source.db import database_manager as dbm
from source.utils import (
    annotation_utilites,
    dataset_utilities,
    image_utilities,
    video_utilities,
)
from source.utils.annotation_utilites import fix_broken_annotations
from source.utils.logger import logging

log = logging.getLogger(__name__)
log.setLevel(settings.LOG_LEVEL)


# pylint: disable=no-value-for-parameter
#         Disabled, because the dbm-function receive the
#         cursor parameter from the decorator.


def ingest_current_images(
    data_origin_name: str,
    image_folder: Path,
    dataset_name: str,
    data_category_name: str,
):
    """
    Insert the images from the real world footage as a new dataset.

    The images are entered into the database.

    Args:
        data_origin_name: Name of the data origin to associate
            with the data.
        image_folder: The folder that contains the images
        dataset_name: Name of the dataset that will be created
            from the images.
        data_category_name: Name of the data category that
            will be used for the data.
    """
    # Insert Cranfield's data origin
    log.info("Ingesting with data origin = %s", data_origin_name)
    dbm.insert_data_origin(data_origin_name)

    log.info("Inserting dataset %s ...", dataset_name)
    dataset_id = dbm.insert_dataset(name=dataset_name)

    data_category_id = dbm.get_data_category_id_for_name(data_category_name)

    log.info("Inserting images ...")
    images_paths = list(sorted(image_folder.iterdir()))

    data_subset = []

    # For every individual image:
    for image_path in tqdm.tqdm(images_paths):

        image_id = dbm.insert_image(
            image_utilities.create_image_from_path(
                path=image_path, data_origin=data_origin_name
            )
        )

        data_subset.append(
            {
                "dataset_id": dataset_id,
                "image_id": image_id,
                "data_category_id": data_category_id,
            }
        )

    dbm.insert_data_subset(data_subset)


def ingest_cranfield_data(
    data_origin_name: str,
    dataset_name: str,
    image_folder: Path,
    mask_folder: Path,
    mask_mode: str,
    validation_stride: int,
):
    # Insert Cranfield's data origin
    log.info("Ingesting Cranfield data with origin = %s", data_origin_name)
    dbm.insert_data_origin(data_origin_name)

    log.info("Inserting images ...")
    cranfield_images_paths = list(sorted(image_folder.iterdir()))
    for image_path in tqdm.tqdm(cranfield_images_paths):
        dbm.insert_image(
            image_utilities.create_image_from_path(
                path=image_path, data_origin=data_origin_name
            )
        )

    log.info("Inserting annotations ...")
    cranfield_masks_paths = list(sorted(mask_folder.iterdir()))
    for cranfield_masks_path in tqdm.tqdm(cranfield_masks_paths):
        image_name = annotation_utilites.get_image_name_for_cranfield_mask_name(
            str(cranfield_masks_path.name)
        )
        image_id = dbm.get_image_id_for_image_name_and_data_origin(
            image_name=image_name, data_origin=data_origin_name
        )
        annotations_per_image = (
            annotation_utilites.get_annotations_from_segmentation_mask(
                image_id=image_id,
                mask_mode=mask_mode,
                mask_path=cranfield_masks_path,
                black_value=100,
            )
        )
        dbm.insert_annotations(annotations_per_image)

    # Insert dataset
    dbm.insert_dataset(name=dataset_name)

    log.info("Splitting dataset into training and validation ...")
    # Split into training and validation and record the split in the database
    dataset_utilities.split_dataset_into_train_and_val_and_insert_to_db(
        dataset_name=dataset_name,
        data_origin=data_origin_name,
        validation_stride=validation_stride,
    )


def ingest_training_data(
    data_origin_name: str,
    dataset_name: str,
    image_folder: Path,
    mask_folder: Path,
    mask_mode: str,
    consider_validation: bool,
    validation_stride: int,
):
    """
    Ingest image and annotation data into the database.
    Inserts every `validation_stride` element as validation data.

    Assumes:

    - The annotations need to be created from segmentation masks.
    - The image file and the segmentation mask file have the same name.
    - Every image has a segmentation mask image.

    Args:
        data_origin_name: The name of the data origin that
            the new data will be associated with.
        dataset_name: Name of the new dataset in the database.
        image_folder: The path where the new images lie.
        mask_folder: The path where the new segmentation masks lie.
        mask_mode: Options are:
            - "cluster": All drones are shown in white and clusters of
                white pixels are grouped to form a bounding box.
            - "color_based": Each drone is shown in either green, red,
                or blue. Hence, there can a maximum of three drones
                in the image.
        consider_validation: If part of the training data will become
            validation data.
        validation_stride: Every validation_stride image becomes
            validation data.
    """
    # Insert data origin
    log.info(
        "Ingesting data with origin %s and dataset name %s",
        data_origin_name,
        dataset_name,
    )
    dbm.insert_data_origin(name=data_origin_name)
    dataset_id = dbm.insert_dataset(name=dataset_name)
    train_category_id = dbm.get_data_category_id_for_name(
        name=names.DataCategoryNames.training
    )
    val_category_id = dbm.get_data_category_id_for_name(
        name=names.DataCategoryNames.validation
    )

    log.info("Matching images to segmentation mask images")
    image_paths = list(sorted(image_folder.iterdir()))
    mask_paths = list(sorted(mask_folder.iterdir()))

    if tuple(image.name for image in image_paths) != tuple(
        mask.name for mask in mask_paths
    ):
        raise RuntimeError("Not every image has a matching segmentation mask file.")

    log.info("Inserting images and annotations ...")
    for i, (image_path, mask_path) in tqdm.tqdm(
        enumerate(zip(image_paths, mask_paths))
    ):
        image_id = dbm.insert_image(
            image_utilities.create_image_from_path(
                path=image_path, data_origin=data_origin_name
            )
        )
        annotations_per_image = (
            annotation_utilites.get_annotations_from_segmentation_mask(
                image_id=image_id,
                mask_path=mask_path,
                mask_mode=mask_mode,
                black_value=100,
            )
        )
        dbm.insert_annotations(annotation_list=annotations_per_image)
        if consider_validation and (i % validation_stride == 0):
            dbm.insert_dataset_element(
                dataset_id=dataset_id,
                image_id=image_id,
                data_category_id=val_category_id,
            )
        else:
            dbm.insert_dataset_element(
                dataset_id=dataset_id,
                image_id=image_id,
                data_category_id=train_category_id,
            )


def ingest_drone_vs_bird_data():
    # Insert Drone versus Bird data origin
    # dbm.insert_data_origin(names.DataOriginNames.drone_vs_bird)
    # Insert video meta data to database
    log.info("Ingesting drone vs. bird video meta data into database.")
    videos = []
    for video_path in tqdm.tqdm(sorted(locations.DroneVsBird.videos.iterdir())):
        videos.append(
            video_utilities.create_video_from_path(
                path=video_path, data_origin_name=names.DataOriginNames.drone_vs_bird
            )
        )
    dbm.insert_videos(videos)
    log.info(
        "Turning the images on the disk into images in the database.\n"
        "Also recording the relationship between video, video frame and image."
    )
    video_utilities.video_to_images(data_origin=names.DataOriginNames.drone_vs_bird)
    # Ingest the annotation data
    log.info("Ingesting drone vs. bird annotations into database.")
    annotation_utilites.insert_all_dvb_annotations_into_db()


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

    Merging the training subsets with each other,
    the validation subsets with each other,
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


def fix_annotations_cranfield_default():
    data_origin_name = names.DataOriginNames.cranfield_distractors
    mask_folder = locations.CranfieldDistractors.masks

    log.info("Matching images to segmentation mask images")
    mask_paths = list(sorted(mask_folder.iterdir()))

    log.info("Inserting images and annotations ...")
    for mask_path in tqdm.tqdm(mask_paths):
        image_id = dbm.get_image_id_for_image_name_and_data_origin(
            image_name=annotation_utilites.get_image_name_for_cranfield_mask_name(
                mask_name=mask_path.name
            ),
            data_origin=data_origin_name,
        )
        annotations_per_image = (
            annotation_utilites.get_annotations_from_segmentation_mask(
                image_id=image_id,
                mask_path=mask_path,
                mask_mode="cluster",
                black_value=100,
            )
        )
        dbm.insert_annotations(annotation_list=annotations_per_image)


if __name__ == "__main__":
    # # Create the database tables
    # database_initializer.initialize_database()
    # # Insert the default dataset categories
    # dbm.insert_data_categories()
    # # Insert the Cranfield data
    # ingest_cranfield_data(
    #     data_origin_name=names.DataOriginNames.cranfield_default,
    #     dataset_name=names.DatasetNames.cranfield_default,
    #     image_folder=locations.Cranfield.images,
    #     mask_folder=locations.Cranfield.masks,
    #     validation_stride=10,
    # )
    # ingest_cranfield_data(
    #     data_origin_name=names.DataOriginNames.cranfield_distractors,
    #     dataset_name=names.DatasetNames.cranfield_distractors,
    #     image_folder=locations.CranfieldDistractors.images,
    #     mask_folder=locations.CranfieldDistractors.masks,
    #     validation_stride=20,
    # )
    # combining_datasets(
    #     dataset_name=names.DatasetNames.cranfield_combined,
    #     first_dataset_name=names.DatasetNames.cranfield_default,
    #     second_dataset_name=names.DatasetNames.cranfield_distractors,
    # )
    # fix_broken_annotations()
    # # Insert the Drone versus Bird data
    # ingest_drone_vs_bird_data()
    # # Create Drone versus Bird dataset
    # dbm.insert_dataset(name=names.DatasetNames.drone_vs_bird)
    # # Create the testing Drone versus Bird data subset
    # dataset_utilities.create_drone_vs_bird_testing_data_subset()
    # # Insert a dataset for each drone vs. bird video
    # create_drone_vs_bird_video_datasets()

    # # Insert current work's data into the database
    # ingest_training_data(
    #     data_origin_name=names.DataOriginNames.current_work,
    #     dataset_name=names.DatasetNames.forrest_hut_phantom,
    #     image_folder=locations.ForrestHutPhantom.images,
    #     mask_folder=locations.ForrestHutPhantom.masks,
    #     mask_mode="cluster",
    #     validation_stride=20,
    # )
    # ingest_training_data(
    #     data_origin_name=names.DataOriginNames.current_work,
    #     dataset_name=names.DatasetNames.forrest_hut_mini,
    #     image_folder=locations.ForrestHutMini.images,
    #     mask_folder=locations.ForrestHutMini.masks,
    #     mask_mode="cluster",
    #     validation_stride=20,
    # )
    # ingest_training_data(
    #     data_origin_name=names.DataOriginNames.current_work,
    #     dataset_name=names.DatasetNames.forrest_hut_inspire,
    #     image_folder=locations.ForrestHutInspire.images,
    #     mask_folder=locations.ForrestHutInspire.masks,
    #     mask_mode="cluster",
    #     validation_stride=20,
    # )
    # ingest_training_data(
    #     data_origin_name=names.DataOriginNames.current_work,
    #     dataset_name=names.DatasetNames.campus,
    #     image_folder=locations.Campus.images,
    #     mask_folder=locations.Campus.masks,
    #     mask_mode="color_based",
    #     validation_stride=20,
    # )
    # ingest_training_data(
    #     data_origin_name=names.DataOriginNames.current_work,
    #     dataset_name=names.DatasetNames.hill,
    #     image_folder=locations.Hill.images,
    #     mask_folder=locations.Hill.masks,
    #     mask_mode="color_based",
    #     validation_stride=20,
    # )
    # ingest_training_data(
    #     data_origin_name=names.DataOriginNames.current_work,vim.g.tagbar_iconchars = { "▶", "▼" }
    #     dataset_name=names.DatasetNames.drone_school_far,
    #     image_folder=locations.DroneSchoolFar.images,
    #     mask_folder=locations.DroneSchoolFar.masks,tauscht man gleich die PU gegen bessere. Schade, dass Ibanez nicht etwas mehr investiert hat für einen Graphtech-Sattel... ist leider nicht ganz stimmrein, was bei dem Preis aber völlig ok ist. Rest ist prima, die Verarbeitung bis auf eine etwas unschöne Kante am Pick-Guard (mit einmal drüberschleifen weg) wunderbar, und gemessen am Preis ist die Gitarre wohl fast unschlagbar. Ich habe 15 Gitarren, die teilweise das Zehnfache kosten... trotzdem spiele ich nun dauernd diese kleine Japanerin (die ja aus Indonesien kommt...). Klare Kaufempfehlung.
    #     mask_mode="cluster",
    #     validation_stride=20,
    # )
    # ingest_training_data(
    #     data_origin_name=names.DataOriginNames.current_work,
    #     dataset_name=names.DatasetNames.drone_school_mid,
    #     image_folder=locations.DroneSchoolMid.images,
    #     mask_folder=locations.DroneSchoolMid.masks,
    #     mask_mode="cluster",
    #     validation_stride=20,
    # )
    # ingest_training_data(
    #     data_origin_name=names.DataOriginNames.current_work,
    #     dataset_name=names.DatasetNames.drone_school_mini,
    #     image_folder=locations.DroneSchoolMini.images,
    #     mask_folder=locations.DroneSchoolMini.masks,
    #     mask_mode="cluster",
    #     validation_stride=20,
    # )
    # ingest_training_data(
    #     data_origin_name=names.DataOriginNames.current_work,
    #     dataset_name=names.DatasetNames.drone_school_inspire,
    #     image_folder=locations.DroneSchoolInspire.images,
    #     mask_folder=locations.DroneSchoolInspire.masks,
    #     mask_mode="cluster",
    #     validation_stride=20,
    # )
    # ingest_training_data(
    #     data_origin_name=names.DataOriginNames.current_work,
    #     dataset_name=names.DatasetNames.drone_school_phantom,
    #     image_folder=locations.DroneSchoolPhantom.images,
    #     mask_folder=locations.DroneSchoolPhantom.masks,
    #     mask_mode="cluster",
    #     validation_stride=20,
    # )
    # dataset_names = [
    #     names.DatasetNames.cranfield_combined,
    #     names.DatasetNames.forrest_hut_phantom,
    #     names.DatasetNames.forrest_hut_mini,
    #     names.DatasetNames.forrest_hut_inspire,
    #     names.DatasetNames.campus,
    #     names.DatasetNames.hill,
    #     names.DatasetNames.drone_school_phantom,
    #     names.DatasetNames.drone_school_mini,
    #     names.DatasetNames.drone_school_inspire,
    #     names.DatasetNames.drone_school_mid,
    #     names.DatasetNames.drone_school_far,
    # ]
    # dbm.combine_datasets(
    #     new_dataset_name=names.DatasetNames.all_combined, names_to_combine=dataset_names
    # )
    # dbm.combine_datasets(
    #     new_dataset_name=names.DatasetNames.drone_schools,
    #     names_to_combine=(
    #         names.DatasetNames.drone_school_far,
    #         names.DatasetNames.drone_school_mid,
    #         names.DatasetNames.drone_school_inspire,
    #         names.DatasetNames.drone_school_mini,
    #         names.DatasetNames.drone_school_phantom,
    #     ),
    # )
    # dbm.combine_datasets(
    #     new_dataset_name=names.DatasetNames.hill_campus_forrest,
    #     names_to_combine=[
    #         names.DatasetNames.campus,
    #         names.DatasetNames.hill,
    #         names.DatasetNames.forrest_hut_inspire,
    #         names.DatasetNames.forrest_hut_mini,
    #         names.DatasetNames.forrest_hut_phantom,
    #     ],
    # )
    # dbm.decimate_dataset(
    #     new_dataset_name=names.DatasetNames.campus_section,
    #     name_to_decimate=names.DatasetNames.campus,
    #     stride=10,
    #     start=0,
    # )
    # dbm.decimate_dataset(
    #     new_dataset_name=names.DatasetNames.hill_section,
    #     name_to_decimate=names.DatasetNames.hill,
    #     stride=10,
    #     start=0,
    # )
    # dbm.decimate_dataset(
    #     new_dataset_name=names.DatasetNames.forrest_hut_phantom_section,
    #     name_to_decimate=names.DatasetNames.forrest_hut_phantom,
    #     stride=14,
    #     start=0,
    # )
    # dbm.decimate_dataset(
    #     new_dataset_name=names.DatasetNames.forrest_hut_mini_section,
    #     name_to_decimate=names.DatasetNames.forrest_hut_mini,
    #     stride=14,
    #     start=4,
    # )
    # dbm.decimate_dataset(
    #     new_dataset_name=names.DatasetNames.forrest_hut_inspire_section,
    #     name_to_decimate=names.DatasetNames.forrest_hut_inspire,
    #     stride=14,
    #     start=8,
    # )
    # dbm.decimate_dataset(
    #     new_dataset_name=names.DatasetNames.drone_school_mid_section,
    #     name_to_decimate=names.DatasetNames.drone_school_mid,
    #     stride=10,
    #     start=0,
    # )
    # dbm.combine_datasets(
    #     new_dataset_name=names.DatasetNames.campus_forrest_hill_school_section,
    #     names_to_combine=[
    #         names.DatasetNames.campus_section,
    #         names.DatasetNames.hill_section,
    #         names.DatasetNames.forrest_hut_inspire_section,
    #         names.DatasetNames.forrest_hut_mini_section,
    #         names.DatasetNames.forrest_hut_phantom_section,
    #         names.DatasetNames.drone_school_mid_section,
    #     ],
    # )
    # dbm.combine_datasets(
    #     new_dataset_name=names.DatasetNames.cranfield_default_and_current_section,
    #     names_to_combine=[
    #         names.DatasetNames.cranfield_default,
    #         names.DatasetNames.campus_forrest_hill_school_section,
    #     ],
    # )
    # dbm.insert_data_origin(name=names.DataOriginNames.current_work_real_life)
    # for image_folder in locations.CurrentWorkRealWorldFootage.images.iterdir():
    #     ingest_current_images(
    #         data_origin_name=names.DataOriginNames.current_work_real_life,
    #         image_folder=image_folder,
    #         dataset_name=image_folder.name,
    #         data_category_name=names.DataCategoryNames.testing,
    #     )
    # fix_annotations_cranfield_default()
    # ingest_training_data(
    #     data_origin_name=names.DataOriginNames.current_work_city,
    #     dataset_name=names.DatasetNames.city,
    #     image_folder=locations.City.images,
    #     mask_folder=locations.City.masks,
    #     mask_mode="cluster",
    #     consider_validation=False,
    #     validation_stride=1_000_000,
    # )
    # ingest_training_data(
    #     data_origin_name=names.DataOriginNames.current_work_meadow,
    #     dataset_name=names.DatasetNames.meadow,
    #     image_folder=locations.Meadow.images,
    #     mask_folder=locations.Meadow.masks,
    #     mask_mode="cluster",
    #     consider_validation=False,
    #     validation_stride=1_000_000,
    # )
    # ingest_training_data(
    #     data_origin_name=names.DataOriginNames.current_work_lake,
    #     dataset_name=names.DatasetNames.lake,
    #     image_folder=locations.Lake.images,
    #     mask_folder=locations.Lake.masks,
    #     mask_mode="cluster",
    #     consider_validation=False,
    #     validation_stride=1_000_000,
    # )
    # dbm.decimate_dataset(
    #     new_dataset_name=names.DatasetNames.city_section,
    #     name_to_decimate=names.DatasetNames.city,
    #     stride=4,
    #     start=0,
    # )
    # dbm.decimate_dataset(
    #     new_dataset_name=names.DatasetNames.meadow_section,
    #     name_to_decimate=names.DatasetNames.meadow,
    #     stride=4,
    #     start=0,
    # )
    # dbm.decimate_dataset(
    #     new_dataset_name=names.DatasetNames.lake_se-ction,
    #     name_to_decimate=names.DatasetNames.lake,
    #     stride=4,
    #     start=0,
    # )
    # dbm.combine_datasets(
    #     new_dataset_name=names.DatasetNames.best_shot,
    #     names_to_combine=[
    #         names.DatasetNames.campus_section,
    #         names.DatasetNames.hill_section,
    #         names.DatasetNames.forrest_hut_inspire_section,
    #         names.DatasetNames.forrest_hut_mini_section,
    #         names.DatasetNames.forrest_hut_phantom_section,
    #         names.DatasetNames.city_section,
    #         names.DatasetNames.lake_section,
    #         names.DatasetNames.meadow_section,
    #         names.DatasetNames.cranfield_combined,
    #     ],
    # )
    # ingest_current_images(
    #     data_origin_name=names.DataOriginNames.current_work_disorder,
    #     image_folder=locations.Disorder.images,
    #     dataset_name=names.DatasetNames.disorder,
    #     data_category_name=names.DataCategoryNames.training,
    # )
    # dbm.combine_datasets(
    #     new_dataset_name=names.DatasetNames.next_best_shot,
    #     names_to_combine=[
    #         names.DatasetNames.best_shot,
    #         names.DatasetNames.disorder,
    #     ],
    # )
    # ingest_current_images(
    #     data_origin_name=names.DataOriginNames.current_work_lanterns,
    #     image_folder=locations.Lanterns.images,
    #     dataset_name=names.DatasetNames.lanterns,
    #     data_category_name=names.DataCategoryNames.training,
    # )
    # ingest_current_images(
    #     data_origin_name=names.DataOriginNames.current_work_container,
    #     image_folder=locations.Container.images,
    #     dataset_name=names.DatasetNames.container,
    #     data_category_name=names.DataCategoryNames.training,
    # )
    # dbm.combine_datasets(
    #     new_dataset_name=names.DatasetNames.optimization_0,
    #     names_to_combine=[
    #         names.DatasetNames.next_best_shot,
    #         names.DatasetNames.lanterns,
    #         names.DatasetNames.container,
    #     ],
    # )
    # dbm.decimate_dataset(
    #     new_dataset_name=names.DatasetNames.cranfield_combined_section,
    #     name_to_decimate=names.DatasetNames.cranfield_combined,
    #     stride=5,
    #     start=0,
    # )
    # dbm.combine_datasets(
    #     new_dataset_name=names.DatasetNames.optimization_1,
    #     names_to_combine=[
    #         names.DatasetNames.campus_section,
    #         names.DatasetNames.hill_section,
    #         names.DatasetNames.forrest_hut_inspire_section,
    #         names.DatasetNames.forrest_hut_mini_section,
    #         names.DatasetNames.forrest_hut_phantom_section,
    #         names.DatasetNames.city_section,
    #         names.DatasetNames.lake_section,
    #         names.DatasetNames.meadow_section,
    #         names.DatasetNames.cranfield_combined_section,
    #         names.DatasetNames.lanterns,
    #         names.DatasetNames.container,
    #     ],
    # )
    # ingest_current_images(
    #     data_origin_name=names.DataOriginNames.current_work_normal_twigs,
    #     image_folder=locations.Trees.images_normal,
    #     dataset_name=names.DatasetNames.twigs_tree,
    #     data_category_name=names.DataCategoryNames.training,
    # )
    # ingest_current_images(
    #     data_origin_name=names.DataOriginNames.current_work_pine_twigs,
    #     image_folder=locations.Trees.images_pine,
    #     dataset_name=names.DatasetNames.twigs_pine,
    #     data_category_name=names.DataCategoryNames.training,
    # )
    # ingest_current_images(
    #     data_origin_name=names.DataOriginNames.current_work_container_full,
    #     image_folder=locations.Container.images_full,
    #     dataset_name=names.DatasetNames.container_full,
    #     data_category_name=names.DataCategoryNames.training,
    # )
    # ingest_current_images(
    #     data_origin_name=names.DataOriginNames.current_work_disorder_full,
    #     image_folder=locations.Disorder.images_full,
    #     dataset_name=names.DatasetNames.disorder_full,
    #     data_category_name=names.DataCategoryNames.training,
    # )
    # ingest_current_images(
    #     data_origin_name=names.DataOriginNames.current_work_lanterns_full,
    #     image_folder=locations.Lanterns.images_full,
    #     dataset_name=names.DatasetNames.lanterns_full,
    #     data_category_name=names.DataCategoryNames.training,
    # )
    # dbm.combine_datasets(
    #     new_dataset_name=names.DatasetNames.thesis_new,
    #     names_to_combine=[
    #         names.DatasetNames.cranfield_default,
    #         names.DatasetNames.forrest_hut_inspire_section,
    #         names.DatasetNames.forrest_hut_mini_section,
    #         names.DatasetNames.forrest_hut_phantom_section,
    #         names.DatasetNames.hill_section,
    #         names.DatasetNames.campus_section,
    #         names.DatasetNames.city_section,
    #         names.DatasetNames.meadow_section,
    #         names.DatasetNames.lake_section,
    #         names.DatasetNames.disorder_full,
    #         names.DatasetNames.lanterns_full,
    #         names.DatasetNames.container_full,
    #         names.DatasetNames.twigs_tree,
    #         names.DatasetNames.twigs_pine,
    #     ],
    # )
    # ingest_training_data(
    #     data_origin_name=names.DataOriginNames.current_work_parrot_one,
    #     dataset_name=names.DatasetNames.parrot_one,
    #     image_folder=locations.ParrotOne.images,
    #     mask_folder=locations.ParrotOne.masks,
    #     mask_mode="cluster",
    #     consider_validation=False,
    #     validation_stride=1_000_000,
    # )
    # ingest_training_data(
    #     data_origin_name=names.DataOriginNames.current_work_parrot_two,
    #     dataset_name=names.DatasetNames.parrot_two,
    #     image_folder=locations.ParrotTwo.images,
    #     mask_folder=locations.ParrotTwo.masks,
    #     mask_mode="cluster",
    #     consider_validation=False,
    #     validation_stride=1_000_000,
    # )
    # ingest_current_images(
    #     data_origin_name=names.DataOriginNames.current_work_moon,
    #     image_folder=locations.Moon.images,
    #     dataset_name=names.DatasetNames.moon,
    #     data_category_name=names.DataCategoryNames.training,
    # )
    # dbm.combine_datasets(
    #     new_dataset_name=names.DatasetNames.optimization_3,
    #     names_to_combine=[
    #         names.DatasetNames.cranfield_default,
    #         names.DatasetNames.forrest_hut_inspire_section,
    #         names.DatasetNames.forrest_hut_mini_section,
    #         names.DatasetNames.forrest_hut_phantom_section,
    #         names.DatasetNames.hill_section,
    #         names.DatasetNames.campus_section,
    #         names.DatasetNames.city_section,
    #         names.DatasetNames.meadow_section,
    #         names.DatasetNames.lake_section,
    #         names.DatasetNames.disorder_full,
    #         names.DatasetNames.lanterns_full,
    #         names.DatasetNames.container_full,
    #         names.DatasetNames.twigs_tree,
    #         names.DatasetNames.twigs_pine,
    #         names.DatasetNames.parrot_one,
    #         names.DatasetNames.parrot_two,
    #         names.DatasetNames.moon,
    #     ],
    # )
    pass
