"""This module validates the model prediction quality using COCO evaluation"""

import json
import shutil
from dataclasses import asdict
from pathlib import Path

import torch as th
import tqdm

import coco_evaluation_utilities
import configuration
import database_manager as dbm
import datasets
import datatypes as dt
import image_utilities as image_utils
import locations
import models
import names as n
import video_utilities as video_utils
from logger import logging

log = logging.getLogger(__name__)
log.setLevel(configuration.LOG_LEVEL)

# pylint: disable=no-value-for-parameter
#         Disabled, because the dbm-function receive the
#         cursor parameter from the decorator.


def prediction_to_coco_prediction_entry(
    prediction: dict[str, th.Tensor], image_id: int
) -> list[dt.CocoPredictionEntry]:
    labels = prediction["labels"].detach().to("cpu").tolist()
    boxes = prediction["boxes"].detach().to("cpu").tolist()
    scores = prediction["scores"].detach().to("cpu").tolist()
    if isinstance(image_id, th.Tensor):
        image_id = image_id.item()
    entries = []
    for box, label, score in zip(boxes, labels, scores):
        annotation = dt.CocoPredictionEntry(
            image_id=image_id,
            category_id=label,
            score=score,
            bbox=(box[0], box[1], box[2] - box[0], box[3] - box[1]),
        )
        entries.append(annotation)
    return entries


def infer_one_epoch(validation_model, data_loader):
    coco_predictions = []
    with th.no_grad():
        validation_model.eval()
        for images, labels in tqdm.tqdm(data_loader):
            # Moving input to the right device:
            images = list(image.to(device=configuration.DEVICE) for image in images)
            predictions = validation_model(images)
            for prediction, label, image in zip(predictions, labels, images):
                coco_predictions.extend(
                    prediction_to_coco_prediction_entry(
                        prediction=prediction, image_id=label["image_id"]
                    )
                )
            del images
            if configuration.DEVICE == "cuda":
                th.cuda.empty_cache()
    return tuple(coco_predictions)


def generate_coco_predictions_for_model(
    model_state_id: int,
    data_loader: th.utils.data.DataLoader,
    coco_prediction_file: Path,
):
    restored_model = models.restore_model_state(model_state_id=model_state_id)
    coco_prediction = infer_one_epoch(restored_model, data_loader)
    if not locations.Results.coco_jsons.exists():
        locations.Results.coco_jsons.mkdir(parents=True)
    with open(
        coco_prediction_file,
        "w",
        encoding="utf-8",
    ) as coco_prediction_f:
        json.dump([asdict(entry) for entry in coco_prediction], coco_prediction_f)


def evaluate_on_cranfield(model_state_id: int):
    ground_truth_json = locations.Results.coco_jsons / "cranfield_targets.json"
    log.info("Getting the ground truth ...")
    coco_evaluation_utilities.create_coco_ground_truth_json(
        dataset_name=n.DatasetNames.cranfield_default,
        data_category=n.DataCategoryNames.validation,
        ground_truth_json=ground_truth_json,
    )
    log.info("Generating the predictions ...")
    detection_json = locations.Results.coco_jsons / f"cranfield_prediction_model_state_{model_state_id:03d}.json"
    generate_coco_predictions_for_model(
        model_state_id=model_state_id,
        data_loader=datasets.get_cranfield_default_dataloader_validation(
            normalization_data_id=dbm.get_normalization_data_id_for_dataset_id(
                dbm.get_dataset_id_for_model_state_id(model_state_id)
            )
        ),
        coco_prediction_file=detection_json,
    )
    log.info("Evaluating ...")
    coco_evaluation_utilities.evaluate_model_on_ground_truth(
        detection_json=detection_json, ground_truth_json=ground_truth_json
    )


def evaluate_on_dvb_video(video_name: str, model_state_id: int) -> float:
    ground_truth_json = (
        locations.Results.coco_jsons / f"dvb_{video_name}_ground_truth.json"
    )
    log.info("Getting the ground truth ...")
    coco_evaluation_utilities.create_coco_ground_truth_json(
        dataset_name=video_name,
        data_category=n.DataCategoryNames.testing,
        ground_truth_json=ground_truth_json,
    )
    log.info("Generating the predictions ...")
    detection_json = (
        locations.Results.coco_jsons
        / f"dvb_{video_name}_model_state_{model_state_id:03d}_prediction.json"
    )
    log.info("Evaluating ...")
    generate_coco_predictions_for_model(
        model_state_id=model_state_id,
        data_loader=datasets.get_drone_vs_bird_single_video_dataloader(
            video_name=video_name,
            normalization_data_id=dbm.get_normalization_data_id_for_dataset_id(
                dataset_id=dbm.get_dataset_id_for_model_state_id(model_state_id)
            ),
        ),
        coco_prediction_file=detection_json,
    )
    return coco_evaluation_utilities.evaluate_model_on_ground_truth(
        detection_json=detection_json, ground_truth_json=ground_truth_json
    )


def create_video_from_coco_result_file(
    video_name_stem: str, json_path: Path, video_target_path: Path
):
    """
    Create a video from the results of a drone prediction run.

    Args:
        video_name_stem:
            Name of the video to illustrate the prediction and ground truth in.
        video_target_path:
            Location for the video to be stored in.
        json_path:
            Location of the Coco result json-file.

            It is structured like this::

                [
                  {
                    "image_id": 121090,
                    "category_id": 1,
                    "bbox": [
                      1174.987,  # x_min
                      349.123,   # y_min
                      87.533,  # width
                      56.181  # height
                    ],
                    "score": 0.990
                  },
                  ...
                ]
    """
    log.debug("Loading json file")
    with open(json_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    log.debug("Creating temporary folder ...")
    temp_folder = locations.TEMP_DIR / json_path.stem
    if temp_folder.exists():
        shutil.rmtree(temp_folder)
    temp_folder.mkdir()

    log.debug("Retrieving key info from the database ...")
    video_id = dbm.get_video_id_for_video_name_stem(video_name_stem=video_name_stem)
    image_ids = dbm.get_image_ids_for_video_id(video_id=video_id)

    log.debug("Sorting the predictions to image_ids ... ")
    predictions_for_images = {}
    for result in results:
        prediction = dt.Annotation(
            image_id=result["image_id"],
            label_id=result["category_id"],
            x_min=result["bbox"][0],
            y_min=result["bbox"][1],
            x_max=result["bbox"][0] + result["bbox"][2],
            y_max=result["bbox"][1] + result["bbox"][3],
            score=result["score"],
        )
        if prediction.image_id not in predictions_for_images:
            predictions_for_images[prediction.image_id] = [prediction]
        else:
            predictions_for_images[prediction.image_id].append(prediction)

    log.debug("Creating the images ...")
    image_utils.create_images_with_prediction_and_ground_truth(
        image_ids=image_ids,
        predictions_for_images=predictions_for_images,
        target_folder=temp_folder,
    )

    log.debug("Creating the video from the images ...")
    width, height = dbm.get_video_shape_for_video_id(video_id=video_id)
    if not video_target_path.parent.exists():
        video_target_path.parent.mkdir(parents=True)
    video_utils.create_video_from_images_in_folder(
        image_folder=temp_folder,
        width=width,
        height=height,
        video_target_path=video_target_path,
    )


if __name__ == "__main__":

    ap_05_l = []
    model_state = 69

    for video_name in n.DroneVsBirdVideos.video_names:

        ap_05 = evaluate_on_dvb_video(
            video_name=video_name,
            model_state_id=model_state)

        print(f"{video_name}: {ap_05}")
        ap_05_l.append(ap_05)

    for video_name, ap_05 in zip(n.DroneVsBirdVideos.video_names, ap_05_l):
        print(f"{video_name}: {ap_05:0.3f}")

        # create_video_from_coco_result_file(
        #     video_name_stem=video_name,
        #     json_path=locations.Results.coco_jsons
        #     / f"dvb_{video_name}_model_state_{model_state:03d}_prediction.json",
        #     video_target_path=locations.Results.coco_videos
        #     / f"dvb_{video_name}_model_state_{model_state:03d}_prediction.avi",
        # )
