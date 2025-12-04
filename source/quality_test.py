"""This module validates the model prediction quality using COCO evaluation"""

import json
import multiprocessing as mp
import shutil
from dataclasses import asdict
from pathlib import Path

import PIL.Image
import torch as th
import tqdm
from torchvision.transforms import v2

from source import datasets, models
from source.config import locations, names, settings
from source.data import datatypes as dt
from source.db import database_manager as dbm
from source.utils import coco_evaluation_utilities, image_utilities, video_utilities
from source.utils.logger import logging

log = logging.getLogger(__name__)
log.setLevel(settings.LOG_LEVEL)


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
            images = list(image.to(device=settings.DEVICE) for image in images)
            predictions = validation_model(images)
            for prediction, label, image in zip(predictions, labels, images):
                coco_predictions.extend(
                    prediction_to_coco_prediction_entry(
                        prediction=prediction, image_id=label["image_id"]
                    )
                )
            del images
            del labels
            del predictions
            if settings.DEVICE == "cuda":
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
        dataset_name=names.DatasetNames.cranfield_default,
        data_category=names.DataCategoryNames.validation,
        ground_truth_json=ground_truth_json,
    )
    log.info("Generating the predictions ...")
    detection_json = (
        locations.Results.coco_jsons
        / f"cranfield_prediction_model_state_{model_state_id:03d}.json"
    )
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
        data_category=names.DataCategoryNames.testing,
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
    video_name_stem: str,
    json_path: Path,
    video_target_path: Path,
    frame_rate: float = 60,
):
    """
    Create a video from the results of a drone prediction run.

    Args:
        video_name_stem:
            Name of the video to illustrate the prediction and ground truth in.
        video_target_path:
            Location for the video to be stored in.
        frame_rate: The frame rate of the video (optional). Default is 60 FPS.
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
    image_utilities.create_images_with_prediction_and_ground_truth(
        image_ids=image_ids,
        predictions_for_images=predictions_for_images,
        target_folder=temp_folder,
    )

    log.debug("Creating the video from the images ...")
    width, height = dbm.get_video_shape_for_video_id(video_id=video_id)
    if not video_target_path.parent.exists():
        video_target_path.parent.mkdir(parents=True)
    video_utilities.create_video_from_images_in_folder(
        image_folder=temp_folder,
        width=width,
        height=height,
        video_target_path=video_target_path,
        frame_rate=frame_rate,
    )


def create_video_worker(video_name: str, model_state_id: int):
    create_video_from_coco_result_file(
        video_name_stem=video_name,
        json_path=locations.Results.coco_jsons
        / f"dvb_{video_name}_model_state_{model_state_id:03d}_prediction.json",
        video_target_path=locations.Results.coco_videos
        / "hard_videos"
        / f"dvb_{video_name}_model_state_{model_state_id:03d}_prediction.avi",
        frame_rate=45.0,
    )
    print(f"Video {video_name} completed.")


def create_hard_videos():

    model_state = 80
    hard_videos = names.DroneVsBirdVideos.hard_videos

    with mp.Pool(processes=mp.cpu_count()) as pool:
        # `map` preserves order; `imap_unordered` yields results as they finish
        pool.starmap(
            create_video_worker,
            ((hard_video, model_state) for hard_video in hard_videos),
        )
        # evaluate_on_dvb_video(
        #     video_name=video_name,
        #     model_state_id=model_state)


def infer_one_image(image_path: Path, model_state_id: int):

    image = PIL.Image.open(image_path, mode="r").convert("RGB")

    normalization_data = dbm.get_normalization_data_for_id(
        normalization_data_id=dbm.get_normalization_data_id_for_dataset_id(
            dataset_id=dbm.get_dataset_id_for_model_state_id(model_state_id)
        )
    )

    transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(th.float32, scale=True),
            v2.Normalize(mean=normalization_data.mean, std=normalization_data.std),
        ]
    )
    input_image_tensor = transforms(image).to(settings.DEVICE)

    restored_model = models.restore_model_state(model_state_id=model_state_id)
    restored_model.eval()
    with th.no_grad():
        predictions = restored_model([input_image_tensor])

    annotations = []
    labels = predictions[0]["labels"].detach().to("cpu").tolist()
    boxes = predictions[0]["boxes"].detach().to("cpu").tolist()
    scores = predictions[0]["scores"].detach().to("cpu").tolist()
    for label, box, score in zip(labels, boxes, scores):
        annotations.append(
            dt.Annotation(
                image_id=-1,
                label_id=label,
                x_min=box[0],
                y_min=box[1],
                x_max=box[2],
                y_max=box[3],
                id=-1,
                score=score,
            )
        )

    image_utilities.add_bounding_boxes_to_image(
        img=image,
        bboxes=annotations,
        legend_entry=dt.ImageLegendEntry(xy=(5, 5), text="Prediction"),
    ).show()


if __name__ == "__main__":
    # create_hard_videos()
    # model_states = [69, 80]
    # file = locations.RESULTS_DIR / f"ap05_model_state_{model_state}.csv"
    #
    # with open(file, "a", encoding="utf-8") as f:
    #     f.write("Video Name,AP@0.5\n")
    #
    # for video_name in ["fixed_wing_over_hill_1"]:
    #     ap_05 = evaluate_on_dvb_video(
    #         video_name=video_name,
    #         model_state_id=model_state)
    #
    #     with open(file, "a", encoding="utf-8") as f:
    #         f.write(f"{video_name},{ap_05}\n")
    #
    #     print(f"{video_name},{ap_05}")

    infer_one_image(
        image_path=locations.TEMP_DIR / "image_size_experiment" / "larger" / "0.png",
        model_state_id=80,
    )

    infer_one_image(
        image_path=locations.TEMP_DIR / "image_size_experiment" / "same" / "0.png",
        model_state_id=80,
    )

    infer_one_image(
        image_path=locations.TEMP_DIR / "image_size_experiment" / "smaller" / "0.png",
        model_state_id=80,
    )

    infer_one_image(
        image_path=locations.TEMP_DIR / "image_size_experiment" / "smallest" / "0.png",
        model_state_id=80,
    )
