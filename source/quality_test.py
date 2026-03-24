"""This module validates the model prediction quality using COCO evaluation"""

import json
import multiprocessing as mp
import shutil
import uuid
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import PIL.Image
import torch
import tqdm
from torchvision.transforms import v2

from source import datasets, models
from source.config import locations, names, settings
from source.config.model_register import ModelRegister
from source.data import datatypes as dt
from source.db import database_manager as dbm
from source.utils import coco_evaluation_utilities, image_utilities, video_utilities
from source.utils.logger import logging

log = logging.getLogger(__name__)
log.setLevel(settings.LOG_LEVEL)


# pylint: disable=no-value-for-parameter
#         Disabled, because the dbm-function receive the
#         cursor parameter from the decorator.


def generate_prediction_video(
    model_id: int,
    model: torch.nn.Module,
    dataset_name: str,
    target: Path,
):
    dataset_id = dbm.get_dataset_id_for_dataset_name(dataset_name=dataset_name)
    image_ids = dbm.get_image_ids_for_dataset_id(dataset_id=dataset_id)
    data_loader = datasets.get_dataloader(
        dataset_name=dataset_name,
        data_category_names=[names.DataCategoryNames.testing],
        augment=False,
    )
    predictions = infer_one_epoch(model, data_loader)
    predictions_for_images = {}
    for prediction in predictions:
        if prediction.image_id not in predictions_for_images:
            predictions_for_images[prediction.image_id] = [prediction.to_annotation()]
        else:
            predictions_for_images[prediction.image_id].append(
                prediction.to_annotation()
            )

    image_folder = locations.Results.coco_images / dataset_name / f"{model_id}"
    if not image_folder.exists():
        image_folder.mkdir(parents=True)
    log.info("Creating the images ...")
    image_utilities.create_images_with_prediction_and_ground_truth(
        image_ids=image_ids,
        predictions_for_images=predictions_for_images,
        target_folder=image_folder,
    )
    log.info("Creating the video ... ")
    if not target.parent.exists():
        target.parent.mkdir(parents=True)
    width, height = dbm.get_image_width_and_height_for_image_id(image_ids[0])
    video_utilities.create_video_from_images_in_folder(
        image_folder=image_folder,
        video_target_path=target,
        width=width,
        height=height,
        frame_rate=25,
    )


def prediction_to_coco_prediction_entry(
    prediction: dict[str, torch.Tensor], image_id: int
) -> list[dt.CocoPredictionEntry]:
    labels = prediction["labels"].detach().to("cpu").tolist()
    boxes = prediction["boxes"].detach().to("cpu").tolist()
    scores = prediction["scores"].detach().to("cpu").tolist()
    if isinstance(image_id, torch.Tensor):
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


def infer_one_epoch(
    inference_model: torch.nn.Module, data_loader: torch.utils.data.DataLoader
) -> tuple[dt.CocoPredictionEntry]:
    was_training = inference_model.training
    coco_predictions = []
    with torch.inference_mode():
        inference_model.eval()
        for images, labels in tqdm.tqdm(data_loader):
            # Moving input to the right device:
            images = list(image.to(device=settings.DEVICE) for image in images)
            predictions = inference_model(images)
            for prediction, label in zip(predictions, labels):
                coco_predictions.extend(
                    prediction_to_coco_prediction_entry(
                        prediction=prediction, image_id=label["image_id"]
                    )
                )
            del images
            del labels
            del predictions
    if was_training:
        inference_model.train()
    return tuple(coco_predictions)


def generate_coco_predictions_for_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    coco_prediction_file: Path,
):
    coco_predictions = infer_one_epoch(model, data_loader)
    if not coco_prediction_file.parent.exists():
        coco_prediction_file.parent.mkdir(parents=True)
    with open(
        coco_prediction_file,
        "w",
        encoding="utf-8",
    ) as coco_prediction_f:
        json.dump([asdict(entry) for entry in coco_predictions], coco_prediction_f)


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
        model=models.restore_model_state(model_id=model_state_id),
        data_loader=datasets.get_dataloader(
            dataset_name=names.DatasetNames.cranfield_default,
            data_category_names=[names.DataCategoryNames.validation],
            augment=False,
            get_untransformed_func=datasets.get_untransformed_cranfield,
        ),
        coco_prediction_file=detection_json,
    )
    log.info("Evaluating ...")
    coco_evaluation_utilities.evaluate_model_on_ground_truth(
        detection_json=detection_json, ground_truth_json=ground_truth_json
    )


def evaluate_on_dataset(
    dataset_name: str, eval_model: torch.nn.Module, model_state_id: int = -1
) -> float:
    """
    Determine model's the AP@05 quality value on the given video.

    Args:
        dataset_name: The dataset name that shall be used for evaluation.
            It it can be a drone video from Drone versus Bird.
        eval_model: The model to test
        model_state_id: (Optional) Give the model state id from the database.
            This will be used in the coco detection json-file's name.

    Returns:
        The AP@05 value of the model on the video
    """
    log.info("Getting the ground truth for dataset %s ...", dataset_name)
    ground_truth_json = get_dvb_ground_truth_coco_json_path(video_name=dataset_name)
    log.info("Ground truth json is: %s", ground_truth_json)
    if not ground_truth_json.exists():
        coco_evaluation_utilities.create_coco_ground_truth_json(
            dataset_name=dataset_name,
            data_category=names.DataCategoryNames.testing,
            ground_truth_json=ground_truth_json,
        )
    if model_state_id == -1:
        detection_json = (
            locations.Results.coco_jsons / f"{dataset_name}_{uuid.uuid4()}.json"
        )
    else:
        detection_json = get_dvb_detection_coco_json_path(
            dataset_name=dataset_name, model_state_id=model_state_id
        )
    log.info("Generating the prediction json file in %s ...", str(detection_json))
    if not detection_json.exists():
        generate_coco_predictions_for_model(
            model=eval_model,
            data_loader=datasets.get_dataloader(
                dataset_name=dataset_name,
                data_category_names=[names.DataCategoryNames.testing],
                augment=False,
                get_untransformed_func=datasets.get_untransformed_uncached_function,
            ),
            coco_prediction_file=detection_json,
        )
    log.info("Computing the COCO quality values ...")
    return coco_evaluation_utilities.evaluate_model_on_ground_truth(
        detection_json=detection_json, ground_truth_json=ground_truth_json
    )


def create_iamges_from_coco_result_file(
    dataset_name: str,
    prediction_json_paths: list[Path],
    image_path: Path,
    model_colors: list[str],
    legend_names: list[str],
):
    """
    Create a video from the results of a drone prediction run.

    Args:
        video_name_stem:
            Name of the video to illustrate the prediction and ground truth in.
        video_target_path:
            Location for the video to be stored in.
        image_path:
            Path for the images
        frame_rate: The frame rate of the video (optional). Default is 60 FPS.
        prediction_json_path:
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
    image_predictions_per_model = []
    for prediction_json_path in prediction_json_paths:
        log.debug("Loading json file %s ...", prediction_json_path.name)
        with open(prediction_json_path, "r", encoding="utf-8") as json_file:
            results = json.load(json_file)

        log.debug("Creating image folder %s ...", image_path)
        if not image_path.exists():
            image_path.mkdir(parents=True)

        log.debug("Retrieving key info from the database for %s ...", dataset_name)
        dataset_id = dbm.get_dataset_id_for_dataset_name(dataset_name=dataset_name)
        image_ids = dbm.get_image_ids_for_dataset_id(dataset_id=dataset_id)

        log.debug("Sorting the predictions to image_ids for %s ... ", dataset_name)
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
        image_predictions_per_model.append(predictions_for_images)

    log.debug("Creating the images for %s ...", dataset_name)
    image_utilities.create_images_with_prediction_and_ground_truth(
        image_ids=image_ids,
        image_predictions_per_model=image_predictions_per_model,
        target_folder=image_path,
        legend_names=legend_names,
        model_colors=model_colors,
    )

    if False:
        log.debug("Creating the video from the images ...")
        width, height = dbm.get_video_shape_for_video_id(video_id=video_id)
        if not video_target_path.parent.exists():
            video_target_path.parent.mkdir(parents=True)
        video_utilities.create_video_from_images_in_folder(
            image_folder=image_path,
            width=width,
            height=height,
            video_target_path=video_target_path,
            frame_rate=frame_rate,
        )


def get_dvb_detection_coco_json_path(dataset_name: str, model_state_id: int) -> Path:
    return (
        locations.Results.coco_jsons
        / "dvb"
        / f"{model_state_id}"
        / f"box_nms_{settings.BOX_NMS_THRESH:0.1f}"
        / f"{dataset_name}.json"
    )


def get_dvb_ground_truth_coco_json_path(video_name: str) -> Path:
    return locations.DroneVsBird.coco_ground_truths / f"{video_name}.json"


def create_images_worker(
    dataset_name: str,
    model_state_ids: list[int],
    model_colors: list[str],
    legend_names: list[str],
):
    # Establish target folder name
    if len(model_state_ids) > 1:
        model_folder_name = (
            f"{"_".join([str(model_id) for model_id in model_state_ids])}"
        )
    else:
        model_folder_name = f"{model_state_ids[0]}"

    # Construct expected prediction paths
    prediction_json_paths = [
        get_dvb_detection_coco_json_path(
            dataset_name=dataset_name, model_state_id=model_state_id
        )
        for model_state_id in model_state_ids
    ]

    # Run image creation
    create_iamges_from_coco_result_file(
        dataset_name=dataset_name,
        prediction_json_paths=prediction_json_paths,
        model_colors=model_colors,
        legend_names=legend_names,
        image_path=locations.Results.coco_images
        / "dvb"
        / model_folder_name
        / f"box_nms_{settings.BOX_NMS_THRESH}"
        / f"{dataset_name}",
    )
    print(f"Video {dataset_name} completed.")


def create_videos_with_annotations(
    dataset_names: Sequence[str],
    model_state_ids: list[int],
    model_colors: list[str],
    legend_names: list[str],
):
    # Check if the predictions exist:
    for model_id in model_state_ids:
        for dataset_name in dataset_names:
            prediction_json_path = get_dvb_detection_coco_json_path(
                dataset_name=dataset_name, model_state_id=model_id
            )
            if not prediction_json_path.exists():
                evaluate_on_dataset(
                    dataset_name, models.restore_model_state(model_id), model_id
                )
    with mp.Pool(processes=mp.cpu_count()) as pool:
        # `map` preserves order; `imap_unordered` yields results as they finish
        pool.starmap(
            create_images_worker,
            (
                (dataset_name, model_state_ids, model_colors, legend_names)
                for dataset_name in dataset_names
            ),
        )


def infer_one_image(image_path: Path, model_state_id: int):

    image = PIL.Image.open(image_path, mode="r").convert("RGB")
    transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    input_image_tensor = transforms(image).to(settings.DEVICE)

    restored_model = models.restore_model_state(model_id=model_state_id)
    restored_model.eval()
    with torch.no_grad():
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


def create_ap05_for_dataset(
    model_to_check: torch.nn.Module,
    dataset_names: Sequence[str],
    model_state_id: int = -1,
) -> dict[str, float]:
    ap05_for_videos = {}
    for dataset in dataset_names:
        ap05_for_videos[dataset] = evaluate_on_dataset(
            dataset_name=dataset,
            eval_model=model_to_check,
            model_state_id=model_state_id,
        )
    return ap05_for_videos


def create_ap05_json_for_dataset(model_state_id: int, dataset_names: Sequence[str]):
    check_model = models.restore_model_state(model_state_id)
    ap05_results = create_ap05_for_dataset(
        model_to_check=check_model,
        dataset_names=dataset_names,
        model_state_id=model_state_id,
    )
    with open(
        locations.Results.ap05_jsons
        / f"model_state_{model_state_id}_nms_{settings.BOX_NMS_THRESH:0.1f}.json",
        mode="w",
        encoding="utf-8",
    ) as f:
        json.dump(ap05_results, fp=f, indent=True)


if __name__ == "__main__":

    # create_ap05_json_for_dataset(
    #     model_state_id=ModelRegister.additional_data,
    #     dataset_names=names.DroneVsBirdVideos.video_names,
    # )

    # ap05_result = create_ap05_for_dataset(
    #     dataset_names=[names.DatasetNames.drone_vs_bird],
    #     model_state_id=ModelRegister.parrot_data,
    #     model_to_check=models.restore_model_state(ModelRegister.parrot_data),
    # )
    # with open(
    #     locations.Results.ap05_jsons
    #     / f"model_state_{ModelRegister.parrot_data}_nms_{settings.BOX_NMS_THRESH:0.1f}_single_value.json",
    #     mode="w",
    #     encoding="utf-8",
    # ) as f:
    #     json.dump(ap05_result, fp=f, indent=True)
    #
    # ap05s = {}
    # for i, model_state in enumerate(MODEL_STATES):
    #     evaluation_model = models.restore_model_state(model_id=model_state)
    #     print(f"Evaluating model {model_state} ({i + 1} / {len(MODEL_STATES)})")
    #     ap05 = evaluate_on_dataset(
    #         dataset_name=names.DatasetNames.drone_vs_bird,
    #         eval_model=evaluation_model,
    #         model_state_id=model_state,
    #     )
    #     ap05s[model_state] = ap05
    # with open(
    #     locations.Results.ap05_jsons
    #     / f"model_state_{model_state}_single_value_only.json",
    #     mode="w",
    #     encoding="utf-8",
    # ) as f:
    #     json.dump(ap05s, fp=f, indent=True)
    # print(ap05s)

    create_videos_with_annotations(
        dataset_names=["trees-background"],
        model_state_ids=[
            # ModelRegiste.starting_point,
            # ModelRegister.baseline,
            ModelRegister.additional_data,
        ],
        legend_names=[
            "Additional Data"
            # "Baseline"
            # "Starting Point"
        ],  # ,"Starting Point", "Baseline", "Additinal Data"],
        model_colors=[
            "DeepPink",
            # "DarkOrange",
            # "DodgerBlue",
            # "GreenYellow",
            # "SkyBlue"
        ],
    )
