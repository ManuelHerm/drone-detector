"""This module provides functionality around AI-models."""

import io

import torch as th
import torchvision as tv
from torchvision.models.detection import FasterRCNN, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from source.config import settings
from source.db import database_manager as dbm

# pylint: disable=no-value-for-parameter
#         Disabled, because the dbm-function receive the
#         cursor parameter from the decorator.


def get_faster_r_cnn_model(num_classes) -> FasterRCNN:
    """
    Get a Faster R-CNN model for fine-tuning.

    It has ResNet50_FPN weights and a new model head.

    Args:
        num_classes: The number of classes that can be detected.

    Returns:
        The Faster R-CNN model on the configuration.DEVICE.
    """
    faster_r_cnn_model = tv.models.detection.fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
        box_score_thresh=settings.BOX_SCORE_THRESH,
    )
    in_features = faster_r_cnn_model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    faster_r_cnn_model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_classes
    )
    return faster_r_cnn_model.to(settings.DEVICE)


@th.no_grad()
def save_torch_state(precious) -> bytes:
    with io.BytesIO() as buffer:
        th.save(precious.state_dict(), buffer)
        return buffer.getvalue()


def save_progress(
    model_to_save,
    optimizer_to_save: th.optim.Optimizer,
    lr_scheduler_to_save: th.optim.lr_scheduler.LRScheduler,
    epochs_trained: int,
    dataset_id: int,
    samples_trained: int,
):
    dbm.insert_model_state(
        dataset_id=dataset_id,
        model_type=model_to_save.__class__.__name__,
        optimizer_type=optimizer_to_save.__class__.__name__,
        lr_scheduler_type=lr_scheduler_to_save.__class__.__name__,
        model_state=save_torch_state(model_to_save),
        optimizer_state=save_torch_state(optimizer_to_save),
        lr_scheduler_state=save_torch_state(lr_scheduler_to_save),
        epochs_trained=epochs_trained,
        samples_trained=samples_trained,
    )


def restore_model_state(model_state_id: int) -> th.nn.Module:
    model_blob = dbm.get_model_state(model_state_id)
    model_state_dict = th.load(io.BytesIO(model_blob), map_location=settings.DEVICE)
    model = get_faster_r_cnn_model(settings.NUM_CLASSES)
    model.load_state_dict(model_state_dict)
    return model
