"""This module provides functionality around AI-models."""

import io

import torch
import torchvision as tv
from torchvision.models.detection import FasterRCNN, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import AnchorGenerator, FastRCNNPredictor

from source.config import settings
from source.db import database_manager as dbm

# pylint: disable=no-value-for-parameter
#         Disabled, because the dbm-function receive the
#         cursor parameter from the decorator.


def get_faster_r_cnn_model(num_classes, constructor_kwargs) -> FasterRCNN:
    """
    Get a Faster R-CNN model for fine-tuning.

    It has ResNet50_FPN weights and a new model head.

    Args:
        num_classes: The number of classes that can be detected.
        constructor_kwargs: Settings for the FasterRCNN constructor
            as a dictionary

    Returns:
        The Faster R-CNN model on the configuration.DEVICE.
    """
    faster_r_cnn_model = tv.models.detection.fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
        box_score_thresh=settings.BOX_SCORE_THRESH,
        box_nms_thresh=settings.BOX_NMS_THRESH,
        **constructor_kwargs
    )
    in_features = faster_r_cnn_model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    faster_r_cnn_model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_classes
    )
    return faster_r_cnn_model.to(settings.DEVICE)


@torch.no_grad()
def save_torch_state(precious) -> bytes:
    if precious is None:
        dict_to_save = {}
    else:
        dict_to_save = precious.state_dict()
    with io.BytesIO() as buffer:
        torch.save(dict_to_save, buffer)
        return buffer.getvalue()


def restore_model_state(model_id: int) -> torch.nn.Module:
    model_state_dict = torch.load(
        io.BytesIO(dbm.get_model_state(model_id=model_id)),
        map_location=settings.DEVICE,
    )
    model_settings_dict = dbm.get_model_setting(model_id=model_id)
    model = get_faster_r_cnn_model(settings.NUM_CLASSES, model_settings_dict)
    model.load_state_dict(model_state_dict)
    model.box_score_thresh = (settings.BOX_SCORE_THRESH,)
    model.box_nms_thresh = (settings.BOX_NMS_THRESH,)
    return model


if __name__ == "__main__":
    pass
