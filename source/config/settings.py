"""This module allows setting the program-wide configuration."""

import logging

import torch
from torchvision.models.detection.faster_rcnn import AnchorGenerator

LOG_LEVEL = logging.DEBUG

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data Quality Criteria
BBOX_MIN_AREA = 9  # Pixel. Used during data ingestion
BBOX_MIN_SIZE = 4.0  # Pixel. Gets used in SanitizeBoundingBoxes transformation.

# Model Parameters
# ----------------

# Threshold to include a box at inference
BOX_SCORE_THRESH = 0.05
# IoU theshold with better higher confidence
# predictions
BOX_NMS_THRESH = 0.5  # Deafault = 0.5
NUM_CLASSES = 2  # Drone and Not-Drone

# Model parameters for new models
MODEL_KWARGS = {
    "rpn_anchor_generator": AnchorGenerator(
        sizes=((8,), (16,), (32,), (64,), (128,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    ),
    "min_size": 1080,
    "max_size": 1920,
    "trainable_backbone_layers": 0,
}

# Training Hyperparameters
TRAIN_BACKBONE = False
BASE_LR_BACKBONE = 0.0003  # 1e-4
BASE_LR_HEADS = 0.0005  # 5e-4
WARMUP_BATCHES = 800
WARMUP_FACTOR = 1.0 / 100.0
OPTIMIZIER_MOMENTUM = 0.9
OPTIMIZIER_WEIGHT_DECAY = 0.0005
OPTIMIZER_NESTEROV = True  # False
MAIN_LR_MILESTONES = []  # [8]  # [3, 6, 9]
MAIN_LR_GAMMA = 0.1

# Parameters for training
NUM_WORKERS_TRAIN = 3
BATCH_SIZE_TRAIN = 6
EPOCHS = 10
VALIDATION_FREQUENCY = 1
SAVING_FREQUENCY = 1

# Parameters for Inference
BATCH_SIZE_TEST = 20
NUM_WORKERS_TEST = 20

# Debug-Section
HAS_DATASET_LENGTH_LIMIT = False
LIM_DATASET_LENGTH = -10

# Image Formatting
BBOX_TRANSPARENCY = True
BBOX_LINE_WIDTH = 2
FONT_SIZE_DIVISOR = 50
IMAGE_SAVING_FORMAT = "png"
IMAGE_SAVING_KWARGS = {
    # "format": "JPEG",
    # "quality": 95,
}
