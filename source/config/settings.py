"""This module allows setting the program-wide configuration."""

import logging

import torch as th

LOG_LEVEL = logging.DEBUG

DEVICE = "cuda" if th.cuda.is_available() else "cpu"

# Data Quality Criteria
BBOX_MIN_AREA = 9  # Pixel

# Model Parameters
BOX_SCORE_THRESH = (
    0.1  # The model needs to be this sure it is a drone during inference.
)
NUM_CLASSES = 2  # Drone and Not-Drone

# Training Hyperparameters
LEARNING_RATE = 2e-05

# Parameters for training
NUM_WORKERS_TRAIN = 5
BATCH_SIZE_TRAIN = 10
EPOCHS = 200

# Parameters for Inference
BATCH_SIZE_TEST = 20
NUM_WORKERS_TEST = 4

# Debug-Section
HAS_DATASET_LENGTH_LIMIT = False
LIM_DATASET_LENGTH = -1
