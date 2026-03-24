"""Module provides convergence monitoring."""

import dataclasses
import datetime
from dataclasses import asdict

import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from source.config import locations
from source.data import datatypes as dt

# pylint: disable=missing-function-docstring


class ConvergenceMonitor:
    """Monitors convergence for Faster R-CNN models."""

    def __init__(self, number_of_samples: int = 0):
        self.number_of_samples: int = number_of_samples
        self.train_losses: pd.DataFrame = self._init_loss_df()
        self.val_losses: pd.DataFrame = self._init_loss_df()
        self.writer = self._init_tensorboard_writer()

    def add_training_point_per_batch(
        self, scalars: dt.FasterRCNNLoss, samples_trained_in_batch: int
    ):
        self.number_of_samples += samples_trained_in_batch
        self.train_losses.loc[len(self.train_losses)] = asdict(scalars)

    def add_validation_point_per_batch(self, scalars: dt.FasterRCNNLoss):
        self.val_losses.loc[len(self.val_losses)] = asdict(scalars)

    def add_training_point_per_epoch(self):
        scalars = self.train_losses.mean().to_dict()
        self.train_losses = self._init_loss_df()
        self.writer.add_scalars("training", scalars, global_step=self.number_of_samples)

    def add_validation_point_per_epoch(self):
        scalars = self.val_losses.mean().to_dict()
        self.val_losses = self._init_loss_df()
        self.writer.add_scalars(
            "validation", scalars, global_step=self.number_of_samples
        )

    def add_dvb_ap05_per_epoch(self, dvb_ap05: dict[str, float]):
        self.writer.add_scalars(
            "drone_vs_bird", dvb_ap05, global_step=self.number_of_samples
        )

    @staticmethod
    def _init_loss_df():
        return pd.DataFrame(
            columns=tuple(field.name for field in dataclasses.fields(dt.FasterRCNNLoss))
        )

    @staticmethod
    def _init_tensorboard_writer() -> SummaryWriter:
        log_dir = locations.Logs.logs / str(
            datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
        )
        return SummaryWriter(log_dir=str(log_dir.resolve()))
