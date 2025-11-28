"""Module provides convergence monitoring."""

import dataclasses
import datetime
from dataclasses import asdict

import pandas as pd
from torch.utils.tensorboard import SummaryWriter

import datatypes as dt
import locations


class ConvergenceMonitor:
    """Monitors convergence for Faster R-CNN models."""

    def __init__(self):
        self.number_of_samples: int = 0
        self.train_losses: pd.DataFrame = self._init_loss_df()
        self.val_losses: pd.DataFrame = self._init_loss_df()
        self.train_writer, self.val_writer = self._init_tensorboard_writers()

    def add_training_point_per_batch(
        self, scalars: dt.FasterRCNNLoss, samples_trained_in_batch: int
    ):
        self.number_of_samples += samples_trained_in_batch
        self._add_training_losses(scalars)

    def add_validation_point_per_batch(self, scalars: dt.FasterRCNNLoss):
        self._add_validation_losses(scalars)

    def _add_training_losses(self, scalars: dt.FasterRCNNLoss):
        self.train_losses.loc[len(self.train_losses)] = asdict(scalars)

    def _add_validation_losses(self, scalars: dt.FasterRCNNLoss):
        self.val_losses.loc[len(self.val_losses)] = asdict(scalars)

    def add_training_point_per_epoch(self):
        scalars = self.train_losses.mean().to_dict()
        self.train_losses = self._init_loss_df()
        self.train_writer.add_scalars(
            "epoch/training", scalars, global_step=self.number_of_samples
        )

    def add_validation_point_per_epoch(self):
        scalars = self.val_losses.mean().to_dict()
        self.val_losses = self._init_loss_df()
        self.val_writer.add_scalars(
            "epoch/validation", scalars, global_step=self.number_of_samples
        )

    @staticmethod
    def _init_loss_df():
        return pd.DataFrame(
            columns=tuple(field.name for field in dataclasses.fields(dt.FasterRCNNLoss))
        )

    @staticmethod
    def _init_tensorboard_writers() -> tuple[SummaryWriter, SummaryWriter]:
        training_log_dir = locations.Logs.training / str(
            datetime.datetime.now().strftime("%Y-%m-%d_%H:%M_%S")
        )
        val_log_dir = locations.Logs.validation / str(
            datetime.datetime.now().strftime("%Y-%m-%d_%H:%M_%S")
        )
        return (
            SummaryWriter(log_dir=str(training_log_dir.resolve())),
            SummaryWriter(log_dir=str(val_log_dir.resolve())),
        )
