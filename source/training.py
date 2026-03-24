"""
This module facilitates the model training.
"""

from io import BytesIO

import torch
import tqdm

from source import datasets, models, quality_test
from source.config import names, settings
from source.data import datatypes as dt
from source.db import database_manager as dbm
from source.utils import dataset_utilities
from source.utils.convergance_monitor import ConvergenceMonitor
from source.utils.logger import logging

log = logging.getLogger(__name__)
log.setLevel(settings.LOG_LEVEL)

# pylint: disable=no-value-for-parameter
#         Disabled, because the dbm-function receive the
#         cursor parameter from the decorator.


class Trainer:

    def __init__(self, dataset_name: str, from_model_state: int = -1):

        # Dataloader
        # ----------

        self.training_loader = datasets.get_dataloader(
            dataset_name=dataset_name,
            data_category_names=[
                names.DataCategoryNames.training,
                names.DataCategoryNames.validation,
            ],
            augment=True,
            get_untransformed_func=datasets.get_untransformed_uncached_function,
        )
        self.dataset_id = dbm.get_dataset_id_for_dataset_name(dataset_name=dataset_name)

        # Model
        # -----
        if from_model_state != -1:
            self.model = models.restore_model_state(from_model_state)
            self.model_state = from_model_state
        else:
            self.model = models.get_faster_r_cnn_model(
                settings.NUM_CLASSES, settings.MODEL_KWARGS
            )

        # Optimizer Setup
        # ---------------

        # Parameter splitting
        backbone_params = []
        head_params = []

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if "backbone" in name:
                if settings.TRAIN_BACKBONE:
                    backbone_params.append(p)
                else:
                    p.requires_grad = False
            else:
                head_params.append(p)

        # Learning rate assignment for ...
        # 1) Head parameters:
        optimizer_learning_rates = [
            {"params": head_params, "lr": settings.BASE_LR_HEADS}
        ]
        # 2) Backbone parameters:
        if settings.TRAIN_BACKBONE:
            optimizer_learning_rates.append(
                {"params": backbone_params, "lr": settings.BASE_LR_BACKBONE}
            )

        # Optimizer
        self.optimizer = torch.optim.SGD(
            optimizer_learning_rates,
            momentum=settings.OPTIMIZIER_MOMENTUM,
            weight_decay=settings.OPTIMIZIER_WEIGHT_DECAY,
            nesterov=settings.OPTIMIZER_NESTEROV,
        )
        if from_model_state != -1:
            optim_state_dict = torch.load(
                f=BytesIO(dbm.get_optimizer_state(model_state_id=from_model_state)),
                map_location=settings.DEVICE,
            )
            self.optimizer.load_state_dict(optim_state_dict)

        # Tracking
        # --------

        if from_model_state != -1:
            self.batches_trained, self.epochs_trained, self.samples_trained = (
                dbm.get_training_metrics(model_state_id=from_model_state)
            )
        else:
            self.batches_trained = 0
            self.epochs_trained = 0
            self.samples_trained = 0
        self.lr_history = []  # list of (global_step, [lr_group0, lr_group1, ...])

        # Learning rate schedulers
        # ------------------------

        # Warmup
        self.warmup_batches = min(
            settings.WARMUP_BATCHES, len(self.training_loader) - 1
        )
        self.warmup_lr_scheduler = None
        if self.batches_trained < self.warmup_batches:
            self.warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=settings.WARMUP_FACTOR,
                total_iters=self.warmup_batches,
            )
            if from_model_state != -1:
                warmup_state_dict = torch.load(
                    f=BytesIO(dbm.get_lr_warmup_state(state_id=from_model_state)),
                    map_location=settings.DEVICE,
                )
                self.warmup_lr_scheduler.load_state_dict(warmup_state_dict)

        # Main
        self.main_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=settings.MAIN_LR_MILESTONES,
            gamma=settings.MAIN_LR_GAMMA,
        )
        if from_model_state != -1:
            main_lr_state_dict = torch.load(
                f=BytesIO(dbm.get_lr_main_state(state_id=from_model_state)),
                map_location=settings.DEVICE,
            )
            self.main_lr_scheduler.load_state_dict(main_lr_state_dict)

        # Convergence Monitor
        # -------------------

        self.convergence_monitor = ConvergenceMonitor(
            number_of_samples=self.samples_trained
        )

    def _store_lrs(self):
        self.lr_history.append(
            (
                self.batches_trained,
                tuple(float(pg["lr"]) for pg in self.optimizer.param_groups),
            )
        )

    def _print_backbone_in_optimizer(self):
        backbone_param_ids = {id(p) for p in self.model.backbone.parameters()}
        opt_param_ids = {
            id(p) for g in self.optimizer.param_groups for p in g["params"]
        }
        in_opt = len(backbone_param_ids & opt_param_ids) > 0
        log.info(
            "Backbone is in optimizer." if in_opt else "Backbone is not in optimizer."
        )

    def print_learning_rates(self):
        for batch_nr, (lr_1, lr_2) in self.lr_history:
            print(f"{batch_nr:03}: {lr_1:0.3e} {lr_2:0.3e}")

    def train(self) -> int:
        log.info("Training on %s", settings.DEVICE)

        self._print_backbone_in_optimizer()

        for epoch in range(self.epochs_trained, settings.EPOCHS):
            self.model.train()
            log.info("Epoch number: %s", epoch)

            log.info(
                "Learning rates for batch %s: %0.3e, %0.3e",
                self.batches_trained,
                self.optimizer.param_groups[0]["lr"],
                (
                    self.optimizer.param_groups[1]["lr"]
                    if settings.TRAIN_BACKBONE
                    else 0.0
                ),
            )

            for images, targets in tqdm.tqdm(self.training_loader):

                self._store_lrs()

                # Moving input to the right device:
                images = list(image.to(device=settings.DEVICE) for image in images)
                targets = [
                    {
                        key: value.to(device=settings.DEVICE)
                        for key, value in target.items()
                    }
                    for target in targets
                ]

                # Computing the loss
                loss_dict = self.model(images, targets)
                losses = sum(individual_loss for individual_loss in loss_dict.values())

                # Resetting the gradients
                self.optimizer.zero_grad()

                # Backpropagation
                losses.backward()

                # Applying the changes
                self.optimizer.step()

                # Updating the learning rates (warmup)
                if self.warmup_lr_scheduler is not None:
                    if self.batches_trained < self.warmup_batches:
                        self.warmup_lr_scheduler.step()

                # Monitoring
                self.batches_trained += 1
                self.samples_trained += len(images)
                self.convergence_monitor.add_training_point_per_batch(
                    dt.FasterRCNNLoss(
                        box_reg=loss_dict["loss_box_reg"].detach().cpu().item(),
                        classifier=loss_dict["loss_classifier"].detach().cpu().item(),
                        objectness=loss_dict["loss_objectness"].detach().cpu().item(),
                        rpn_box_reg=loss_dict["loss_rpn_box_reg"].detach().cpu().item(),
                    ),
                    samples_trained_in_batch=len(images),
                )

                # Emptying the memory for the next cycle
                del images
                del targets
                del loss_dict
                del losses

            # Updating the learning rates (main)
            self.main_lr_scheduler.step()

            self.epochs_trained += 1

            # Saving the status
            if self.epochs_trained % settings.SAVING_FREQUENCY == 0:
                self.model_state = dbm.insert_model_state(
                    dataset_id=self.dataset_id,
                    model_state=models.save_torch_state(self.model),
                    optimizer_state=models.save_torch_state(self.optimizer),
                    warmup_lr_scheduler_state=models.save_torch_state(
                        self.warmup_lr_scheduler
                    ),
                    main_lr_scheduler_state=models.save_torch_state(
                        self.main_lr_scheduler
                    ),
                    epochs_trained=self.epochs_trained,
                    samples_trained=self.samples_trained,
                    batches_trained=self.batches_trained,
                    model_setting=settings.MODEL_KWARGS,
                )

            # Convergence monitor udpate
            self.convergence_monitor.add_training_point_per_epoch()
            if self.epochs_trained % settings.VALIDATION_FREQUENCY == 0:
                self.convergence_monitor.add_dvb_ap05_per_epoch(
                    quality_test.create_ap05_for_dataset(
                        model_to_check=self.model,
                        dataset_names=names.DroneVsBirdVideos.validation_videos,
                        model_state_id=self.model_state,
                    )
                )
        return self.model_state


if __name__ == "__main__":
    trainer = Trainer(
        dataset_name=names.DatasetNames.optimization_3,
    )
    last_model_state = trainer.train()
    print(f"{last_model_state = }")
    quality_test.create_ap05_json_for_dataset(
        model_state_id=last_model_state,
        dataset_names=names.DroneVsBirdVideos.video_names,
    )
