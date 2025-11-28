"""
This module facilitates the model training.
"""

import torch as th
import tqdm

import configuration
import database_manager as dbm
import datasets
import datatypes as dt
import models
import names
from convergance_monitor import ConvergenceMonitor
from logger import logging

log = logging.getLogger(__name__)
log.setLevel(configuration.LOG_LEVEL)

# pylint: disable=no-value-for-parameter
#         Disabled, because the dbm-function receive the
#         cursor parameter from the decorator.


def train_one_epoch(training_model, data_loader, optim, lr_scheduler, conv_monitor):
    training_model.train()
    for images, targets in tqdm.tqdm(data_loader):
        # Moving input to the right device:
        images = list(image.to(device=configuration.DEVICE) for image in images)
        targets = [
            {
                key: value.to(device=configuration.DEVICE)
                for key, value in target.items()
            }
            for target in targets
        ]
        # Computing the loss
        loss_dict = training_model(images, targets)
        losses = sum(individual_loss for individual_loss in loss_dict.values())
        # Resetting the gradients
        optim.zero_grad()
        # Backpropagation
        losses.backward()
        # Applying the changes
        optim.step()
        # Updating the learning rates
        lr_scheduler.step()
        # Monitoring
        with th.no_grad():
            conv_monitor.add_training_point_per_batch(
                dt.FasterRCNNLoss(
                    box_reg=loss_dict["loss_box_reg"].detach().cpu().item(),
                    classifier=loss_dict["loss_classifier"].detach().cpu().item(),
                    objectness=loss_dict["loss_objectness"].detach().cpu().item(),
                    rpn_box_reg=loss_dict["loss_rpn_box_reg"].detach().cpu().item(),
                ),
                len(images),
            )
        # Emptying the memory for the next cycle
        del images
        del targets
        del loss_dict
        del losses
        if configuration.DEVICE == "cuda":
            th.cuda.empty_cache()


def validate_one_epoch(validation_model, data_loader, conv_monitor):
    with th.no_grad():
        validation_model.train()
        for images, targets in tqdm.tqdm(data_loader):
            # Moving input to the right device:
            images = list(image.to(device=configuration.DEVICE) for image in images)
            targets = [
                {
                    key: value.to(device=configuration.DEVICE)
                    for key, value in target.items()
                }
                for target in targets
            ]
            # Computing the loss
            loss_dict = validation_model(images, targets)
            conv_monitor.add_validation_point_per_batch(
                dt.FasterRCNNLoss(
                    box_reg=loss_dict["loss_box_reg"].detach().cpu().item(),
                    classifier=loss_dict["loss_classifier"].detach().cpu().item(),
                    objectness=loss_dict["loss_objectness"].detach().cpu().item(),
                    rpn_box_reg=loss_dict["loss_rpn_box_reg"].detach().cpu().item(),
                )
            )


def train_on_cranfield_default():
    log.info("Training on %s", configuration.DEVICE)
    model = models.get_faster_r_cnn_model(num_classes=configuration.NUM_CLASSES)
    training_loader = datasets.get_cranfield_default_dataloader_training()
    validation_loader = datasets.get_cranfield_default_dataloader_validation(
        normalization_data_id=dbm.get_normalization_data_for_dataset_id(
            dbm.get_dataset_id_for_dataset_name(names.DatasetNames.cranfield_default)
        )
    )
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = th.optim.SGD(
        params, lr=configuration.LEARNING_RATE, momentum=0.9, weight_decay=0.0005
    )
    learning_rate_scheduler = th.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0 / 1000.0,
        total_iters=min(1000, len(training_loader) - 1),
    )
    convergence_monitor = ConvergenceMonitor()
    for epoch in range(configuration.EPOCHS):
        log.info("Epoch number: %s", epoch)
        train_one_epoch(
            model,
            training_loader,
            optimizer,
            learning_rate_scheduler,
            convergence_monitor,
        )
        epochs_trained = epoch + 1
        if epochs_trained % 10 == 0:
            models.save_progress(
                model_to_save=model,
                optimizer_to_save=optimizer,
                lr_scheduler_to_save=learning_rate_scheduler,
                epochs_trained=epochs_trained,
                dataset_id=training_loader.dataset.dataset_id,
            )
        convergence_monitor.add_training_point_per_epoch()
        validate_one_epoch(model, validation_loader, convergence_monitor)
        convergence_monitor.add_validation_point_per_epoch()


def train_on_cranfield_combined():
    log.info("Training on %s", configuration.DEVICE)
    model = models.get_faster_r_cnn_model(num_classes=configuration.NUM_CLASSES)
    training_loader = datasets.get_dataloader(
        dataset_name=names.DatasetNames.cranfield_combined,
        data_category_name=names.DataCategoryNames.training,
        augment=True,
        get_untransformed_func=datasets.get_untransformed_uncached_function,
    )
    validation_loader = datasets.get_dataloader(
        dataset_name=names.DatasetNames.cranfield_combined,
        data_category_name=names.DataCategoryNames.validation,
        augment=False,
        get_untransformed_func=datasets.get_untransformed_uncached_function,
    )
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = th.optim.SGD(
        params, lr=configuration.LEARNING_RATE, momentum=0.9, weight_decay=0.0005
    )
    learning_rate_scheduler = th.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0 / 1000.0,
        total_iters=min(1000, len(training_loader) - 1),
    )
    convergence_monitor = ConvergenceMonitor()
    for epoch in range(configuration.EPOCHS):
        log.info("Epoch number: %s", epoch)
        train_one_epoch(
            model,
            training_loader,
            optimizer,
            learning_rate_scheduler,
            convergence_monitor,
        )
        epochs_trained = epoch + 1
        if epochs_trained % 10 == 0:
            models.save_progress(
                model_to_save=model,
                optimizer_to_save=optimizer,
                lr_scheduler_to_save=learning_rate_scheduler,
                epochs_trained=epochs_trained,
                dataset_id=training_loader.dataset.dataset_id,
            )
        convergence_monitor.add_training_point_per_epoch()
        validate_one_epoch(model, validation_loader, convergence_monitor)
        convergence_monitor.add_validation_point_per_epoch()


if __name__ == "__main__":
    train_on_cranfield_combined()
