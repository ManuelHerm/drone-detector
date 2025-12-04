"""
This module facilitates the model training.
"""

from io import BytesIO

import torch as th
import tqdm

from source import datasets, models
from source.config import names, settings
from source.data import datatypes as dt
from source.db import database_manager as dbm
from source.utils.convergance_monitor import ConvergenceMonitor
from source.utils.logger import logging

log = logging.getLogger(__name__)
log.setLevel(settings.LOG_LEVEL)

# pylint: disable=no-value-for-parameter
#         Disabled, because the dbm-function receive the
#         cursor parameter from the decorator.


def train_one_epoch(training_model, data_loader, optim, lr_scheduler, conv_monitor):
    training_model.train()
    for images, targets in tqdm.tqdm(data_loader):
        # Moving input to the right device:
        images = list(image.to(device=settings.DEVICE) for image in images)
        targets = [
            {key: value.to(device=settings.DEVICE) for key, value in target.items()}
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
        if settings.DEVICE == "cuda":
            th.cuda.empty_cache()


def validate_one_epoch(validation_model, data_loader, conv_monitor):
    with th.no_grad():
        validation_model.train()
        for images, targets in tqdm.tqdm(data_loader):
            # Moving input to the right device:
            images = list(image.to(device=settings.DEVICE) for image in images)
            targets = [
                {key: value.to(device=settings.DEVICE) for key, value in target.items()}
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
    log.info("Training on %s", settings.DEVICE)
    model = models.get_faster_r_cnn_model(num_classes=settings.NUM_CLASSES)
    training_loader = datasets.get_cranfield_default_dataloader_training()
    validation_loader = datasets.get_cranfield_default_dataloader_validation(
        normalization_data_id=dbm.get_normalization_data_for_dataset_id(
            dbm.get_dataset_id_for_dataset_name(names.DatasetNames.cranfield_default)
        )
    )
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = th.optim.SGD(
        params, lr=settings.LEARNING_RATE, momentum=0.9, weight_decay=0.0005
    )
    learning_rate_scheduler = th.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0 / 1000.0,
        total_iters=min(1000, len(training_loader) - 1),
    )
    convergence_monitor = ConvergenceMonitor()
    for epoch in range(settings.EPOCHS):
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
    log.info("Training on %s", settings.DEVICE)
    model = models.get_faster_r_cnn_model(num_classes=settings.NUM_CLASSES)
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
        params, lr=settings.LEARNING_RATE, momentum=0.9, weight_decay=0.0005
    )
    learning_rate_scheduler = th.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0 / 1000.0,
        total_iters=min(1000, len(training_loader) - 1),
    )
    convergence_monitor = ConvergenceMonitor()
    for epoch in range(settings.EPOCHS):
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


def continue_to_train_on_cranfield_combined(model_state_id: int):
    log.info("Training on %s", settings.DEVICE)

    # Restoring model from database
    model = models.get_faster_r_cnn_model(num_classes=settings.NUM_CLASSES)
    state_dict = th.load(
        f=BytesIO(dbm.get_model_state(model_state_id=model_state_id)),
        map_location=settings.DEVICE,
    )
    model.load_state_dict(state_dict)

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
        params, lr=settings.LEARNING_RATE, momentum=0.9, weight_decay=0.0005
    )
    optim_state_dict = th.load(
        f=BytesIO(dbm.get_optimizer_state(model_state_id=model_state_id)),
        map_location=settings.DEVICE,
    )
    optimizer.load_state_dict(optim_state_dict)

    learning_rate_scheduler = th.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0 / 1000.0,
        total_iters=min(1000, len(training_loader) - 1),
    )
    lr_scheduler_state_dict = th.load(
        f=dbm.get_learning_rate_scheduler_state(state_id=model_state_id),
        map_location=settings.DEVICE,
    )
    learning_rate_scheduler.load_state_dict(lr_scheduler_state_dict)

    convergence_monitor = ConvergenceMonitor(
        number_of_samples=dbm.get_samples_trained(model_state_id=model_state_id)
    )

    epochs_trained = dbm.get_epochs_trained(model_state_id=model_state_id)

    for epoch in range(epochs_trained, settings.EPOCHS):
        log.info("Epoch number: %s", epoch)
        train_one_epoch(
            model,
            training_loader,
            optimizer,
            learning_rate_scheduler,
            convergence_monitor,
        )
        epochs_trained = epoch + 1
        if epochs_trained % 5 == 0:
            models.save_progress(
                model_to_save=model,
                optimizer_to_save=optimizer,
                lr_scheduler_to_save=learning_rate_scheduler,
                epochs_trained=epochs_trained,
                dataset_id=training_loader.dataset.dataset_id,
                samples_trained=convergence_monitor.number_of_samples,
            )
        convergence_monitor.add_training_point_per_epoch()
        validate_one_epoch(model, validation_loader, convergence_monitor)
        convergence_monitor.add_validation_point_per_epoch()


if __name__ == "__main__":
    continue_to_train_on_cranfield_combined(model_state_id=77)
