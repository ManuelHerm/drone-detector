"""Module provides the PyTorch datasets."""

import random
from typing import Callable, Sequence, TypeVar

import torch as th
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2

from source.config import locations, names, settings
from source.data.cache import get_cache
from source.db import database_manager as dbm
from source.utils import image_utilities
from source.utils.logger import logging

T = TypeVar("T")
random.seed(918)
cranfield_cache = get_cache(locations.Cache.cranfield)
drone_vs_bird_cache = get_cache(locations.Cache.drone_vs_bird)

log = logging.getLogger(__name__)
log.setLevel(settings.LOG_LEVEL)

# pylint: disable=no-value-for-parameter
#         Disabled, because the dbm-function receive the
#         cursor parameter from the decorator.


def random_subset(seq: Sequence[T]) -> list[T]:
    """
    Return a random subset.

    The empty list is chosen with the same probability as any other
    possible size (0 … len(seq)).

    Args:
        seq: The sequence to choose from.

    Returns:
        A list with either one element from seq or an empty list.
    """
    max_len = len(seq)
    # Choose a size uniformly from 0 … max_len
    k = random.randint(0, max_len)
    # Randomly pick *k* distinct elements
    return random.sample(seq, k)


def batch_to_tuple(batch):
    """
    From Cranfield source code.
    """
    return tuple(zip(*batch))


def get_untransformed_uncached_function(
    image_id: int,
) -> tuple[tv_tensors.Image, dict[str, tv_tensors.TVTensor]]:
    """
    Get an image and target tensor unmodified.

    The return values represent the data in the database exactly.

    Args:
        image_id: The image_id of the image to retrieve from the database.

    Returns:
        The image as a tv_tensor and the target in the form

            {"boxes": tv_tensor.BoundingBox, "labels": torch.tensor, "image_id": int}.
    """
    image_blob = dbm.get_image_for_id(image_id)
    image_tensor = image_utilities.image_blob_to_tensor(image_blob, scale=True)
    try:
        annotations = dbm.get_annotations_for_image_id(image_id)
        boxes = tv_tensors.BoundingBoxes(
            th.tensor([[a.x_min, a.y_min, a.x_max, a.y_max] for a in annotations]),
            format="XYXY",
            canvas_size=(image_tensor.shape[-2], image_tensor.shape[-1]),
        )
        labels = th.tensor([a.label_id for a in annotations])
    except IndexError:
        # There are no annotations for the image
        boxes = tv_tensors.BoundingBoxes(
            th.zeros((0, 4), dtype=th.float32),
            format="XYXY",
            canvas_size=(image_tensor.shape[-2], image_tensor.shape[-1]),
        )
        labels = th.zeros((0,), dtype=th.int64)

    target = {"boxes": boxes, "labels": labels, "image_id": th.tensor(image_id)}
    return image_tensor, target


# Using the caching decorator for the Cranfield dataset specifically.
@cranfield_cache.memoize(typed=True)
def get_untransformed_cranfield(
    image_id: int,
) -> tuple[tv_tensors.Image, dict[str, tv_tensors.TVTensor]]:
    return get_untransformed_uncached_function(image_id)


class DroneDataset(Dataset):
    """Class to provide the drone datasets."""

    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-positional-arguments
    #         These arguments are necessary.
    def __init__(
        self,
        dataset_name: str,
        data_category_name: str,
        augment: bool,
        normalization_data_id: int,
        get_untransformed_function: Callable[
            [int], tuple[tv_tensors.Image, dict[str, tv_tensors.TVTensor]]
        ],
    ):
        """
        Load the images from the database.

        Args:
            dataset_name: Name of the dataset.
            data_category_name: Name of the data category
                (like "training" or "validation").
            augment: Whether to augment the data using transforms.
            get_untransformed_function: Function that retrieves untransformed
                samples from the database.
        """
        self.data_category_id = dbm.get_data_category_id_for_name(
            name=data_category_name
        )
        self.dataset_id = dbm.get_dataset_id_for_dataset_name(dataset_name=dataset_name)
        self.image_ids = dbm.get_image_ids_for_data_category_and_dataset_id(
            dataset_id=self.dataset_id, data_category_id=self.data_category_id
        )
        normalization_data = dbm.get_normalization_data_for_id(
            normalization_data_id=normalization_data_id
        )
        self.augment = augment
        self.mean = normalization_data.mean
        self.std = normalization_data.std
        self.get_untransformed = get_untransformed_function

    def __len__(self):
        if settings.HAS_DATASET_LENGTH_LIMIT:
            return min(settings.LIM_DATASET_LENGTH, len(self.image_ids))
        return len(self.image_ids)

    def __getitem__(self, index):
        image, target = self.get_untransformed(self.image_ids[index])
        chosen_transforms = []
        if self.augment:
            shorter_side_length = min(image.cpu().detach().shape[1:])
            channel_averages = image.mean(dim=(1, 2)).cpu().detach().tolist()
            transforms_geometry = tuple(
                [
                    v2.RandomResize(min_size=600, max_size=shorter_side_length),
                    v2.RandomCrop(size=(600, 600)),
                    v2.RandomZoomOut(fill=channel_averages),
                    v2.RandomPerspective(fill=channel_averages, distortion_scale=0.5),
                ]
            )
            chosen_transforms.extend(random_subset(transforms_geometry))
            transforms_gauss = tuple(
                [
                    v2.GaussianBlur(kernel_size=5, sigma=(0.4, 0.6)),
                    v2.GaussianNoise(sigma=0.1),
                ]
            )
            chosen_transforms.extend(random_subset(transforms_gauss))
            transforms_color = tuple([v2.RandomPhotometricDistort()])
            chosen_transforms.extend(random_subset(transforms_color))
            transforms_common = tuple([v2.RandomHorizontalFlip()])
            chosen_transforms.extend(transforms_common)
        transforms_finish = tuple(
            [v2.Normalize(mean=self.mean, std=self.std), v2.SanitizeBoundingBoxes()]
        )
        chosen_transforms.extend(transforms_finish)
        transforms = v2.Compose(chosen_transforms)
        return transforms(image, target)


def get_drone_vs_bird_dataloader(normalization_data_id: int):
    """
    Initialize the entire Drone vs. Bird DroneDataset and generate dataloader.

    Uses the training data-category.
    Considers settings from the configuration module.

    Args:
        normalization_data_id: The id of the normalization data that
            shall be used for normalization.

    Returns:
        Drone versus Bird testing dataloader.
    """
    dataset = DroneDataset(
        dataset_name=names.DatasetNames.drone_vs_bird,
        data_category_name=names.DataCategoryNames.testing,
        augment=False,
        normalization_data_id=normalization_data_id,
        # Cache was gigantic for drone versus bird. Therefore,
        # the uncached version here.
        get_untransformed_function=get_untransformed_uncached_function,
    )
    return th.utils.data.DataLoader(
        dataset,
        batch_size=settings.BATCH_SIZE_TEST,
        num_workers=settings.NUM_WORKERS_TEST,
        collate_fn=batch_to_tuple,
        pin_memory=True,
    )


def get_cranfield_default_dataloader_training() -> DataLoader:
    """
    Initialize the CranfieldDataset for training and generate dataloader.

    Considers settings from the configuration module.

    Returns:
        Cranfield training dataloader.
    """
    dataset = DroneDataset(
        dataset_name=names.DatasetNames.cranfield_default,
        data_category_name=names.DataCategoryNames.training,
        augment=True,
        normalization_data_id=dbm.get_normalization_data_id_for_dataset_id(
            dbm.get_dataset_id_for_dataset_name(names.DatasetNames.cranfield_default)
        ),
        get_untransformed_function=get_untransformed_cranfield,
    )
    return th.utils.data.DataLoader(
        dataset,
        batch_size=settings.BATCH_SIZE_TRAIN,
        num_workers=settings.NUM_WORKERS_TRAIN,
        collate_fn=batch_to_tuple,
        pin_memory=True,
    )


def get_dataloader(
    dataset_name: str,
    data_category_name: str,
    augment: bool,
    get_untransformed_func: Callable[
        [int], tuple[tv_tensors.Image, dict[str, tv_tensors.TVTensor]]
    ],
) -> DataLoader:
    """
    Initialize the dataset and generate dataloader.

    Considers settings from the configuration module.

    Returns:
        Dataloader
    """
    dataset = DroneDataset(
        dataset_name=dataset_name,
        data_category_name=data_category_name,
        augment=augment,
        normalization_data_id=dbm.get_normalization_data_id_for_dataset_id(
            dbm.get_dataset_id_for_dataset_name(dataset_name)
        ),
        get_untransformed_function=get_untransformed_func,
    )
    return th.utils.data.DataLoader(
        dataset,
        batch_size=settings.BATCH_SIZE_TRAIN,
        num_workers=settings.NUM_WORKERS_TRAIN,
        collate_fn=batch_to_tuple,
        pin_memory=True,
    )


def get_cranfield_default_dataloader_validation(
    normalization_data_id: int,
) -> DataLoader:
    """
    Initialize the CranfieldDataset for validation and generate dataloader.

    Considers settings from the configuration module.

    Args:
        normalization_data_id: The id of the normalization data that
            shall be used for normalization.

    Returns:
        Cranfield validation dataloader.
    """
    dataset = DroneDataset(
        dataset_name=names.DatasetNames.cranfield_default,
        data_category_name=names.DataCategoryNames.validation,
        augment=False,
        normalization_data_id=normalization_data_id,
        get_untransformed_function=get_untransformed_cranfield,
    )
    return th.utils.data.DataLoader(
        dataset,
        batch_size=settings.BATCH_SIZE_TEST,
        num_workers=settings.NUM_WORKERS_TEST,
        collate_fn=batch_to_tuple,
        pin_memory=True,
    )


def get_drone_vs_bird_single_video_dataloader(
    video_name: str, normalization_data_id: int
) -> DataLoader:
    """
    Initialize a Drone vs. Bird dataset for testing and generate dataloader.

    Considers settings from the configuration module for:

    - Batch size
    - Number of workers

    Args:
        video_name: Video name of the video that is the basis
            for the dataset.
        normalization_data_id: ID of the normalization data that was used for
            training the model under test.

    Returns:
        Drone versus Bird dataloader.
    """
    dataset = DroneDataset(
        dataset_name=video_name,
        data_category_name=names.DataCategoryNames.testing,
        augment=False,
        normalization_data_id=normalization_data_id,
        get_untransformed_function=get_untransformed_uncached_function,
    )
    return th.utils.data.DataLoader(
        dataset,
        batch_size=settings.BATCH_SIZE_TEST,
        num_workers=settings.NUM_WORKERS_TEST,
        collate_fn=batch_to_tuple,
        pin_memory=True,
    )


if __name__ == "__main__":
    pass
