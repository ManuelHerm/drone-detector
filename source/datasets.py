"""Module provides the PyTorch datasets."""

import random
from pathlib import Path
from typing import Callable, Sequence, TypeVar

import torch
import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.functional import to_pil_image

from source.config import locations, names, settings
from source.data import datatypes as dt
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
            torch.tensor([[a.x_min, a.y_min, a.x_max, a.y_max] for a in annotations]),
            format="XYXY",
            canvas_size=(image_tensor.shape[-2], image_tensor.shape[-1]),
        )
        labels = torch.tensor([a.label_id for a in annotations])
    except IndexError:
        # There are no annotations for the image
        boxes = tv_tensors.BoundingBoxes(
            torch.zeros((0, 4), dtype=torch.float32),
            format="XYXY",
            canvas_size=(image_tensor.shape[-2], image_tensor.shape[-1]),
        )
        labels = torch.zeros((0,), dtype=torch.int64)

    target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor(image_id)}
    return image_tensor, target


# Using the caching decorator for the Cranfield dataset specifically.
@cranfield_cache.memoize(typed=True)
def get_untransformed_cranfield(
    image_id: int,
) -> tuple[tv_tensors.Image, dict[str, tv_tensors.TVTensor]]:
    return get_untransformed_uncached_function(image_id)


class DroneDataset(Dataset):
    """Class to provide the drone datasets."""

    def __init__(
        self,
        dataset_name: str,
        data_category_names: list[str],
        augment: bool,
        get_untransformed_function: Callable[
            [int], tuple[tv_tensors.Image, dict[str, tv_tensors.TVTensor]]
        ],
    ):
        """
        Load the images from the database.

        Args:
            dataset_name: Name of the dataset.
            data_category_names: Names of the data categories
                (like "training" or "validation").
            augment: Whether to augment the data using transforms.
            get_untransformed_function: Function that retrieves untransformed
                samples from the database.
        """
        self.data_category_ids = [
            dbm.get_data_category_id_for_name(name=name) for name in data_category_names
        ]

        self.dataset_id = dbm.get_dataset_id_for_dataset_name(dataset_name=dataset_name)

        self.image_ids = []

        for data_category_id in self.data_category_ids:
            self.image_ids.extend(
                dbm.get_image_ids_for_data_category_and_dataset_id(
                    dataset_id=self.dataset_id, data_category_id=data_category_id
                )
            )

        self.augment = augment

        self._aug = v2.Compose(
            [
                v2.RandomApply([v2.GaussianNoise(mean=0.0, sigma=0.05)]),
                v2.RandomPerspective(fill=(0.485, 0.456, 0.406), distortion_scale=0.3),
                v2.RandomApply([v2.RandomCrop((800, 800))], p=0.2),
                v2.RandomApply([v2.RandomPhotometricDistort()], p=0.3),
                v2.RandomHorizontalFlip(p=0.5),  # already random
                v2.SanitizeBoundingBoxes(min_size=settings.BBOX_MIN_SIZE),
            ]
        )
        self._noaug = v2.Compose(
            [
                v2.SanitizeBoundingBoxes(min_size=settings.BBOX_MIN_SIZE),
            ]
        )

        self.get_untransformed = get_untransformed_function

    def __len__(self):
        if settings.HAS_DATASET_LENGTH_LIMIT:
            return min(settings.LIM_DATASET_LENGTH, len(self.image_ids))
        return len(self.image_ids)

    def __getitem__(self, index):
        image, target = self.get_untransformed(self.image_ids[index])
        return (self._aug if self.augment else self._noaug)(image, target)


def get_dataloader(
    dataset_name: str,
    data_category_names: list[str],
    augment: bool,
    get_untransformed_func: Callable[
        [int], tuple[tv_tensors.Image, dict[str, tv_tensors.TVTensor]]
    ] = get_untransformed_uncached_function,
) -> DataLoader:
    """
    Initialize the dataset and generate dataloader.

    Considers settings from the configuration module.

    Returns:
        Dataloader
    """
    dataset = DroneDataset(
        dataset_name=dataset_name,
        data_category_names=data_category_names,
        augment=augment,
        get_untransformed_function=get_untransformed_func,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=settings.BATCH_SIZE_TRAIN,
        num_workers=settings.NUM_WORKERS_TRAIN,
        collate_fn=batch_to_tuple,
        pin_memory=True,
        shuffle=True,
    )


@torch.inference_mode()
def save_training_sample(
    image: tv_tensors.Image,
    bounding_boxes: tv_tensors.BoundingBoxes,
    image_id: torch.Tensor,
    target_path: Path,
):
    pil_image = to_pil_image(image)
    image_id_int = int(image_id.detach().to("cpu").item())
    annotations = []
    for box in bounding_boxes:
        annotation = dt.Annotation(
            image_id=image_id_int,
            label_id=1,
            x_min=int(box[0].detach().to("cpu").item()),
            y_min=int(box[1].detach().to("cpu").item()),
            x_max=int(box[2].detach().to("cpu").item()),
            y_max=int(box[3].detach().to("cpu").item()),
            id=-1,
        )
        annotations.append(annotation)
    sample_image = image_utilities.add_bounding_boxes_to_image(
        img=pil_image,
        bboxes=annotations,
        legend_entry=dt.ImageLegendEntry(xy=(10, 10), text=f"{image_id_int = }"),
    )
    sample_image.save(target_path)


if __name__ == "__main__":
    DATASET_NAME = names.DatasetNames.optimization_3
    dl = get_dataloader(
        dataset_name=DATASET_NAME,
        data_category_names=[names.DataCategoryNames.training],
        augment=True,
    )
    for i, (images, targets) in tqdm.tqdm(enumerate(dl)):
        for j, (image, target) in enumerate(zip(images, targets)):
            save_training_sample(
                image=image,
                bounding_boxes=target["boxes"],
                image_id=target["image_id"],
                target_path=locations.TEMP_DIR / "dataset_control" / f"{i}_{j}.png",
            )
