import json
import random
from dataclasses import asdict
from typing import Any

import pycocotools.coco
import torch
import torchvision
import tqdm
from PIL import ImageDraw
from torchvision import tv_tensors
from torchvision.models.detection import (FasterRCNN,
                                          FasterRCNN_ResNet50_FPN_Weights)
from torchvision.transforms import v2

import annotation_utilites as annotation_utils
import coco_evaluation_utilities
import configuration
import database_manager as dbm
import dataset_utilities as dataset_utils
import datasets
import datatypes as dt
import image_utilities as image_utils
import locations
import models
import names as n

random.seed(918)


def show_tensor(tensor: torch.Tensor):
    img = torchvision.transforms.ToPILImage()(tensor)
    img.show()


def show_tensor_and_boxes(tensor: torch.Tensor, boxes_ground_truth: torch.Tensor, boxes_prediction: torch.Tensor = None):
    img = torchvision.transforms.ToPILImage()(tensor)
    boxes_list = boxes_ground_truth.cpu().detach().tolist()
    drw = ImageDraw.Draw(img, "RGB")
    for box in boxes_list:
        box = [(box[0], box[1]), (box[2], box[3])]
        drw.rectangle(xy=box, fill=None, outline="blue", width=1)
    if boxes_prediction is not None:
        predictions_list = boxes_prediction.cpu().detach().tolist()
        for box in predictions_list:
            box = [(box[0], box[1]), (box[2], box[3])]
            drw.rectangle(xy=box, fill=None, outline="orange", width=1)
    img.show()


def checkout_resize(img_t: torch.Tensor, target_t: dict[str, tv_tensors.TVTensor]):
    shorter_side_length = min(img_t.cpu().shape[1:])
    transform = v2.RandomResize(min_size=600, max_size=shorter_side_length)
    img_t, target_t = transform(img_t, target_t)
    show_tensor_and_boxes(img_t, target_t["boxes"])


def checkout_horizontal_flip(
    img_t: torch.Tensor, target_t: dict[str, tv_tensors.TVTensor]
):
    transform = v2.RandomHorizontalFlip(p=1.0)
    img_t, target_t = transform(img_t, target_t)
    show_tensor_and_boxes(img_t, target_t["boxes"])


def checkout_crop(img_t: torch.Tensor, target_t: dict[str, tv_tensors.TVTensor]):
    transform = v2.Compose([v2.RandomCrop(size=(600, 600)), v2.SanitizeBoundingBoxes()])
    img_t, target_t = transform(img_t, target_t)
    show_tensor_and_boxes(img_t, target_t["boxes"])


def checkout_zoom_out(img_t: torch.Tensor, target_t: dict[str, tv_tensors.TVTensor]):
    channel_average = img_t.mean(dim=(1, 2))
    transform = v2.RandomZoomOut(fill=channel_average.tolist(), p=1.0)
    img_t, target_t = transform(img_t, target_t)
    show_tensor_and_boxes(img_t, target_t["boxes"])


def checkout_photometric_distort(
    img_t: torch.Tensor, target_t: dict[str, tv_tensors.TVTensor]
):
    transform = v2.RandomPhotometricDistort(p=1.0)
    img_t, target_t = transform(img_t, target_t)
    show_tensor_and_boxes(img_t, target_t["boxes"])


def checkout_random_perspective(
    img_t: torch.Tensor, target_t: dict[str, tv_tensors.TVTensor]
):
    channel_average = img_t.mean(dim=(1, 2)).cpu().tolist()
    transform = v2.RandomPerspective(p=1.0, fill=channel_average, distortion_scale=0.5)
    img_t, target_t = transform(img_t, target_t)
    show_tensor_and_boxes(img_t, target_t["boxes"])


def checkout_gaussian_blur(
    img_t: torch.Tensor, target_t: dict[str, tv_tensors.TVTensor]
):
    transform = v2.GaussianBlur(kernel_size=5, sigma=(0.4, 0.6))
    img_t, target_t = transform(img_t, target_t)
    show_tensor_and_boxes(img_t, target_t["boxes"])


def checkout_gaussian_noise(
    img_t: torch.Tensor, target_t: dict[str, tv_tensors.TVTensor]
):
    transform = v2.GaussianNoise(sigma=0.1)
    img_t, target_t = transform(img_t, target_t)
    show_tensor_and_boxes(img_t, target_t["boxes"])


def random_subset(seq):
    """
    Return a random subset of *seq*.
    The empty list is chosen with the same probability as any other
    possible size (0 … len(seq)).
    """
    max_len = len(seq)
    # Choose a size uniformly from 0 … max_len
    k = random.randint(0, max_len)
    # Randomly pick *k* distinct elements
    return random.sample(seq, k)


def checkout_pipeline(img_t: torch.Tensor, target_t: dict[str, tv_tensors.TVTensor]):
    shorter_side_length = min(img_t.cpu().detach().shape[1:])
    channel_averages = img_t.mean(dim=(1, 2)).cpu().detach().tolist()
    chosen_transforms = []
    transforms_geometry = tuple(
        [
            v2.RandomResize(min_size=600, max_size=shorter_side_length),
            v2.Compose([v2.RandomCrop(size=(600, 600)), v2.SanitizeBoundingBoxes()]),
            v2.RandomZoomOut(fill=channel_averages),
            v2.RandomPerspective(fill=channel_averages, distortion_scale=0.5),
        ]
    )
    transforms_gauss = tuple(
        [v2.GaussianBlur(kernel_size=5, sigma=(0.4, 0.6)), v2.GaussianNoise(sigma=0.1)]
    )
    transforms_color = tuple([v2.RandomPhotometricDistort()])
    transforms_common = tuple([v2.RandomHorizontalFlip()])
    chosen_transforms.extend(random_subset(transforms_geometry))
    chosen_transforms.extend(random_subset(transforms_gauss))
    chosen_transforms.extend(random_subset(transforms_color))
    chosen_transforms.extend(transforms_common)
    transforms = v2.Compose(chosen_transforms)
    img_t, target_t = transforms(img_t, target_t)
    show_tensor_and_boxes(img_t, target_t["boxes"])


def pytorch_example_fasterrcnn():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    # For training
    images, boxes = torch.rand(4, 3, 600, 1200), torch.rand(4, 11, 4)
    boxes[:, :, 2:4] = boxes[:, :, 0:2] + boxes[:, :, 2:4]
    labels = torch.randint(1, 91, (4, 11))
    images = list(image for image in images)
    targets = []
    for i in range(len(images)):
        d = {}
        d['boxes'] = boxes[i]
        d['labels'] = labels[i]
        targets.append(d)
    model.train()
    output = model(images, targets)
    print(output)
    # For inference
    model.eval()
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    predictions = model(x)


def check_model(model: torch.nn.Module, check_images: tv_tensors.TVTensor, label: tv_tensors.BoundingBoxes):
    with torch.no_grad():
        model.eval()
        check_images = list(check_image.to(device=configuration.DEVICE) for check_image in check_images)
        predictions = model(check_images)
        normalization_data = dbm.get_normalization_data_for_dataset_id(1)
        mean = torch.tensor(normalization_data.mean).view(-1, 1, 1)
        std = torch.tensor(normalization_data.std).view(-1, 1, 1)
        if configuration.DEVICE != "cpu":
            check_images = list(check_image.to(device="cpu") for check_image in check_images)
            predictions = [
                {
                    key: value.to(device="cpu")
                    for key, value in prediction.items()
                }
                for prediction in predictions
            ]
        for single_image, single_label, single_result in zip(check_images, label, predictions):
            img_original = single_image * std + mean
            show_tensor_and_boxes(img_original, single_label["boxes"], single_result["boxes"])
            pass


if __name__ == "__main__":
    # restored_model = models.restore_model_state(63)
    # val_loader = datasets.get_cranfield_dataloader_validation(
    #     normalization_data_id=dbm.get_normalization_data_for_dataset_id(
    #         dataset_id=dbm.get_dataset_id_for_model_state_id(63)
    #     )
    # )
    # for image, target in val_loader:
    #     check_model(restored_model, image, target)
    #     break
    # cranfield_val_coco = coco_evaluation_utilities.get_coco_dataset(dataset_name=n.DatasetNames.cranfield_default, data_category=n.DataCategoryNames.training)
    # json.dump(asdict(cranfield_val_coco), open("cranfield_train_coco.json", "w"))
    # coco = pycocotools.coco.COCO("cranfield_train_coco.json")
    # image = dbm.get_image_for_id(125228)
    # image = image_utils.image_blob_to_image(image)
    # image.show()

    # restored_model = models.restore_model_state(63)
    # drone_vs_bird_dataloader = datasets.get_drone_vs_bird_dataset_testing()
    # for image, target in drone_vs_bird_dataloader:
    #     check_model(restored_model, image, target)
    #     break

    pass

