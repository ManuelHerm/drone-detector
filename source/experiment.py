import io
import json
import multiprocessing as mp
import random
from dataclasses import asdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import pycocotools.coco
import torch
import torchvision
import tqdm
from PIL import ImageDraw
from torchvision import tv_tensors
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import v2

from source import datasets, main, models
from source.config import locations, names, settings
from source.data import datatypes
from source.db import database_manager as dbm
from source.utils import annotation_utilites, coco_evaluation_utilities
from source.utils import image_utilities
from source.utils import image_utilities as image_utils
from source.utils import video_utilities
from source.utils.logger import logging

log = logging.getLogger(__name__)
log.setLevel(settings.LOG_LEVEL)

random.seed(918)

mpl.rcParams["figure.dpi"] = 600
plt.style.use("seaborn-v0_8")

# pylint: skip-file
# Is just a testing ground file.


def show_tensor(tensor: torch.Tensor):
    img = torchvision.transforms.ToPILImage()(tensor)
    img.show()


def show_tensor_and_boxes(
    tensor: torch.Tensor,
    boxes_ground_truth: torch.Tensor,
    boxes_prediction: torch.Tensor = None,
):
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
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    # For training
    images, boxes = torch.rand(4, 3, 600, 1200), torch.rand(4, 11, 4)
    boxes[:, :, 2:4] = boxes[:, :, 0:2] + boxes[:, :, 2:4]
    labels = torch.randint(1, 91, (4, 11))
    images = list(image for image in images)
    targets = []
    for i in range(len(images)):
        d = {}
        d["boxes"] = boxes[i]
        d["labels"] = labels[i]
        targets.append(d)
    model.train()
    output = model(images, targets)
    print(output)
    # For inference
    model.eval()
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    predictions = model(x)


def check_model(
    model: torch.nn.Module,
    check_images: tv_tensors.TVTensor,
    label: tv_tensors.BoundingBoxes,
):
    with torch.no_grad():
        model.eval()
        check_images = list(
            check_image.to(device=settings.DEVICE) for check_image in check_images
        )
        predictions = model(check_images)
        normalization_data = dbm.get_normalization_data_for_dataset_id(1)
        mean = torch.tensor(normalization_data.mean).view(-1, 1, 1)
        std = torch.tensor(normalization_data.std).view(-1, 1, 1)
        if settings.DEVICE != "cpu":
            check_images = list(
                check_image.to(device="cpu") for check_image in check_images
            )
            predictions = [
                {key: value.to(device="cpu") for key, value in prediction.items()}
                for prediction in predictions
            ]
        for single_image, single_label, single_result in zip(
            check_images, label, predictions
        ):
            img_original = single_image * std + mean
            show_tensor_and_boxes(
                img_original, single_label["boxes"], single_result["boxes"]
            )
            pass


def create_dataset_video_worker(
    dataset_name: str,
    image_path: Path,
    video_path: Path,
    width: int,
    height: int,
    fps: int,
):
    log.info("Creating images for dataset %s", dataset_name)
    image_utils.draw_bounding_boxes_for_dataset(
        dataset_name=dataset_name, target_folder=image_path
    )
    log.info("Creating video for dataset %s", dataset_name)
    # video_utilities.create_video_from_images_in_folder(
    #     image_folder=image_path,
    #     video_target_path=video_path,
    #     width=width,
    #     height=height,
    #     frame_rate=fps,
    # )
    # log.info(f"Video for dataset %s completed.", dataset_name)


def control_dataset_annotations():

    datasets = {
        # "forrest_hut_phantom": names.DatasetNames.forrest_hut_phantom,
        # "forrest_hut_mini": names.DatasetNames.forrest_hut_mini,
        "forrest_hut_inspire": names.DatasetNames.forrest_hut_inspire,
        # "campus": names.DatasetNames.campus,
        # "hill": names.DatasetNames.hill,
        # "drone_school_phantom": names.DatasetNames.drone_school_phantom,
        # "drone_school_mini": names.DatasetNames.drone_school_mini,
        # "drone_school_inspire": names.DatasetNames.drone_school_inspire,
        # "drone_school_mid": names.DatasetNames.drone_school_mid,
        # "drone_school_far": names.DatasetNames.drone_school_far,
    }

    image_paths = tuple(
        locations.TEMP_DIR / "images" / name for name in datasets.keys()
    )
    video_paths = tuple(
        locations.TEMP_DIR / "videos" / name for name in datasets.keys()
    )
    for image_path, video_path in zip(image_paths, video_paths):
        if not image_path.exists():
            image_path.mkdir(parents=True)
        if not video_path.exists():
            video_path.mkdir(parents=True)

    with mp.Pool(processes=mp.cpu_count()) as pool:
        # `map` preserves order; `imap_unordered` yields results as they finish
        pool.starmap(
            create_dataset_video_worker,
            (
                (dataset_name, image_path, video_path, 1980, 1080, 30)
                for dataset_name, image_path, video_path in zip(
                    datasets.values(), image_paths, video_paths
                )
            ),
        )


def get_dvb_annotation_short_side_length() -> list[float]:
    image_ids = dbm.get_image_ids_for_data_origin_name(
        data_origin_name=names.DataOriginNames.drone_vs_bird
    )
    annotation_short_sides = []
    pytorch_min_size = 1080
    pytorch_max_size = 1920
    for image_id in tqdm.tqdm(image_ids):
        # Getting the image scaling factor
        width, height = dbm.get_image_width_and_height_for_image_id(image_id=image_id)
        shortest_side = min(width, height)
        longest_side = max(width, height)
        scale = pytorch_min_size / shortest_side
        if scale * longest_side > pytorch_max_size:
            scale = pytorch_max_size / longest_side
        # Getting the annotations
        try:
            image_annotations = dbm.get_annotations_for_image_id(image_id=image_id)
            for image_annotation in image_annotations:
                annotation_width = (
                    image_annotation.x_max - image_annotation.x_min
                ) * scale
                annotation_height = (
                    image_annotation.y_max - image_annotation.y_min
                ) * scale
                annotation_short_sides.append(min(annotation_height, annotation_width))
        except IndexError:
            continue
    return annotation_short_sides


def get_dvb_drone_size_overview():
    anno = get_dvb_annotation_short_side_length()
    anno_ser = pd.Series(anno)
    anno_ser.to_json("drone_size_larger.json")
    # anno_ser = pd.read_json("drone_size.json", orient="index")
    print(f"{anno_ser.median() = }")
    print(f"{anno_ser.quantile(q=0.1) = }")
    ax = anno_ser.plot.hist(bins=350)
    # ax.set_xlim(0, 50)
    plt.tight_layout()
    plt.savefig("drone_size_larger.png")


@torch.inference_mode()
def rpn_proposal_recall(
    model,  # torchvision FasterRCNN
    data_loader,
    iou_thresh=0.5,
    max_images=None,
):
    model.eval()

    matched = 0
    total = 0

    for idx, (images, targets) in tqdm.tqdm(enumerate(data_loader)):
        if max_images is not None and idx >= max_images:
            break

        # Move images/targets
        images = [img.to(settings.DEVICE) for img in images]
        targets = [{k: v.to(settings.DEVICE) for k, v in t.items()} for t in targets]

        # 1)Apply the model's internal transform (resize/normalize)
        transformed, _ = model.transform(images, targets)

        # 2) Backbone features
        features = model.backbone(transformed.tensors)
        if isinstance(features, torch.Tensor):
            features = {"0": features}  # for non-FPN backbones

        # 3) RPN proposals
        proposals, _ = model.rpn(transformed, features)  # list[Tensor], per image

        # 4) Compare proposals to GT
        for props, t in zip(proposals, targets):
            gt = t["boxes"]
            if gt.numel() == 0:
                continue

            # If RPN returns no proposals (rare), count as 0 recall for those GT boxes
            if props is None or props.numel() == 0:
                total += gt.shape[0]
                continue

            # IoU: (M, N)
            ious = torchvision.ops.box_iou(gt, props)
            best_iou_per_gt = ious.max(dim=1).values
            matched += (best_iou_per_gt >= iou_thresh).sum().item()
            total += gt.shape[0]
    recall = matched / total if total > 0 else 0.0
    return recall


def model_variation_experiment():

    MODEL_STATE = 182

    model_blob = dbm.get_model_state(model_id=MODEL_STATE)
    model_state_dict = torch.load(io.BytesIO(model_blob), map_location=settings.DEVICE)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
        box_score_thresh=settings.BOX_SCORE_THRESH,
        rpn_anchor_generator=torchvision.models.detection.faster_rcnn.AnchorGenerator(
            sizes=((8,), (16,), (32,), (64,), (128,)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5,
        ),
        rpn_fg_iou_thresh=0.7,
        rpn_bg_iou_thresh=0.3,
        min_size=1080,
        max_size=1920,
        rpn_pre_nms_top_n_test=4000,
        rpn_post_nms_top_n_test=2000,
        rpn_nms_thresh=0.9,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = (
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, settings.NUM_CLASSES
        )
    )
    model = model.to(settings.DEVICE)
    model.load_state_dict(model_state_dict)

    data_loader = datasets.get_dataloader(
        dataset_name="2019_09_02_GOPR5871_1058_solo",
        data_category_names=[names.DataCategoryNames.testing],
        augment=False,
        get_untransformed_func=datasets.get_untransformed_uncached_function,
    )
    recall = rpn_proposal_recall(
        model=model,
        data_loader=data_loader,
        iou_thresh=0.7,
    )
    print(recall)


if __name__ == "__main__":

    annotation_utilites.seg_mask_to_bbox(
        mask_path=locations.DATA_DIR / "current" / "mask-experiment" / "sample_mask.png"
    )
