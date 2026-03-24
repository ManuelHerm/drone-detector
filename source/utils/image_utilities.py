"""This module provides functionality around images."""

import io
import shutil
from io import BytesIO
from pathlib import Path

import PIL
import PIL.Image
import PIL.ImageColor
import PIL.ImageDraw
import torch as th
import torchvision.tv_tensors
import tqdm
from torchvision.transforms import v2

from source.config import locations, settings
from source.data import datatypes as dt
from source.db import database_manager as dbm

# pylint: disable=no-value-for-parameter
#         Disabled, because the dbm-function receive the
#         cursor parameter from the decorator.
# pylint: disable=no-member
#         Disabled, because pylint does not find the
#         cv2 functions.


def create_image_from_path(path: Path, data_origin: str) -> dt.Image:
    data_origin_id = dbm.get_data_origin_id_for_name(data_origin)
    image = PIL.Image.open(path, mode="r").convert("RGB")
    width, height = image.size
    with io.BytesIO() as buffer:
        image.save(buffer, format="PNG")
        png_bytes = buffer.getvalue()
    return dt.Image(
        name=path.name,
        data_origin_id=data_origin_id,
        width=width,
        height=height,
        image=png_bytes,
    )


def add_bounding_boxes_to_certain_image(image_id: int):
    image_blob = dbm.get_image_for_id(image_id)
    image = image_blob_to_image(image_blob)
    try:
        annotations = dbm.get_annotations_for_image_id(image_id)
    except IndexError:
        annotations = []
    image = add_bounding_boxes_to_image(image, annotations)
    image.show()


def add_bounding_boxes_to_image(
    img: PIL.Image.Image,
    bboxes: list[dt.Annotation],
    color: str = "blue",
    legend_entry: dt.ImageLegendEntry = None,
    offset: int = 0,
) -> PIL.Image.Image:

    base = img.convert("RGBA")
    overlay = PIL.Image.new("RGBA", base.size, (0, 0, 0, 0))
    drw = PIL.ImageDraw.Draw(overlay, "RGBA")
    rgb = PIL.ImageColor.getrgb(color)

    if legend_entry:
        drw.text(
            xy=legend_entry.xy,
            fill=(*rgb, 255),
            text=legend_entry.text,
            font_size=base.size[1] / settings.FONT_SIZE_DIVISOR,
        )

    for bbox in bboxes:
        box = [(bbox.x_min, bbox.y_min), (bbox.x_max, bbox.y_max)]

        score = 1.0 if bbox.score is None else float(bbox.score)
        if settings.BBOX_TRANSPARENCY:
            opacity = max(0.05, min(1.0, score))
        else:
            opacity = 1.0
        alpha = int(round(255 * opacity))

        drw.rectangle(
            xy=box, fill=None, outline=(*rgb, alpha), width=settings.BBOX_LINE_WIDTH
        )

        if bbox.score is not None:
            # bbox is in upper left quadrant
            if bbox.x_min <= base.width // 2 and bbox.y_min <= base.height // 2:
                x = bbox.x_max + 5 + 25 * offset
                y = bbox.y_max
                anchor = "lb"
            # bbox is in upper right quadrant
            elif bbox.x_min > base.width // 2 and bbox.y_min <= base.height // 2:
                x = bbox.x_min
                y = bbox.y_max + 5 + 10 * offset
                anchor = "lt"
            # bbox is in lower left quadrant
            elif bbox.x_min <= base.width // 2 and bbox.y_min > base.height // 2:
                x = bbox.x_max + 5 + 25 * offset
                y = bbox.y_min
                anchor = "lt"
            # bbox is in lower right quadrant
            else:
                x = bbox.x_min
                y = bbox.y_min - 5 - 10 * offset
                anchor = "lb"
            drw.text(
                xy=(x, y), fill=(*rgb, alpha), text=f"{bbox.score:0.2f}", anchor=anchor
            )

    return PIL.Image.alpha_composite(base, overlay).convert(img.mode)


def add_broken_bounding_boxes_to_image(
    img: PIL.Image.Image, broken_bboxes: list[dt.Annotation]
) -> PIL.Image.Image:
    drw = PIL.ImageDraw.Draw(img, "RGB")
    for broken_bbox in broken_bboxes:
        box = [
            (broken_bbox.x_min - 10, broken_bbox.y_min - 10),
            (broken_bbox.x_max + 10, broken_bbox.y_max + 10),
        ]
        drw.rectangle(xy=box, fill=None, outline="pink", width=3)
    return img


def create_images_with_prediction_and_ground_truth(
    image_ids: list[int],
    image_predictions_per_model: list[dict[int, list[dt.Annotation]]],
    legend_names: list[str],
    model_colors: list[str],
    target_folder: Path,
):
    for image_id in image_ids:
        # Adding the predictions to the image
        image = image_blob_to_image(dbm.get_image_for_id(image_id=image_id))
        for i, (predictions_for_images, model_color, legend_name) in enumerate(
            zip(image_predictions_per_model, model_colors, legend_names)
        ):
            if image_id in predictions_for_images:
                image = add_bounding_boxes_to_image(
                    img=image,
                    bboxes=predictions_for_images[image_id],
                    color=model_color,
                    legend_entry=dt.ImageLegendEntry(
                        xy=(5, i * image.size[1] / settings.FONT_SIZE_DIVISOR * 1.5),
                        text=legend_name,
                    ),
                    offset=i,
                )
        # Adding the ground truth to the image
        try:
            ground_truth = dbm.get_annotations_for_image_id(image_id)
        except IndexError:
            ground_truth = None
        if ground_truth:
            image = add_bounding_boxes_to_image(
                img=image,
                bboxes=ground_truth,
                legend_entry=dt.ImageLegendEntry(
                    xy=(
                        5,
                        len(legend_names)
                        * image.size[1]
                        / settings.FONT_SIZE_DIVISOR
                        * 1.5,
                    ),
                    text="Ground Truth",
                ),
                color="DeepPink",
            )
        image.save(
            target_folder / f"{image_id}.{settings.IMAGE_SAVING_FORMAT}",
            **settings.IMAGE_SAVING_KWARGS,
        )


def image_blob_to_image(img_blob: bytes) -> PIL.Image.Image:
    """
    Turn database image blob into an image.

    Args:
        img_blob: The bytes from the database.

            They were saved there using::

                cv2.imencode(".png", frame).tobytes()
    """
    return PIL.Image.open(BytesIO(img_blob))


def image_blob_to_tensor(img_blob: bytes, scale: bool) -> torchvision.tv_tensors.Image:
    """
    Turn a database image blob into an Image tensor.

    Args:
        img_blob: The image blob from the database.
        scale: Whether to scale the image.
            Scaling brings the values into the range [0 ... 1].

    Returns:
        Image as a tensor
    """
    image_pil = image_blob_to_image(img_blob)
    transforms = v2.Compose([v2.ToImage(), v2.ToDtype(th.float32, scale=scale)])
    return transforms(image_pil)


def draw_bounding_boxes_for_video(video_id: int, video_name: str = "None"):
    image_ids = dbm.get_image_ids_for_video_id(video_id)
    for image_id in image_ids:
        image = dbm.get_image_for_id(image_id)
        image = image_blob_to_image(image)
        try:
            annotations = dbm.get_annotations_for_image_id(image_id)
            image = add_bounding_boxes_to_image(image, annotations)
        # If no annotation for this image exists, just skip
        # drawing one.
        except IndexError:
            pass
        if video_name == "None":
            image.save(locations.CACHE_DIR / f"{image_id}.jpg")
        else:
            image.save(locations.CACHE_DIR / f"{video_name}_{image_id}.jpg")


def draw_bounding_boxes_for_dataset(dataset_name: str, target_folder: Path):
    dataset_id = dbm.get_dataset_id_for_dataset_name(dataset_name=dataset_name)
    image_ids = dbm.get_image_ids_for_dataset_id(dataset_id=dataset_id)
    for image_id in image_ids:
        image = dbm.get_image_for_id(image_id)
        image = image_blob_to_image(image)
        try:
            annotations = dbm.get_annotations_for_image_id(image_id)
            image = add_bounding_boxes_to_image(image, annotations)
        # If no annotation for this image exists, just skip
        # drawing one.
        except IndexError:
            pass
        image.save(target_folder / f"{image_id}.png")


def move_images_to_folders(path: Path):
    folder_names = set()
    for image in tqdm.tqdm(sorted(path.iterdir())):
        name_components = str(image.stem).split("_")
        folder_name = "_".join(name_components[0:-1])
        if folder_name not in folder_names:
            folder_names.add(folder_name)
            (locations.CACHE_DIR / folder_name).mkdir()
        shutil.move(src=image, dst=locations.CACHE_DIR / folder_name)


def turn_all_videos_to_images_with_bounding_boxes():
    videos = dbm.get_videos_for_origin("Drone versus Bird")
    for video in tqdm.tqdm(videos):
        draw_bounding_boxes_for_video(video.id, Path(video.name).stem)


if __name__ == "__main__":
    add_bounding_boxes_to_certain_image(109128)
