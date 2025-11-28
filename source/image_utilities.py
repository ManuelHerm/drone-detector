"""This module provides functionality around images."""

import io
import shutil
from io import BytesIO
from pathlib import Path

import PIL
import PIL.Image
import PIL.ImageDraw
import torch as th
import torchvision.tv_tensors
import tqdm
from torchvision.transforms import v2

import database_manager as dbm
import datatypes as dt
import locations as loca
from datatypes import Annotation

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
    # Encode to JPEG in memory
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
    annotations = dbm.get_annotations_for_image_id(image_id)
    image = add_bounding_boxes_to_image(image, annotations)
    image.show()


def add_bounding_boxes_to_image(
    img: PIL.Image.Image,
    bboxes: list[Annotation],
    color: str = "blue",
    legend_entry: dt.ImageLegendEntry = None,
) -> PIL.Image.Image:
    drw = PIL.ImageDraw.Draw(img, "RGB")
    if legend_entry:
        drw.text(xy=legend_entry.xy, fill=color, text=legend_entry.text)
    for bbox in bboxes:
        box = [(bbox.x_min, bbox.y_min), (bbox.x_max, bbox.y_max)]
        drw.rectangle(xy=box, fill=None, outline=color, width=2)
        if bbox.score:
            # bbox is in upper left quadrant
            if bbox.x_min <= img.width // 2 and bbox.y_min <= img.height // 2:
                x = bbox.x_max + 5
                y = bbox.y_max
                anchor = "lb"
            # bbox is in upper right quadrant
            elif bbox.x_min > img.width // 2 and bbox.y_min <= img.height // 2:
                x = bbox.x_min
                y = bbox.y_max + 5
                anchor = "lt"
            # bbox is in lower left quadrant
            elif bbox.x_min <= img.width // 2 and bbox.y_min > img.height // 2:
                x = bbox.x_max + 5
                y = bbox.y_min
                anchor = "lt"
            # bbox is in lower right quadrant
            else:
                x = bbox.x_min
                y = bbox.y_min - 5
                anchor = "lb"
            drw.text(xy=(x, y), fill=color, text=f"{bbox.score:0.2f}", anchor=anchor)

    return img


def add_broken_bounding_boxes_to_image(
    img: PIL.Image.Image, broken_bboxes: list[Annotation]
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
    predictions_for_images: dict[int, list[Annotation]],
    target_folder: Path,
):
    for image_id in tqdm.tqdm(image_ids):
        # Adding the predictions to the image
        image = image_blob_to_image(dbm.get_image_for_id(image_id=image_id))
        if image_id in predictions_for_images:
            image = add_bounding_boxes_to_image(
                img=image,
                bboxes=predictions_for_images[image_id],
                color="blue",
                legend_entry=dt.ImageLegendEntry(xy=(5, 5), text="Prediction"),
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
                legend_entry=dt.ImageLegendEntry(xy=(5, 20), text="Ground Truth"),
                color="green",
            )
        image.save(target_folder / f"{image_id}.png")


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
        scale: Whether to scale the image. Scaling brings the values into the range [0 ... 1].

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
            image.save(loca.CACHE_DIR / f"{image_id}.jpg")
        else:
            image.save(loca.CACHE_DIR / f"{video_name}_{image_id}.jpg")


def move_images_to_folders(path: Path):
    folder_names = set()
    for image in tqdm.tqdm(sorted(path.iterdir())):
        name_components = str(image.stem).split("_")
        folder_name = "_".join(name_components[0:-1])
        if folder_name not in folder_names:
            folder_names.add(folder_name)
            (loca.CACHE_DIR / folder_name).mkdir()
        shutil.move(src=image, dst=loca.CACHE_DIR / folder_name)


def turn_all_videos_to_images_with_bounding_boxes():
    videos = dbm.get_videos_for_origin("Drone versus Bird")
    for video in tqdm.tqdm(videos):
        draw_bounding_boxes_for_video(video.id, Path(video.name).stem)


if __name__ == "__main__":
    add_bounding_boxes_to_certain_image(107108)
