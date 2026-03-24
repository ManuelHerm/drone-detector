"""This module provides functionality around videos."""

from __future__ import annotations

from os import PathLike
from pathlib import Path

import cv2
import tqdm

from source.data import datatypes as dt
from source.db import database_manager as dbm

# pylint: disable=no-value-for-parameter
#         Disabled, because the dbm-function receive the
#         cursor parameter from the decorator.
# pylint: disable=no-member
#         Disabled, because pylint does not find the
#         cv2 functions.


def video_to_images(data_origin: str):
    """
    Turn video from the database into images and add them with metadata to the database.

    This means:
        - Adding the image with a fitting name to the images table.
        - Recording the image with its frame number in the video_frames table.

    Idea from here:
        https://github.com/KostadinovShalon/UAVDetectionTrackingBenchmark/blob/main/detection/utils/video_to_images.py

    Args:
        data_origin: The name of the video set used in the database.
    """
    video_data = dbm.get_videos_for_origin(data_origin)
    data_orign_id = dbm.get_data_origin_id_for_name(data_origin)

    for video in tqdm.tqdm(video_data):
        capture = cv2.VideoCapture(video.path_abs)

        frame_number = 0
        while True:
            has_frames, frame = capture.read()
            if not has_frames:
                break

            ret, buffer = cv2.imencode(".png", frame)
            if not ret:
                raise RuntimeError(f"Failed to encode frame {frame_number}")
            img_bytes = buffer.tobytes()
            img_id = dbm.insert_image(
                dt.Image(
                    name=f"{Path(video.name).stem}_{frame_number}.png",
                    data_origin_id=data_orign_id,
                    width=video.properties.width,
                    height=video.properties.height,
                    image=img_bytes,
                )
            )
            dbm.insert_video_frame(video.id, img_id, frame_number)
            frame_number += 1


def get_video_properties(video_path: PathLike) -> dt.VideoProperties:
    video_capture = cv2.VideoCapture(str(video_path))
    video_property = dt.VideoProperties(
        width=round(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=round(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        frame_rate=video_capture.get(cv2.CAP_PROP_FPS),
        frame_count=round(video_capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        convert_rgb=bool(round(video_capture.get(cv2.CAP_PROP_CONVERT_RGB))),
    )
    video_capture.release()
    return video_property


def create_video_from_path(path: Path, data_origin_name: str) -> dt.Video:
    return dt.Video(
        name=path.name,
        path_abs=str(path),
        sha_256=dt.Video.get_sha_256(path),
        data_origin_id=dbm.get_data_origin_id_for_name(data_origin_name),
        properties=get_video_properties(path),
    )


def create_video_from_images_in_folder(
    image_folder: Path,
    video_target_path: Path,
    width: int,
    height: int,
    frame_rate: float,
):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # use "XVID" or "avc1" if needed
    out = cv2.VideoWriter(
        filename=str(video_target_path),
        fourcc=fourcc,
        fps=frame_rate,
        frameSize=(width, height),
    )

    for file_name in sorted(image_folder.iterdir()):
        img = cv2.imread(str(file_name), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Warning: skipping unreadable file {file_name}")
            continue
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        # Resize if different from first frame
        if (img.shape[1], img.shape[0]) != (width, height):
            img = cv2.resize(img, (width, height))
        out.write(img)

    out.release()


if __name__ == "__main__":
    pass
