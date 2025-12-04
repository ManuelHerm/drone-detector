"""This module provides database management functionality."""

from __future__ import annotations

import sqlite3
from dataclasses import asdict

from source.data import datatypes as dt
from source.db.connection import database_connection
from source.db.database_initializer import initialize_database

# pylint: disable=no-value-for-parameter
#         Disabled, because the dbm-function receive the
#         cursor parameter from the decorator.
# pylint: disable=missing-function-docstring
#         There are many functions following the same pattern.
#         They are simple and can be understood by their signature.


@database_connection
def insert_data_origin(name: str, cursor: sqlite3.Cursor) -> int:
    """
    Insert data origin into database.

    Args:
        name: Name of the data origin
        cursor: Database cursor. Supplied by the decorator.

    Returns:
        ID value of the inserted data origin.
    """
    cursor.execute("INSERT INTO data_origins (name) VALUES (?);", (name,))
    return cursor.lastrowid


@database_connection
def insert_data_subset(
    data_subset_content: list[dict[str, int]], cursor: sqlite3.Cursor
):
    """
    Insert data subset into database.

    Args:
        data_subset_content: Each element in the list is formed like this:

            {
                "dataset_id": <dataset_id>,
                "image_id": <image_id>,
                "data_category_id": <data_category_id>
            }

        cursor: Database cursor. Supplied by the decorator.
    """
    for data_subset in data_subset_content:
        cursor.execute(
            """INSERT INTO data_subsets (dataset_id, image_id, data_category_id)
               VALUES (:dataset_id, :image_id, :data_category_id);""",
            data_subset,
        )
    return cursor.lastrowid


@database_connection
def insert_image(image: dt.Image, cursor: sqlite3.Cursor) -> int:
    """
    Insert an image into the database.

    Args:
        image: Image to insert.
        cursor: Database cursor. Supplied by the decorator.

    Returns:
        The id of the inserted image.
    """
    data = asdict(image)
    cursor.execute(
        """
        INSERT INTO images
            (name, data_origin_id, width, height, image)
        VALUES (:name, :data_origin_id, :width, :height, :image)
        """,
        data,
    )
    return cursor.lastrowid


@database_connection
def insert_model_state(
    dataset_id: int,
    model_type: str,
    optimizer_type: str,
    lr_scheduler_type: str,
    model_state: bytes,
    optimizer_state: bytes,
    lr_scheduler_state: bytes,
    epochs_trained: int,
    samples_trained: int,
    cursor: sqlite3.Cursor,
) -> int:
    cursor.execute(
        """
        INSERT INTO model_states (dataset_id,
                                  model_type,
                                  optimizer_type,
                                  lr_scheduler_type,
                                  model_state,
                                  optimizer_state,
                                  lr_scheduler_state,
                                  epochs_trained,
                                  samples_trained)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            dataset_id,
            model_type,
            optimizer_type,
            lr_scheduler_type,
            model_state,
            optimizer_state,
            lr_scheduler_state,
            epochs_trained,
            samples_trained,
        ),
    )
    return cursor.lastrowid


@database_connection
def insert_video_frame(
    video_id: int, img_id: int, frame_number: int, cursor: sqlite3.Cursor
) -> int:
    """
    Insert video frame into database.

    Args:
        video_id: Video ID number where the image belongs to.
        img_id: Image ID of the image that belongs to the frame.
        frame_number: The frame number in the video.
        cursor: Database cursor. Supplied by the decorator.

    Returns:
        The id of the inserted video frame.
    """
    cursor.execute(
        """
        INSERT INTO video_frames
            (video_id, image_id, frame_number)
        VALUES (?, ?, ?)
        """,
        (video_id, img_id, frame_number),
    )
    return cursor.lastrowid


@database_connection
def insert_videos(videos: list[dt.Video], cursor: sqlite3.Cursor):
    """
    Insert videos into the database.

    Args:
        videos: Videos to insert.
        cursor: Database cursor. Supplied by the decorator.
    """
    for video in videos:
        cursor.execute(
            """
            INSERT INTO videos (name,
                                data_origin_id,
                                width,
                                height,
                                frame_rate,
                                frame_count,
                                convert_rgb,
                                path,
                                sha_256)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                video.name,
                video.data_origin_id,
                video.properties.width,
                video.properties.height,
                video.properties.frame_rate,
                video.properties.frame_count,
                video.properties.convert_rgb,
                video.path_abs,
                video.sha_256,
            ),
        )


@database_connection
def insert_annotations(annotation_list: list[dt.Annotation], cursor: sqlite3.Cursor):
    """
    Insert annotations into the database.

    Args:
        annotation_list: List of annotations to insert.
        cursor: Database cursor. Supplied by the decorator.
    """
    for annotation in annotation_list:
        cursor.execute(
            """
            INSERT INTO annotations
                (image_id, label_id, x_min, y_min, x_max, y_max)
            VALUES (:image_id, :label_id, :x_min, :y_min, :x_max, :y_max)
            """,
            asdict(annotation),
        )


@database_connection
def insert_data_categories(cursor: sqlite3.Cursor):
    """
    Insert the default categories: training, validation, and testing.

    Args:
        cursor: Database cursor. Supplied by the decorator.
    """
    cursor.execute(
        """
        INSERT INTO data_categories (name)
        VALUES ('training'),
               ('validation'),
               ('testing');
        """
    )


@database_connection
def insert_dataset(name: str, cursor: sqlite3.Cursor) -> int:
    """
    Insert a dataset name.

    Args:
        name: Name of the dataset
        cursor: Database cursor. Supplied by the decorator.

    Returns:
        The id of the inserted dataset.
    """
    cursor.execute(
        """
        INSERT INTO datasets
            (name)
        VALUES (?);
        """,
        (name,),
    )
    return cursor.lastrowid


@database_connection
def insert_normalization_dataset(data: dt.NormalizationData, cursor: sqlite3.Cursor):
    """
    Insert a normalization dataset into the database.

    Args:
        data: The normalization data to insert.
        cursor: Database cursor. Supplied by the decorator.
    """
    cursor.execute(
        """
        INSERT INTO normalization_data (
            dataset_id, 
            mean_ch_0, 
            mean_ch_1, 
            mean_ch_2, 
            std_dev_ch_0, 
            std_dev_ch_1, 
            std_dev_ch_2
        )
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """,
        (
            data.dataset_id,
            data.mean[0],
            data.mean[1],
            data.mean[2],
            data.std[0],
            data.std[1],
            data.std[2],
        ),
    )


def _create_annotation_from_dict(annotation_row: dict) -> dt.Annotation:
    return dt.Annotation(
        id=annotation_row["id"],
        image_id=annotation_row["image_id"],
        label_id=annotation_row["label_id"],
        x_min=annotation_row["x_min"],
        y_min=annotation_row["y_min"],
        x_max=annotation_row["x_max"],
        y_max=annotation_row["y_max"],
    )


@database_connection
def get_annotations(cursor: sqlite3.Cursor) -> list[dt.Annotation]:
    """
    Get all annotations in the database.

    Args:
        cursor: Database cursor. Supplied by the decorator.

    Returns:
        All annotations in the database.
    """
    cursor.execute(
        """
        SELECT id, image_id, label_id, x_min, y_min, x_max, y_max
        FROM annotations;
        """,
    )
    rows = cursor.fetchall()
    if rows:
        return [_create_annotation_from_dict(row) for row in rows]
    raise IndexError("No annotations found.")


@database_connection
def get_annotations_for_image_id(
    image_id: int, cursor: sqlite3.Cursor
) -> list[dt.Annotation]:
    cursor.execute(
        """SELECT id, image_id, label_id, x_min, y_min, x_max, y_max
           FROM annotations
           WHERE image_id = ?""",
        (image_id,),
    )
    rows = cursor.fetchall()
    if rows:
        return [_create_annotation_from_dict(row) for row in rows]
    raise IndexError(
        f"No annotations found.\nSearched for annotations with image_id {image_id}."
    )


@database_connection
def get_annotation_for_annotation_id(
    annotation_id: int, cursor: sqlite3.Cursor
) -> dt.Annotation:
    cursor.execute(
        """SELECT id, image_id, label_id, x_min, y_min, x_max, y_max
           FROM annotations
           WHERE id = ?""",
        (annotation_id,),
    )
    row = cursor.fetchone()
    if row:
        return _create_annotation_from_dict(row)
    raise IndexError(f"No annotation with id {annotation_id} found.")


@database_connection
def get_data_origin_id_for_name(name: str, cursor: sqlite3.Cursor) -> int:
    """
    Look up the id of the data origin for the given name.

    Args:
        name: Name of the data origin.
        cursor: Database cursor. Supplied by the decorator.

    Returns:
        `id` of the data origin, if the row exists.

    Raises:
        IndexError: If the row does not exist.
    """
    cursor.execute("SELECT id from data_origins WHERE name = ?", (name,))
    row = cursor.fetchone()
    if row:
        return row["id"]
    raise IndexError(f'There is no data origin with the name "{name}".')


@database_connection
def get_data_category_id_for_name(name: str, cursor: sqlite3.Cursor) -> int:
    cursor.execute("SELECT id from data_categories WHERE name = ?", (name,))
    row = cursor.fetchone()
    if row:
        return row["id"]
    raise IndexError(f'There is no data category with the name "{name}".')


@database_connection
def get_dataset_id_for_dataset_name(dataset_name: str, cursor: sqlite3.Cursor) -> int:
    cursor.execute("SELECT id from datasets WHERE name = ?", (dataset_name,))
    row = cursor.fetchone()
    if row:
        return row["id"]
    raise IndexError(f'There is no dataset with the name "{dataset_name}".')


@database_connection
def get_dataset_id_for_model_state_id(
    model_state_id: int, cursor: sqlite3.Cursor
) -> int:
    cursor.execute(
        "SELECT dataset_id from model_states WHERE id = ?", (model_state_id,)
    )
    row = cursor.fetchone()
    if row:
        return row["dataset_id"]
    raise IndexError(f'There is no model_state with the id "{model_state_id}".')


@database_connection
def get_dataset_name_for_id(dataset_id: int, cursor: sqlite3.Cursor) -> str:
    cursor.execute("SELECT name from datasets WHERE id = ?", (dataset_id,))
    row = cursor.fetchone()
    if row:
        return row["name"]
    raise IndexError(f"There is no dataset with the id = {dataset_id}.")


@database_connection
def get_epochs_trained(model_state_id: int, cursor: sqlite3.Cursor) -> int:
    cursor.execute(
        "SELECT epochs_trained from model_states WHERE id = ?", (model_state_id,)
    )
    row = cursor.fetchone()
    if row:
        return row["epochs_trained"]
    raise IndexError(f"There is no dataset with the id = {model_state_id}.")


@database_connection
def get_image_id_for_video_id_and_frame_number(
    video_id: int, frame_number: int, cursor: sqlite3.Cursor
) -> int:
    cursor.execute(
        "SELECT image_id from video_frames WHERE video_id = ? AND frame_number = ?;",
        (video_id, frame_number),
    )
    row = cursor.fetchone()
    if row:
        return row[0]
    raise ValueError(
        "Data origin not found.\n"
        f"Searched for frame_number {frame_number} in video {video_id}"
    )


@database_connection
def get_image_id_for_image_name(image_name: str, cursor: sqlite3.Cursor) -> int:
    cursor.execute("SELECT id from images WHERE name = ?", (image_name,))
    row = cursor.fetchone()
    if row:
        return row[0]
    raise IndexError(f'There is no image with name "{image_name}"')


@database_connection
def get_image_ids_for_data_origin_name(
    data_origin_name: str, cursor: sqlite3.Cursor, has_limit=False, limit=None
) -> list[int]:
    data_origin_id = get_data_origin_id_for_name(data_origin_name)
    if has_limit:
        cursor.execute(
            "SELECT id from images WHERE data_origin_id = ? LIMIT ?",
            (data_origin_id, limit),
        )
    else:
        cursor.execute(
            "SELECT id from images WHERE data_origin_id = ?", (data_origin_id,)
        )
    rows = cursor.fetchall()
    if rows:
        return [row["id"] for row in rows]
    raise IndexError(
        f"No image ids found.\n"
        f"Searched for images that beĺong to the data origin {data_origin_name}."
    )


@database_connection
def get_image_ids_for_data_category_and_dataset_id(
    dataset_id: int,
    data_category_id: str,
    cursor: sqlite3.Cursor,
    has_limit: bool = False,
    limit: int = None,
) -> list[int]:
    if has_limit:
        cursor.execute(
            """SELECT image_id
               from data_subsets
               WHERE dataset_id = ?
                 AND data_category_id = ?
               LIMIT ?;""",
            (dataset_id, data_category_id, limit),
        )
    else:
        cursor.execute(
            """SELECT image_id
               from data_subsets
               WHERE dataset_id = ?
                 AND data_category_id = ?;
            """,
            (dataset_id, data_category_id),
        )
    rows = cursor.fetchall()
    if rows:
        return [row["image_id"] for row in rows]
    raise IndexError(
        "No image ids found.\n"
        f"Searched for images in the data_category with id {data_category_id} "
        f"and that belong to the dataset with id {dataset_id}."
    )


@database_connection
def get_images_for_image_ids(
    image_ids: list[int], cursor: sqlite3.Cursor
) -> list[bytes]:
    images = []
    for image_id in image_ids:
        cursor.execute("SELECT image from images WHERE id = ?", (image_id,))
        row = cursor.fetchone()
        if row:
            images.append(row["image"])
        else:
            raise IndexError(f"Image with id {image_id} not found.")
    return images


@database_connection
def get_image_ids_for_video_id(video_id: int, cursor: sqlite3.Cursor) -> list[int]:
    cursor.execute(
        "SELECT image_id from video_frames WHERE video_id = ? order by frame_number;",
        (video_id,),
    )
    rows = cursor.fetchall()
    if rows:
        # Each row contains a tuple with a single elements.
        # We do not need the tuple, just the contained integer.
        return [img_id[0] for img_id in rows]
    raise ValueError(
        f"No image ids found.\nSearched for video_id {video_id} in video_frames."
    )


@database_connection
def get_image_for_id(image_id: int, cursor: sqlite3.Cursor) -> bytes:
    cursor.execute("SELECT image from images WHERE id = ?", (image_id,))
    row = cursor.fetchone()
    if row:
        return row[0]
    raise ValueError(f"Data origin not found.\nSearched for image with id {image_id}.")


@database_connection
def get_image_width_and_height_for_image_id(
    image_id: int, cursor: sqlite3.Cursor
) -> tuple[int, int]:
    cursor.execute("SELECT width, height from images WHERE id = ?", (image_id,))
    row = cursor.fetchone()
    if row:
        return row["width"], row["height"]
    raise ValueError(f"Image with id {image_id} not found.")


@database_connection
def get_labels(cursor: sqlite3.Cursor) -> list[dt.Label]:
    cursor.execute("SELECT id, name, number from labels")
    rows = cursor.fetchall()
    if rows:
        return [
            dt.Label(id=row["id"], name=row["name"], number=row["number"])
            for row in rows
        ]
    raise ValueError("No labels found.")


@database_connection
def get_learning_rate_scheduler_state(state_id: int, cursor: sqlite3.Cursor) -> bytes:
    cursor.execute(
        "SELECT lr_scheduler_state from model_states WHERE id = ?", (state_id,)
    )
    row = cursor.fetchone()
    if row:
        return row[0]
    raise ValueError(
        f"No learning rate scheduler with state with id = {state_id} found."
    )


@database_connection
def get_model_state(model_state_id: int, cursor: sqlite3.Cursor) -> bytes:
    cursor.execute(
        "SELECT model_state from model_states WHERE id = ?", (model_state_id,)
    )
    row = cursor.fetchone()
    if row:
        return row[0]
    raise ValueError(f"No model state with id = {model_state_id} found.")


@database_connection
def get_model_state_ids(cursor: sqlite3.Cursor) -> list[int]:
    cursor.execute("SELECT id from model_states")
    rows = cursor.fetchall()
    if rows:
        return [row["id"] for row in rows]
    raise ValueError("No model states found.")


@database_connection
def get_normalization_data_for_dataset_id(
    dataset_id: int, cursor: sqlite3.Cursor
) -> dt.NormalizationData:
    cursor.execute(
        """
        SELECT mean_ch_0, mean_ch_1, mean_ch_2, std_dev_ch_0, std_dev_ch_1, std_dev_ch_2
        FROM normalization_data
        WHERE dataset_id = ?
        """,
        (dataset_id,),
    )
    row = cursor.fetchone()
    if row:
        return dt.NormalizationData(
            dataset_id=dataset_id,
            mean=(row["mean_ch_0"], row["mean_ch_1"], row["mean_ch_2"]),
            std=(row["std_dev_ch_0"], row["std_dev_ch_1"], row["std_dev_ch_2"]),
        )
    raise ValueError(f"Normalization data not found.\nSearched for id {dataset_id}.")


@database_connection
def get_normalization_data_id_for_dataset_id(
    dataset_id: int, cursor: sqlite3.Cursor
) -> int:
    cursor.execute(
        """
        SELECT id
        FROM normalization_data
        WHERE dataset_id = ?""",
        (dataset_id,),
    )
    row = cursor.fetchone()
    if row:
        return row["id"]
    raise ValueError(f"No normalization data with dataset id = {dataset_id} found.")


@database_connection
def get_normalization_data_for_id(
    normalization_data_id: int, cursor: sqlite3.Cursor
) -> dt.NormalizationData:
    cursor.execute(
        """
        SELECT 
            dataset_id, 
            mean_ch_0, 
            mean_ch_1, 
            mean_ch_2, 
            std_dev_ch_0, 
            std_dev_ch_1, 
            std_dev_ch_2
        FROM normalization_data
        WHERE id = ?""",
        (normalization_data_id,),
    )
    row = cursor.fetchone()
    if row:
        return dt.NormalizationData(
            dataset_id=row["dataset_id"],
            mean=(row["mean_ch_0"], row["mean_ch_1"], row["mean_ch_2"]),
            std=(row["std_dev_ch_0"], row["std_dev_ch_1"], row["std_dev_ch_2"]),
        )
    raise ValueError(f"No normalization data found with id {normalization_data_id}.")


@database_connection
def get_number_of_drones_for_images(
    image_ids: list[int], cursor: sqlite3.Cursor
) -> list[tuple[int, int]]:
    """
    Count the number of drones in each image from the image_ids list.

    Args:
        image_ids: The ids for the images to consider.
        cursor: sqlite cursor. Supplied by the decorator.

    Returns:
        A list of (image_id, number of drones).
    """
    drone_count_list = []
    for image_id in image_ids:
        cursor.execute(
            """
            SELECT COUNT(id) as number_of_drones
            FROM annotations
            WHERE image_id = ?;
            """,
            (image_id,),
        )
        row = cursor.fetchone()
        drone_count_list.append((image_id, row["number_of_drones"]))
    return drone_count_list


@database_connection
def get_optimizer_state(model_state_id: int, cursor: sqlite3.Cursor) -> bytes:
    cursor.execute(
        "SELECT optimizer_state from model_states WHERE id = ?", (model_state_id,)
    )
    row = cursor.fetchone()
    if row:
        return row[0]
    raise ValueError(f"No optimizer state with id = {model_state_id} found.")


@database_connection
def get_video_shape_for_video_id(
    video_id: int, cursor: sqlite3.Cursor
) -> tuple[int, int]:
    cursor.execute(
        """
        SELECT width,
               height
        FROM videos
        WHERE id = ?
        """,
        (video_id,),
    )
    row = cursor.fetchone()
    if row:
        return row["width"], row["height"]
    raise IndexError(f"Video with id = {video_id} not found.")


@database_connection
def get_videos_for_origin(origin: str, cursor: sqlite3.Cursor) -> list[dt.Video]:
    origin_id = get_data_origin_id_for_name(origin)
    cursor.execute(
        """
        SELECT id,
               name,
               data_origin_id,
               width,
               height,
               frame_rate,
               frame_count,
               convert_rgb,
               path,
               sha_256
        FROM videos
        WHERE data_origin_id = ?
        """,
        (origin_id,),
    )
    rows = cursor.fetchall()
    if rows:
        return [
            dt.Video(
                id=row["id"],
                name=row["name"],
                path_abs=row["path"],
                sha_256=row["sha_256"],
                data_origin_id=row["data_origin_id"],
                properties=dt.VideoProperties(
                    width=row["width"],
                    height=row["height"],
                    frame_rate=row["frame_rate"],
                    frame_count=row["frame_count"],
                    convert_rgb=row["convert_rgb"],
                ),
            )
            for row in rows
        ]
    raise ValueError(f"No data for given origin = {origin}")


@database_connection
def get_video_id_for_video_name_stem(
    video_name_stem: str, cursor: sqlite3.Cursor
) -> int:
    """
    Get video id for a video name stem.

    Args:
        video_name_stem: Name of the video to get the id from
        cursor: Database cursor. Supplied by the decorator.

    Returns:
        The id that matches the video name stem.
    """
    cursor.execute(
        "SELECT id from videos WHERE name LIKE (? || '%');",
        (video_name_stem,),
    )
    row = cursor.fetchone()
    if row:
        return row[0]
    raise ValueError(f"No video found with this name stem: {video_name_stem}.")


@database_connection
def get_samples_trained(model_state_id: int, cursor: sqlite3.Cursor) -> int:
    cursor.execute(
        "SELECT samples_trained from model_states WHERE id = ?;",
        (model_state_id,),
    )
    row = cursor.fetchone()
    if row:
        return row[0]
    raise ValueError(
        f"No model state with id {model_state_id} found "
        "while searching for samples_trained."
    )


@database_connection
def remove_annotations(annotations_to_remove: set[int], cursor: sqlite3.Cursor):
    """
    Remove annotations from the database.

    Args:
        annotations_to_remove: The ids for the annotations to remove.
        cursor: sqlite cursor. Supplied by the decorator.
    """
    for annotation_id in annotations_to_remove:
        cursor.execute(
            """
        DELETE FROM annotations
        WHERE id = ?;
        """,
            (annotation_id,),
        )


if __name__ == "__main__":
    initialize_database()
