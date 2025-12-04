"""Module for database initialization."""

from __future__ import annotations

import sqlite3

from source.db.connection import database_connection


@database_connection
def initialize_database(cursor: sqlite3.Cursor):
    def create_data_origin(cur: sqlite3.Cursor):
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS data_origins
            (
                id   INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                name TEXT    NOT NULL UNIQUE
            );
            """
        )

    def create_video_table(cur: sqlite3.Cursor):
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS videos
            (
                id             INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                name           TEXT    NOT NULL UNIQUE,
                data_origin_id INTEGER NOT NULL,
                width          INTEGER NOT NULL,
                height         INTEGER NOT NULL,
                frame_rate     FLOAT   NOT NULL,
                frame_count    INTEGER NOT NULL,
                convert_rgb    BOOLEAN NOT NULL,
                path           TEXT    NOT NULL UNIQUE,
                sha_256        TEXT    NOT NULL UNIQUE,
                FOREIGN KEY (data_origin_id) REFERENCES data_origins (id)
            );
            """
        )

    def create_image_table(cur: sqlite3.Cursor):
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS images
            (
                id             INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                name           TEXT    NOT NULL UNIQUE,
                data_origin_id INTEGER NOT NULL,
                width          INTEGER NOT NULL,
                height         INTEGER NOT NULL,
                image          BLOB    NOT NULL UNIQUE,
                FOREIGN KEY (data_origin_id) REFERENCES data_origins (id)
            );
            """
        )

    def create_video_frame_table(cur: sqlite3.Cursor):
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS video_frames
            (
                id           INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                video_id     INTEGER NOT NULL,
                image_id     INTEGER NOT NULL,
                frame_number INTEGER NOT NULL,
                FOREIGN KEY (video_id) REFERENCES videos (id),
                FOREIGN KEY (image_id) REFERENCES images (id),
                UNIQUE (video_id, image_id)
            );
            """
        )

    def create_labels_table(cur: sqlite3.Cursor):
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS labels
            (
                id     INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                name   TEXT    NOT NULL UNIQUE,
                number INTEGER NOT NULL UNIQUE
            );
            """
        )

    def create_annotations_table(cur: sqlite3.Cursor):
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS annotations
            (
                id       INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER NOT NULL,
                label_id INTEGER NOT NULL,
                x_min    INTEGER NOT NULL,
                y_min    INTEGER NOT NULL,
                x_max    INTEGER NOT NULL,
                y_max    INTEGER NOT NULL,
                FOREIGN KEY (image_id) REFERENCES images (id),
                FOREIGN KEY (label_id) REFERENCES labels (id)
            );
            """
        )

    def create_datasets_table(cur: sqlite3.Cursor):
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS datasets
            (
                id   Integer NOT NULL PRIMARY KEY AUTOINCREMENT,
                name TEXT    NOT NULL UNIQUE
            );
            """
        )

    def create_data_categories_table(cur: sqlite3.Cursor):
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS data_categories
            (
                id   INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                name TEXT    NOT NULL UNIQUE
            );
            """
        )

    def create_data_subsets_table(cur: sqlite3.Cursor):
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS data_subsets
            (
                id               Integer NOT NULL PRIMARY KEY AUTOINCREMENT,
                dataset_id       INTEGER NOT NULL,
                image_id         INTEGER NOT NULL,
                data_category_id INTEGER NOT NULL,
                FOREIGN KEY (dataset_id) REFERENCES datasets (id),
                FOREIGN KEY (image_id) REFERENCES images (id),
                FOREIGN KEY (data_category_id) REFERENCES data_categories (id),
                UNIQUE (dataset_id, image_id)
            );
            """
        )

    def create_normalization_data_table(cur: sqlite3.Cursor):
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS normalization_data
            (
                id           Integer NOT NULL PRIMARY KEY AUTOINCREMENT,
                dataset_id   INTEGER NOT NULL UNIQUE,
                mean_ch_0    FLOAT   NOT NULL,
                mean_ch_1    FLOAT   NOT NULL,
                mean_ch_2    FLOAT   NOT NULL,
                std_dev_ch_0 FLOAT   NOT NULL,
                std_dev_ch_1 FLOAT   NOT NULL,
                std_dev_ch_2 FLOAT   NOT NULL,
                FOREIGN KEY (dataset_id) REFERENCES datasets (id)
            );
            """
        )

    def create_model_states_table(cur: sqlite3.Cursor):
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS model_states
            (
                id                 INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                dataset_id         INTEGER NOT NULL,
                model_type         TEXT    NOT NULL,
                optimizer_type     TEXT    NOT NULL,
                lr_scheduler_type  TEXT    NOT NULL,
                epochs_trained     INTEGER NOT NULL,
                samples_trained    INTEGER NOT NULL,
                timestamp          TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP,
                model_state        BLOB    NOT NULL,
                optimizer_state    BLOB    NOT NULL,
                lr_scheduler_state BLOB    NOT NULL,
                FOREIGN KEY (dataset_id) REFERENCES datasets (id)
            );
            """
        )

    create_data_origin(cursor)
    create_video_table(cursor)
    create_image_table(cursor)
    create_video_frame_table(cursor)
    create_labels_table(cursor)
    create_annotations_table(cursor)
    create_datasets_table(cursor)
    create_data_categories_table(cursor)
    create_data_subsets_table(cursor)
    create_normalization_data_table(cursor)
    create_model_states_table(cursor)
