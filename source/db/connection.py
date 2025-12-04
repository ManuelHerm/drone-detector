"""Module provides the standard database connection decorator."""

from __future__ import annotations

import functools
import sqlite3
from typing import Callable

from source.config.locations import DB_PATH


def database_connection(func) -> Callable:
    """
    Manage the database connection.

    - Begins and ends the transaction.
    - Carries out rollback in case of an error.

    The decorated function needs a parameter called cursor of type ``sqlite3.Cursor``.
    This allows the decorated function to use the database.

    Args:
        func: Function to decorate.

    Returns:
        Return value of the decorated function.

    Raises:
        sqlite3.Error, if there was a database error.
    """

    @functools.wraps(func)
    def wrapper_database_connection(*args, **kwargs):
        con = sqlite3.connect(DB_PATH)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON;")
        cur = con.cursor()
        try:
            # database function gets the cursor
            # from the decorator
            con.execute("BEGIN TRANSACTION;")
            return_value = func(*args, cursor=cur, **kwargs)
            con.commit()
            return return_value
        except sqlite3.Error as e:
            con.rollback()
            raise sqlite3.Error(e)
        finally:
            cur.close()
            con.close()

    return wrapper_database_connection
