"""Logging module from the book Deep Learning with PyTorch by Manning"""

import logging

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Some libraries attempt to add their own root logger handlers.
# This is annoying, and so we get rid of them.
for handler in list(root_logger.handlers):
    root_logger.removeHandler(handler)

LOGFMT_STR = (
    "%(asctime)s %(levelname)-8s pid:%(process)d "
    "%(name)s:%(lineno)03d:%(funcName)s %(message)s"
)
FORMATTER = logging.Formatter(LOGFMT_STR)

STREAM_HANDLER = logging.StreamHandler()
STREAM_HANDLER.setFormatter(FORMATTER)
STREAM_HANDLER.setLevel(logging.DEBUG)

root_logger.addHandler(STREAM_HANDLER)
