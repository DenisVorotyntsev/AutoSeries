# pylint: disable=logging-fstring-interpolation, broad-except
"""common"""
import logging
import importlib
import sys


class ModelApiError(Exception):
    """Model api error"""


def get_logger(verbosity_level, name, use_error_log=False):
    """Set logging format to something like:
        2019-04-25 12:52:51,924 INFO score.py: <message>
    """
    logger = logging.getLogger(name)
    logging_level = getattr(logging, verbosity_level)
    logger.setLevel(logging_level)
    formatter = logging.Formatter(
        fmt='%(asctime)s %(levelname)s %(filename)s: %(message)s')
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging_level)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    if use_error_log:
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging.WARNING)
        stderr_handler.setFormatter(formatter)
        logger.addHandler(stderr_handler)
    logger.propagate = False
    return logger


VERBOSITY_LEVEL = 'INFO'
LOGGER = get_logger(VERBOSITY_LEVEL, __file__)


def _check_umodel_methed(umodel):
    # Check if the model has methods `train`, `predict`, `save`, `load`.
    for attr in ['train', 'update', 'predict', 'save', 'load']:
        if not hasattr(umodel, attr):
            raise ModelApiError("Your model object doesn't have the method ")


def import_umodel():
    """import user model"""
    model_cls = importlib.import_module('model').Model
    _check_umodel_methed(model_cls)

    return model_cls


def init_usermodel(dataset):
    """initialize user model"""
    return import_umodel()(
        dataset.get_metadata(), dataset.get_test_timestamp(),
        dataset.get_pred_timestamp())
