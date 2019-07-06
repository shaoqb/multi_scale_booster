import logging
import os.path as osp
from mmcv.runner import get_time_str


def add_file_handler(logger,
                     filename=None,
                     mode='w',
                     level=logging.INFO):
    # TODO: move this method out of runner
    file_handler = logging.FileHandler(filename, mode)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    file_handler.setLevel(level)
    logger.addHandler(file_handler)
    return logger


def init_logger(log_dir=None, level=logging.INFO):
    """Init the logger.

    Args:
        log_dir(str, optional): Log file directory. If not specified, no
            log file will be used.
        level (int or str): See the built-in python logging module.

    Returns:
        :obj:`~logging.Logger`: Python logger.
    """
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s', level=level)
    logger = logging.getLogger(__name__)
    if log_dir:
        filename = '{}.log'.format(get_time_str())
        log_file = osp.join(log_dir, filename)
        add_file_handler(logger, log_file, level=level)
    return logger


