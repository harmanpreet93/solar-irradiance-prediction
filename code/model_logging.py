import logging
from datetime import datetime
import tensorflow as tf

logger = None


def get_logger():
    global logger
    if not logger:
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        file_name = 'log/' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '.log'
        line_format = "%(asctime)s %(levelname)s: %(filename)s:%(funcName)s():%(lineno)d - %(message)s"
        logging.basicConfig(
            level=logging.DEBUG,
            format=line_format,
            filename=file_name,
            filemode="w"
        )
        logger = logging.getLogger(__name__)
        logger.info("Logger initialized")
    return logger


def get_summary_writer():
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'log/gradient_tape/' + current_time + '/train'
    test_log_dir = 'log/gradient_tape/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    return train_summary_writer, test_summary_writer
