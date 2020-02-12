import logging
from datetime import datetime

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
