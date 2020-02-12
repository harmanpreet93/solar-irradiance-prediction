import logging
from datetime import datetime

logger = None


def get_logger():
    global logger
    if not logger:
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)s: %(filename)s:%(funcName)s():%(lineno)d - %(message)s",
            filename="log/{0}.log".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            filemode="w"
        )
        logger = logging.getLogger(__name__)
        logger.info("Logger initialized")
    return logger
