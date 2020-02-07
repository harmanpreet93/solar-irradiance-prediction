import logging

logger = None


def get_logger():
    global logger
    if not logger:
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)s: %(filename)s:%(funcName)s():%(lineno)d - %(message)s",
            filename="log/logging.log",
            filemode="w"
        )
        logger = logging.getLogger(__name__)
        logger.info("Logger initialized")
    return logger
