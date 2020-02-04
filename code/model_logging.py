import logging

logger = None


def get_logger():
    global logger
    if not logger:
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logFormat = "%(asctime)s %(levelname)s: %(filename)s:%(funcName)s():%(lineno)d - %(message)s"
        logging.basicConfig(
            level=logging.DEBUG,
            format=logFormat,
            filename="log/logging.log",
            filemode="w"
        )
        logger = logging.getLogger(__name__)
        logger.info("Logger initialized")
    return logger
