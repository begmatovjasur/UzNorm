import logging
from logging.handlers import RotatingFileHandler


def configure(folder):
    folder.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("uznorm_studio")
    if not logger.handlers:
        handler = RotatingFileHandler(folder / "studio.log", maxBytes=1024**2, backupCount=2, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger

