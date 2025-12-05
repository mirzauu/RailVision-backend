import logging
from logging import StreamHandler, Formatter
from typing import Optional
from src.config.settings import settings

_configured: Optional[bool] = False

def setup_logging() -> None:
    global _configured
    if _configured:
        return
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(level=level)
    logger = logging.getLogger()
    for h in logger.handlers:
        h.setLevel(level)
        if isinstance(h, StreamHandler):
            h.setFormatter(Formatter('%(asctime)s %(levelname)s %(name)s %(message)s'))
    _configured = True
