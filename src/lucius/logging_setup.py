from __future__ import annotations

import logging
import os

_CONFIGURED = False


def configure_logging(level: str | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=(level or os.environ.get("LUCIUS_LOG", "INFO")).upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name if name.startswith("lucius") else f"lucius.{name}")
