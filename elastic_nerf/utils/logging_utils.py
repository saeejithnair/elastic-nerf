from pathlib import Path
from typing import Union

from gonas.utils import logging_utils as gonas_lu

LOGGER = gonas_lu.configure_logger("elastic_nerf")


def info(message) -> None:
    gonas_lu.info(message, LOGGER)


def warning(message) -> None:
    gonas_lu.warning(message, LOGGER)


def error(message) -> None:
    gonas_lu.error(message, LOGGER)


def get_latest_checkpoint(checkpoint_dir: Path) -> Union[Path, None]:
    """Returns the latest checkpoint in the checkpoint directory."""
    checkpoints = checkpoint_dir.rglob("*.ckpt")
    if not checkpoints:
        return None
    latest_checkpoint = sorted(checkpoints, key=lambda x: x.stat().st_mtime)[-1]
    return latest_checkpoint
