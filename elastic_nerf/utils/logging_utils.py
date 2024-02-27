import errno
import fcntl
import os
import time
from pathlib import Path
from typing import Union

import torch
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


def robust_torch_save(obj, path, max_retries=5, initial_retry_delay=1):
    """
    Saves a PyTorch object to an NFS path with robust handling for NFS peculiarities.
    This includes file locking, exponential backoff retries, and cleanup of partial files on failure.

    Args:
        obj: PyTorch object to be saved.
        path: NFS file path for saving the object.
        max_retries: Maximum number of retry attempts (default: 5).
        initial_retry_delay: Initial delay between retries in seconds, which doubles after each retry (default: 1).

    Raises:
        Exception: If saving fails after the maximum number of retries or due to an unexpected error.
    """

    retry_delay = initial_retry_delay
    for attempt in range(max_retries):
        try:
            with open(path, "wb") as file_handle:
                fcntl.flock(
                    file_handle, fcntl.LOCK_EX
                )  # Attempt to acquire an exclusive lock
                torch.save(obj, file_handle)
                return  # Success, exit the function
        except IOError as e:
            if e.errno != errno.EAGAIN:
                if os.path.exists(path):  # Clean up any partially written file
                    os.remove(path)
                raise  # Reraise the exception if it's not a "try again" error
        finally:
            # This block ensures the file lock is released even if an error occurs
            if "file_handle" in locals() and not file_handle.closed:
                fcntl.flock(file_handle, fcntl.LOCK_UN)

        time.sleep(retry_delay)  # Wait before retrying
        retry_delay *= 2  # Exponential backoff

    # If the loop completes without returning, it means all retries have been exhausted
    raise Exception(f"Failed to save object to {path} after {max_retries} attempts.")
