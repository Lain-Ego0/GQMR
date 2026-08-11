"""Shared exporter validation and atomic output helpers."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np

from gqmr.core.errors import GQMRError
from gqmr.core.motion import RobotMotion


class ExportError(GQMRError, ValueError):
    """Raised when a motion cannot be exported without hiding invalid data."""


def require_exportable(motion: RobotMotion) -> None:
    if motion.frame_count < 3:
        raise ExportError("export requires at least three frames")
    if not np.all(motion.frame_valid):
        invalid = np.flatnonzero(~motion.frame_valid)
        raise ExportError(
            f"motion contains {len(invalid)} invalid frames; select or repair a valid range"
        )


def atomic_write(path: str | os.PathLike[str], writer: Callable[[object], None]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination
