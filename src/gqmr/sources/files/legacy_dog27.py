"""Strict reader for the historical AI4Animation 27-point CSV-like TXT files."""

from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from gqmr import __version__
from gqmr.core.errors import GQMRError
from gqmr.core.motion import AnimalMotion
from gqmr.skeletons import AnimalSkeleton, get_skeleton

_VALUE_COUNT = 27 * 3
_MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024


class LegacyDog27Error(GQMRError, ValueError):
    """Raised when a legacy 27-point file is malformed or unsafe."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rows(path: Path) -> np.ndarray:
    try:
        size = path.stat().st_size
    except OSError as error:
        raise LegacyDog27Error(f"cannot stat input {path}: {error}") from error
    if size <= 0 or size > _MAX_FILE_SIZE:
        raise LegacyDog27Error("legacy dog-27 input is empty or exceeds 2 GiB")
    rows: list[list[float]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            for line_number, row in enumerate(reader, start=1):
                if len(row) != _VALUE_COUNT:
                    raise LegacyDog27Error(
                        f"line {line_number} must contain exactly {_VALUE_COUNT} values, got {len(row)}"
                    )
                try:
                    values = [float(value) for value in row]
                except ValueError as error:
                    raise LegacyDog27Error(
                        f"line {line_number} contains a non-numeric value"
                    ) from error
                if not np.all(np.isfinite(values)):
                    raise LegacyDog27Error(
                        f"line {line_number} contains NaN or infinity"
                    )
                rows.append(values)
    except (OSError, csv.Error, UnicodeError) as error:
        if isinstance(error, LegacyDog27Error):
            raise
        raise LegacyDog27Error(f"cannot read legacy dog-27 input: {error}") from error
    if not rows:
        raise LegacyDog27Error("legacy dog-27 input has no frames")
    return np.asarray(rows, dtype=np.float64).reshape(-1, 27, 3)


def _legacy_to_world(positions: np.ndarray) -> np.ndarray:
    # Historical importer: Rx(+90 deg), then Rz(0.47*pi). No robot-specific scale.
    rotation = Rotation.from_euler("z", 0.47 * np.pi) * Rotation.from_euler(
        "x", 0.5 * np.pi
    )
    return np.ascontiguousarray(rotation.apply(positions.reshape(-1, 3)).reshape(positions.shape))


def inspect_legacy_dog27(
    path: str | os.PathLike[str], *, fps: float = 60.0
) -> dict[str, Any]:
    if not np.isfinite(fps) or fps <= 0.0:
        raise LegacyDog27Error("fps must be finite and positive")
    input_path = Path(path)
    rows = _read_rows(input_path)
    duplicate_error = float(np.max(np.linalg.norm(rows[:, 0] - rows[:, 1], axis=1)))
    return {
        "path": str(input_path),
        "format": "legacy_ai4animation_dog27",
        "frames": len(rows),
        "fps": float(fps),
        "duration_seconds": (len(rows) - 1) / float(fps),
        "keypoints": 27,
        "pelvis_duplicate_max_error": duplicate_error,
        "source_sha256": _sha256_file(input_path),
        "license": "CC-BY-NC-4.0",
    }


def load_legacy_dog27(
    path: str | os.PathLike[str],
    *,
    fps: float = 60.0,
    start_frame: int = 0,
    end_frame: int | None = None,
    skeleton: AnimalSkeleton | None = None,
) -> AnimalMotion:
    if not np.isfinite(fps) or fps <= 0.0:
        raise LegacyDog27Error("fps must be finite and positive")
    input_path = Path(path)
    rows = _read_rows(input_path)
    stop = len(rows) if end_frame is None else end_frame
    if start_frame < 0 or stop <= start_frame or stop > len(rows):
        raise LegacyDog27Error("invalid start/end frame range")
    rows = rows[start_frame:stop]
    skeleton = skeleton or get_skeleton("dog-27")
    if len(skeleton.keypoints) != 27:
        raise LegacyDog27Error("legacy input requires a 27-point skeleton")
    positions = _legacy_to_world(rows).astype(np.float32)
    frames = len(positions)
    timestamps = np.arange(frames, dtype=np.float64) / float(fps)
    return AnimalMotion(
        timestamps=timestamps,
        keypoint_names=skeleton.names,
        positions=positions,
        confidence=np.ones((frames, 27), dtype=np.float32),
        valid_mask=np.ones((frames, 27), dtype=np.bool_),
        contact_probability=np.full((frames, 4), np.nan, dtype=np.float32),
        frame_valid=np.ones(frames, dtype=np.bool_),
        metadata={
            "coordinate_frame": "gqmr_world_x_forward_y_left_z_up",
            "length_unit": "m",
            "time_unit": "s",
            "skeleton_id": skeleton.id,
            "skeleton_sha256": skeleton.sha256,
            "contact_order": ["FL", "FR", "RL", "RR"],
            "contact_source": "unknown",
            "source": {
                "format": "legacy_ai4animation_dog27",
                "path": str(input_path),
                "sha256": _sha256_file(input_path),
                "fps": float(fps),
                "start_frame": start_frame,
                "end_frame": stop,
                "coordinate_transform": "Rz(0.47*pi) @ Rx(0.5*pi)",
                "license": "CC-BY-NC-4.0",
            },
            "created_by": {"gqmr_version": __version__},
        },
    )
