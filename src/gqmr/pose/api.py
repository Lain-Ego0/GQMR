"""Public pose plugin API v1."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gqmr.core.errors import GQMRError


class PoseDataError(GQMRError, ValueError):
    """Raised when pose-plugin data violates API v1."""


@dataclass(frozen=True, slots=True)
class VideoFrameBatch:
    frames: ArrayLike
    pts: ArrayLike
    time_base_numerator: int
    time_base_denominator: int

    def __post_init__(self) -> None:
        frames = np.ascontiguousarray(self.frames, dtype=np.uint8)
        pts = np.ascontiguousarray(self.pts, dtype=np.int64)
        if frames.ndim != 4 or frames.shape[-1] not in {3, 4}:
            raise PoseDataError("video frames must have shape [B,H,W,3|4]")
        if pts.shape != (len(frames),):
            raise PoseDataError("video PTS must match frame count")
        if self.time_base_numerator <= 0 or self.time_base_denominator <= 0:
            raise PoseDataError("video time base must be positive")
        object.__setattr__(self, "frames", frames)
        object.__setattr__(self, "pts", pts)

    @property
    def timestamps(self) -> NDArray[np.float64]:
        return self.pts.astype(np.float64) * (
            self.time_base_numerator / self.time_base_denominator
        )


@dataclass(frozen=True, slots=True)
class KeypointBatch:
    timestamps: ArrayLike
    keypoint_names: tuple[str, ...]
    instance_ids: tuple[str, ...]
    positions: ArrayLike
    confidence: ArrayLike
    valid_mask: ArrayLike
    coordinate_frame: str
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        timestamps = np.ascontiguousarray(self.timestamps, dtype=np.float64)
        positions = np.ascontiguousarray(self.positions, dtype=np.float32)
        confidence = np.ascontiguousarray(self.confidence, dtype=np.float32)
        valid = np.ascontiguousarray(self.valid_mask, dtype=np.bool_)
        if timestamps.ndim != 1 or not len(timestamps):
            raise PoseDataError("keypoint timestamps must be a non-empty 1D array")
        if not np.all(np.isfinite(timestamps)) or (
            len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0.0)
        ):
            raise PoseDataError("keypoint timestamps must be finite and strictly increasing")
        if len(set(self.keypoint_names)) != len(self.keypoint_names) or any(
            not name for name in self.keypoint_names
        ):
            raise PoseDataError("keypoint names must be non-empty and unique")
        if len(set(self.instance_ids)) != len(self.instance_ids) or any(
            not value for value in self.instance_ids
        ):
            raise PoseDataError("instance IDs must be non-empty and unique")
        expected_prefix = (
            len(timestamps),
            len(self.instance_ids),
            len(self.keypoint_names),
        )
        if positions.ndim != 4 or positions.shape[:3] != expected_prefix or positions.shape[3] not in {2, 3}:
            raise PoseDataError("positions must have shape [T,I,K,2|3]")
        if confidence.shape != expected_prefix or valid.shape != expected_prefix:
            raise PoseDataError("confidence/valid mask shape does not match positions")
        if np.any(~np.isfinite(confidence)) or np.any((confidence < 0.0) | (confidence > 1.0)):
            raise PoseDataError("confidence must be finite in [0,1]")
        if np.any(valid & ~np.all(np.isfinite(positions), axis=-1)):
            raise PoseDataError("valid keypoints must have finite positions")
        if not self.coordinate_frame:
            raise PoseDataError("coordinate_frame is required")
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "valid_mask", valid)

    @property
    def dimensions(self) -> int:
        return self.positions.shape[-1]


@dataclass(frozen=True, slots=True)
class PoseBackendInfo:
    api_version: Literal[1]
    name: str
    package: str
    package_version: str
    skeleton_ids: tuple[str, ...]
    dimensions: tuple[Literal[2, 3], ...]
    multi_instance: bool
    batch_range: tuple[int, int]
    devices: tuple[str, ...]
    output_coordinate_frame: str


class CancelToken(Protocol):
    @property
    def cancelled(self) -> bool: ...


class PoseBackendV1(Protocol):
    api_version: Literal[1]

    def describe(self) -> PoseBackendInfo: ...

    def load(self, config: dict[str, Any]) -> None: ...

    def infer(self, batch: VideoFrameBatch, cancel: CancelToken) -> KeypointBatch: ...

    def close(self) -> None: ...


def discover_pose_backends() -> dict[str, type[PoseBackendV1]]:
    discovered: dict[str, type[PoseBackendV1]] = {}
    for entry_point in entry_points(group="gqmr.pose_backends"):
        backend = entry_point.load()
        if getattr(backend, "api_version", None) != 1:
            continue
        discovered[entry_point.name] = backend
    return discovered
