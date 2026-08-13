"""Incremental, timestamp-safe video inference for pose backend API v1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from gqmr.pose.api import (
    KeypointBatch,
    PoseBackendInfo,
    PoseBackendV1,
    PoseDataError,
    VideoFrameBatch,
)
from gqmr.sources.video import iter_video_frame_batches


class _NeverCancelled:
    @property
    def cancelled(self) -> bool:
        return False


def _strict_json_bytes(value: Any, *, label: str) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise PoseDataError(f"{label} is not strict JSON: {error}") from error


def _file_sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
    except OSError as error:
        raise PoseDataError(f"cannot hash source video: {error}") from error
    return digest.hexdigest(), size


def _padded_batch(batch: VideoFrameBatch, minimum_frames: int) -> VideoFrameBatch:
    if len(batch.frames) >= minimum_frames:
        return batch
    missing = minimum_frames - len(batch.frames)
    if len(batch.pts) > 1:
        step = max(1, int(np.median(np.diff(batch.pts))))
    else:
        step = 1
    extra_pts = batch.pts[-1] + step * np.arange(1, missing + 1, dtype=np.int64)
    return VideoFrameBatch(
        frames=np.concatenate(
            [batch.frames, np.repeat(batch.frames[-1:], missing, axis=0)], axis=0
        ),
        pts=np.concatenate([batch.pts, extra_pts]),
        time_base_numerator=batch.time_base_numerator,
        time_base_denominator=batch.time_base_denominator,
    )


def _trim_keypoints(batch: KeypointBatch, frames: int) -> KeypointBatch:
    return KeypointBatch(
        timestamps=batch.timestamps[:frames],
        keypoint_names=batch.keypoint_names,
        instance_ids=batch.instance_ids,
        positions=batch.positions[:frames],
        confidence=batch.confidence[:frames],
        valid_mask=batch.valid_mask[:frames],
        coordinate_frame=batch.coordinate_frame,
        metadata=batch.metadata,
    )


def _validate_output(output: KeypointBatch, source: VideoFrameBatch) -> None:
    if len(output.timestamps) != len(source.frames):
        raise PoseDataError("pose backend must return exactly one result per video frame")
    if not np.allclose(output.timestamps, source.timestamps, rtol=0.0, atol=1e-7):
        raise PoseDataError("pose backend timestamps do not match video presentation timestamps")


def infer_video_with_backend(
    backend: PoseBackendV1,
    video_path: str | Path,
    *,
    backend_info: PoseBackendInfo | None = None,
    backend_config: dict[str, Any] | None = None,
    batch_size: int = 16,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
    max_frames: int | None = None,
) -> KeypointBatch:
    """Run an already-loaded backend over a video and merge validated batches."""

    info = backend_info or backend.describe()
    minimum_batch, maximum_batch = info.batch_range
    if minimum_batch <= 0 or maximum_batch < minimum_batch:
        raise PoseDataError("pose backend declared an invalid batch range")
    if not minimum_batch <= batch_size <= maximum_batch:
        raise PoseDataError(
            f"batch_size must be within backend range [{minimum_batch},{maximum_batch}]"
        )
    config = backend_config or {}
    config_sha256 = hashlib.sha256(
        _strict_json_bytes(config, label="pose backend config")
    ).hexdigest()
    outputs: list[KeypointBatch] = []
    metadata_hashes: list[str] = []
    cancel = _NeverCancelled()
    for video_batch in iter_video_frame_batches(
        video_path,
        batch_size=batch_size,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        max_frames=max_frames,
    ):
        inference_batch = _padded_batch(video_batch, minimum_batch)
        output = backend.infer(inference_batch, cancel)
        _validate_output(output, inference_batch)
        output = _trim_keypoints(output, len(video_batch.frames))
        outputs.append(output)
        metadata_hashes.append(
            hashlib.sha256(
                _strict_json_bytes(output.metadata, label="pose backend metadata")
            ).hexdigest()
        )
    first = outputs[0]
    for output in outputs[1:]:
        if (
            output.keypoint_names != first.keypoint_names
            or output.instance_ids != first.instance_ids
            or output.dimensions != first.dimensions
            or output.coordinate_frame != first.coordinate_frame
        ):
            raise PoseDataError("pose backend output schema changed between video batches")
    timestamps = np.concatenate([output.timestamps for output in outputs])
    if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0.0):
        raise PoseDataError("pose backend produced duplicate or out-of-order video timestamps")
    source_path = Path(video_path)
    source_sha256, source_size = _file_sha256(source_path)
    metadata = {
        "format": "gqmr_video_pose_v1",
        "backend": {
            "api_version": info.api_version,
            "name": info.name,
            "package": info.package,
            "package_version": info.package_version,
            "skeleton_ids": list(info.skeleton_ids),
        },
        "backend_config_sha256": config_sha256,
        "backend_output_metadata_sha256": metadata_hashes,
        "source_video": {
            "path": str(source_path),
            "sha256": source_sha256,
            "size_bytes": source_size,
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,
            "max_frames": max_frames,
        },
        "inference": {"batch_size": batch_size, "batches": len(outputs)},
    }
    return KeypointBatch(
        timestamps=timestamps,
        keypoint_names=first.keypoint_names,
        instance_ids=first.instance_ids,
        positions=np.concatenate([output.positions for output in outputs], axis=0),
        confidence=np.concatenate([output.confidence for output in outputs], axis=0),
        valid_mask=np.concatenate([output.valid_mask for output in outputs], axis=0),
        coordinate_frame=first.coordinate_frame,
        metadata=metadata,
    )
