"""Spawn-isolated execution for trusted pose and exporter plugins."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from gqmr.exporters.api import discover_exporters
from gqmr.jobs import ProcessJob
from gqmr.jobs.process import ProcessJobError
from gqmr.pose import (
    KeypointBatch,
    VideoFrameBatch,
    discover_pose_backends,
    infer_video_with_backend,
)


class _NeverCancelled:
    @property
    def cancelled(self) -> bool:
        return False


def _pose_child(entry_name: str, config: dict[str, Any], batch: VideoFrameBatch) -> KeypointBatch:
    backends = discover_pose_backends()
    if entry_name not in backends:
        raise ValueError(f"pose backend {entry_name!r} is not installed")
    backend = backends[entry_name]()
    try:
        backend.load(config)
        return backend.infer(batch, _NeverCancelled())
    finally:
        backend.close()


def _pose_video_child(
    entry_name: str,
    config: dict[str, Any],
    video_path: str,
    batch_size: int,
    start_seconds: float,
    end_seconds: float | None,
    max_frames: int | None,
) -> KeypointBatch:
    backends = discover_pose_backends()
    if entry_name not in backends:
        raise ValueError(f"pose backend {entry_name!r} is not installed")
    backend = backends[entry_name]()
    try:
        info = backend.describe()
        backend.load(config)
        return infer_video_with_backend(
            backend,
            video_path,
            backend_info=info,
            backend_config=config,
            batch_size=batch_size,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            max_frames=max_frames,
        )
    finally:
        backend.close()


def _export_child(
    entry_name: str,
    motion,
    destination: str,
    config: dict[str, Any],
):
    exporters = discover_exporters()
    if entry_name not in exporters:
        raise ValueError(f"exporter {entry_name!r} is not installed")
    exporter = exporters[entry_name]()
    exporter.validate(motion, config)
    return exporter.export(motion, Path(destination), config, _NeverCancelled())


def run_pose_backend_plugin(
    entry_name: str,
    config: dict[str, Any],
    batch: VideoFrameBatch,
    *,
    timeout: float | None = None,
) -> KeypointBatch:
    with ProcessJob(
        "gqmr.plugins.runner:_pose_child", args=(entry_name, config, batch)
    ) as job:
        return job.result(timeout=timeout)


def run_pose_video_backend_plugin(
    entry_name: str,
    config: dict[str, Any],
    video_path: str | Path,
    *,
    batch_size: int = 16,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
    max_frames: int | None = None,
    timeout: float | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> KeypointBatch:
    with ProcessJob(
        "gqmr.plugins.runner:_pose_video_child",
        args=(
            entry_name,
            config,
            str(video_path),
            batch_size,
            start_seconds,
            end_seconds,
            max_frames,
        ),
    ) as job:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if cancelled is not None and cancelled():
                raise ProcessJobError("pose video job was cancelled")
            wait_seconds = 0.2
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise ProcessJobError("pose video job timed out")
                wait_seconds = min(wait_seconds, remaining)
            try:
                return job.result(timeout=wait_seconds)
            except ProcessJobError as error:
                if str(error) != "job result timed out":
                    raise
                if not job.running:
                    raise ProcessJobError(
                        "pose video job exited without returning a result"
                    ) from error


def run_exporter_plugin(
    entry_name: str,
    motion,
    destination: str | Path,
    config: dict[str, Any],
    *,
    timeout: float | None = None,
):
    with ProcessJob(
        "gqmr.plugins.runner:_export_child",
        args=(entry_name, motion, str(destination), config),
    ) as job:
        return job.result(timeout=timeout)
