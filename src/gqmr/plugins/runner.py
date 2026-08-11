"""Spawn-isolated execution for trusted pose and exporter plugins."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gqmr.exporters.api import discover_exporters
from gqmr.jobs import ProcessJob
from gqmr.pose import KeypointBatch, VideoFrameBatch, discover_pose_backends


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
