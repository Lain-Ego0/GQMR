"""Public exporter plugin API v1."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Literal, Protocol

from gqmr.core.motion import RobotMotion


@dataclass(frozen=True, slots=True)
class ExporterInfo:
    api_version: Literal[1]
    name: str
    package: str
    package_version: str
    formats: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExportReport:
    destination: Path
    format_name: str
    frames: int
    warnings: tuple[str, ...] = ()


class CancelToken(Protocol):
    @property
    def cancelled(self) -> bool: ...


class ExporterV1(Protocol):
    api_version: Literal[1]

    def describe(self) -> ExporterInfo: ...

    def validate(self, motion: RobotMotion, config: dict[str, Any]) -> dict[str, Any]: ...

    def export(
        self,
        motion: RobotMotion,
        destination: Path,
        config: dict[str, Any],
        cancel: CancelToken,
    ) -> ExportReport: ...


def discover_exporters() -> dict[str, type[ExporterV1]]:
    discovered: dict[str, type[ExporterV1]] = {}
    for entry_point in entry_points(group="gqmr.exporters"):
        exporter = entry_point.load()
        if getattr(exporter, "api_version", None) == 1:
            discovered[entry_point.name] = exporter
    return discovered
