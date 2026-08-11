"""Strict project.json and edits.json v1 models."""

from __future__ import annotations

import hashlib
import mimetypes
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from gqmr import __version__
from gqmr.core.errors import GQMRError


class ProjectError(GQMRError, ValueError):
    """Raised when a project document or resource is invalid."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _resource_snapshot(path: Path) -> tuple[os.stat_result, str]:
    before = path.stat()
    digest = _sha256_file(path)
    after = path.stat()
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise ProjectError(f"project resource changed while importing: {path}")
    return after, digest


class ProjectResource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    resource_id: str
    uri: str = Field(min_length=1)
    media_type: str = Field(min_length=1)
    size: int = Field(ge=0)
    mtime_ns: int = Field(ge=0)
    sha256: str
    embedded: bool = False

    @field_validator("resource_id")
    @classmethod
    def valid_uuid(cls, value: str) -> str:
        try:
            uuid.UUID(value)
        except ValueError as error:
            raise ValueError("resource_id must be a UUID") from error
        return value

    @field_validator("sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("sha256 must be 64 lowercase hexadecimal characters")
        return value

    @field_validator("uri")
    @classmethod
    def safe_uri(cls, value: str) -> str:
        if "\x00" in value:
            raise ValueError("resource URI contains NUL")
        return value

    @model_validator(mode="after")
    def embedded_uri_is_internal(self) -> "ProjectResource":
        if self.embedded and not self.uri.startswith("embedded/"):
            raise ValueError("embedded resource URI must start with embedded/")
        return self


class EditCommand(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    command_id: str
    kind: Literal[
        "trim", "time_scale", "root_transform", "contact_override", "resample"
    ]
    resource_id: str
    parameters: dict[str, Any]
    created_at: str

    @field_validator("command_id", "resource_id")
    @classmethod
    def valid_uuid(cls, value: str) -> str:
        try:
            uuid.UUID(value)
        except ValueError as error:
            raise ValueError("command/resource ID must be a UUID") from error
        return value


class ProjectDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["gqmr.project"] = "gqmr.project"
    schema_version: Literal["1.0"] = "1.0"
    project_id: str
    created_at: str
    updated_at: str
    gqmr_version: str
    resources: dict[str, ProjectResource] = Field(default_factory=dict)
    active_animal_motion: str | None = None
    robot: dict[str, Any] = Field(default_factory=dict)
    mapping: dict[str, Any] = Field(default_factory=dict)
    retarget: dict[str, Any] = Field(default_factory=dict)
    timeline: dict[str, Any] = Field(
        default_factory=lambda: {"start": 0.0, "end": 0.0, "loop": False}
    )
    export_presets: list[dict[str, Any]] = Field(default_factory=list)
    ui_state: dict[str, Any] = Field(default_factory=dict)
    edits: tuple[EditCommand, ...] = ()

    @field_validator("project_id")
    @classmethod
    def valid_uuid(cls, value: str) -> str:
        try:
            uuid.UUID(value)
        except ValueError as error:
            raise ValueError("project_id must be a UUID") from error
        return value

    @field_validator("created_at", "updated_at")
    @classmethod
    def valid_timestamp(cls, value: str) -> str:
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("project timestamps must be RFC3339") from error
        return value

    @model_validator(mode="after")
    def validate_references(self) -> "ProjectDocument":
        for key, resource in self.resources.items():
            if key != resource.resource_id:
                raise ValueError("resource mapping key must equal resource_id")
        for field in (
            self.active_animal_motion,
            self.retarget.get("active_robot_motion"),
        ):
            if field is not None and field not in self.resources:
                raise ValueError("active motion references an unknown resource")
        if any(edit.resource_id not in self.resources for edit in self.edits):
            raise ValueError("edit references an unknown resource")
        timeline_keys = {"start", "end", "loop"}
        if set(self.timeline) != timeline_keys:
            raise ValueError("timeline must contain start, end, loop")
        start, end = self.timeline["start"], self.timeline["end"]
        if (
            not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
            or start < 0.0
            or end < start
            or not isinstance(self.timeline["loop"], bool)
        ):
            raise ValueError("timeline values are invalid")
        return self


def new_project() -> ProjectDocument:
    now = _utc_now()
    return ProjectDocument(
        project_id=str(uuid.uuid4()),
        created_at=now,
        updated_at=now,
        gqmr_version=__version__,
    )


def add_resource(
    project: ProjectDocument,
    path: str | Path,
    *,
    media_type: str | None = None,
    make_active: Literal["animal", "robot"] | None = None,
) -> ProjectDocument:
    resource_path = Path(path).expanduser().resolve(strict=True)
    if not resource_path.is_file():
        raise ProjectError(f"project resource is not a file: {resource_path}")
    stat, digest = _resource_snapshot(resource_path)
    for resource_id, existing in project.resources.items():
        if (
            existing.size == stat.st_size
            and existing.sha256 == digest
        ):
            update: dict[str, Any] = {"updated_at": _utc_now()}
            if make_active == "animal":
                update["active_animal_motion"] = resource_id
            elif make_active == "robot":
                retarget = dict(project.retarget)
                retarget["active_robot_motion"] = resource_id
                update["retarget"] = retarget
            return project.model_copy(update=update)
    resource_id = str(uuid.uuid4())
    resource = ProjectResource(
        resource_id=resource_id,
        uri=str(resource_path),
        media_type=media_type
        or mimetypes.guess_type(resource_path.name)[0]
        or "application/octet-stream",
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        sha256=digest,
        embedded=False,
    )
    resources = dict(project.resources)
    resources[resource_id] = resource
    update: dict[str, Any] = {"resources": resources, "updated_at": _utc_now()}
    if make_active == "animal":
        update["active_animal_motion"] = resource_id
    elif make_active == "robot":
        retarget = dict(project.retarget)
        retarget["active_robot_motion"] = resource_id
        update["retarget"] = retarget
    return project.model_copy(update=update)
