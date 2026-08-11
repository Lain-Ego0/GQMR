"""Safe, atomic `.gqmr` ZIP64 project persistence."""

from __future__ import annotations

import json
import hashlib
import os
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import ValidationError
from platformdirs import user_cache_path

from gqmr.project.model import ProjectDocument, ProjectError, ProjectResource, _utc_now
from gqmr.core.json import StrictJSONError, loads_strict_json

_BASE_MEMBERS = {"project.json", "edits.json"}
_MAX_MEMBERS = 4096
_MAX_METADATA_SIZE = 16 * 1024 * 1024
_MAX_TOTAL_SIZE = 16 * 1024 * 1024 * 1024


def _json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ProjectError(f"project data is not strict JSON: {error}") from error


def _documents(project: ProjectDocument) -> tuple[bytes, bytes]:
    document = project.model_dump(mode="json", exclude={"edits"})
    edits = [edit.model_dump(mode="json") for edit in project.edits]
    return _json_bytes(document), _json_bytes(edits)


def _write_archive(
    destination: Path,
    project: ProjectDocument,
    embedded_files: dict[str, Path],
) -> None:
    project_json, edits_json = _documents(project)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    try:
        with zipfile.ZipFile(
            temporary_name, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True
        ) as archive:
            archive.writestr("project.json", project_json)
            archive.writestr("edits.json", edits_json)
            for uri, source in sorted(embedded_files.items()):
                archive.write(source, uri)
        with open(temporary_name, "rb") as stream:
            os.fsync(stream.fileno())
        if destination.exists():
            backup = destination.with_suffix(destination.suffix + ".bak")
            shutil.copy2(destination, backup)
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


def _embedded_files(
    project: ProjectDocument,
    *,
    source_path: Path | None,
    cache_dir: Path,
) -> dict[str, Path]:
    if source_path is None or not source_path.is_file():
        raise ProjectError(
            "embedded resources require the original portable project archive"
        )
    result: dict[str, Path] = {}
    for resource_id, resource in project.resources.items():
        if resource.embedded:
            result[resource.uri] = materialize_resource(
                source_path, project, resource_id, cache_dir=cache_dir
            )
    return result


def save_project(
    path: str | Path,
    project: ProjectDocument,
    *,
    source_path: str | Path | None = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    updated = project.model_copy(update={"updated_at": _utc_now()})
    if any(resource.embedded for resource in updated.resources.values()):
        source = Path(source_path) if source_path is not None else destination
        with tempfile.TemporaryDirectory(
            prefix=f".{destination.name}.resources.", dir=destination.parent
        ) as temporary_directory:
            embedded = _embedded_files(
                updated, source_path=source, cache_dir=Path(temporary_directory)
            )
            _write_archive(destination, updated, embedded)
    else:
        _write_archive(destination, updated, {})
    return destination


def pack_project(
    path: str | Path,
    project: ProjectDocument,
    *,
    source_path: str | Path | None = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    resources: dict[str, ProjectResource] = {}
    embedded_files: dict[str, Path] = {}
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.resources.", dir=destination.parent
    ) as temporary_directory:
        portable_source = Path(source_path) if source_path is not None else None
        materialized = (
            _embedded_files(
                project,
                source_path=portable_source,
                cache_dir=Path(temporary_directory),
            )
            if any(resource.embedded for resource in project.resources.values())
            else {}
        )
        for resource_id, resource in project.resources.items():
            source = materialized.get(resource.uri, Path(resource.uri))
            if not source.is_file():
                raise ProjectError(f"cannot pack missing resource {source}")
            uri = f"embedded/{resource_id}/{source.name}"
            resources[resource_id] = resource.model_copy(
                update={"uri": uri, "embedded": True}
            )
            embedded_files[uri] = source
        packed = project.model_copy(
            update={"resources": resources, "updated_at": _utc_now()}
        )
        _write_archive(destination, packed, embedded_files)
    return destination


def _safe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts and "." not in path.parts


def load_project(path: str | Path) -> ProjectDocument:
    source = Path(path)
    if not source.is_file():
        raise ProjectError(f"project file does not exist: {source}")
    try:
        with zipfile.ZipFile(source) as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            if len(members) > _MAX_MEMBERS or len(names) != len(set(names)):
                raise ProjectError("project archive has too many or duplicate members")
            if not _BASE_MEMBERS.issubset(names):
                raise ProjectError("project archive is missing project.json or edits.json")
            total = 0
            for member in members:
                if (
                    member.is_dir()
                    or member.flag_bits & 0x1
                    or not _safe_member(member.filename)
                ):
                    raise ProjectError(f"unsafe project member {member.filename!r}")
                if member.filename not in _BASE_MEMBERS and not (
                    member.filename.startswith("embedded/")
                    or member.filename.startswith("thumbnails/")
                ):
                    raise ProjectError(f"unexpected project member {member.filename!r}")
                total += member.file_size
                if total > _MAX_TOTAL_SIZE:
                    raise ProjectError("project archive exceeds the size limit")
                if (
                    member.file_size > 1024 * 1024
                    and member.file_size > max(member.compress_size, 1) * 10_000
                ):
                    raise ProjectError("project member compression ratio exceeds the limit")
            if any(archive.getinfo(name).file_size > _MAX_METADATA_SIZE for name in _BASE_MEMBERS):
                raise ProjectError("project metadata exceeds the size limit")
            project_raw = loads_strict_json(archive.read("project.json").decode("utf-8"))
            edits_raw = loads_strict_json(archive.read("edits.json").decode("utf-8"))
            if not isinstance(project_raw, dict) or not isinstance(edits_raw, list):
                raise ProjectError("project.json/edits.json have invalid top-level types")
            project_raw["edits"] = edits_raw
            project = ProjectDocument.model_validate(project_raw)
            expected_embedded = {
                resource.uri
                for resource in project.resources.values()
                if resource.embedded
            }
            missing = expected_embedded - set(names)
            if missing:
                raise ProjectError(f"project is missing embedded resources: {sorted(missing)}")
            for resource in project.resources.values():
                if not resource.embedded:
                    continue
                member = archive.getinfo(resource.uri)
                if member.file_size != resource.size:
                    raise ProjectError(
                        f"embedded resource size mismatch: {resource.uri}"
                    )
                digest = hashlib.sha256()
                with archive.open(member) as stream:
                    while chunk := stream.read(1024 * 1024):
                        digest.update(chunk)
                if digest.hexdigest() != resource.sha256:
                    raise ProjectError(
                        f"embedded resource hash mismatch: {resource.uri}"
                    )
            return project
    except (
        OSError,
        UnicodeError,
        zipfile.BadZipFile,
        json.JSONDecodeError,
        StrictJSONError,
        ValidationError,
    ) as error:
        if isinstance(error, ProjectError):
            raise
        raise ProjectError(f"cannot load project {source}: {error}") from error


def materialize_resource(
    project_path: str | Path,
    project: ProjectDocument,
    resource_id: str,
    *,
    cache_dir: str | Path | None = None,
) -> Path:
    try:
        resource = project.resources[resource_id]
    except KeyError as error:
        raise ProjectError(f"unknown project resource {resource_id}") from error
    if not resource.embedded:
        path = Path(resource.uri)
        if not path.is_file() or path.stat().st_size != resource.size:
            raise ProjectError(f"external project resource is missing or changed: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != resource.sha256:
            raise ProjectError(f"external project resource hash changed: {path}")
        return path
    root = (
        Path(cache_dir)
        if cache_dir is not None
        else user_cache_path("gqmr") / "projects"
    )
    destination = root / resource.sha256 / Path(resource.uri).name
    if destination.is_file() and destination.stat().st_size == resource.size:
        digest = hashlib.sha256()
        with destination.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() == resource.sha256:
            return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        digest = hashlib.sha256()
        with zipfile.ZipFile(project_path) as archive, archive.open(resource.uri) as source:
            with os.fdopen(descriptor, "wb") as output:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
        if digest.hexdigest() != resource.sha256:
            raise ProjectError("embedded resource changed while materializing")
        os.replace(temporary_name, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return destination
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
