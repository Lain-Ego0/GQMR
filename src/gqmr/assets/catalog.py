"""Load and validate built-in asset manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib import resources
from pathlib import PurePosixPath
from typing import Any, Mapping

from gqmr.core.errors import AssetError

_ASSET_IDS = ("unitree-go2", "unitree-b2")
_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True, slots=True)
class AssetFile:
    path: str
    sha256: str
    size: int


@dataclass(frozen=True, slots=True)
class AssetSpec:
    asset_id: str
    display_name: str
    repository: str
    commit: str
    archive_url: str
    archive_sha256: str
    archive_prefix: str
    license_spdx: str
    license_path: str
    model_path: str
    model_sha256: str
    files: tuple[AssetFile, ...]
    document: Mapping[str, Any]

    @property
    def total_size(self) -> int:
        return sum(item.size for item in self.files)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX for character in value)
    )


def _safe_relative_path(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise AssetError(f"manifest {field} must be a non-empty POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise AssetError(f"manifest {field} is unsafe: {value!r}")
    return value


def _canonical_json(document: Mapping[str, Any]) -> bytes:
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def manifest_sha256(spec: AssetSpec) -> str:
    return hashlib.sha256(_canonical_json(spec.document)).hexdigest()


def _calculate_model_sha256(files: tuple[AssetFile, ...], license_path: str) -> str:
    digest = hashlib.sha256()
    for item in files:
        if item.path == license_path:
            continue
        digest.update(item.path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.sha256.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(item.size).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _parse_manifest(document: Mapping[str, Any]) -> AssetSpec:
    if document.get("schema_id") != "gqmr.asset_manifest" or document.get(
        "schema_version"
    ) != 1:
        raise AssetError("unsupported built-in asset manifest schema")
    source = document.get("source")
    license_info = document.get("license")
    raw_files = document.get("files")
    if not isinstance(source, Mapping) or not isinstance(license_info, Mapping):
        raise AssetError("asset manifest source/license must be objects")
    if not isinstance(raw_files, list) or not raw_files:
        raise AssetError("asset manifest files must be a non-empty list")
    files: list[AssetFile] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_files):
        if not isinstance(raw, Mapping):
            raise AssetError(f"asset manifest file {index} must be an object")
        path = _safe_relative_path(raw.get("path"), field=f"files[{index}].path")
        sha256 = raw.get("sha256")
        size = raw.get("size")
        if path in seen:
            raise AssetError(f"asset manifest has duplicate file path {path!r}")
        if not _is_sha256(sha256) or not isinstance(size, int) or size < 0:
            raise AssetError(f"asset manifest file {path!r} has invalid hash/size")
        seen.add(path)
        files.append(AssetFile(path=path, sha256=sha256, size=size))
    files_tuple = tuple(files)
    license_path = _safe_relative_path(license_info.get("path"), field="license.path")
    model_path = _safe_relative_path(document.get("model_path"), field="model_path")
    if license_path not in seen or model_path not in seen:
        raise AssetError("asset manifest license/model path is not in files")
    for field, value in (
        ("source.archive_sha256", source.get("archive_sha256")),
        ("model_sha256", document.get("model_sha256")),
    ):
        if not _is_sha256(value):
            raise AssetError(f"asset manifest {field} is not a SHA-256")
    calculated = _calculate_model_sha256(files_tuple, license_path)
    if calculated != document["model_sha256"]:
        raise AssetError("asset manifest model_sha256 is internally inconsistent")
    return AssetSpec(
        asset_id=str(document["asset_id"]),
        display_name=str(document["display_name"]),
        repository=str(source["repository"]),
        commit=str(source["commit"]),
        archive_url=str(source["archive_url"]),
        archive_sha256=str(source["archive_sha256"]),
        archive_prefix=str(source["archive_prefix"]),
        license_spdx=str(license_info["spdx"]),
        license_path=license_path,
        model_path=model_path,
        model_sha256=str(document["model_sha256"]),
        files=files_tuple,
        document=dict(document),
    )


def available_assets() -> tuple[str, ...]:
    return _ASSET_IDS


def get_asset_spec(asset_id: str) -> AssetSpec:
    if asset_id not in _ASSET_IDS:
        raise AssetError(
            f"unknown asset {asset_id!r}; available: {', '.join(_ASSET_IDS)}"
        )
    resource = resources.files("gqmr.assets.manifests").joinpath(f"{asset_id}.json")
    try:
        document = json.loads(resource.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AssetError(f"cannot load built-in manifest for {asset_id}: {error}") from error
    if not isinstance(document, dict):
        raise AssetError(f"built-in manifest for {asset_id} is not an object")
    spec = _parse_manifest(document)
    if spec.asset_id != asset_id:
        raise AssetError(f"built-in manifest ID mismatch for {asset_id}")
    return spec
