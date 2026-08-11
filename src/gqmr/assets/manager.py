"""Install, verify, pack, and unpack trusted robot assets."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tarfile
import tempfile
import urllib.error
import urllib.request
import uuid
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Mapping

from gqmr.assets.catalog import AssetFile, AssetSpec, get_asset_spec, manifest_sha256
from gqmr.core.errors import AssetError

_DOWNLOAD_LIMIT = 512 * 1024 * 1024
_ARCHIVE_MEMBER_LIMIT = 100_000
_ARCHIVE_UNCOMPRESSED_LIMIT = 4 * 1024 * 1024 * 1024
_PACK_MEMBER_LIMIT = 256
_PACK_UNCOMPRESSED_LIMIT = 512 * 1024 * 1024
_INSTALL_METADATA = "gqmr-install.json"


@dataclass(frozen=True, slots=True)
class AssetStatus:
    asset_id: str
    state: str
    valid: bool
    install_path: str
    model_path: str
    model_sha256: str
    source_commit: str
    license: str
    expected_size: int
    installed_size: int
    missing_files: tuple[str, ...] = ()
    corrupt_files: tuple[str, ...] = ()
    unexpected_files: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_spec(asset: str | AssetSpec) -> AssetSpec:
    return get_asset_spec(asset) if isinstance(asset, str) else asset


def default_cache_root() -> Path:
    try:
        from platformdirs import user_cache_path
    except ImportError as error:
        raise AssetError(
            "platformdirs is required to locate the GQMR asset cache; "
            "install project dependencies or pass --cache-dir"
        ) from error
    return user_cache_path("gqmr")


def _cache_root(cache_dir: str | os.PathLike[str] | None) -> Path:
    return Path(cache_dir) if cache_dir is not None else default_cache_root()


def _install_path(spec: AssetSpec, cache_root: Path) -> Path:
    return cache_root / "assets" / spec.asset_id / spec.commit


def _sha256_stream(stream: BinaryIO, destination: BinaryIO | None = None) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while True:
        chunk = stream.read(1024 * 1024)
        if not chunk:
            break
        size += len(chunk)
        digest.update(chunk)
        if destination is not None:
            destination.write(chunk)
    return digest.hexdigest(), size


def _sha256_file(path: Path) -> tuple[str, int]:
    with path.open("rb") as stream:
        return _sha256_stream(stream)


def status_asset(
    asset: str | AssetSpec, *, cache_dir: str | os.PathLike[str] | None = None
) -> AssetStatus:
    spec = _resolve_spec(asset)
    install_path = _install_path(spec, _cache_root(cache_dir))
    if not install_path.is_dir():
        return AssetStatus(
            asset_id=spec.asset_id,
            state="missing",
            valid=False,
            install_path=str(install_path),
            model_path=str(install_path / spec.model_path),
            model_sha256=spec.model_sha256,
            source_commit=spec.commit,
            license=spec.license_spdx,
            expected_size=spec.total_size,
            installed_size=0,
            missing_files=tuple(item.path for item in spec.files),
        )
    missing: list[str] = []
    corrupt: list[str] = []
    installed_size = 0
    expected_paths = {item.path for item in spec.files}
    for item in spec.files:
        path = install_path / item.path
        if not path.is_file() or path.is_symlink():
            missing.append(item.path)
            continue
        try:
            actual_hash, actual_size = _sha256_file(path)
        except OSError:
            corrupt.append(item.path)
            continue
        installed_size += actual_size
        if actual_size != item.size or actual_hash != item.sha256:
            corrupt.append(item.path)
    actual_paths: set[str] = set()
    unsafe_paths: set[str] = set()
    for path in install_path.rglob("*"):
        relative = path.relative_to(install_path).as_posix()
        if path.is_symlink():
            unsafe_paths.add(relative)
        elif path.is_file():
            actual_paths.add(relative)
    unexpected = sorted(actual_paths - expected_paths - {_INSTALL_METADATA})
    unexpected.extend(sorted(unsafe_paths))
    metadata_path = install_path / _INSTALL_METADATA
    try:
        metadata = _strict_json(metadata_path.read_bytes(), name=_INSTALL_METADATA)
    except (OSError, AssetError):
        corrupt.append(_INSTALL_METADATA)
    else:
        if metadata != _installation_document(spec):
            corrupt.append(_INSTALL_METADATA)
    valid = not missing and not corrupt and not unexpected
    return AssetStatus(
        asset_id=spec.asset_id,
        state="ok" if valid else "corrupt",
        valid=valid,
        install_path=str(install_path),
        model_path=str(install_path / spec.model_path),
        model_sha256=spec.model_sha256,
        source_commit=spec.commit,
        license=spec.license_spdx,
        expected_size=spec.total_size,
        installed_size=installed_size,
        missing_files=tuple(missing),
        corrupt_files=tuple(corrupt),
        unexpected_files=tuple(unexpected),
    )


def _download_archive(spec: AssetSpec, destination: Path) -> None:
    request = urllib.request.Request(
        spec.archive_url, headers={"User-Agent": "GQMR asset installer/0.0.1"}
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response, destination.open(
            "wb"
        ) as output:
            total = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > _DOWNLOAD_LIMIT:
                    raise AssetError("asset download exceeds the 512 MiB safety limit")
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
    except (OSError, urllib.error.URLError) as error:
        raise AssetError(f"cannot download {spec.asset_id}: {error}") from error


def _validate_archive_members(archive: tarfile.TarFile) -> dict[str, tarfile.TarInfo]:
    members = archive.getmembers()
    if len(members) > _ARCHIVE_MEMBER_LIMIT:
        raise AssetError("source archive contains too many members")
    result: dict[str, tarfile.TarInfo] = {}
    total_size = 0
    for member in members:
        name = member.name
        path = PurePosixPath(name)
        if name in result:
            raise AssetError(f"source archive has duplicate member {name!r}")
        if path.is_absolute() or ".." in path.parts:
            raise AssetError(f"source archive has unsafe path {name!r}")
        if not (member.isfile() or member.isdir()):
            raise AssetError(f"source archive has unsupported member type {name!r}")
        if member.isfile():
            total_size += member.size
            if total_size > _ARCHIVE_UNCOMPRESSED_LIMIT:
                raise AssetError("source archive exceeds the uncompressed safety limit")
        result[name] = member
    return result


def _extract_tar_to_directory(spec: AssetSpec, archive_path: Path, destination: Path) -> None:
    try:
        archive_hash, archive_size = _sha256_file(archive_path)
    except OSError as error:
        raise AssetError(f"cannot read source archive {archive_path}: {error}") from error
    if archive_size > _DOWNLOAD_LIMIT or archive_hash != spec.archive_sha256:
        raise AssetError(
            f"archive hash mismatch for {spec.asset_id}: expected "
            f"{spec.archive_sha256}, got {archive_hash}"
        )
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            members = _validate_archive_members(archive)
            for item in spec.files:
                source_name = spec.archive_prefix + item.path
                member = members.get(source_name)
                if member is None or not member.isfile():
                    raise AssetError(f"source archive is missing {item.path!r}")
                if member.size != item.size:
                    raise AssetError(f"source archive size mismatch for {item.path!r}")
                source = archive.extractfile(member)
                if source is None:
                    raise AssetError(f"cannot read source archive member {item.path!r}")
                target = destination / item.path
                target.parent.mkdir(parents=True, exist_ok=True)
                with source, target.open("wb") as output:
                    actual_hash, actual_size = _sha256_stream(source, output)
                    output.flush()
                    os.fsync(output.fileno())
                if actual_size != item.size or actual_hash != item.sha256:
                    raise AssetError(f"file hash mismatch for {item.path!r}")
    except (tarfile.TarError, OSError) as error:
        if isinstance(error, AssetError):
            raise
        raise AssetError(f"cannot read source archive: {error}") from error


def _installation_document(spec: AssetSpec) -> dict[str, Any]:
    return {
        "schema_id": "gqmr.asset_installation",
        "schema_version": 1,
        "asset_id": spec.asset_id,
        "source_commit": spec.commit,
        "manifest_sha256": manifest_sha256(spec),
        "model_sha256": spec.model_sha256,
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    payload = json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    with path.open("wb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())


def _activate_install(temp_directory: Path, install_path: Path, *, repair: bool) -> None:
    install_path.parent.mkdir(parents=True, exist_ok=True)
    backup: Path | None = None
    if install_path.exists():
        if not repair:
            raise AssetError(
                f"asset path already exists but is invalid: {install_path}; use --repair"
            )
        backup = install_path.with_name(f"{install_path.name}.bak-{uuid.uuid4().hex}")
        os.replace(install_path, backup)
    try:
        os.replace(temp_directory, install_path)
        _fsync_directory(install_path.parent)
    except BaseException:
        if backup is not None and not install_path.exists():
            os.replace(backup, install_path)
        raise
    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)


def install_asset(
    asset: str | AssetSpec,
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    archive_path: str | os.PathLike[str] | None = None,
    repair: bool = False,
) -> AssetStatus:
    spec = _resolve_spec(asset)
    cache_root = _cache_root(cache_dir)
    current = status_asset(spec, cache_dir=cache_root)
    if current.valid:
        return current
    cache_root.mkdir(parents=True, exist_ok=True)
    temporary_archive: Path | None = None
    temporary_directory: Path | None = None
    try:
        if archive_path is None:
            descriptor, name = tempfile.mkstemp(
                prefix=f".{spec.asset_id}.", suffix=".tar.gz", dir=cache_root
            )
            os.close(descriptor)
            temporary_archive = Path(name)
            _download_archive(spec, temporary_archive)
            source_archive = temporary_archive
        else:
            source_archive = Path(archive_path)
        install_path = _install_path(spec, cache_root)
        install_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_directory = Path(
            tempfile.mkdtemp(prefix=f".{spec.commit}.", dir=install_path.parent)
        )
        _extract_tar_to_directory(spec, source_archive, temporary_directory)
        _write_json(temporary_directory / _INSTALL_METADATA, _installation_document(spec))
        _activate_install(temporary_directory, install_path, repair=repair)
    except BaseException:
        if temporary_directory is not None:
            shutil.rmtree(temporary_directory, ignore_errors=True)
        raise
    finally:
        if temporary_archive is not None:
            temporary_archive.unlink(missing_ok=True)
    result = status_asset(spec, cache_dir=cache_root)
    if not result.valid:
        raise AssetError(f"installed asset failed verification: {result.to_dict()}")
    return result


def _canonical_json(document: Mapping[str, Any]) -> bytes:
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def pack_asset(
    asset: str | AssetSpec,
    destination: str | os.PathLike[str],
    *,
    cache_dir: str | os.PathLike[str] | None = None,
) -> Path:
    spec = _resolve_spec(asset)
    cache_root = _cache_root(cache_dir)
    status = status_asset(spec, cache_dir=cache_root)
    if not status.valid:
        raise AssetError(f"cannot pack unverified asset {spec.asset_id}: {status.state}")
    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    pack_document = {
        "schema_id": "gqmr.asset_pack",
        "schema_version": 1,
        "asset_id": spec.asset_id,
        "manifest_sha256": manifest_sha256(spec),
        "model_sha256": spec.model_sha256,
    }
    try:
        with zipfile.ZipFile(
            temporary_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True
        ) as archive:
            archive.writestr("pack.json", _canonical_json(pack_document))
            archive.writestr("manifest.json", _canonical_json(spec.document))
            root = Path(status.install_path)
            for item in spec.files:
                archive.write(root / item.path, arcname=f"files/{item.path}")
        with temporary_path.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_path, output_path)
        _fsync_directory(output_path.parent)
    except BaseException as error:
        temporary_path.unlink(missing_ok=True)
        if isinstance(error, OSError):
            raise AssetError(f"cannot create asset pack {output_path}: {error}") from error
        raise
    return output_path


def _strict_json(payload: bytes, *, name: str) -> dict[str, Any]:
    def object_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AssetError(f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=object_hook,
            parse_constant=lambda value: (_ for _ in ()).throw(
                AssetError(f"{name} contains non-finite value {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AssetError(f"invalid {name}: {error}") from error
    if not isinstance(value, dict):
        raise AssetError(f"{name} must be a JSON object")
    return value


def _inspect_pack(archive: zipfile.ZipFile, spec: AssetSpec) -> dict[str, zipfile.ZipInfo]:
    members = archive.infolist()
    if len(members) > _PACK_MEMBER_LIMIT:
        raise AssetError("asset pack contains too many members")
    names = [member.filename for member in members]
    if len(names) != len(set(names)):
        raise AssetError("asset pack contains duplicate member names")
    expected = {"pack.json", "manifest.json"} | {
        f"files/{item.path}" for item in spec.files
    }
    if set(names) != expected:
        raise AssetError("asset pack member set does not match the trusted manifest")
    total = 0
    result: dict[str, zipfile.ZipInfo] = {}
    for member in members:
        path = PurePosixPath(member.filename)
        mode = member.external_attr >> 16
        file_type = stat.S_IFMT(mode)
        if path.is_absolute() or ".." in path.parts or member.is_dir():
            raise AssetError(f"asset pack contains unsafe path {member.filename!r}")
        if file_type and file_type != stat.S_IFREG:
            raise AssetError(f"asset pack contains non-regular file {member.filename!r}")
        total += member.file_size
        if total > _PACK_UNCOMPRESSED_LIMIT:
            raise AssetError("asset pack exceeds the uncompressed safety limit")
        result[member.filename] = member
    return result


def _preinspect_pack(archive: zipfile.ZipFile) -> None:
    members = archive.infolist()
    if len(members) > _PACK_MEMBER_LIMIT:
        raise AssetError("asset pack contains too many members")
    names = [member.filename for member in members]
    if len(names) != len(set(names)):
        raise AssetError("asset pack contains duplicate member names")
    if "pack.json" not in names:
        raise AssetError("asset pack is missing pack.json")
    total = 0
    for member in members:
        path = PurePosixPath(member.filename)
        if path.is_absolute() or ".." in path.parts or member.is_dir():
            raise AssetError(f"asset pack contains unsafe path {member.filename!r}")
        total += member.file_size
        if total > _PACK_UNCOMPRESSED_LIMIT:
            raise AssetError("asset pack exceeds the uncompressed safety limit")
        if (
            member.file_size > 1024 * 1024
            and member.file_size > max(member.compress_size, 1) * 10_000
        ):
            raise AssetError("asset pack contains an unsafe compression ratio")
    if archive.getinfo("pack.json").file_size > 64 * 1024:
        raise AssetError("asset pack header exceeds the safety limit")


def unpack_asset(
    source: str | os.PathLike[str],
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    repair: bool = False,
    asset: str | AssetSpec | None = None,
) -> AssetStatus:
    source_path = Path(source)
    if not source_path.is_file():
        raise AssetError(f"asset pack does not exist: {source_path}")
    try:
        with zipfile.ZipFile(source_path) as archive:
            _preinspect_pack(archive)
            raw_pack = _strict_json(archive.read("pack.json"), name="pack.json")
            asset_id = raw_pack.get("asset_id")
            spec = _resolve_spec(asset if asset is not None else str(asset_id))
            members = _inspect_pack(archive, spec)
            if (
                raw_pack.get("schema_id") != "gqmr.asset_pack"
                or raw_pack.get("schema_version") != 1
                or asset_id != spec.asset_id
                or raw_pack.get("manifest_sha256") != manifest_sha256(spec)
                or raw_pack.get("model_sha256") != spec.model_sha256
            ):
                raise AssetError("asset pack header does not match the trusted manifest")
            packed_manifest = _strict_json(
                archive.read(members["manifest.json"]), name="manifest.json"
            )
            if packed_manifest != spec.document:
                raise AssetError("packed manifest differs from the built-in trusted manifest")
            cache_root = _cache_root(cache_dir)
            install_path = _install_path(spec, cache_root)
            install_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_directory = Path(
                tempfile.mkdtemp(prefix=f".{spec.commit}.", dir=install_path.parent)
            )
            try:
                for item in spec.files:
                    target = temporary_directory / item.path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(members[f"files/{item.path}"]) as input_stream, target.open(
                        "wb"
                    ) as output:
                        actual_hash, actual_size = _sha256_stream(input_stream, output)
                        output.flush()
                        os.fsync(output.fileno())
                    if actual_hash != item.sha256 or actual_size != item.size:
                        raise AssetError(f"packed file hash mismatch for {item.path!r}")
                _write_json(
                    temporary_directory / _INSTALL_METADATA,
                    _installation_document(spec),
                )
                _activate_install(temporary_directory, install_path, repair=repair)
            except BaseException:
                shutil.rmtree(temporary_directory, ignore_errors=True)
                raise
    except (zipfile.BadZipFile, KeyError, OSError) as error:
        if isinstance(error, AssetError):
            raise
        raise AssetError(f"cannot read asset pack: {error}") from error
    result = status_asset(spec, cache_dir=_cache_root(cache_dir))
    if not result.valid:
        raise AssetError("unpacked asset failed verification")
    return result
