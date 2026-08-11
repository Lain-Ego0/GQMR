from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from gqmr.assets.catalog import AssetFile, AssetSpec, get_asset_spec
from gqmr.assets.manager import (
    default_asset_root,
    install_asset,
    pack_asset,
    status_asset,
    unpack_asset,
)
from gqmr.core.errors import AssetError


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _model_sha256(files: tuple[AssetFile, ...], license_path: str) -> str:
    digest = hashlib.sha256()
    for item in files:
        if item.path == license_path:
            continue
        digest.update(item.path.encode())
        digest.update(b"\0")
        digest.update(item.sha256.encode())
        digest.update(b"\0")
        digest.update(str(item.size).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _write_tar(path: Path, payloads: dict[str, bytes], *, unsafe: bool = False) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in payloads.items():
            info = tarfile.TarInfo(f"fixture-root/{name}")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        if unsafe:
            payload = b"escape"
            info = tarfile.TarInfo("../escape.txt")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def _fixture_spec(archive: Path, payloads: dict[str, bytes]) -> AssetSpec:
    files = tuple(
        AssetFile(path=name, sha256=_sha256(payload), size=len(payload))
        for name, payload in payloads.items()
    )
    model_sha = _model_sha256(files, "LICENSE")
    archive_sha = _sha256(archive.read_bytes())
    document = {
        "schema_id": "gqmr.asset_manifest",
        "schema_version": 1,
        "asset_id": "fixture-robot",
        "display_name": "Fixture Robot",
        "source": {
            "repository": "https://example.invalid/fixture",
            "commit": "1" * 40,
            "archive_url": "https://example.invalid/fixture.tar.gz",
            "archive_sha256": archive_sha,
            "archive_prefix": "fixture-root/",
        },
        "license": {"spdx": "MIT", "path": "LICENSE"},
        "model_path": "robot/scene.xml",
        "model_sha256": model_sha,
        "files": [
            {"path": item.path, "sha256": item.sha256, "size": item.size}
            for item in files
        ],
    }
    return AssetSpec(
        asset_id="fixture-robot",
        display_name="Fixture Robot",
        repository="https://example.invalid/fixture",
        commit="1" * 40,
        archive_url="https://example.invalid/fixture.tar.gz",
        archive_sha256=archive_sha,
        archive_prefix="fixture-root/",
        license_spdx="MIT",
        license_path="LICENSE",
        model_path="robot/scene.xml",
        model_sha256=model_sha,
        files=files,
        document=document,
    )


@pytest.fixture
def fixture_asset(tmp_path: Path) -> tuple[Path, AssetSpec]:
    payloads = {
        "LICENSE": b"fixture license\n",
        "robot/scene.xml": b"<mujoco model='fixture'/>\n",
        "robot/assets/mesh.bin": b"mesh-data\x00\x01",
    }
    archive = tmp_path / "fixture.tar.gz"
    _write_tar(archive, payloads)
    return archive, _fixture_spec(archive, payloads)


def test_builtin_manifests_are_internally_consistent() -> None:
    go2 = get_asset_spec("unitree-go2")
    b2 = get_asset_spec("unitree-b2")
    assert len(go2.files) == 19
    assert len(b2.files) == 34
    assert go2.license_spdx == b2.license_spdx == "BSD-3-Clause"
    assert go2.archive_sha256 == b2.archive_sha256


def test_repository_assets_are_the_verified_default() -> None:
    repository = Path(__file__).resolve().parents[2]
    assert default_asset_root() == repository
    assert status_asset("unitree-go2").valid
    assert status_asset("unitree-b2").valid


def test_install_status_corruption_repair_and_offline_round_trip(
    tmp_path: Path, fixture_asset: tuple[Path, AssetSpec]
) -> None:
    archive, spec = fixture_asset
    first_cache = tmp_path / "cache-one"
    second_cache = tmp_path / "cache-two"
    assert status_asset(spec, cache_dir=first_cache).state == "missing"

    installed = install_asset(spec, cache_dir=first_cache, archive_path=archive)
    assert installed.valid
    model_path = Path(installed.model_path)
    assert model_path.read_bytes().startswith(b"<mujoco")

    model_path.write_bytes(b"tampered")
    corrupt = status_asset(spec, cache_dir=first_cache)
    assert corrupt.state == "corrupt"
    assert "robot/scene.xml" in corrupt.corrupt_files
    with pytest.raises(AssetError, match="--repair"):
        install_asset(spec, cache_dir=first_cache, archive_path=archive)

    repaired = install_asset(
        spec, cache_dir=first_cache, archive_path=archive, repair=True
    )
    assert repaired.valid

    pack = tmp_path / "fixture.gqmr-assets"
    pack_asset(spec, pack, cache_dir=first_cache)
    unpacked = unpack_asset(pack, cache_dir=second_cache, asset=spec)
    assert unpacked.valid
    assert Path(unpacked.model_path).read_bytes() == model_path.read_bytes()


def test_archive_hash_mismatch_is_rejected(
    tmp_path: Path, fixture_asset: tuple[Path, AssetSpec]
) -> None:
    archive, spec = fixture_asset
    archive.write_bytes(archive.read_bytes() + b"tampered")
    with pytest.raises(AssetError, match="archive hash mismatch"):
        install_asset(spec, cache_dir=tmp_path / "cache", archive_path=archive)


def test_tar_path_traversal_is_rejected(tmp_path: Path) -> None:
    payloads = {
        "LICENSE": b"license",
        "robot/scene.xml": b"<mujoco/>",
    }
    archive = tmp_path / "unsafe.tar.gz"
    _write_tar(archive, payloads, unsafe=True)
    spec = _fixture_spec(archive, payloads)
    with pytest.raises(AssetError, match="unsafe path"):
        install_asset(spec, cache_dir=tmp_path / "cache", archive_path=archive)
    assert not (tmp_path / "escape.txt").exists()


def test_pack_tampering_is_rejected(
    tmp_path: Path, fixture_asset: tuple[Path, AssetSpec]
) -> None:
    archive, spec = fixture_asset
    cache = tmp_path / "cache"
    install_asset(spec, cache_dir=cache, archive_path=archive)
    pack = tmp_path / "fixture.gqmr-assets"
    pack_asset(spec, pack, cache_dir=cache)
    payload = bytearray(pack.read_bytes())
    payload[-20] ^= 0xFF
    pack.write_bytes(payload)
    with pytest.raises(AssetError):
        unpack_asset(pack, cache_dir=tmp_path / "other", asset=spec)
