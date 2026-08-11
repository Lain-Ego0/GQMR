"""Hash and load user-supplied v1 MJCF asset directories."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from gqmr.robots.config import load_robot_config
from gqmr.robots.model import RobotModel, RobotModelError

_MAX_FILES = 10000
_MAX_TOTAL_BYTES = 8 * 1024 * 1024 * 1024


def external_asset_sha256(root: str | os.PathLike[str]) -> str:
    asset_root = Path(root).resolve(strict=True)
    if not asset_root.is_dir():
        raise RobotModelError("external asset root must be a directory")
    files: list[Path] = []
    total = 0
    for path in sorted(asset_root.rglob("*")):
        if path.is_symlink():
            raise RobotModelError(f"external asset tree contains a symlink: {path}")
        if not path.is_file():
            continue
        files.append(path)
        total += path.stat().st_size
        if len(files) > _MAX_FILES or total > _MAX_TOTAL_BYTES:
            raise RobotModelError("external asset tree exceeds file/size limits")
    if not files:
        raise RobotModelError("external asset tree has no files")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(asset_root).as_posix()
        file_digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                file_digest.update(chunk)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.hexdigest().encode("ascii"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_external_robot_model(
    config_path: str | os.PathLike[str], asset_root: str | os.PathLike[str]
) -> RobotModel:
    config = load_robot_config(config_path)
    root = Path(asset_root).resolve(strict=True)
    actual_hash = external_asset_sha256(root)
    if actual_hash != config.model_sha256:
        raise RobotModelError("external asset hash does not match robot config")
    model_path = (root / config.model).resolve(strict=True)
    try:
        model_path.relative_to(root)
    except ValueError as error:
        raise RobotModelError("external robot model escapes the asset root") from error
    return RobotModel.from_xml_path(model_path, config)
