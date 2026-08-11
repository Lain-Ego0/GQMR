"""Deterministic cache keys for pose jobs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def pose_cache_key(
    inputs: list[str | Path],
    *,
    backend_package: str,
    backend_version: str,
    config: dict[str, Any],
    algorithm_version: str,
) -> str:
    digest = hashlib.sha256()
    for value in inputs:
        path = Path(value)
        digest.update(str(path.resolve()).encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\n")
    payload = {
        "backend_package": backend_package,
        "backend_version": backend_version,
        "config": config,
        "algorithm_version": algorithm_version,
    }
    digest.update(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return digest.hexdigest()
