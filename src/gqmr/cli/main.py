"""GQMR command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from gqmr import __version__
from gqmr.core.errors import GQMRError
from gqmr.core.io import load_motion


def _motion_summary(
    path: Path, *, expected_model_sha256: str | None = None
) -> dict[str, object]:
    motion = load_motion(path, expected_model_sha256=expected_model_sha256)
    summary: dict[str, object] = {
        "path": str(path),
        "schema_id": motion.schema_id,
        "schema_version": motion.schema_version,
        "frames": motion.frame_count,
        "duration_seconds": motion.duration,
        "valid_frames": int(motion.frame_valid.sum()),
    }
    if hasattr(motion, "keypoint_names"):
        summary["keypoints"] = list(motion.keypoint_names)
    if hasattr(motion, "dof_names"):
        summary["dofs"] = list(motion.dof_names)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gqmr")
    parser.add_argument("--version", action="version", version=f"gqmr {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser(
        "inspect", help="inspect a canonical motion NPZ"
    )
    inspect_parser.add_argument("path", type=Path)
    validate_parser = subparsers.add_parser(
        "validate", help="validate a canonical motion NPZ"
    )
    validate_parser.add_argument("path", type=Path)
    validate_parser.add_argument(
        "--model-sha256",
        help="require a RobotMotion to be bound to this exact model hash",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = _motion_summary(
            args.path,
            expected_model_sha256=getattr(args, "model_sha256", None),
        )
        if args.command == "validate":
            summary = {"valid": True, **summary}
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except GQMRError as error:
        result = {
            "valid": False,
            "error_type": type(error).__name__,
            "message": str(error),
            "path": str(args.path),
        }
        print(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
