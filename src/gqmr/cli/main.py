"""GQMR command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from gqmr import __version__
from gqmr.assets import (
    available_assets,
    install_asset,
    pack_asset,
    status_asset,
    unpack_asset,
)
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
    assets_parser = subparsers.add_parser("assets", help="manage trusted robot assets")
    asset_commands = assets_parser.add_subparsers(dest="asset_command", required=True)

    status_parser = asset_commands.add_parser("status", help="verify installed assets")
    status_parser.add_argument("asset", nargs="?", choices=available_assets())
    status_parser.add_argument("--cache-dir", type=Path)

    install_parser = asset_commands.add_parser("install", help="install a trusted asset")
    install_parser.add_argument("asset", choices=available_assets())
    install_parser.add_argument("--cache-dir", type=Path)
    install_parser.add_argument(
        "--archive", type=Path, help="use a pre-downloaded fixed-commit tar.gz"
    )
    install_parser.add_argument("--repair", action="store_true")

    pack_parser = asset_commands.add_parser("pack", help="create a verified offline pack")
    pack_parser.add_argument("asset", choices=available_assets())
    pack_parser.add_argument("destination", type=Path)
    pack_parser.add_argument("--cache-dir", type=Path)

    unpack_parser = asset_commands.add_parser(
        "unpack", help="install a verified offline asset pack"
    )
    unpack_parser.add_argument("source", type=Path)
    unpack_parser.add_argument("--cache-dir", type=Path)
    unpack_parser.add_argument("--repair", action="store_true")
    return parser


def _run_assets(args: argparse.Namespace) -> tuple[dict[str, object], int]:
    if args.asset_command == "status":
        asset_ids = (args.asset,) if args.asset else available_assets()
        statuses = [
            status_asset(asset_id, cache_dir=args.cache_dir).to_dict()
            for asset_id in asset_ids
        ]
        return {"assets": statuses}, 0 if all(item["valid"] for item in statuses) else 1
    if args.asset_command == "install":
        status = install_asset(
            args.asset,
            cache_dir=args.cache_dir,
            archive_path=args.archive,
            repair=args.repair,
        )
        return status.to_dict(), 0
    if args.asset_command == "pack":
        destination = pack_asset(
            args.asset, args.destination, cache_dir=args.cache_dir
        )
        return {"asset_id": args.asset, "pack": str(destination), "valid": True}, 0
    status = unpack_asset(
        args.source, cache_dir=args.cache_dir, repair=args.repair
    )
    return status.to_dict(), 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "assets":
            result, exit_code = _run_assets(args)
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            return exit_code
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
            "resource": str(
                getattr(
                    args,
                    "path",
                    getattr(args, "source", getattr(args, "asset", "unknown")),
                )
            ),
        }
        print(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
