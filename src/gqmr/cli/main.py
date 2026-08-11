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
from gqmr.core.motion import RobotMotion
from gqmr.robots import available_robot_configs, load_robot_model
from gqmr.robots.model import RobotModelError


def _motion_summary(
    path: Path,
    *,
    expected_model_sha256: str | None = None,
    expected_dof_order: tuple[str, ...] | None = None,
) -> dict[str, object]:
    motion = load_motion(path, expected_model_sha256=expected_model_sha256)
    if expected_dof_order is not None:
        if not isinstance(motion, RobotMotion):
            raise RobotModelError("--robot can only validate RobotMotion files")
        if motion.dof_names != expected_dof_order:
            raise RobotModelError(
                "RobotMotion dof_names do not match the configured business order"
            )
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
    model_binding = validate_parser.add_mutually_exclusive_group()
    model_binding.add_argument(
        "--model-sha256",
        help="require a RobotMotion to be bound to this exact model hash",
    )
    model_binding.add_argument(
        "--robot",
        choices=available_robot_configs(),
        help="validate against an installed and verified robot configuration",
    )
    validate_parser.add_argument(
        "--cache-dir", type=Path, help="override the trusted asset cache root"
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

    robots_parser = subparsers.add_parser(
        "robots", help="inspect validated MuJoCo robot bindings"
    )
    robot_commands = robots_parser.add_subparsers(dest="robot_command", required=True)
    robot_inspect = robot_commands.add_parser(
        "inspect", help="show model dimensions and business-name mappings"
    )
    robot_inspect.add_argument("robot", choices=available_robot_configs())
    robot_inspect.add_argument("--cache-dir", type=Path)
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


def _run_robots(args: argparse.Namespace) -> dict[str, object]:
    robot = load_robot_model(args.robot, cache_dir=args.cache_dir)
    return {
        "robot_id": robot.config.id,
        "robot_config_sha256": robot.config.sha256,
        "model_sha256": robot.config.model_sha256,
        "nq": robot.model.nq,
        "nv": robot.model.nv,
        "nu": robot.model.nu,
        "root_joint_id": robot.root_joint_id,
        "root_qpos_address": robot.root_qpos_address,
        "root_dof_address": robot.root_dof_address,
        "dof_order": list(robot.config.dof_order),
        "qpos_addresses": robot.qpos_addresses.tolist(),
        "dof_addresses": robot.dof_addresses.tolist(),
        "joint_ranges": robot.joint_ranges.tolist(),
        "feet": {
            leg: {
                "body_id": binding.body_id,
                "local_position": binding.local_position.tolist(),
                "contact_geom_ids": list(binding.contact_geom_ids),
            }
            for leg, binding in robot.feet.items()
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "assets":
            result, exit_code = _run_assets(args)
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            return exit_code
        if args.command == "robots":
            result = _run_robots(args)
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        expected_model_sha256 = getattr(args, "model_sha256", None)
        expected_dof_order = None
        if getattr(args, "robot", None):
            robot = load_robot_model(args.robot, cache_dir=args.cache_dir)
            expected_model_sha256 = robot.config.model_sha256
            expected_dof_order = robot.config.dof_order
        summary = _motion_summary(
            args.path,
            expected_model_sha256=expected_model_sha256,
            expected_dof_order=expected_dof_order,
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
                    getattr(
                        args,
                        "source",
                        getattr(args, "asset", getattr(args, "robot", "unknown")),
                    ),
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
