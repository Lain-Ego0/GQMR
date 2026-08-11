"""GQMR command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from gqmr import __version__
from gqmr.assets import (
    available_assets,
    install_asset,
    pack_asset,
    status_asset,
    unpack_asset,
)
from gqmr.core.errors import GQMRError
from gqmr.core.io import load_motion, save_motion
from gqmr.core.motion import AnimalMotion, RobotMotion
from gqmr.exporters import export_deepmimic_json, export_isaaclab_amp_v232
from gqmr.project import (
    add_resource,
    load_project,
    new_project,
    pack_project,
    save_project,
)
from gqmr.retarget import FastRetargetConfig, replay_quality_report, retarget_fast
from gqmr.robots import available_robot_configs, load_robot_model
from gqmr.robots.model import RobotModelError
from gqmr.skeletons import get_skeleton
from gqmr.sources.files import inspect_legacy_dog27, load_legacy_dog27
from gqmr.synthetic import generate_dog27_motion


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


def _inspect_path(path: Path) -> dict[str, object]:
    if path.suffix.lower() == ".npz":
        return _motion_summary(path)
    return inspect_legacy_dog27(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gqmr")
    parser.add_argument("--version", action="version", version=f"gqmr {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser(
        "inspect", help="inspect a canonical NPZ or legacy dog-27 text file"
    )
    inspect_parser.add_argument("path", type=Path)
    convert_parser = subparsers.add_parser(
        "convert", help="convert a legacy dog-27 text file to canonical AnimalMotion"
    )
    convert_parser.add_argument("path", type=Path)
    convert_parser.add_argument("--skeleton", choices=("dog-27",), default="dog-27")
    convert_parser.add_argument("--fps", type=float, default=60.0)
    convert_parser.add_argument("--start-frame", type=int, default=0)
    convert_parser.add_argument("--end-frame", type=int)
    convert_parser.add_argument("--output", type=Path, required=True)
    synthetic_parser = subparsers.add_parser(
        "synthetic", help="generate an MIT-licensed canonical dog-27 motion"
    )
    synthetic_parser.add_argument("gait", choices=("walk", "trot", "pace", "turn"))
    synthetic_parser.add_argument("--duration", type=float, default=2.0)
    synthetic_parser.add_argument("--fps", type=float, default=60.0)
    synthetic_parser.add_argument("--output", type=Path, required=True)
    retarget_parser = subparsers.add_parser(
        "retarget", help="retarget canonical AnimalMotion to a v1 MuJoCo robot"
    )
    retarget_parser.add_argument("path", type=Path)
    retarget_parser.add_argument("--robot", choices=available_robot_configs(), required=True)
    retarget_parser.add_argument("--cache-dir", type=Path)
    retarget_parser.add_argument("--output", type=Path, required=True)
    retarget_parser.add_argument("--max-iterations", type=int, default=24)
    retarget_parser.add_argument("--damping", type=float, default=0.025)
    retarget_parser.add_argument("--residual-tolerance", type=float, default=0.03)
    retarget_parser.add_argument("--unreachable-residual", type=float, default=0.10)
    play_parser = subparsers.add_parser(
        "play", help="replay a RobotMotion through MuJoCo FK and report quality"
    )
    play_parser.add_argument("path", type=Path)
    play_parser.add_argument("--robot", choices=available_robot_configs(), required=True)
    play_parser.add_argument("--cache-dir", type=Path)
    export_parser = subparsers.add_parser(
        "export", help="export RobotMotion to a training or compatibility format"
    )
    export_parser.add_argument("path", type=Path)
    export_parser.add_argument(
        "--format",
        choices=("canonical", "isaaclab_amp_v232", "deepmimic"),
        required=True,
    )
    export_parser.add_argument("--robot", choices=available_robot_configs(), required=True)
    export_parser.add_argument("--cache-dir", type=Path)
    export_parser.add_argument("--fps", type=int, default=60)
    export_parser.add_argument("--output", type=Path, required=True)
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
    project_parser = subparsers.add_parser("project", help="manage .gqmr projects")
    project_commands = project_parser.add_subparsers(
        dest="project_command", required=True
    )
    project_new = project_commands.add_parser("new", help="create an empty project")
    project_new.add_argument("destination", type=Path)
    project_info = project_commands.add_parser("info", help="inspect a project")
    project_info.add_argument("path", type=Path)
    project_add = project_commands.add_parser("add", help="add an external resource")
    project_add.add_argument("path", type=Path, help="project path")
    project_add.add_argument("resource", type=Path)
    project_add.add_argument("--kind", choices=("animal", "robot", "other"), default="other")
    project_pack = project_commands.add_parser("pack", help="create a portable project")
    project_pack.add_argument("path", type=Path)
    project_pack.add_argument("destination", type=Path)
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


def _run_convert(args: argparse.Namespace) -> dict[str, object]:
    motion = load_legacy_dog27(
        args.path,
        fps=args.fps,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        skeleton=get_skeleton(args.skeleton),
    )
    save_motion(args.output, motion)
    return {
        "input": str(args.path),
        "output": str(args.output),
        "schema_id": motion.schema_id,
        "frames": motion.frame_count,
        "duration_seconds": motion.duration,
        "skeleton_id": args.skeleton,
    }


def _run_synthetic(args: argparse.Namespace) -> dict[str, object]:
    motion = generate_dog27_motion(args.gait, duration=args.duration, fps=args.fps)
    save_motion(args.output, motion)
    return {
        "output": str(args.output),
        "schema_id": motion.schema_id,
        "frames": motion.frame_count,
        "duration_seconds": motion.duration,
        "skeleton_id": motion.metadata["skeleton_id"],
        "gait": args.gait,
        "license": motion.metadata["source"]["license"],
    }


def _run_retarget(args: argparse.Namespace) -> dict[str, object]:
    source = load_motion(args.path)
    if not isinstance(source, AnimalMotion):
        raise RobotModelError("retarget input must be a canonical AnimalMotion")
    robot = load_robot_model(args.robot, cache_dir=args.cache_dir)
    config = FastRetargetConfig(
        max_iterations=args.max_iterations,
        damping=args.damping,
        residual_tolerance=args.residual_tolerance,
        unreachable_residual=args.unreachable_residual,
    )
    motion, diagnostics = retarget_fast(source, robot, config=config)
    save_motion(args.output, motion)
    residual = motion.solver_residual[motion.frame_valid]
    return {
        "input": str(args.path),
        "output": str(args.output),
        "robot_id": args.robot,
        "frames": motion.frame_count,
        "valid_frames": int(np.count_nonzero(motion.frame_valid)),
        "valid_frame_ratio": float(np.mean(motion.frame_valid)),
        "foot_target_rmse_m": (
            float(np.sqrt(np.mean(residual * residual))) if len(residual) else None
        ),
        "root_translation_scale": diagnostics.root_translation_scale,
        "leg_motion_scales": diagnostics.leg_motion_scales,
    }


def _run_play(args: argparse.Namespace) -> dict[str, object]:
    motion = load_motion(args.path)
    if not isinstance(motion, RobotMotion):
        raise RobotModelError("play input must be a canonical RobotMotion")
    robot = load_robot_model(args.robot, cache_dir=args.cache_dir)
    return replay_quality_report(motion, robot)


def _run_export(args: argparse.Namespace) -> dict[str, object]:
    motion = load_motion(args.path)
    if not isinstance(motion, RobotMotion):
        raise RobotModelError("export input must be a canonical RobotMotion")
    robot = load_robot_model(args.robot, cache_dir=args.cache_dir)
    if motion.metadata["model_sha256"] != robot.config.model_sha256:
        raise RobotModelError("RobotMotion model hash does not match export robot")
    if args.format == "canonical":
        save_motion(args.output, motion)
    elif args.format == "isaaclab_amp_v232":
        export_isaaclab_amp_v232(motion, robot, args.output, fps=args.fps)
    else:
        export_deepmimic_json(motion, args.output)
    return {
        "input": str(args.path),
        "output": str(args.output),
        "format": args.format,
        "robot_id": args.robot,
        "frames": motion.frame_count,
        "valid": True,
    }


def _project_summary(path: Path, project) -> dict[str, object]:
    return {
        "path": str(path),
        "project_id": project.project_id,
        "schema_version": project.schema_version,
        "resources": len(project.resources),
        "embedded_resources": sum(
            resource.embedded for resource in project.resources.values()
        ),
        "active_animal_motion": project.active_animal_motion,
        "active_robot_motion": project.active_robot_motion,
        "edits": len(project.edits),
    }


def _run_project(args: argparse.Namespace) -> dict[str, object]:
    if args.project_command == "new":
        project = new_project()
        save_project(args.destination, project)
        return _project_summary(args.destination, project)
    project = load_project(args.path)
    if args.project_command == "info":
        return _project_summary(args.path, project)
    if args.project_command == "add":
        active = args.kind if args.kind in {"animal", "robot"} else None
        project = add_resource(project, args.resource, make_active=active)
        save_project(args.path, project)
        return _project_summary(args.path, project)
    pack_project(args.destination, project)
    packed = load_project(args.destination)
    return _project_summary(args.destination, packed)


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
        if args.command == "project":
            result = _run_project(args)
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.command == "convert":
            result = _run_convert(args)
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.command == "synthetic":
            result = _run_synthetic(args)
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.command == "retarget":
            result = _run_retarget(args)
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.command == "play":
            result = _run_play(args)
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.command == "export":
            result = _run_export(args)
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.command == "inspect":
            result = _inspect_path(args.path)
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
