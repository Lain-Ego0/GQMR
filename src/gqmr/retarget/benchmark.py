"""Repeatable cross-robot motion-suite evaluation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from gqmr.retarget.fast import retarget_fast
from gqmr.retarget.high_quality import retarget_high_quality
from gqmr.retarget.quality import replay_quality_report
from gqmr.robots import load_robot_model
from gqmr.synthetic import available_motion_presets, generate_dog27_preset


def evaluate_motion_suite(
    robot_ids: Sequence[str],
    *,
    cache_dir: Path | None = None,
    high_quality: bool = False,
    duration: float | None = None,
    fps: float | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Evaluate every stable motion preset against each selected robot."""

    presets = available_motion_presets()
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for robot_id in robot_ids:
        try:
            robot = load_robot_model(robot_id, cache_dir=cache_dir)
        except Exception as error:
            errors.append({"robot_id": robot_id, "error": str(error)})
            continue
        for preset in presets:
            if cancelled is not None and cancelled():
                return _suite_report(
                    robot_ids, presets, rows, errors, high_quality, cancelled=True
                )
            try:
                animal = generate_dog27_preset(
                    preset.id, duration=duration, fps=fps
                )
                motion, _ = (
                    retarget_high_quality(animal, robot)
                    if high_quality
                    else retarget_fast(animal, robot)
                )
                report = replay_quality_report(motion, robot)
                rows.append(
                    {
                        "robot_id": robot_id,
                        "preset_id": preset.id,
                        "action": preset.label,
                        "valid_frame_ratio": report["valid_frame_ratio"],
                        "solver_residual_rmse_m": report["solver_residual_rmse_m"],
                        "solver_residual_p95_m": report["solver_residual_p95_m"],
                        "joint_limit_violation_frames": report[
                            "joint_limit_violation_frames"
                        ],
                        "self_collision_frames": report["self_collision_frames"],
                        "maximum_ground_penetration_m": report[
                            "maximum_ground_penetration_m"
                        ],
                        "mean_contact_foot_speed_mps": report[
                            "mean_contact_foot_speed_mps"
                        ],
                    }
                )
            except Exception as error:
                errors.append(
                    {
                        "robot_id": robot_id,
                        "preset_id": preset.id,
                        "error": str(error),
                    }
                )
    return _suite_report(robot_ids, presets, rows, errors, high_quality)


def _suite_report(
    robot_ids,
    presets,
    rows: list[dict[str, Any]],
    errors: list[dict[str, str]],
    high_quality: bool,
    *,
    cancelled: bool = False,
) -> dict[str, Any]:
    valid_ratios = [row["valid_frame_ratio"] for row in rows]
    residuals = [
        row["solver_residual_rmse_m"]
        for row in rows
        if row["solver_residual_rmse_m"] is not None
    ]
    total = len(robot_ids) * len(presets)
    return {
        "benchmark": "gqmr_dog27_motion_presets_v1",
        "mode": "high-quality" if high_quality else "fast",
        "robots": list(robot_ids),
        "actions": [preset.id for preset in presets],
        "cancelled": cancelled,
        "summary": {
            "requested_evaluations": total,
            "completed_evaluations": len(rows),
            "failed_evaluations": len(errors),
            "mean_valid_frame_ratio": (
                sum(valid_ratios) / len(valid_ratios) if valid_ratios else None
            ),
            "mean_solver_residual_rmse_m": (
                sum(residuals) / len(residuals) if residuals else None
            ),
            "joint_limit_violation_frames": sum(
                row["joint_limit_violation_frames"] for row in rows
            ),
            "self_collision_frames": sum(
                row["self_collision_frames"] for row in rows
            ),
        },
        "results": rows,
        "errors": errors,
    }
