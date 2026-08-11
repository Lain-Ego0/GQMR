"""Kinematic replay and quality metrics for canonical RobotMotion."""

from __future__ import annotations

from typing import Any

import numpy as np

from gqmr.core.motion import RobotMotion
from gqmr.robots import RobotModel
from gqmr.robots.model import RobotModelError


def replay_quality_report(motion: RobotMotion, robot: RobotModel) -> dict[str, Any]:
    """Replay finite frames through MuJoCo FK and return JSON-safe metrics."""

    if motion.metadata["model_sha256"] != robot.config.model_sha256:
        raise RobotModelError("RobotMotion model hash does not match replay robot")
    if motion.dof_names != robot.config.dof_order:
        raise RobotModelError("RobotMotion DOF order does not match replay robot")
    feet = np.full((motion.frame_count, 4, 3), np.nan, dtype=np.float64)
    replayed = np.zeros(motion.frame_count, dtype=np.bool_)
    lower, upper = robot.joint_ranges[:, 0], robot.joint_ranges[:, 1]
    finite_pose = (
        np.all(np.isfinite(motion.root_position), axis=1)
        & np.all(np.isfinite(motion.root_rotation), axis=1)
        & np.all(np.isfinite(motion.dof_position), axis=1)
    )
    for frame in np.flatnonzero(finite_pose):
        try:
            robot.set_pose(
                motion.root_position[frame],
                motion.root_rotation[frame],
                motion.dof_position[frame],
            )
        except RobotModelError:
            continue
        feet[frame] = robot.foot_positions()
        replayed[frame] = True

    joint_limit_violation = np.any(
        (motion.dof_position < lower) | (motion.dof_position > upper), axis=1
    )
    residual = motion.solver_residual[motion.frame_valid]
    residual = residual[np.isfinite(residual)]
    contact_slide_samples: list[float] = []
    if motion.frame_count >= 2:
        delta_time = np.diff(motion.timestamps)
        foot_speed = np.linalg.norm(np.diff(feet, axis=0), axis=2) / delta_time[:, None]
        contact = (
            motion.foot_contact_probability[:-1] >= 0.5
        ) & (motion.foot_contact_probability[1:] >= 0.5)
        usable_contact = contact & np.isfinite(foot_speed)
        contact_slide_samples = foot_speed[usable_contact].tolist()
    finite_foot_z = feet[..., 2][np.isfinite(feet[..., 2])]
    valid_count = int(np.count_nonzero(motion.frame_valid))
    return {
        "robot_id": robot.config.id,
        "frames": motion.frame_count,
        "replayed_frames": int(np.count_nonzero(replayed)),
        "valid_frames": valid_count,
        "valid_frame_ratio": valid_count / motion.frame_count,
        "joint_limit_violation_frames": int(np.count_nonzero(joint_limit_violation)),
        "solver_residual_rmse_m": (
            float(np.sqrt(np.mean(residual * residual))) if len(residual) else None
        ),
        "solver_residual_p95_m": (
            float(np.percentile(residual, 95.0)) if len(residual) else None
        ),
        "mean_contact_foot_speed_mps": (
            float(np.mean(contact_slide_samples)) if contact_slide_samples else None
        ),
        "minimum_foot_height_m": (
            float(np.min(finite_foot_z)) if len(finite_foot_z) else None
        ),
        "status_counts": {
            str(int(status)): int(count)
            for status, count in zip(
                *np.unique(motion.solver_status, return_counts=True)
            )
        },
    }
