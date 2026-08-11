"""MuJoCo PD tracking replay for kinematic motion quality diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from gqmr.core.coordinates import wxyz_to_xyzw
from gqmr.core.motion import RobotMotion
from gqmr.robots import RobotModel
from gqmr.robots.model import RobotModelError


@dataclass(frozen=True, slots=True)
class PDReplayConfig:
    kp: float = 35.0
    kd: float = 1.0
    fall_height_ratio: float = 0.45
    fall_tilt_degrees: float = 60.0

    def __post_init__(self) -> None:
        if any(
            not np.isfinite(value) or value <= 0.0
            for value in (self.kp, self.kd, self.fall_height_ratio, self.fall_tilt_degrees)
        ):
            raise ValueError("PD replay configuration must be finite and positive")


def simulate_pd_tracking(
    motion: RobotMotion,
    robot: RobotModel,
    *,
    config: PDReplayConfig | None = None,
) -> dict[str, Any]:
    """Track joint targets without claiming the kinematic clip is dynamically stable."""

    config = config or PDReplayConfig()
    if motion.metadata["model_sha256"] != robot.config.model_sha256:
        raise RobotModelError("RobotMotion model hash does not match dynamics robot")
    if motion.dof_names != robot.config.dof_order or not np.all(motion.frame_valid):
        raise RobotModelError("PD replay requires matching DOFs and fully valid frames")
    if np.any(robot.actuator_ids < 0):
        raise RobotModelError("PD replay requires one direct actuator per business DOF")
    robot.set_pose(
        motion.root_position[0], motion.root_rotation[0], motion.dof_position[0]
    )
    robot.data.qvel[robot.root_dof_address : robot.root_dof_address + 3] = (
        motion.root_linear_velocity[0]
    )
    robot.data.qvel[robot.dof_addresses] = motion.dof_velocity[0]
    model_timestep = float(robot.model.opt.timestep)
    steps = max(1, int(np.ceil(motion.duration / model_timestep)))
    root_errors: list[float] = []
    joint_errors: list[float] = []
    torque_peaks = np.zeros(len(robot.config.dof_order), dtype=np.float64)
    contact_speeds: list[float] = []
    previous_feet = robot.foot_positions()
    previous_time = 0.0
    fall_time: float | None = None
    default_height = float(robot.config.default_root_position[2])
    fall_cosine = np.cos(np.deg2rad(config.fall_tilt_degrees))

    for step in range(steps + 1):
        simulation_time = min(step * model_timestep, motion.duration)
        right = int(np.searchsorted(motion.timestamps, simulation_time, side="right"))
        right = min(max(right, 1), motion.frame_count - 1)
        left = right - 1
        interval = motion.timestamps[right] - motion.timestamps[left]
        alpha = float((simulation_time - motion.timestamps[left]) / interval)
        target_q = (1.0 - alpha) * motion.dof_position[left] + alpha * motion.dof_position[right]
        target_dq = (1.0 - alpha) * motion.dof_velocity[left] + alpha * motion.dof_velocity[right]
        desired_root = (1.0 - alpha) * motion.root_position[left] + alpha * motion.root_position[right]
        current_q = robot.data.qpos[robot.qpos_addresses]
        current_dq = robot.data.qvel[robot.dof_addresses]
        command = config.kp * (target_q - current_q) + config.kd * (target_dq - current_dq)
        limited = robot.model.actuator_ctrllimited[robot.actuator_ids].astype(bool)
        ranges = robot.model.actuator_ctrlrange[robot.actuator_ids]
        command[limited] = np.clip(
            command[limited], ranges[limited, 0], ranges[limited, 1]
        )
        robot.data.ctrl[:] = 0.0
        robot.data.ctrl[robot.actuator_ids] = command
        torque_peaks = np.maximum(torque_peaks, np.abs(command))
        root_errors.append(
            float(np.linalg.norm(robot.data.qpos[robot.root_qpos_address : robot.root_qpos_address + 3] - desired_root))
        )
        joint_errors.append(float(np.sqrt(np.mean((current_q - target_q) ** 2))))
        feet = robot.foot_positions()
        if step > 0:
            speed = np.linalg.norm(feet - previous_feet, axis=1) / (
                simulation_time - previous_time
            )
            contact_index = min(left, motion.frame_count - 1)
            active = motion.foot_contact_probability[contact_index] >= 0.5
            contact_speeds.extend(speed[active & np.isfinite(speed)].tolist())
        previous_feet = feet
        previous_time = simulation_time
        root_quaternion = robot.data.qpos[
            robot.root_qpos_address + 3 : robot.root_qpos_address + 7
        ]
        up = Rotation.from_quat(wxyz_to_xyzw(root_quaternion)).apply([0.0, 0.0, 1.0])
        root_height = float(robot.data.qpos[robot.root_qpos_address + 2])
        if fall_time is None and (
            root_height < config.fall_height_ratio * default_height or up[2] < fall_cosine
        ):
            fall_time = simulation_time
        if step < steps:
            mujoco.mj_step(robot.model, robot.data)

    return {
        "robot_id": robot.config.id,
        "duration_seconds": motion.duration,
        "simulated_steps": steps,
        "model_timestep_seconds": model_timestep,
        "fall_time_seconds": fall_time,
        "root_tracking_rmse_m": float(np.sqrt(np.mean(np.square(root_errors)))),
        "joint_tracking_rmse_rad": float(np.sqrt(np.mean(np.square(joint_errors)))),
        "mean_contact_foot_speed_mps": (
            float(np.mean(contact_speeds)) if contact_speeds else None
        ),
        "peak_actuator_command": torque_peaks.tolist(),
        "peak_actuator_command_max": float(np.max(torque_peaks)),
        "claim": "diagnostic_pd_tracking_only",
    }
