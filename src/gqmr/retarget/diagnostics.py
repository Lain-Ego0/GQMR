"""Per-frame diagnostics for interactive retarget inspection and repair."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from gqmr.core.motion import RobotMotion, SolverStatus
from gqmr.retarget.fast import RetargetDiagnostics
from gqmr.robots import RobotModel


@dataclass(frozen=True, slots=True)
class MotionDiagnostics:
    invalid: NDArray[np.bool_]
    unreachable: NDArray[np.bool_]
    joint_limit_proximity: NDArray[np.float32]
    ground_penetration: NDArray[np.float32]
    self_collision: NDArray[np.bool_]
    foot_slip_speed: NDArray[np.float32]
    root_position_correction: NDArray[np.float32]
    root_rotation_correction: NDArray[np.float32]

    @property
    def problem_frames(self) -> NDArray[np.int64]:
        problem = (
            self.invalid
            | self.unreachable
            | self.self_collision
            | (self.ground_penetration > 0.003)
            | (np.max(self.joint_limit_proximity, axis=1) > 0.95)
            | (np.max(self.foot_slip_speed, axis=1) > 0.15)
        )
        return np.flatnonzero(problem)

    def frame_messages(self, frame: int) -> tuple[str, ...]:
        messages: list[str] = []
        if self.invalid[frame]:
            messages.append("无效帧")
        if self.unreachable[frame]:
            messages.append("足端目标不可达")
        if self.self_collision[frame]:
            messages.append("自碰撞")
        if self.ground_penetration[frame] > 0.003:
            messages.append(
                f"穿地 {self.ground_penetration[frame] * 1000.0:.1f} mm"
            )
        limit = float(np.max(self.joint_limit_proximity[frame]))
        if limit > 0.95:
            messages.append(f"关节限位 {limit * 100.0:.0f}%")
        slip = float(np.max(self.foot_slip_speed[frame]))
        if slip > 0.15:
            messages.append(f"接触足滑移 {slip:.2f} m/s")
        return tuple(messages)


def diagnose_motion(
    motion: RobotMotion,
    robot: RobotModel,
    retarget: RetargetDiagnostics | None = None,
) -> MotionDiagnostics:
    """Replay a clip and compute arrays suitable for an interactive timeline."""

    frames = motion.frame_count
    lower, upper = robot.joint_ranges[:, 0], robot.joint_ranges[:, 1]
    center = 0.5 * (lower + upper)
    half_range = np.maximum(0.5 * (upper - lower), 1e-9)
    proximity = np.abs(motion.dof_position - center) / half_range
    feet = np.full((frames, 4, 3), np.nan, dtype=np.float64)
    penetration = np.zeros(frames, dtype=np.float32)
    collision = np.zeros(frames, dtype=np.bool_)
    finite = (
        np.all(np.isfinite(motion.root_position), axis=1)
        & np.all(np.isfinite(motion.root_rotation), axis=1)
        & np.all(np.isfinite(motion.dof_position), axis=1)
    )
    for frame in np.flatnonzero(finite):
        robot.set_pose(
            motion.root_position[frame],
            motion.root_rotation[frame],
            motion.dof_position[frame],
        )
        feet[frame] = robot.foot_positions()
        self_contacts, ground = robot.collision_metrics()
        collision[frame] = self_contacts > 0
        penetration[frame] = ground
    slip = np.zeros((frames, 4), dtype=np.float32)
    if frames > 1:
        dt = np.diff(motion.timestamps)
        speed = np.linalg.norm(np.diff(feet, axis=0), axis=2) / dt[:, None]
        contact = (
            (motion.foot_contact_probability[:-1] >= 0.5)
            & (motion.foot_contact_probability[1:] >= 0.5)
        )
        speed = np.where(contact & np.isfinite(speed), speed, 0.0)
        slip[1:] = speed
        slip[0] = slip[1]
    zero = np.zeros((frames, 3), dtype=np.float32)
    position_correction = (
        zero
        if retarget is None or retarget.root_position_correction is None
        else retarget.root_position_correction
    )
    rotation_correction = (
        zero
        if retarget is None or retarget.root_rotation_correction is None
        else retarget.root_rotation_correction
    )
    return MotionDiagnostics(
        invalid=np.ascontiguousarray(~motion.frame_valid),
        unreachable=np.ascontiguousarray(
            motion.solver_status == int(SolverStatus.UNREACHABLE)
        ),
        joint_limit_proximity=np.ascontiguousarray(proximity, dtype=np.float32),
        ground_penetration=penetration,
        self_collision=collision,
        foot_slip_speed=slip,
        root_position_correction=np.ascontiguousarray(
            position_correction, dtype=np.float32
        ),
        root_rotation_correction=np.ascontiguousarray(
            rotation_correction, dtype=np.float32
        ),
    )
