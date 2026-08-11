"""Sliding-window contact-aware kinematic refinement."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np

from gqmr.core.derivatives import linear_velocity
from gqmr.core.motion import AnimalMotion, RobotMotion, SolverStatus
from gqmr.retarget.fast import FastRetargetConfig, RetargetDiagnostics, retarget_fast
from gqmr.robots import RobotModel
from gqmr.skeletons import AnimalSkeleton


@dataclass(frozen=True, slots=True)
class HighQualityRetargetConfig:
    window_seconds: float = 1.0
    passes: int = 3
    max_iterations: int = 20
    damping: float = 0.02
    smoothness: float = 0.015
    max_joint_step: float = 0.12
    contact_threshold: float = 0.5
    residual_tolerance: float = 0.03
    unreachable_residual: float = 0.10
    minimum_foot_height: float = 0.02
    avoid_self_collision: bool = True

    def __post_init__(self) -> None:
        values = (
            self.window_seconds,
            self.damping,
            self.smoothness,
            self.max_joint_step,
            self.residual_tolerance,
            self.unreachable_residual,
        )
        if self.passes <= 0 or self.max_iterations <= 0 or any(
            not np.isfinite(value) or value <= 0.0 for value in values
        ):
            raise ValueError("high-quality configuration values must be positive")
        if not 0.0 <= self.contact_threshold <= 1.0:
            raise ValueError("contact_threshold must be in [0,1]")
        if not np.isfinite(self.minimum_foot_height) or self.minimum_foot_height < 0.0:
            raise ValueError("minimum_foot_height must be finite and non-negative")


def _contact_locked_targets(
    target: np.ndarray,
    achieved: np.ndarray,
    probability: np.ndarray,
    config: HighQualityRetargetConfig,
) -> np.ndarray:
    desired = target.copy()
    frames = len(target)
    for leg in range(4):
        mask = np.isfinite(probability[:, leg]) & (
            probability[:, leg] >= config.contact_threshold
        )
        start = 0
        while start < frames:
            if not mask[start]:
                start += 1
                continue
            stop = start + 1
            while stop < frames and mask[stop]:
                stop += 1
            anchor = np.median(achieved[start:stop, leg], axis=0)
            anchor[2] = max(anchor[2], config.minimum_foot_height)
            desired[start:stop, leg] = anchor
            start = stop
    desired[..., 2] = np.maximum(desired[..., 2], config.minimum_foot_height)
    return desired


def retarget_high_quality(
    motion: AnimalMotion,
    robot: RobotModel,
    *,
    skeleton: AnimalSkeleton | None = None,
    fast_config: FastRetargetConfig | None = None,
    config: HighQualityRetargetConfig | None = None,
) -> tuple[RobotMotion, RetargetDiagnostics]:
    """Run fast retargeting then contact-aware windowed Gauss-Seidel refinement."""

    config = config or HighQualityRetargetConfig()
    fast_motion, fast_diagnostics = retarget_fast(
        motion, robot, skeleton=skeleton, config=fast_config
    )
    desired = _contact_locked_targets(
        fast_diagnostics.target_foot_positions.astype(np.float64),
        fast_diagnostics.achieved_foot_positions.astype(np.float64),
        fast_motion.foot_contact_probability,
        config,
    )
    q = fast_motion.dof_position.astype(np.float64).copy()
    achieved = fast_diagnostics.achieved_foot_positions.astype(np.float64).copy()
    lower, upper = robot.joint_ranges[:, 0], robot.joint_ranges[:, 1]
    dofs = q.shape[1]
    identity = np.eye(dofs, dtype=np.float64)
    median_dt = float(np.median(np.diff(motion.timestamps)))
    radius = max(1, int(round(config.window_seconds / median_dt / 2.0)))
    iterations = np.zeros(motion.frame_count, dtype=np.int16)

    for pass_index in range(config.passes):
        frame_order = (
            range(motion.frame_count)
            if pass_index % 2 == 0
            else range(motion.frame_count - 1, -1, -1)
        )
        for frame in frame_order:
            if not fast_motion.frame_valid[frame]:
                continue
            window_start = max(0, frame - radius)
            window_stop = min(motion.frame_count, frame + radius + 1)
            reference = np.median(q[window_start:window_stop], axis=0)
            current_q = q[frame].copy()
            for iteration in range(1, config.max_iterations + 1):
                robot.set_pose(
                    fast_motion.root_position[frame],
                    fast_motion.root_rotation[frame],
                    current_q,
                )
                current_feet = robot.foot_positions()
                error = (desired[frame] - current_feet).reshape(-1)
                residual = float(np.sqrt(np.mean(error * error)))
                if residual <= config.residual_tolerance * 0.35:
                    break
                jacobian = robot.foot_jacobians().reshape(12, dofs)
                normal = (
                    jacobian.T @ jacobian
                    + (config.damping**2 + config.smoothness) * identity
                )
                right = jacobian.T @ error + config.smoothness * (
                    reference - current_q
                )
                step = np.linalg.solve(normal, right)
                step = np.clip(step, -config.max_joint_step, config.max_joint_step)
                current_q = np.clip(current_q + step, lower, upper)
            q[frame] = current_q
            robot.set_pose(
                fast_motion.root_position[frame],
                fast_motion.root_rotation[frame],
                current_q,
            )
            if config.avoid_self_collision and robot.collision_metrics()[0] > 0:
                default_q = np.asarray(
                    robot.config.default_dof_position, dtype=np.float64
                )
                for blend in np.linspace(0.1, 1.0, 10):
                    candidate = np.clip(
                        (1.0 - blend) * current_q + blend * default_q,
                        lower,
                        upper,
                    )
                    robot.set_pose(
                        fast_motion.root_position[frame],
                        fast_motion.root_rotation[frame],
                        candidate,
                    )
                    if robot.collision_metrics()[0] == 0:
                        current_q = candidate
                        q[frame] = candidate
                        break
            achieved[frame] = robot.foot_positions()
            iterations[frame] = max(iterations[frame], iteration)

    error = desired - achieved
    residual = np.sqrt(np.mean(error * error, axis=(1, 2)))
    status = fast_motion.solver_status.copy()
    valid = fast_motion.frame_valid.copy()
    for frame in np.flatnonzero(valid):
        if residual[frame] >= config.unreachable_residual:
            status[frame] = SolverStatus.UNREACHABLE
            valid[frame] = False
        elif residual[frame] > config.residual_tolerance:
            status[frame] = SolverStatus.MAX_ITER
        elif status[frame] not in {SolverStatus.DEGRADED_ROOT}:
            status[frame] = SolverStatus.OK
    metadata = dict(fast_motion.metadata)
    retarget_config = dict(metadata["retarget_config"])
    retarget_config.update(
        {"mode": "high_quality_contact_v1", "high_quality": asdict(config)}
    )
    metadata["retarget_config"] = retarget_config
    result = replace(
        fast_motion,
        dof_position=q,
        dof_velocity=linear_velocity(motion.timestamps, q),
        frame_valid=valid,
        solver_status=status,
        solver_residual=residual,
        metadata=metadata,
    )
    diagnostics = RetargetDiagnostics(
        target_foot_positions=np.ascontiguousarray(desired, dtype=np.float32),
        achieved_foot_positions=np.ascontiguousarray(achieved, dtype=np.float32),
        iterations=iterations,
        root_translation_scale=fast_diagnostics.root_translation_scale,
        leg_motion_scales=fast_diagnostics.leg_motion_scales,
    )
    return result, diagnostics
