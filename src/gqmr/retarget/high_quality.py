"""Sliding-window contact-aware kinematic refinement."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
from scipy.spatial.transform import Rotation

from gqmr.core.coordinates import wxyz_to_xyzw
from gqmr.core.derivatives import linear_velocity
from gqmr.core.motion import AnimalMotion, RobotMotion, SolverStatus
from gqmr.retarget.fast import FastRetargetConfig, RetargetDiagnostics, retarget_fast
from gqmr.robots import RobotModel
from gqmr.skeletons import AnimalSkeleton


@dataclass(frozen=True, slots=True)
class HighQualityRetargetConfig:
    window_seconds: float = 0.5
    passes: int = 3
    max_iterations: int = 48
    damping: float = 0.005
    smoothness: float = 0.0002
    max_joint_step: float = 0.12
    contact_threshold: float = 0.5
    residual_tolerance: float = 0.001
    unreachable_residual: float = 0.10
    minimum_foot_height: float = 0.02
    avoid_self_collision: bool = True
    optimize_root_position: bool = True
    root_tracking: float = 0.02
    root_smoothness: float = 0.002
    contact_weight: float = 2.0
    max_root_step: float = 0.01
    max_root_horizontal_offset: float = 0.08
    max_root_vertical_offset: float = 0.12

    def __post_init__(self) -> None:
        values = (
            self.window_seconds,
            self.damping,
            self.smoothness,
            self.max_joint_step,
            self.residual_tolerance,
            self.unreachable_residual,
            self.root_tracking,
            self.root_smoothness,
            self.contact_weight,
            self.max_root_step,
        )
        if self.passes <= 0 or self.max_iterations <= 0 or any(
            not np.isfinite(value) or value <= 0.0 for value in values
        ):
            raise ValueError("high-quality configuration values must be positive")
        if not 0.0 <= self.contact_threshold <= 1.0:
            raise ValueError("contact_threshold must be in [0,1]")
        if not np.isfinite(self.minimum_foot_height) or self.minimum_foot_height < 0.0:
            raise ValueError("minimum_foot_height must be finite and non-negative")
        root_offsets = (
            self.max_root_horizontal_offset,
            self.max_root_vertical_offset,
        )
        if any(not np.isfinite(value) or value < 0.0 for value in root_offsets):
            raise ValueError("maximum root offsets must be finite and non-negative")


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


def _close_loop_endpoint(
    desired: np.ndarray,
    original_target: np.ndarray,
    root_position: np.ndarray,
    root_rotation: np.ndarray,
    probability: np.ndarray,
) -> bool:
    """Make a detected loop close without merging distinct stance intervals."""

    if len(desired) < 2 or not np.allclose(
        probability[0], probability[-1], atol=1e-6, equal_nan=True
    ):
        return False
    rotations = Rotation.from_quat(wxyz_to_xyzw(root_rotation))
    first_local = rotations[0].inv().apply(original_target[0] - root_position[0])
    last_local = rotations[-1].inv().apply(original_target[-1] - root_position[-1])
    if np.max(np.abs(first_local - last_local)) > 1e-4:
        return False
    desired_local = rotations[0].inv().apply(desired[0] - root_position[0])
    desired[-1] = root_position[-1] + rotations[-1].apply(desired_local)
    return True


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
    loop_detected = _close_loop_endpoint(
        desired,
        fast_diagnostics.target_foot_positions.astype(np.float64),
        fast_motion.root_position.astype(np.float64),
        fast_motion.root_rotation.astype(np.float64),
        fast_motion.foot_contact_probability,
    )
    q = fast_motion.dof_position.astype(np.float64).copy()
    root_position = fast_motion.root_position.astype(np.float64).copy()
    root_correction = np.zeros_like(root_position)
    achieved = fast_diagnostics.achieved_foot_positions.astype(np.float64).copy()
    lower, upper = robot.joint_ranges[:, 0], robot.joint_ranges[:, 1]
    dofs = q.shape[1]
    identity = np.eye(dofs, dtype=np.float64)
    root_identity = np.eye(3, dtype=np.float64)
    root_translation_jacobian = np.tile(np.eye(3, dtype=np.float64), (4, 1))
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
            joint_reference = np.median(q[window_start:window_stop], axis=0)
            root_reference = np.median(
                root_correction[window_start:window_stop], axis=0
            )
            current_q = q[frame].copy()
            current_root_correction = root_correction[frame].copy()
            optimize_root_for_frame = config.optimize_root_position and not loop_detected
            minimum_vertical_correction = -config.max_root_vertical_offset
            for iteration in range(1, config.max_iterations + 1):
                current_root_position = (
                    fast_motion.root_position[frame] + current_root_correction
                )
                robot.set_pose(
                    current_root_position,
                    fast_motion.root_rotation[frame],
                    current_q,
                )
                if optimize_root_for_frame:
                    ground_penetration = robot.collision_metrics()[1]
                    if ground_penetration > 0.0:
                        minimum_vertical_correction = min(
                            config.max_root_vertical_offset,
                            current_root_correction[2]
                            + ground_penetration
                            + 1e-4,
                        )
                        current_root_correction[2] = max(
                            current_root_correction[2],
                            minimum_vertical_correction,
                        )
                        current_root_position = (
                            fast_motion.root_position[frame]
                            + current_root_correction
                        )
                        robot.set_pose(
                            current_root_position,
                            fast_motion.root_rotation[frame],
                            current_q,
                        )
                current_feet = robot.foot_positions()
                error = (desired[frame] - current_feet).reshape(-1)
                residual = float(np.sqrt(np.mean(error * error)))
                if residual <= config.residual_tolerance * 0.35:
                    break
                joint_jacobian = robot.foot_jacobians().reshape(12, dofs)
                contact_probability = np.nan_to_num(
                    fast_motion.foot_contact_probability[frame], nan=0.0
                )
                leg_weights = 1.0 + config.contact_weight * contact_probability
                row_weights = np.repeat(leg_weights, 3)
                weighted_error = error * row_weights
                if optimize_root_for_frame:
                    jacobian = np.column_stack(
                        (root_translation_jacobian, joint_jacobian)
                    )
                    weighted_jacobian = jacobian * row_weights[:, None]
                    regularizer = np.zeros((3 + dofs, 3 + dofs), dtype=np.float64)
                    regularizer[:3, :3] = (
                        config.damping**2
                        + config.root_tracking
                        + config.root_smoothness
                    ) * root_identity
                    regularizer[3:, 3:] = (
                        config.damping**2 + config.smoothness
                    ) * identity
                    normal = weighted_jacobian.T @ weighted_jacobian + regularizer
                    right = weighted_jacobian.T @ weighted_error
                    right[:3] += (
                        config.root_smoothness
                        * (root_reference - current_root_correction)
                        - config.root_tracking * current_root_correction
                    )
                    right[3:] += config.smoothness * (
                        joint_reference - current_q
                    )
                    step = np.linalg.solve(normal, right)
                    root_step = np.clip(
                        step[:3], -config.max_root_step, config.max_root_step
                    )
                    current_root_correction += root_step
                    horizontal_norm = float(
                        np.linalg.norm(current_root_correction[:2])
                    )
                    if horizontal_norm > config.max_root_horizontal_offset:
                        current_root_correction[:2] *= (
                            config.max_root_horizontal_offset / horizontal_norm
                        )
                    current_root_correction[2] = np.clip(
                        current_root_correction[2],
                        minimum_vertical_correction,
                        config.max_root_vertical_offset,
                    )
                    joint_step = step[3:]
                else:
                    normal = joint_jacobian.T @ joint_jacobian + (
                        config.damping**2 + config.smoothness
                    ) * identity
                    right = joint_jacobian.T @ error
                    right += config.smoothness * (joint_reference - current_q)
                    joint_step = np.linalg.solve(normal, right)
                joint_step = np.clip(
                    joint_step, -config.max_joint_step, config.max_joint_step
                )
                current_q = np.clip(current_q + joint_step, lower, upper)
            if optimize_root_for_frame:
                for _ in range(4):
                    current_root_position = (
                        fast_motion.root_position[frame]
                        + current_root_correction
                    )
                    robot.set_pose(
                        current_root_position,
                        fast_motion.root_rotation[frame],
                        current_q,
                    )
                    current_feet = robot.foot_positions()
                    foot_clearance = config.minimum_foot_height - float(
                        np.min(current_feet[:, 2])
                    )
                    ground_penetration = robot.collision_metrics()[1]
                    required_lift = max(foot_clearance, ground_penetration + 1e-4)
                    if required_lift <= 1e-6:
                        break
                    current_root_correction[2] += required_lift
            q[frame] = current_q
            root_correction[frame] = current_root_correction
            root_position[frame] = (
                fast_motion.root_position[frame] + current_root_correction
            )
            robot.set_pose(
                root_position[frame],
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
                        root_position[frame],
                        fast_motion.root_rotation[frame],
                        candidate,
                    )
                    if robot.collision_metrics()[0] == 0:
                        current_q = candidate
                        q[frame] = candidate
                        break
            if optimize_root_for_frame:
                current_feet = robot.foot_positions()
                required_lift = max(
                    config.minimum_foot_height - float(np.min(current_feet[:, 2])),
                    robot.collision_metrics()[1] + 1e-4,
                )
                if required_lift > 1e-6:
                    current_root_correction[2] += required_lift
                    root_correction[frame] = current_root_correction
                    root_position[frame] = (
                        fast_motion.root_position[frame]
                        + current_root_correction
                    )
                    robot.set_pose(
                        root_position[frame],
                        fast_motion.root_rotation[frame],
                        current_q,
                    )
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
        {"mode": "high_quality_root_contact_v3", "high_quality": asdict(config)}
    )
    metadata["retarget_config"] = retarget_config
    float32_margin = 4.0 * np.finfo(np.float32).eps * np.maximum(
        1.0, np.maximum(np.abs(lower), np.abs(upper))
    )
    q = np.clip(q, lower + float32_margin, upper - float32_margin)
    result = replace(
        fast_motion,
        root_position=root_position,
        root_linear_velocity=linear_velocity(motion.timestamps, root_position),
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
