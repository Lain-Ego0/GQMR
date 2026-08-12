"""Sliding-window contact-aware kinematic refinement."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
from scipy.spatial.transform import Rotation

from gqmr.core.coordinates import wxyz_to_xyzw, xyzw_to_wxyz
from gqmr.core.derivatives import angular_velocity_world, linear_velocity
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
    optimize_root_rotation: bool = True
    root_rotation_tracking: float = 0.03
    root_rotation_smoothness: float = 0.003
    contact_rotation_stability: float = 2.0
    flight_rotation_tracking_scale: float = 0.15
    yaw_tracking_multiplier: float = 20.0
    max_root_rotation_step: float = np.deg2rad(2.0)
    max_root_tilt_correction: float = np.deg2rad(28.0)
    max_root_yaw_correction: float = np.deg2rad(5.0)
    stable_roll_limit: float = np.deg2rad(18.0)
    stable_pitch_limit: float = np.deg2rad(20.0)

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
            self.root_rotation_tracking,
            self.root_rotation_smoothness,
            self.contact_rotation_stability,
            self.flight_rotation_tracking_scale,
            self.yaw_tracking_multiplier,
            self.max_root_rotation_step,
            self.max_root_tilt_correction,
            self.max_root_yaw_correction,
            self.stable_roll_limit,
            self.stable_pitch_limit,
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


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64
    )


def _smooth_stable_rotation_targets(
    root_rotation: np.ndarray,
    config: HighQualityRetargetConfig,
) -> tuple[Rotation, np.ndarray]:
    """Preserve heading while smoothly projecting excessive roll/pitch."""

    source = Rotation.from_quat(wxyz_to_xyzw(root_rotation))
    euler = source.as_euler("xyz")
    projected = euler.copy()
    projected[:, 0] = config.stable_roll_limit * np.tanh(
        euler[:, 0] / config.stable_roll_limit
    )
    projected[:, 1] = config.stable_pitch_limit * np.tanh(
        euler[:, 1] / config.stable_pitch_limit
    )
    stable = Rotation.from_euler("xyz", projected)
    preferred_correction = (stable * source.inv()).as_rotvec()
    return source, preferred_correction


def _bounded_rotation_correction(
    correction: np.ndarray, config: HighQualityRetargetConfig
) -> np.ndarray:
    result = correction.copy()
    tilt = float(np.linalg.norm(result[:2]))
    if tilt > config.max_root_tilt_correction:
        result[:2] *= config.max_root_tilt_correction / tilt
    result[2] = np.clip(
        result[2], -config.max_root_yaw_correction, config.max_root_yaw_correction
    )
    return result


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
    frame_range: tuple[int, int] | None = None,
    initial_motion: RobotMotion | None = None,
) -> tuple[RobotMotion, RetargetDiagnostics]:
    """Run contact-aware root-pose refinement, optionally on one frame interval."""

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
    if initial_motion is not None:
        if (
            initial_motion.frame_count != motion.frame_count
            or initial_motion.dof_names != fast_motion.dof_names
            or initial_motion.metadata["model_sha256"]
            != fast_motion.metadata["model_sha256"]
        ):
            raise ValueError("initial_motion is incompatible with the refinement input")
        initial = initial_motion
    else:
        initial = fast_motion
    if frame_range is None:
        active_start, active_stop = 0, motion.frame_count
    else:
        active_start, active_stop_inclusive = frame_range
        active_stop = active_stop_inclusive + 1
        if not 0 <= active_start < active_stop <= motion.frame_count:
            raise ValueError("frame_range must be an inclusive interval inside the clip")
    active_frames = np.arange(active_start, active_stop, dtype=np.int32)

    q = initial.dof_position.astype(np.float64).copy()
    initial_q = q.copy()
    root_position = initial.root_position.astype(np.float64).copy()
    initial_root_position = root_position.copy()
    root_correction = (
        root_position - fast_motion.root_position.astype(np.float64)
    )
    source_root_rotation, preferred_rotation_correction = (
        _smooth_stable_rotation_targets(fast_motion.root_rotation, config)
    )
    initial_root_rotation = Rotation.from_quat(
        wxyz_to_xyzw(initial.root_rotation)
    )
    root_rotation_correction = (
        initial_root_rotation * source_root_rotation.inv()
    ).as_rotvec()
    initial_rotation_correction = root_rotation_correction.copy()
    root_rotation = initial.root_rotation.astype(np.float64).copy()
    achieved = np.empty_like(
        fast_diagnostics.achieved_foot_positions, dtype=np.float64
    )
    for frame in range(motion.frame_count):
        robot.set_pose(root_position[frame], root_rotation[frame], q[frame])
        achieved[frame] = robot.foot_positions()
    lower, upper = robot.joint_ranges[:, 0], robot.joint_ranges[:, 1]
    dofs = q.shape[1]
    identity = np.eye(dofs, dtype=np.float64)
    root_identity = np.eye(3, dtype=np.float64)
    root_translation_jacobian = np.tile(np.eye(3, dtype=np.float64), (4, 1))
    median_dt = float(np.median(np.diff(motion.timestamps)))
    radius = max(1, int(round(config.window_seconds / median_dt / 2.0)))
    iterations = np.zeros(motion.frame_count, dtype=np.int16)

    total_passes = config.passes * (2 if config.optimize_root_rotation else 1)
    for pass_index in range(total_passes):
        frame_order = (
            active_frames
            if pass_index % 2 == 0
            else active_frames[::-1]
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
            rotation_reference = np.median(
                root_rotation_correction[window_start:window_stop], axis=0
            )
            current_q = q[frame].copy()
            current_root_correction = root_correction[frame].copy()
            current_rotation_correction = root_rotation_correction[frame].copy()
            if loop_detected and frame == motion.frame_count - 1:
                current_rotation_correction = root_rotation_correction[0].copy()
            starting_q = current_q.copy()
            starting_root_correction = current_root_correction.copy()
            starting_rotation_correction = current_rotation_correction.copy()
            starting_root_position = (
                fast_motion.root_position[frame] + starting_root_correction
            )
            starting_root_rotation = (
                Rotation.from_rotvec(starting_rotation_correction)
                * source_root_rotation[frame]
            )
            robot.set_pose(
                starting_root_position,
                xyzw_to_wxyz(starting_root_rotation.as_quat()),
                starting_q,
            )
            starting_error = desired[frame] - robot.foot_positions()
            starting_residual = float(
                np.sqrt(np.mean(starting_error * starting_error))
            )
            optimize_root_for_frame = config.optimize_root_position and not loop_detected
            optimize_rotation_for_frame = (
                config.optimize_root_rotation and pass_index >= config.passes
            )
            minimum_vertical_correction = -config.max_root_vertical_offset
            for iteration in range(1, config.max_iterations + 1):
                current_root_position = (
                    fast_motion.root_position[frame] + current_root_correction
                )
                current_root_rotation = (
                    Rotation.from_rotvec(current_rotation_correction)
                    * source_root_rotation[frame]
                )
                robot.set_pose(
                    current_root_position,
                    xyzw_to_wxyz(current_root_rotation.as_quat()),
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
                            xyzw_to_wxyz(current_root_rotation.as_quat()),
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
                if optimize_root_for_frame or optimize_rotation_for_frame:
                    jacobian_parts: list[np.ndarray] = []
                    if optimize_root_for_frame:
                        jacobian_parts.append(root_translation_jacobian)
                    if optimize_rotation_for_frame:
                        rotation_jacobian = np.vstack(
                            [
                                -_skew(foot - current_root_position)
                                for foot in current_feet
                            ]
                        )
                        jacobian_parts.append(rotation_jacobian)
                    jacobian_parts.append(joint_jacobian)
                    jacobian = np.column_stack(jacobian_parts)
                    weighted_jacobian = jacobian * row_weights[:, None]
                    root_variables = 3 * int(optimize_root_for_frame)
                    rotation_variables = 3 * int(optimize_rotation_for_frame)
                    regularizer = np.zeros(
                        (root_variables + rotation_variables + dofs,) * 2,
                        dtype=np.float64,
                    )
                    offset = 0
                    if optimize_root_for_frame:
                        regularizer[:3, :3] = (
                            config.damping**2
                            + config.root_tracking
                            + config.root_smoothness
                        ) * root_identity
                        offset = 3
                    support = float(np.max(contact_probability))
                    if optimize_rotation_for_frame:
                        stability_scale = config.flight_rotation_tracking_scale + (
                            1.0 - config.flight_rotation_tracking_scale
                        ) * support
                        tilt_tracking = config.root_rotation_tracking * stability_scale * (
                            1.0 + config.contact_rotation_stability * support
                        )
                        rotation_tracking = np.array(
                            [
                                tilt_tracking,
                                tilt_tracking,
                                config.root_rotation_tracking
                                * config.yaw_tracking_multiplier,
                            ],
                            dtype=np.float64,
                        )
                        regularizer[offset : offset + 3, offset : offset + 3] = (
                            np.diag(rotation_tracking)
                            + (
                                config.damping**2
                                + config.root_rotation_smoothness
                            )
                            * root_identity
                        )
                        offset += 3
                    regularizer[offset:, offset:] = (
                        config.damping**2 + config.smoothness
                    ) * identity
                    normal = weighted_jacobian.T @ weighted_jacobian + regularizer
                    right = weighted_jacobian.T @ weighted_error
                    offset = 0
                    if optimize_root_for_frame:
                        right[:3] += (
                            config.root_smoothness
                            * (root_reference - current_root_correction)
                            - config.root_tracking * current_root_correction
                        )
                        offset = 3
                    if optimize_rotation_for_frame:
                        preferred = preferred_rotation_correction[frame].copy()
                        preferred[2] = 0.0
                        if loop_detected and frame == motion.frame_count - 1:
                            preferred = root_rotation_correction[0]
                            rotation_tracking *= 4.0
                        right[offset : offset + 3] += (
                            config.root_rotation_smoothness
                            * (rotation_reference - current_rotation_correction)
                            + rotation_tracking
                            * (preferred - current_rotation_correction)
                        )
                        offset += 3
                    right[offset:] += config.smoothness * (joint_reference - current_q)
                    step = np.linalg.solve(normal, right)
                    offset = 0
                    if optimize_root_for_frame:
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
                        offset = 3
                    if optimize_rotation_for_frame:
                        rotation_step = step[offset : offset + 3]
                        rotation_norm = float(np.linalg.norm(rotation_step))
                        if rotation_norm > config.max_root_rotation_step:
                            rotation_step *= config.max_root_rotation_step / rotation_norm
                        current_rotation_correction = _bounded_rotation_correction(
                            current_rotation_correction + rotation_step, config
                        )
                        offset += 3
                    joint_step = step[offset:]
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
                        xyzw_to_wxyz(
                            (
                                Rotation.from_rotvec(current_rotation_correction)
                                * source_root_rotation[frame]
                            ).as_quat()
                        ),
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
            root_rotation_correction[frame] = current_rotation_correction
            root_position[frame] = (
                fast_motion.root_position[frame] + current_root_correction
            )
            root_rotation[frame] = xyzw_to_wxyz(
                (
                    Rotation.from_rotvec(current_rotation_correction)
                    * source_root_rotation[frame]
                ).as_quat()
            )
            robot.set_pose(root_position[frame], root_rotation[frame], current_q)
            candidate_feet = robot.foot_positions()
            candidate_error = desired[frame] - candidate_feet
            candidate_residual = float(
                np.sqrt(np.mean(candidate_error * candidate_error))
            )
            candidate_collision = robot.collision_metrics()[0]
            if candidate_residual > starting_residual + 1e-8 or (
                config.avoid_self_collision and candidate_collision > 0
            ):
                best = (
                    starting_residual,
                    starting_q,
                    starting_root_correction,
                    starting_rotation_correction,
                )
                for blend in (0.75, 0.5, 0.25):
                    blended_q = starting_q + blend * (current_q - starting_q)
                    blended_root = starting_root_correction + blend * (
                        current_root_correction - starting_root_correction
                    )
                    blended_rotation = starting_rotation_correction + blend * (
                        current_rotation_correction - starting_rotation_correction
                    )
                    blended_position = fast_motion.root_position[frame] + blended_root
                    blended_quaternion = xyzw_to_wxyz(
                        (
                            Rotation.from_rotvec(blended_rotation)
                            * source_root_rotation[frame]
                        ).as_quat()
                    )
                    robot.set_pose(blended_position, blended_quaternion, blended_q)
                    blended_feet = robot.foot_positions()
                    blended_error = desired[frame] - blended_feet
                    blended_residual = float(
                        np.sqrt(np.mean(blended_error * blended_error))
                    )
                    if config.avoid_self_collision and robot.collision_metrics()[0] > 0:
                        continue
                    if blended_residual < best[0]:
                        best = (
                            blended_residual,
                            blended_q,
                            blended_root,
                            blended_rotation,
                        )
                _, current_q, current_root_correction, current_rotation_correction = best
                q[frame] = current_q
                root_correction[frame] = current_root_correction
                root_rotation_correction[frame] = current_rotation_correction
                root_position[frame] = (
                    fast_motion.root_position[frame] + current_root_correction
                )
                root_rotation[frame] = xyzw_to_wxyz(
                    (
                        Rotation.from_rotvec(current_rotation_correction)
                        * source_root_rotation[frame]
                    ).as_quat()
                )
            robot.set_pose(
                root_position[frame],
                root_rotation[frame],
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
                        root_rotation[frame],
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
                        root_rotation[frame],
                        current_q,
                    )
            achieved[frame] = robot.foot_positions()
            iterations[frame] = max(iterations[frame], iteration)

        if (
            loop_detected
            and config.optimize_root_rotation
            and pass_index >= config.passes
            and active_start == 0
            and active_stop == motion.frame_count
        ):
            root_rotation_correction[-1] = root_rotation_correction[0]
            root_rotation[-1] = xyzw_to_wxyz(
                (
                    Rotation.from_rotvec(root_rotation_correction[0])
                    * source_root_rotation[-1]
                ).as_quat()
            )
            robot.set_pose(root_position[-1], root_rotation[-1], q[-1])
            achieved[-1] = robot.foot_positions()

    if frame_range is not None and len(active_frames) > 1:
        blend_frames = min(
            max(1, int(round(0.08 / median_dt))), max(1, len(active_frames) // 2)
        )
        distance_to_edge = np.minimum(
            np.arange(len(active_frames)), np.arange(len(active_frames))[::-1]
        )
        blend = np.clip(distance_to_edge / blend_frames, 0.0, 1.0)
        blend = blend * blend * (3.0 - 2.0 * blend)
        q[active_frames] = initial_q[active_frames] + blend[:, None] * (
            q[active_frames] - initial_q[active_frames]
        )
        root_position[active_frames] = initial_root_position[active_frames] + blend[
            :, None
        ] * (root_position[active_frames] - initial_root_position[active_frames])
        root_correction[active_frames] = (
            root_position[active_frames]
            - fast_motion.root_position[active_frames].astype(np.float64)
        )
        root_rotation_correction[active_frames] = initial_rotation_correction[
            active_frames
        ] + blend[:, None] * (
            root_rotation_correction[active_frames]
            - initial_rotation_correction[active_frames]
        )
        for frame in active_frames:
            root_rotation[frame] = xyzw_to_wxyz(
                (
                    Rotation.from_rotvec(root_rotation_correction[frame])
                    * source_root_rotation[frame]
                ).as_quat()
            )
            robot.set_pose(root_position[frame], root_rotation[frame], q[frame])
            achieved[frame] = robot.foot_positions()

    status = initial.solver_status.copy()
    valid = initial.frame_valid.copy()
    residual = initial.solver_residual.astype(np.float64).copy()
    active_error = desired[active_frames] - achieved[active_frames]
    residual[active_frames] = np.sqrt(
        np.mean(active_error * active_error, axis=(1, 2))
    )
    for frame in active_frames[valid[active_frames]]:
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
    if frame_range is not None:
        retarget_config["refined_frame_range"] = [
            int(active_start),
            int(active_stop - 1),
        ]
    metadata["retarget_config"] = retarget_config
    float32_margin = 4.0 * np.finfo(np.float32).eps * np.maximum(
        1.0, np.maximum(np.abs(lower), np.abs(upper))
    )
    q = np.clip(q, lower + float32_margin, upper - float32_margin)
    result = replace(
        fast_motion,
        root_position=root_position,
        root_rotation=root_rotation,
        root_linear_velocity=linear_velocity(motion.timestamps, root_position),
        root_angular_velocity=angular_velocity_world(
            motion.timestamps, root_rotation
        ),
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
        root_position_correction=np.ascontiguousarray(
            root_correction, dtype=np.float32
        ),
        root_rotation_correction=np.ascontiguousarray(
            root_rotation_correction, dtype=np.float32
        ),
    )
    return result, diagnostics
