"""Fast per-frame MuJoCo Jacobian retargeting for v1 quadrupeds."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from gqmr import __version__
from gqmr.assets import get_asset_spec
from gqmr.core.coordinates import wxyz_to_xyzw, xyzw_to_wxyz
from gqmr.core.derivatives import angular_velocity_world, linear_velocity
from gqmr.core.errors import GQMRError
from gqmr.core.io import motion_sha256
from gqmr.core.motion import AnimalMotion, RobotMotion, SolverStatus
from gqmr.retarget.animal_preprocess import (
    estimate_body_scale,
    estimate_contact_probability,
    estimate_root_motion,
)
from gqmr.robots import LEG_ORDER, RobotModel
from gqmr.skeletons import AnimalSkeleton, get_skeleton


class FastRetargetError(GQMRError, ValueError):
    """Raised when fast retargeting cannot safely process its inputs."""


@dataclass(frozen=True, slots=True)
class FastRetargetConfig:
    max_iterations: int = 40
    damping: float = 0.020
    max_joint_step: float = 0.20
    residual_tolerance: float = 0.005
    unreachable_residual: float = 0.10
    root_translation_scale: float | None = None
    foot_motion_scale: float = 1.0
    maximum_leg_reach_ratio: float = 0.98

    def __post_init__(self) -> None:
        numeric = (
            self.damping,
            self.max_joint_step,
            self.residual_tolerance,
            self.unreachable_residual,
            self.foot_motion_scale,
            self.maximum_leg_reach_ratio,
        )
        if self.max_iterations <= 0 or not np.all(np.isfinite(numeric)):
            raise FastRetargetError("fast retarget configuration must be finite and positive")
        if any(value <= 0.0 for value in numeric):
            raise FastRetargetError("fast retarget configuration must be finite and positive")
        if self.unreachable_residual < self.residual_tolerance:
            raise FastRetargetError(
                "unreachable_residual must not be below residual_tolerance"
            )
        if self.root_translation_scale is not None and (
            not np.isfinite(self.root_translation_scale)
            or self.root_translation_scale <= 0.0
        ):
            raise FastRetargetError(
                "root_translation_scale must be finite and positive"
            )
        if self.maximum_leg_reach_ratio > 1.0:
            raise FastRetargetError("maximum_leg_reach_ratio must not exceed 1")


@dataclass(frozen=True, slots=True)
class RetargetDiagnostics:
    target_foot_positions: NDArray[np.float32]
    achieved_foot_positions: NDArray[np.float32]
    iterations: NDArray[np.int16]
    root_translation_scale: float
    leg_motion_scales: dict[str, float]
    root_position_correction: NDArray[np.float32] | None = None
    root_rotation_correction: NDArray[np.float32] | None = None


def _differentiate_linear(timestamps: np.ndarray, values: np.ndarray) -> np.ndarray:
    if len(timestamps) == 1:
        return np.zeros_like(values, dtype=np.float64)
    if len(timestamps) == 2:
        slope = (values[1] - values[0]) / (timestamps[1] - timestamps[0])
        return np.stack((slope, slope))
    return linear_velocity(timestamps, values)


def _differentiate_rotation(timestamps: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    if len(timestamps) == 1:
        return np.zeros((1, 3), dtype=np.float64)
    if len(timestamps) == 2:
        rotations = Rotation.from_quat(wxyz_to_xyzw(quaternion))
        velocity = (rotations[1] * rotations[0].inv()).as_rotvec() / (
            timestamps[1] - timestamps[0]
        )
        return np.stack((velocity, velocity))
    return angular_velocity_world(timestamps, quaternion)


def _heading_rotations(rotations: Rotation) -> Rotation:
    forward = rotations.apply(np.array([1.0, 0.0, 0.0]))
    horizontal_norm = np.linalg.norm(forward[:, :2], axis=1)
    if np.any(horizontal_norm < 1e-8):
        raise FastRetargetError("source root heading is vertically degenerate")
    yaw = np.unwrap(np.arctan2(forward[:, 1], forward[:, 0]))
    rotation_vectors = np.zeros((len(yaw), 3), dtype=np.float64)
    rotation_vectors[:, 2] = yaw
    return Rotation.from_rotvec(rotation_vectors)


def _build_targets(
    motion: AnimalMotion,
    robot: RobotModel,
    skeleton: AnimalSkeleton,
    config: FastRetargetConfig,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    dict[str, float],
]:
    root = estimate_root_motion(motion, skeleton)
    body_scale = estimate_body_scale(motion, skeleton)
    name_to_index = {name: index for index, name in enumerate(motion.keypoint_names)}
    toe_ids = np.array(
        [name_to_index[skeleton.limb_chains[leg][-1]] for leg in LEG_ORDER],
        dtype=np.int32,
    )
    limb_root_ids = np.array(
        [name_to_index[skeleton.limb_chains[leg][0]] for leg in LEG_ORDER],
        dtype=np.int32,
    )

    robot.set_pose(
        robot.config.default_root_position,
        robot.config.default_root_rotation,
        robot.config.default_dof_position,
    )
    default_root_position = np.asarray(robot.config.default_root_position, dtype=np.float64)
    default_root_rotation = Rotation.from_quat(
        wxyz_to_xyzw(robot.config.default_root_rotation)
    )
    default_feet_world = robot.foot_positions()
    default_feet_local = default_root_rotation.inv().apply(
        default_feet_world - default_root_position
    )
    leg_joint_anchors_world = np.stack(
        [
            robot.data.xanchor[robot.joint_ids[index * 3 : (index + 1) * 3]]
            for index in range(len(LEG_ORDER))
        ]
    )
    default_leg_anchors_local = default_root_rotation.inv().apply(
        leg_joint_anchors_world[:, 0] - default_root_position
    )
    maximum_leg_reaches = np.array(
        [
            np.sum(np.linalg.norm(np.diff(anchors, axis=0), axis=1))
            + np.linalg.norm(default_feet_world[index] - anchors[-1])
            for index, anchors in enumerate(leg_joint_anchors_world)
        ],
        dtype=np.float64,
    )

    front_center = 0.5 * (default_feet_local[0] + default_feet_local[1])
    rear_center = 0.5 * (default_feet_local[2] + default_feet_local[3])
    robot_body_length = float(np.linalg.norm(front_center - rear_center))
    root_scale = config.root_translation_scale or robot_body_length / body_scale.torso_length
    leg_scales = {
        leg: float(
            np.linalg.norm(
                default_feet_local[index] - default_leg_anchors_local[index]
            )
            / body_scale.leg_lengths[leg]
        )
        * config.foot_motion_scale
        for index, leg in enumerate(LEG_ORDER)
    }

    source_rotation = Rotation.from_quat(wxyz_to_xyzw(root.rotation))
    source_heading = _heading_rotations(source_rotation)
    source_initial_inverse = source_heading[0].inv()
    source_relative_rotation = source_rotation * source_rotation[0].inv()
    robot_rotation = source_relative_rotation * default_root_rotation
    robot_rotation_wxyz = xyzw_to_wxyz(robot_rotation.as_quat())

    source_delta_world = root.position.astype(np.float64) - root.position[0]
    source_delta_initial = source_initial_inverse.apply(source_delta_world)
    robot_root_position = default_root_position + default_root_rotation.apply(
        source_delta_initial * root_scale
    )

    source_limb_vectors_world = (
        motion.positions[:, toe_ids] - motion.positions[:, limb_root_ids]
    ).astype(np.float64)
    source_limb_vectors_local = np.empty_like(source_limb_vectors_world)
    for frame in range(motion.frame_count):
        source_limb_vectors_local[frame] = source_rotation[frame].inv().apply(
            source_limb_vectors_world[frame]
        )
    source_neutral_limb_vectors = np.empty((len(LEG_ORDER), 3), dtype=np.float64)
    for index in range(len(LEG_ORDER)):
        samples_usable = (
            motion.frame_valid
            & root.valid
            & motion.valid_mask[:, toe_ids[index]]
            & motion.valid_mask[:, limb_root_ids[index]]
            & np.all(np.isfinite(source_limb_vectors_local[:, index]), axis=1)
        )
        if not np.any(samples_usable):
            raise FastRetargetError(
                f"source leg {LEG_ORDER[index]} has no usable limb vectors"
            )
        source_neutral_limb_vectors[index] = np.median(
            source_limb_vectors_local[samples_usable, index], axis=0
        )

    source_limb_delta = (
        source_limb_vectors_local - source_neutral_limb_vectors[None, :, :]
    )
    target_feet = np.empty_like(source_limb_vectors_world)
    for frame in range(motion.frame_count):
        local_target = default_feet_local + np.stack(
            [
                source_limb_delta[frame, index] * leg_scales[leg]
                for index, leg in enumerate(LEG_ORDER)
            ]
        )
        leg_vectors = local_target - default_leg_anchors_local
        leg_vector_norms = np.linalg.norm(leg_vectors, axis=1)
        allowed_reaches = maximum_leg_reaches * config.maximum_leg_reach_ratio
        overextended = leg_vector_norms > allowed_reaches
        if np.any(overextended):
            leg_vectors[overextended] *= (
                allowed_reaches[overextended] / leg_vector_norms[overextended]
            )[:, None]
            local_target[overextended] = (
                default_leg_anchors_local[overextended] + leg_vectors[overextended]
            )
        target_feet[frame] = robot_root_position[frame] + robot_rotation[frame].apply(
            local_target
        )

    usable = (
        motion.frame_valid
        & root.valid
        & np.all(motion.valid_mask[:, toe_ids], axis=1)
        & np.all(motion.valid_mask[:, limb_root_ids], axis=1)
        & np.all(np.isfinite(target_feet), axis=(1, 2))
    )
    return (
        robot_root_position,
        robot_rotation_wxyz,
        target_feet,
        usable,
        root.status,
        float(root_scale),
        leg_scales,
    )


def retarget_fast(
    motion: AnimalMotion,
    robot: RobotModel,
    *,
    skeleton: AnimalSkeleton | None = None,
    config: FastRetargetConfig | None = None,
) -> tuple[RobotMotion, RetargetDiagnostics]:
    """Retarget an AnimalMotion with warm-started damped least-squares IK."""

    skeleton = skeleton or get_skeleton(motion.metadata["skeleton_id"])
    config = config or FastRetargetConfig()
    try:
        (
            root_position,
            root_rotation,
            target_feet,
            input_usable,
            root_status,
            root_scale,
            leg_scales,
        ) = _build_targets(motion, robot, skeleton, config)
    except (KeyError, ValueError) as error:
        if isinstance(error, FastRetargetError):
            raise
        raise FastRetargetError(f"cannot build retarget targets: {error}") from error
    frames = motion.frame_count
    dofs = len(robot.config.dof_order)
    dof_position = np.empty((frames, dofs), dtype=np.float64)
    achieved_feet = np.empty((frames, 4, 3), dtype=np.float64)
    solver_status = np.empty(frames, dtype=np.int16)
    solver_residual = np.full(frames, np.nan, dtype=np.float64)
    iterations = np.zeros(frames, dtype=np.int16)
    frame_valid = np.zeros(frames, dtype=np.bool_)
    q = np.asarray(robot.config.default_dof_position, dtype=np.float64).copy()
    lower = robot.joint_ranges[:, 0]
    upper = robot.joint_ranges[:, 1]
    identity = np.eye(dofs, dtype=np.float64)

    for frame in range(frames):
        if not input_usable[frame]:
            dof_position[frame] = q
            robot.set_pose(root_position[frame], root_rotation[frame], q)
            achieved_feet[frame] = robot.foot_positions()
            solver_status[frame] = SolverStatus.MISSING_INPUT
            continue
        try:
            for iteration in range(1, config.max_iterations + 1):
                robot.set_pose(root_position[frame], root_rotation[frame], q)
                current = robot.foot_positions()
                error = (target_feet[frame] - current).reshape(-1)
                residual = float(np.sqrt(np.mean(error * error)))
                if residual <= config.residual_tolerance:
                    break
                jacobian = robot.foot_jacobians().reshape(12, dofs)
                normal = jacobian.T @ jacobian + config.damping**2 * identity
                step = np.linalg.solve(normal, jacobian.T @ error)
                step = np.clip(step, -config.max_joint_step, config.max_joint_step)
                q = np.clip(q + step, lower, upper)
            robot.set_pose(root_position[frame], root_rotation[frame], q)
            achieved = robot.foot_positions()
            error = (target_feet[frame] - achieved).reshape(-1)
            residual = float(np.sqrt(np.mean(error * error)))
            dof_position[frame] = q
            achieved_feet[frame] = achieved
            solver_residual[frame] = residual
            iterations[frame] = iteration
            if residual >= config.unreachable_residual:
                solver_status[frame] = SolverStatus.UNREACHABLE
            elif residual > config.residual_tolerance:
                solver_status[frame] = SolverStatus.MAX_ITER
                frame_valid[frame] = True
            else:
                solver_status[frame] = (
                    SolverStatus.DEGRADED_ROOT
                    if root_status[frame] == SolverStatus.DEGRADED_ROOT
                    else SolverStatus.OK
                )
                frame_valid[frame] = True
        except (FloatingPointError, ValueError, np.linalg.LinAlgError):
            dof_position[frame] = q
            robot.set_pose(root_position[frame], root_rotation[frame], q)
            achieved_feet[frame] = robot.foot_positions()
            solver_status[frame] = SolverStatus.NUMERICAL_ERROR

    contact = motion.contact_probability.copy()
    if np.any(~np.isfinite(contact)):
        estimated = estimate_contact_probability(motion, skeleton)
        contact = np.where(np.isfinite(contact), contact, estimated)
    dof_velocity = _differentiate_linear(motion.timestamps, dof_position)
    root_linear_velocity = _differentiate_linear(motion.timestamps, root_position)
    root_angular_velocity = _differentiate_rotation(motion.timestamps, root_rotation)
    asset = get_asset_spec(robot.config.asset_id)
    result = RobotMotion(
        timestamps=motion.timestamps,
        dof_names=robot.config.dof_order,
        root_position=root_position.astype(np.float32),
        root_rotation=root_rotation.astype(np.float32),
        dof_position=dof_position.astype(np.float32),
        root_linear_velocity=root_linear_velocity.astype(np.float32),
        root_angular_velocity=root_angular_velocity.astype(np.float32),
        dof_velocity=dof_velocity.astype(np.float32),
        foot_contact_probability=contact.astype(np.float32),
        frame_valid=frame_valid,
        solver_status=solver_status,
        solver_residual=solver_residual.astype(np.float32),
        metadata={
            "coordinate_frame": "gqmr_world_x_forward_y_left_z_up",
            "quaternion_order": "wxyz",
            "root_velocity_frame": "world",
            "model_id": robot.config.id,
            "model_source_commit": asset.commit,
            "model_sha256": robot.config.model_sha256,
            "robot_config_sha256": robot.config.sha256,
            "contact_order": ["FL", "FR", "RL", "RR"],
            "source_motion_sha256": motion_sha256(motion),
            "retarget_config": {
                "mode": "fast_semantic_limb_dls_v2",
                **asdict(config),
                "resolved_root_translation_scale": root_scale,
                "resolved_leg_motion_scales": leg_scales,
            },
            "created_by": {"gqmr_version": __version__},
        },
    )
    diagnostics = RetargetDiagnostics(
        target_foot_positions=np.ascontiguousarray(target_feet, dtype=np.float32),
        achieved_foot_positions=np.ascontiguousarray(achieved_feet, dtype=np.float32),
        iterations=iterations,
        root_translation_scale=root_scale,
        leg_motion_scales=leg_scales,
        root_position_correction=np.zeros((frames, 3), dtype=np.float32),
        root_rotation_correction=np.zeros((frames, 3), dtype=np.float32),
    )
    return result, diagnostics
