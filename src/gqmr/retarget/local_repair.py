"""Serializable local-repair contracts and deterministic solver execution."""

from __future__ import annotations

import uuid
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Literal, Protocol

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
)
from scipy.spatial.transform import Rotation

from gqmr.core.coordinates import wxyz_to_xyzw, xyzw_to_wxyz
from gqmr.core.derivatives import angular_velocity_world, linear_velocity
from gqmr.core.errors import GQMRError
from gqmr.core.io import motion_sha256
from gqmr.core.motion import RobotMotion, SolverStatus
from gqmr.retarget.preprocess import GroundEstimate
from gqmr.robots import LEG_ORDER, RobotModel

FootRepairMode = Literal["auto", "lock", "unlock"]
_SHA256_LENGTH = 64
_ROBOT_ARRAY_FIELDS = (
    "root_position",
    "root_rotation",
    "dof_position",
    "root_linear_velocity",
    "root_angular_velocity",
    "dof_velocity",
    "foot_contact_probability",
    "frame_valid",
    "solver_status",
    "solver_residual",
)


class LocalRepairError(GQMRError, ValueError):
    """Raised when a local repair or its replay violates the A1 contract."""


class FootRepairModes(BaseModel):
    """Explicit per-foot contact policy in canonical FL/FR/RL/RR order."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    FL: FootRepairMode = "auto"
    FR: FootRepairMode = "auto"
    RL: FootRepairMode = "auto"
    RR: FootRepairMode = "auto"


class LocalRepairConfig(BaseModel):
    """Stable, JSON-serializable parameters for one local repair request."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: Literal["1.0"] = "1.0"
    root_height_offset_m: float = Field(default=0.0, ge=-0.25, le=0.25)
    root_translation_scale: float = Field(default=1.0, ge=0.25, le=2.0)
    root_tilt_scale: float = Field(default=1.0, ge=0.0, le=1.5)
    limb_target_scale: float = Field(default=1.0, ge=0.25, le=2.0)
    smoothing_strength: float = Field(default=0.0, ge=0.0, le=1.0)
    foot_modes: FootRepairModes = Field(default_factory=FootRepairModes)
    reestimate_contact: bool = False
    reestimate_ground: bool = False


class LocalRepairDiagnostics(BaseModel):
    """Deterministic solver diagnostics recorded in results and edit commands."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: Literal["1.0"] = "1.0"
    solver: str = Field(min_length=1)
    solver_version: str = Field(min_length=1)
    frames_processed: int = Field(ge=1)
    iterations: int = Field(default=0, ge=0)
    converged: bool
    residual_rmse_before_m: float | None = Field(default=None, ge=0.0)
    residual_rmse_after_m: float | None = Field(default=None, ge=0.0)
    status_counts: dict[str, int] = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    details: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("status_counts")
    @classmethod
    def valid_status_counts(cls, value: dict[str, int]) -> dict[str, int]:
        if any(not key.strip() or count < 0 for key, count in value.items()):
            raise ValueError("status_counts keys must be non-empty and counts non-negative")
        return dict(sorted(value.items()))

    @field_validator("warnings")
    @classmethod
    def valid_warnings(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not warning.strip() for warning in value):
            raise ValueError("warnings must contain non-empty text")
        return value


@dataclass(frozen=True, slots=True)
class LocalRepairSolverOutput:
    """Raw deterministic output produced by an A2/A3 local-repair solver."""

    motion: RobotMotion
    applied_config: LocalRepairConfig
    diagnostics: LocalRepairDiagnostics


class LocalRepairSolver(Protocol):
    def __call__(
        self,
        motion: RobotMotion,
        frame_range: tuple[int, int],
        config: LocalRepairConfig,
    ) -> LocalRepairSolverOutput: ...


@dataclass(frozen=True, slots=True)
class LocalRepairTargets:
    """A2 target trajectories consumed by the A3 interval solver."""

    frame_range: tuple[int, int]
    root_position: np.ndarray
    root_rotation: np.ndarray
    foot_positions: np.ndarray
    contact_probability: np.ndarray
    applied_config: LocalRepairConfig
    ground: GroundEstimate | None = None

    def __post_init__(self) -> None:
        root_position = np.ascontiguousarray(self.root_position, dtype=np.float64)
        root_rotation = np.ascontiguousarray(self.root_rotation, dtype=np.float64)
        foot_positions = np.ascontiguousarray(self.foot_positions, dtype=np.float64)
        contact = np.ascontiguousarray(self.contact_probability, dtype=np.float64)
        frames = len(root_position)
        _validate_frame_range(self.frame_range, frames)
        expected = {
            "root_position": (frames, 3),
            "root_rotation": (frames, 4),
            "foot_positions": (frames, 4, 3),
            "contact_probability": (frames, 4),
        }
        actual = {
            "root_position": root_position.shape,
            "root_rotation": root_rotation.shape,
            "foot_positions": foot_positions.shape,
            "contact_probability": contact.shape,
        }
        for field, shape in expected.items():
            if actual[field] != shape:
                raise LocalRepairError(
                    f"{field} must have shape {shape}, got {actual[field]}"
                )
        start, stop = self.frame_range
        selected = slice(start, stop + 1)
        if not (
            np.all(np.isfinite(root_position[selected]))
            and np.all(np.isfinite(root_rotation[selected]))
            and np.all(np.isfinite(foot_positions[selected]))
        ):
            raise LocalRepairError("selected local repair targets must be finite")
        norms = np.linalg.norm(root_rotation[selected], axis=1)
        if np.any(np.abs(norms - 1.0) >= 1e-5):
            raise LocalRepairError("selected root rotations must be unit quaternions")
        known_contact = np.isfinite(contact)
        if np.any((contact[known_contact] < 0.0) | (contact[known_contact] > 1.0)):
            raise LocalRepairError("contact_probability must be in [0,1] or NaN")
        object.__setattr__(self, "root_position", root_position)
        object.__setattr__(self, "root_rotation", root_rotation)
        object.__setattr__(self, "foot_positions", foot_positions)
        object.__setattr__(self, "contact_probability", contact)


@dataclass(frozen=True, slots=True)
class LocalRepairSolveConfig:
    """Deterministic numerical settings for the A3 interval solver."""

    max_iterations: int = 48
    damping: float = 0.01
    root_position_tracking: float = 2.0
    root_rotation_tracking: float = 2.0
    joint_tracking: float = 0.002
    contact_weight: float = 2.0
    contact_threshold: float = 0.5
    residual_tolerance: float = 0.002
    unreachable_residual: float = 0.10
    max_root_step_m: float = 0.01
    max_root_rotation_step_rad: float = np.deg2rad(2.0)
    max_joint_step_rad: float = 0.12
    maximum_buffer_seconds: float = 0.12
    buffer_duration_ratio: float = 0.20

    def __post_init__(self) -> None:
        positive = (
            self.damping,
            self.root_position_tracking,
            self.root_rotation_tracking,
            self.joint_tracking,
            self.contact_weight,
            self.residual_tolerance,
            self.unreachable_residual,
            self.max_root_step_m,
            self.max_root_rotation_step_rad,
            self.max_joint_step_rad,
            self.maximum_buffer_seconds,
            self.buffer_duration_ratio,
        )
        if self.max_iterations <= 0 or any(
            not np.isfinite(value) or value <= 0.0 for value in positive
        ):
            raise LocalRepairError(
                "local repair solve settings must be finite and positive"
            )
        if not 0.0 <= self.contact_threshold <= 1.0:
            raise LocalRepairError("contact_threshold must be in [0,1]")
        if self.unreachable_residual < self.residual_tolerance:
            raise LocalRepairError(
                "unreachable_residual must not be below residual_tolerance"
            )


@dataclass(frozen=True, slots=True)
class LocalRepairIntervalSolver:
    """Robot-bound A3 solver compatible with the A1 command/replay protocol."""

    robot: RobotModel
    solve_config: LocalRepairSolveConfig = field(
        default_factory=LocalRepairSolveConfig
    )

    def __call__(
        self,
        motion: RobotMotion,
        frame_range: tuple[int, int],
        config: LocalRepairConfig,
    ) -> LocalRepairSolverOutput:
        return solve_local_repair(
            motion,
            self.robot,
            frame_range,
            config,
            solve_config=self.solve_config,
        )


@dataclass(frozen=True, slots=True)
class LocalRepairResult:
    """Validated local repair plus the information needed for exact replay."""

    motion: RobotMotion
    frame_range: tuple[int, int]
    requested_config: LocalRepairConfig
    applied_config: LocalRepairConfig
    diagnostics: LocalRepairDiagnostics
    input_motion_sha256: str
    output_motion_sha256: str

    def __post_init__(self) -> None:
        _validate_frame_range(self.frame_range, self.motion.frame_count)
        _validate_sha256(self.input_motion_sha256, field="input_motion_sha256")
        _validate_sha256(self.output_motion_sha256, field="output_motion_sha256")
        if motion_sha256(self.motion) != self.output_motion_sha256:
            raise LocalRepairError("output_motion_sha256 does not match repaired motion")


class LocalRepairCommandParameters(BaseModel):
    """Strict payload stored inside the generic project edit-command envelope."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    frame_range: tuple[int, int]
    requested_config: LocalRepairConfig
    applied_config: LocalRepairConfig
    diagnostics: LocalRepairDiagnostics
    input_motion_sha256: str
    output_motion_sha256: str

    @field_validator("frame_range")
    @classmethod
    def valid_non_negative_range(cls, value: tuple[int, int]) -> tuple[int, int]:
        if len(value) != 2 or value[0] < 0 or value[1] < value[0]:
            raise ValueError("frame_range must be an inclusive non-negative interval")
        return value

    @field_validator("input_motion_sha256", "output_motion_sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        _validate_sha256(value, field="motion_sha256")
        return value


class LocalRepairCommand(BaseModel):
    """Typed, serializable command that can replay and verify one repair."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    command_id: str
    kind: Literal["local_repair"] = "local_repair"
    resource_id: str
    parameters: LocalRepairCommandParameters
    created_at: str

    @field_validator("command_id", "resource_id")
    @classmethod
    def valid_uuid(cls, value: str) -> str:
        try:
            uuid.UUID(value)
        except ValueError as error:
            raise ValueError("command/resource ID must be a UUID") from error
        return value

    @field_validator("created_at")
    @classmethod
    def valid_timestamp(cls, value: str) -> str:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("created_at must be RFC3339") from error
        if parsed.tzinfo is None:
            raise ValueError("created_at must include a timezone")
        return value

    @classmethod
    def from_result(
        cls,
        result: LocalRepairResult,
        *,
        resource_id: str,
        command_id: str | None = None,
        created_at: str | None = None,
    ) -> "LocalRepairCommand":
        return cls(
            command_id=command_id or str(uuid.uuid4()),
            resource_id=resource_id,
            parameters=LocalRepairCommandParameters(
                frame_range=result.frame_range,
                requested_config=result.requested_config,
                applied_config=result.applied_config,
                diagnostics=result.diagnostics,
                input_motion_sha256=result.input_motion_sha256,
                output_motion_sha256=result.output_motion_sha256,
            ),
            created_at=created_at
            or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        )

    @classmethod
    def from_edit_command(cls, command: Any) -> "LocalRepairCommand":
        if getattr(command, "kind", None) != "local_repair":
            raise LocalRepairError("edit command is not a local repair")
        return cls.model_validate(
            {
                "command_id": command.command_id,
                "kind": command.kind,
                "resource_id": command.resource_id,
                "parameters": command.parameters,
                "created_at": command.created_at,
            }
        )

    def to_edit_command(self) -> Any:
        from gqmr.project.model import EditCommand

        return EditCommand(
            command_id=self.command_id,
            kind=self.kind,
            resource_id=self.resource_id,
            parameters=self.parameters.model_dump(mode="json"),
            created_at=self.created_at,
        )


def _validate_sha256(value: str, *, field: str) -> None:
    if len(value) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise LocalRepairError(f"{field} must be 64 lowercase hexadecimal characters")


def _validate_frame_range(frame_range: tuple[int, int], frame_count: int) -> None:
    if (
        len(frame_range) != 2
        or not all(isinstance(index, int) and not isinstance(index, bool) for index in frame_range)
        or not 0 <= frame_range[0] <= frame_range[1] < frame_count
    ):
        raise LocalRepairError("frame_range must be an inclusive interval inside the motion")


def _validate_repaired_motion(
    before: RobotMotion,
    after: RobotMotion,
    frame_range: tuple[int, int],
) -> None:
    if before.dof_names != after.dof_names or not np.array_equal(
        before.timestamps, after.timestamps
    ):
        raise LocalRepairError("local repair must preserve timestamps and DOF order")
    if before.frame_count != after.frame_count:
        raise LocalRepairError("local repair must preserve frame count")
    start, stop = frame_range
    outside = np.ones(before.frame_count, dtype=np.bool_)
    outside[start : stop + 1] = False
    for field in _ROBOT_ARRAY_FIELDS:
        if not np.array_equal(getattr(before, field)[outside], getattr(after, field)[outside]):
            raise LocalRepairError(f"local repair changed {field} outside frame_range")


def _copy_robot_motion(motion: RobotMotion) -> RobotMotion:
    fields = {
        field: getattr(motion, field).copy() for field in _ROBOT_ARRAY_FIELDS
    }
    return replace(
        motion,
        timestamps=motion.timestamps.copy(),
        metadata=deepcopy(motion.metadata),
        **fields,
    )


def _validate_robot_compatibility(motion: RobotMotion, robot: RobotModel) -> None:
    if motion.dof_names != tuple(robot.config.dof_order):
        raise LocalRepairError("RobotMotion DOF order does not match the repair robot")
    if motion.metadata["model_sha256"] != robot.config.model_sha256:
        raise LocalRepairError("RobotMotion model hash does not match the repair robot")
    if motion.metadata["robot_config_sha256"] != robot.config.sha256:
        raise LocalRepairError("RobotMotion robot-config hash does not match the repair robot")


def _foot_trajectories(motion: RobotMotion, robot: RobotModel) -> np.ndarray:
    result = np.full((motion.frame_count, 4, 3), np.nan, dtype=np.float64)
    for frame in range(motion.frame_count):
        values = (
            motion.root_position[frame],
            motion.root_rotation[frame],
            motion.dof_position[frame],
        )
        if not all(np.all(np.isfinite(value)) for value in values):
            continue
        robot.set_pose(*values)
        result[frame] = robot.foot_positions()
    return result


def _trajectory_speed(timestamps: np.ndarray, positions: np.ndarray) -> np.ndarray:
    if len(timestamps) < 2:
        return np.zeros(positions.shape[:-1], dtype=np.float64)
    edge_order = 2 if len(timestamps) >= 3 else 1
    velocity = np.gradient(positions, timestamps, axis=0, edge_order=edge_order)
    return np.linalg.norm(velocity, axis=-1)


def _estimate_local_ground(
    timestamps: np.ndarray,
    foot_positions: np.ndarray,
    valid: np.ndarray,
    frame_range: tuple[int, int],
) -> GroundEstimate:
    start, stop = frame_range
    selected = np.zeros(len(timestamps), dtype=np.bool_)
    selected[start : stop + 1] = True
    usable = valid[:, None] & selected[:, None] & np.all(
        np.isfinite(foot_positions), axis=2
    )
    samples = foot_positions[usable]
    if not len(samples):
        raise LocalRepairError("selected interval has no finite foot positions")
    speed = _trajectory_speed(timestamps, foot_positions)
    height_limit = float(np.percentile(samples[:, 2], 30.0))
    usable_speeds = speed[usable]
    speed_limit = float(np.percentile(usable_speeds, 45.0))
    candidates = usable & (foot_positions[..., 2] <= height_limit) & (
        speed <= speed_limit
    )
    ground_samples = foot_positions[candidates]
    if len(ground_samples) < 3:
        ground_samples = foot_positions[usable & (foot_positions[..., 2] <= height_limit)]
    if len(ground_samples) < 3:
        height = float(np.percentile(samples[:, 2], 5.0))
        return GroundEstimate(
            point=np.array([0.0, 0.0, height], dtype=np.float32),
            normal=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            rmse=0.0,
            candidate_count=int(len(ground_samples)),
        )
    center = np.mean(ground_samples[:, :2], axis=0)
    design = np.column_stack(
        (
            ground_samples[:, 0] - center[0],
            ground_samples[:, 1] - center[1],
            np.ones(len(ground_samples)),
        )
    )
    coefficients = np.linalg.lstsq(design, ground_samples[:, 2], rcond=None)[0]
    fitted = design @ coefficients
    normal = np.array([-coefficients[0], -coefficients[1], 1.0])
    normal /= np.linalg.norm(normal)
    point = np.array([center[0], center[1], coefficients[2]], dtype=np.float64)
    return GroundEstimate(
        point=np.ascontiguousarray(point, dtype=np.float32),
        normal=np.ascontiguousarray(normal, dtype=np.float32),
        rmse=float(np.sqrt(np.mean((ground_samples[:, 2] - fitted) ** 2))),
        candidate_count=int(len(ground_samples)),
    )


def _estimate_local_contact(
    timestamps: np.ndarray,
    foot_positions: np.ndarray,
    ground: GroundEstimate,
) -> np.ndarray:
    speed = _trajectory_speed(timestamps, foot_positions)
    point = np.asarray(ground.point, dtype=np.float64)
    normal = np.asarray(ground.normal, dtype=np.float64)
    height = np.einsum("tli,i->tl", foot_positions - point, normal)
    with np.errstate(over="ignore", invalid="ignore"):
        height_score = 1.0 / (1.0 + np.exp((height - 0.035) / 0.01))
        speed_score = 1.0 / (1.0 + np.exp((speed - 0.35) / 0.08))
    probability = height_score * speed_score
    probability[~np.all(np.isfinite(foot_positions), axis=2)] = np.nan
    return np.clip(probability, 0.0, 1.0)


def _scale_root_tilt(quaternions: np.ndarray, scale: float) -> np.ndarray:
    rotations = Rotation.from_quat(wxyz_to_xyzw(quaternions))
    forward = rotations.apply(np.array([1.0, 0.0, 0.0]))
    horizontal = np.linalg.norm(forward[:, :2], axis=1)
    if np.any(horizontal < 1e-8):
        raise LocalRepairError("selected root heading is vertically degenerate")
    yaw = np.arctan2(forward[:, 1], forward[:, 0])
    rotation_vectors = np.zeros((len(yaw), 3), dtype=np.float64)
    rotation_vectors[:, 2] = yaw
    heading = Rotation.from_rotvec(rotation_vectors)
    tilt = heading.inv() * rotations
    scaled = heading * Rotation.from_rotvec(tilt.as_rotvec() * scale)
    return xyzw_to_wxyz(scaled.as_quat())


def _smooth_linear(values: np.ndarray, start: int, stop: int, strength: float) -> None:
    if strength <= 0.0 or stop <= start:
        return
    source = values[start : stop + 1].copy()
    previous = np.concatenate((source[:1], source[:-1]), axis=0)
    following = np.concatenate((source[1:], source[-1:]), axis=0)
    smoothed = 0.25 * previous + 0.5 * source + 0.25 * following
    values[start : stop + 1] = source + strength * (smoothed - source)


def _smooth_rotations(
    quaternions: np.ndarray, start: int, stop: int, strength: float
) -> None:
    if strength <= 0.0 or stop <= start:
        return
    source = Rotation.from_quat(wxyz_to_xyzw(quaternions[start : stop + 1]))
    result: list[Rotation] = []
    for index in range(len(source)):
        current = source[index]
        previous = source[max(0, index - 1)]
        following = source[min(len(source) - 1, index + 1)]
        correction = 0.25 * strength * (
            (previous * current.inv()).as_rotvec()
            + (following * current.inv()).as_rotvec()
        )
        result.append(Rotation.from_rotvec(correction) * current)
    quaternions[start : stop + 1] = xyzw_to_wxyz(
        Rotation.concatenate(result).as_quat()
    )


def build_local_repair_targets(
    motion: RobotMotion,
    robot: RobotModel,
    frame_range: tuple[int, int],
    config: LocalRepairConfig,
) -> LocalRepairTargets:
    """Apply A2 target corrections without solving or changing the input motion."""

    _validate_frame_range(frame_range, motion.frame_count)
    _validate_robot_compatibility(motion, robot)
    applied = LocalRepairConfig.model_validate(config.model_dump(mode="json"))
    start, stop = frame_range
    selected = slice(start, stop + 1)
    root_position = motion.root_position.astype(np.float64).copy()
    root_rotation = motion.root_rotation.astype(np.float64).copy()
    foot_positions = _foot_trajectories(motion, robot)
    contact = motion.foot_contact_probability.astype(np.float64).copy()
    if not (
        np.all(np.isfinite(root_position[selected]))
        and np.all(np.isfinite(root_rotation[selected]))
        and np.all(np.isfinite(foot_positions[selected]))
    ):
        raise LocalRepairError("selected interval has non-finite kinematic targets")

    source_root_position = root_position[selected].copy()
    anchor = source_root_position[0]
    root_position[selected] = anchor + applied.root_translation_scale * (
        source_root_position - anchor
    )
    root_position[start : stop + 1, 2] += applied.root_height_offset_m
    root_rotation[selected] = _scale_root_tilt(
        root_rotation[selected], applied.root_tilt_scale
    )

    source_rotation = Rotation.from_quat(wxyz_to_xyzw(motion.root_rotation[selected]))
    target_rotation = Rotation.from_quat(wxyz_to_xyzw(root_rotation[selected]))
    source_root = motion.root_position[selected].astype(np.float64)
    local_feet = np.empty_like(foot_positions[selected])
    for offset in range(stop - start + 1):
        local_feet[offset] = source_rotation[offset].inv().apply(
            foot_positions[start + offset] - source_root[offset]
        )
    neutral_feet = np.median(local_feet, axis=0)
    scaled_local_feet = neutral_feet + applied.limb_target_scale * (
        local_feet - neutral_feet
    )
    for offset in range(stop - start + 1):
        foot_positions[start + offset] = root_position[start + offset] + target_rotation[
            offset
        ].apply(scaled_local_feet[offset])

    ground = (
        _estimate_local_ground(
            motion.timestamps, foot_positions, motion.frame_valid, frame_range
        )
        if applied.reestimate_ground or applied.reestimate_contact
        else None
    )
    if applied.reestimate_contact:
        assert ground is not None
        estimated = _estimate_local_contact(
            motion.timestamps, foot_positions, ground
        )
        contact[selected] = estimated[selected]
    modes = applied.foot_modes.model_dump(mode="json")
    for leg_index, leg in enumerate(LEG_ORDER):
        if modes[leg] == "lock":
            contact[start : stop + 1, leg_index] = 1.0
        elif modes[leg] == "unlock":
            contact[start : stop + 1, leg_index] = 0.0

    _smooth_linear(root_position, start, stop, applied.smoothing_strength)
    _smooth_rotations(root_rotation, start, stop, applied.smoothing_strength)
    _smooth_linear(foot_positions, start, stop, applied.smoothing_strength)
    return LocalRepairTargets(
        frame_range=frame_range,
        root_position=root_position,
        root_rotation=root_rotation,
        foot_positions=foot_positions,
        contact_probability=contact,
        applied_config=applied,
        ground=ground,
    )


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64
    )


def _adaptive_transition_weights(
    timestamps: np.ndarray,
    frame_range: tuple[int, int],
    config: LocalRepairSolveConfig,
) -> tuple[np.ndarray, int]:
    start, stop = frame_range
    count = stop - start + 1
    if count < 3:
        return np.ones(count, dtype=np.float64), 0
    median_dt = float(np.median(np.diff(timestamps)))
    duration = float(timestamps[stop] - timestamps[start])
    buffer_seconds = min(
        config.maximum_buffer_seconds,
        max(median_dt, duration * config.buffer_duration_ratio),
    )
    buffer_frames = min(
        max(1, int(round(buffer_seconds / median_dt))), (count - 1) // 2
    )
    distance = np.minimum(np.arange(count), np.arange(count)[::-1]).astype(
        np.float64
    )
    phase = np.clip(distance / buffer_frames, 0.0, 1.0)
    weights = phase**3 * (10.0 - 15.0 * phase + 6.0 * phase**2)
    return np.ascontiguousarray(weights), buffer_frames


def _blend_rotation_targets(
    source_wxyz: np.ndarray, target_wxyz: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    source = Rotation.from_quat(wxyz_to_xyzw(source_wxyz))
    target = Rotation.from_quat(wxyz_to_xyzw(target_wxyz))
    delta = (target * source.inv()).as_rotvec()
    blended = Rotation.from_rotvec(weights[:, None] * delta) * source
    return np.ascontiguousarray(xyzw_to_wxyz(blended.as_quat()))


def _contact_locked_foot_targets(
    foot_positions: np.ndarray,
    contact_probability: np.ndarray,
    frame_range: tuple[int, int],
    threshold: float,
) -> np.ndarray:
    desired = foot_positions.copy()
    selected_start, selected_stop = frame_range
    for leg in range(4):
        contact = np.isfinite(contact_probability[:, leg]) & (
            contact_probability[:, leg] >= threshold
        )
        frame = selected_start
        while frame <= selected_stop:
            if not contact[frame]:
                frame += 1
                continue
            stop = frame
            while stop < selected_stop and contact[stop + 1]:
                stop += 1
            desired[frame : stop + 1, leg] = np.median(
                foot_positions[frame : stop + 1, leg], axis=0
            )
            frame = stop + 1
    return desired


def _linear_derivative(timestamps: np.ndarray, values: np.ndarray) -> np.ndarray:
    if len(timestamps) == 1:
        return np.zeros_like(values, dtype=np.float64)
    if len(timestamps) == 2:
        slope = (values[1] - values[0]) / (timestamps[1] - timestamps[0])
        return np.stack((slope, slope))
    return linear_velocity(timestamps, values)


def _angular_derivative(timestamps: np.ndarray, quaternions: np.ndarray) -> np.ndarray:
    if len(timestamps) == 1:
        return np.zeros((1, 3), dtype=np.float64)
    if len(timestamps) == 2:
        rotations = Rotation.from_quat(wxyz_to_xyzw(quaternions))
        velocity = (rotations[1] * rotations[0].inv()).as_rotvec() / (
            timestamps[1] - timestamps[0]
        )
        return np.stack((velocity, velocity))
    return angular_velocity_world(timestamps, quaternions)


def solve_local_repair(
    motion: RobotMotion,
    robot: RobotModel,
    frame_range: tuple[int, int],
    config: LocalRepairConfig,
    *,
    solve_config: LocalRepairSolveConfig | None = None,
) -> LocalRepairSolverOutput:
    """Solve A2 targets with joint/root DLS and an SO(3) interval transition."""

    _validate_frame_range(frame_range, motion.frame_count)
    _validate_robot_compatibility(motion, robot)
    settings = solve_config or LocalRepairSolveConfig()
    targets = build_local_repair_targets(motion, robot, frame_range, config)
    start, stop = frame_range
    active = np.arange(start, stop + 1, dtype=np.int32)
    weights, buffer_frames = _adaptive_transition_weights(
        motion.timestamps, frame_range, settings
    )

    root_target = motion.root_position[active].astype(np.float64) + weights[:, None] * (
        targets.root_position[active]
        - motion.root_position[active].astype(np.float64)
    )
    rotation_target = _blend_rotation_targets(
        motion.root_rotation[active], targets.root_rotation[active], weights
    )
    source_feet = _foot_trajectories(motion, robot)
    locked_foot_target = _contact_locked_foot_targets(
        targets.foot_positions,
        targets.contact_probability,
        frame_range,
        settings.contact_threshold,
    )
    foot_target = source_feet[active] + weights[:, None, None] * (
        locked_foot_target[active] - source_feet[active]
    )
    full_foot_target = source_feet.copy()
    full_foot_target[active] = foot_target

    root_position = motion.root_position.astype(np.float64).copy()
    root_rotation = motion.root_rotation.astype(np.float64).copy()
    q = motion.dof_position.astype(np.float64).copy()
    lower, upper = robot.joint_ranges[:, 0], robot.joint_ranges[:, 1]
    dofs = q.shape[1]
    root_translation_jacobian = np.tile(np.eye(3, dtype=np.float64), (4, 1))
    iterations = np.zeros(motion.frame_count, dtype=np.int32)
    numerical_error = np.zeros(motion.frame_count, dtype=np.bool_)
    before_residual = np.empty(len(active), dtype=np.float64)
    after_residual = np.empty(len(active), dtype=np.float64)

    for offset, frame in enumerate(active):
        base_q = q[frame].copy()
        current_position = root_target[offset].copy()
        current_rotation = Rotation.from_quat(
            wxyz_to_xyzw(rotation_target[offset])
        )
        current_q = base_q.copy()
        desired_feet = full_foot_target[frame]
        robot.set_pose(
            motion.root_position[frame], motion.root_rotation[frame], base_q
        )
        initial_error = desired_feet - robot.foot_positions()
        before_residual[offset] = float(
            np.sqrt(np.mean(initial_error * initial_error))
        )
        try:
            for iteration in range(1, settings.max_iterations + 1):
                current_quaternion = xyzw_to_wxyz(current_rotation.as_quat())
                robot.set_pose(current_position, current_quaternion, current_q)
                current_feet = robot.foot_positions()
                error = (desired_feet - current_feet).reshape(-1)
                residual = float(np.sqrt(np.mean(error * error)))
                if residual <= settings.residual_tolerance:
                    break
                contact = np.nan_to_num(
                    targets.contact_probability[frame], nan=0.0
                )
                row_weights = np.repeat(
                    1.0 + settings.contact_weight * contact, 3
                )
                joint_jacobian = robot.foot_jacobians().reshape(12, dofs)
                rotation_jacobian = np.vstack(
                    [
                        -_skew(foot - current_position)
                        for foot in current_feet
                    ]
                )
                jacobian = np.column_stack(
                    (root_translation_jacobian, rotation_jacobian, joint_jacobian)
                )
                weighted_jacobian = jacobian * row_weights[:, None]
                weighted_error = error * row_weights
                normal = weighted_jacobian.T @ weighted_jacobian
                regularizer = np.concatenate(
                    (
                        np.full(3, settings.root_position_tracking),
                        np.full(3, settings.root_rotation_tracking),
                        np.full(dofs, settings.joint_tracking),
                    )
                )
                normal += np.diag(regularizer + settings.damping**2)
                target_rotation_delta = (
                    Rotation.from_quat(wxyz_to_xyzw(rotation_target[offset]))
                    * current_rotation.inv()
                ).as_rotvec()
                tracking_error = np.concatenate(
                    (
                        root_target[offset] - current_position,
                        target_rotation_delta,
                        base_q - current_q,
                    )
                )
                right = weighted_jacobian.T @ weighted_error
                right += regularizer * tracking_error
                step = np.linalg.solve(normal, right)
                root_step = step[:3]
                root_norm = float(np.linalg.norm(root_step))
                if root_norm > settings.max_root_step_m:
                    root_step *= settings.max_root_step_m / root_norm
                rotation_step = step[3:6]
                rotation_norm = float(np.linalg.norm(rotation_step))
                if rotation_norm > settings.max_root_rotation_step_rad:
                    rotation_step *= (
                        settings.max_root_rotation_step_rad / rotation_norm
                    )
                joint_step = np.clip(
                    step[6:],
                    -settings.max_joint_step_rad,
                    settings.max_joint_step_rad,
                )
                current_position += root_step
                current_rotation = (
                    Rotation.from_rotvec(rotation_step) * current_rotation
                )
                current_q = np.clip(current_q + joint_step, lower, upper)
            iterations[frame] = iteration
            root_position[frame] = current_position
            root_rotation[frame] = xyzw_to_wxyz(current_rotation.as_quat())
            q[frame] = current_q
            robot.set_pose(root_position[frame], root_rotation[frame], q[frame])
            final_error = desired_feet - robot.foot_positions()
            after_residual[offset] = float(
                np.sqrt(np.mean(final_error * final_error))
            )
        except (FloatingPointError, ValueError, np.linalg.LinAlgError):
            numerical_error[frame] = True
            iterations[frame] = 0
            root_position[frame] = motion.root_position[frame]
            root_rotation[frame] = motion.root_rotation[frame]
            q[frame] = motion.dof_position[frame]
            after_residual[offset] = before_residual[offset]

    root_linear_velocity = motion.root_linear_velocity.copy()
    root_angular_velocity = motion.root_angular_velocity.copy()
    dof_velocity = motion.dof_velocity.copy()
    recomputed_root_linear = _linear_derivative(motion.timestamps, root_position)
    recomputed_root_angular = _angular_derivative(motion.timestamps, root_rotation)
    recomputed_dof = _linear_derivative(motion.timestamps, q)
    root_linear_velocity[active] = recomputed_root_linear[active]
    root_angular_velocity[active] = recomputed_root_angular[active]
    dof_velocity[active] = recomputed_dof[active]

    contact_probability = motion.foot_contact_probability.copy()
    contact_probability[active] = targets.contact_probability[active]
    frame_valid = motion.frame_valid.copy()
    solver_status = motion.solver_status.copy()
    solver_residual = motion.solver_residual.copy()
    for offset, frame in enumerate(active):
        solver_residual[frame] = after_residual[offset]
        if numerical_error[frame]:
            frame_valid[frame] = False
            solver_status[frame] = SolverStatus.NUMERICAL_ERROR
        elif after_residual[offset] >= settings.unreachable_residual:
            frame_valid[frame] = False
            solver_status[frame] = SolverStatus.UNREACHABLE
        elif after_residual[offset] > settings.residual_tolerance:
            frame_valid[frame] = True
            solver_status[frame] = SolverStatus.MAX_ITER
        else:
            frame_valid[frame] = True
            solver_status[frame] = SolverStatus.OK

    repaired = replace(
        motion,
        root_position=root_position,
        root_rotation=root_rotation,
        dof_position=q,
        root_linear_velocity=root_linear_velocity,
        root_angular_velocity=root_angular_velocity,
        dof_velocity=dof_velocity,
        foot_contact_probability=contact_probability,
        frame_valid=frame_valid,
        solver_status=solver_status,
        solver_residual=solver_residual,
    )
    status_counts: dict[str, int] = {}
    for value in solver_status[active]:
        key = SolverStatus(int(value)).name.lower()
        status_counts[key] = status_counts.get(key, 0) + 1
    converged = bool(
        not np.any(numerical_error[active])
        and np.all(after_residual <= settings.residual_tolerance)
    )
    diagnostics = LocalRepairDiagnostics(
        solver="gqmr-local-joint-root-dls",
        solver_version="1.0",
        frames_processed=len(active),
        iterations=int(np.sum(iterations[active])),
        converged=converged,
        residual_rmse_before_m=float(
            np.sqrt(np.mean(before_residual * before_residual))
        ),
        residual_rmse_after_m=float(
            np.sqrt(np.mean(after_residual * after_residual))
        ),
        status_counts=status_counts,
        warnings=() if converged else ("one or more frames exceeded tolerance",),
        details={
            "buffer_frames": int(buffer_frames),
            "transition_weights": [float(value) for value in weights],
            "velocity_recomputed_range": [int(start), int(stop)],
            "contact_locked_frames": int(
                np.count_nonzero(
                    np.isfinite(targets.contact_probability[active])
                    & (
                        targets.contact_probability[active]
                        >= settings.contact_threshold
                    )
                )
            ),
        },
    )
    return LocalRepairSolverOutput(
        motion=repaired,
        applied_config=targets.applied_config,
        diagnostics=diagnostics,
    )


def run_local_repair(
    motion: RobotMotion,
    frame_range: tuple[int, int],
    config: LocalRepairConfig,
    solver: LocalRepairSolver,
) -> LocalRepairResult:
    """Execute a solver once and normalize its output into the A1 result model."""

    _validate_frame_range(frame_range, motion.frame_count)
    requested = LocalRepairConfig.model_validate(config.model_dump(mode="json"))
    input_hash = motion_sha256(motion)
    output = solver(_copy_robot_motion(motion), frame_range, requested)
    if not isinstance(output, LocalRepairSolverOutput):
        raise LocalRepairError("local repair solver returned an unsupported result")
    _validate_repaired_motion(motion, output.motion, frame_range)
    if output.diagnostics.frames_processed != frame_range[1] - frame_range[0] + 1:
        raise LocalRepairError("diagnostics.frames_processed does not match frame_range")
    metadata = dict(motion.metadata)
    history = list(metadata.get("local_repair_history", []))
    history.append(
        {
            "frame_range": list(frame_range),
            "requested_config": requested.model_dump(mode="json"),
            "applied_config": output.applied_config.model_dump(mode="json"),
            "diagnostics": output.diagnostics.model_dump(mode="json"),
            "input_motion_sha256": input_hash,
        }
    )
    metadata["local_repair_history"] = history
    repaired = replace(output.motion, metadata=metadata)
    output_hash = motion_sha256(repaired)
    return LocalRepairResult(
        motion=repaired,
        frame_range=frame_range,
        requested_config=requested,
        applied_config=output.applied_config,
        diagnostics=output.diagnostics,
        input_motion_sha256=input_hash,
        output_motion_sha256=output_hash,
    )


def replay_local_repair(
    motion: RobotMotion,
    command: LocalRepairCommand,
    solver: LocalRepairSolver,
) -> LocalRepairResult:
    """Replay a persisted command and reject input drift or nondeterministic output."""

    expected = command.parameters
    actual_input_hash = motion_sha256(motion)
    if actual_input_hash != expected.input_motion_sha256:
        raise LocalRepairError(
            "local repair input hash mismatch; command cannot be replayed on this motion"
        )
    result = run_local_repair(
        motion, expected.frame_range, expected.requested_config, solver
    )
    if result.applied_config != expected.applied_config:
        raise LocalRepairError("local repair replay applied different parameters")
    if result.diagnostics != expected.diagnostics:
        raise LocalRepairError("local repair replay produced different diagnostics")
    if result.output_motion_sha256 != expected.output_motion_sha256:
        raise LocalRepairError("local repair replay produced a different content hash")
    return result
