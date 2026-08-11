"""Canonical in-memory representations for Motion Schema v1."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, ClassVar, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gqmr.core.errors import MotionValidationError

CONTACT_ORDER = ("FL", "FR", "RL", "RR")
COORDINATE_FRAME = "gqmr_world_x_forward_y_left_z_up"
_HEX_DIGITS = frozenset("0123456789abcdef")


class SolverStatus(IntEnum):
    OK = 0
    DEGRADED_ROOT = 1
    MAX_ITER = 2
    UNREACHABLE = 3
    MISSING_INPUT = 4
    NUMERICAL_ERROR = 5
    INTERPOLATED = 6


def _array(
    value: ArrayLike,
    dtype: np.dtype[Any] | type[Any],
    *,
    field: str,
    ndim: int,
) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim != ndim:
        raise MotionValidationError(f"expected {ndim} dimensions", field=field)
    return np.ascontiguousarray(array)


def _names(value: ArrayLike, *, field: str) -> tuple[str, ...]:
    array = np.asarray(value)
    if array.ndim != 1:
        raise MotionValidationError("expected a 1D name array", field=field)
    names = tuple(str(item) for item in array.tolist())
    if not names or any(not name or not name.strip() for name in names):
        raise MotionValidationError("names must be non-empty", field=field)
    if len(set(names)) != len(names):
        raise MotionValidationError("names must be unique", field=field)
    return names


def _validate_timestamps(timestamps: NDArray[np.float64]) -> None:
    if len(timestamps) == 0:
        raise MotionValidationError("must contain at least one frame", field="timestamps")
    if not np.all(np.isfinite(timestamps)):
        raise MotionValidationError("contains non-finite values", field="timestamps")
    if timestamps[0] != 0.0:
        raise MotionValidationError("must start at 0.0", field="timestamps")
    if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0.0):
        raise MotionValidationError("must be strictly increasing", field="timestamps")


def _validate_probability(
    values: np.ndarray, *, field: str, allow_nan: bool
) -> None:
    allowed = np.isnan(values) if allow_nan else np.zeros(values.shape, dtype=bool)
    invalid = ~allowed & (~np.isfinite(values) | (values < 0.0) | (values > 1.0))
    if np.any(invalid):
        suffix = " or NaN for unknown values" if allow_nan else ""
        raise MotionValidationError(f"values must be in [0, 1]{suffix}", field=field)


def _require_metadata(
    metadata: Mapping[str, Any], required: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        raise MotionValidationError("must be a JSON object", field="metadata_json")
    result = dict(metadata)
    for key, expected in required.items():
        if key not in result:
            raise MotionValidationError(
                f"missing required field {key!r}", field="metadata_json"
            )
        if expected is not None and result[key] != expected:
            raise MotionValidationError(
                f"field {key!r} must equal {expected!r}", field="metadata_json"
            )
    created_by = result.get("created_by")
    if not isinstance(created_by, Mapping) or not created_by.get("gqmr_version"):
        raise MotionValidationError(
            "created_by.gqmr_version is required", field="metadata_json"
        )
    return result


def _require_nonempty_text(metadata: Mapping[str, Any], key: str) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value.strip():
        raise MotionValidationError(
            f"field {key!r} must be a non-empty string", field="metadata_json"
        )
    return value


def _require_hex(metadata: Mapping[str, Any], key: str, length: int) -> str:
    value = _require_nonempty_text(metadata, key)
    if len(value) != length or any(character not in _HEX_DIGITS for character in value):
        raise MotionValidationError(
            f"field {key!r} must be {length} lowercase hexadecimal characters",
            field="metadata_json",
        )
    return value


@dataclass(frozen=True, slots=True)
class AnimalMotion:
    """Validated canonical animal keypoint motion."""

    schema_id: ClassVar[str] = "gqmr.animal_motion"
    schema_version: ClassVar[str] = "1.0"

    timestamps: ArrayLike
    keypoint_names: tuple[str, ...] | ArrayLike
    positions: ArrayLike
    confidence: ArrayLike
    valid_mask: ArrayLike
    contact_probability: ArrayLike
    frame_valid: ArrayLike
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        timestamps = _array(self.timestamps, np.float64, field="timestamps", ndim=1)
        names = _names(self.keypoint_names, field="keypoint_names")
        positions = _array(self.positions, np.float32, field="positions", ndim=3)
        confidence = _array(self.confidence, np.float32, field="confidence", ndim=2)
        valid_mask = _array(self.valid_mask, np.bool_, field="valid_mask", ndim=2)
        contact = _array(
            self.contact_probability,
            np.float32,
            field="contact_probability",
            ndim=2,
        )
        frame_valid = _array(self.frame_valid, np.bool_, field="frame_valid", ndim=1)
        metadata = _require_metadata(
            self.metadata,
            {
                "coordinate_frame": COORDINATE_FRAME,
                "length_unit": "m",
                "time_unit": "s",
                "skeleton_id": None,
                "skeleton_sha256": None,
                "contact_order": list(CONTACT_ORDER),
                "contact_source": None,
                "source": None,
                "created_by": None,
            },
        )
        _require_nonempty_text(metadata, "skeleton_id")
        _require_hex(metadata, "skeleton_sha256", 64)
        if metadata["contact_source"] not in {
            "unknown",
            "heuristic",
            "mujoco",
            "manual",
            "mixed",
        }:
            raise MotionValidationError(
                "field 'contact_source' has an unsupported value",
                field="metadata_json",
            )
        if not isinstance(metadata["source"], Mapping):
            raise MotionValidationError(
                "field 'source' must be a JSON object", field="metadata_json"
            )
        _validate_timestamps(timestamps)
        frames, keypoints = len(timestamps), len(names)
        expected = {
            "positions": (frames, keypoints, 3),
            "confidence": (frames, keypoints),
            "valid_mask": (frames, keypoints),
            "contact_probability": (frames, 4),
            "frame_valid": (frames,),
        }
        actual = {
            "positions": positions.shape,
            "confidence": confidence.shape,
            "valid_mask": valid_mask.shape,
            "contact_probability": contact.shape,
            "frame_valid": frame_valid.shape,
        }
        for field, shape in expected.items():
            if actual[field] != shape:
                raise MotionValidationError(
                    f"expected shape {shape}, got {actual[field]}", field=field
                )
        _validate_probability(confidence, field="confidence", allow_nan=False)
        _validate_probability(contact, field="contact_probability", allow_nan=True)
        if np.any(valid_mask & ~np.all(np.isfinite(positions), axis=-1)):
            raise MotionValidationError(
                "valid keypoints must have finite positions", field="positions"
            )
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "keypoint_names", names)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "valid_mask", valid_mask)
        object.__setattr__(self, "contact_probability", contact)
        object.__setattr__(self, "frame_valid", frame_valid)
        object.__setattr__(self, "metadata", metadata)

    @property
    def frame_count(self) -> int:
        return len(self.timestamps)

    @property
    def duration(self) -> float:
        return float(self.timestamps[-1])


@dataclass(frozen=True, slots=True)
class RobotMotion:
    """Validated free-root plus scalar-DOF robot motion."""

    schema_id: ClassVar[str] = "gqmr.robot_motion"
    schema_version: ClassVar[str] = "1.0"

    timestamps: ArrayLike
    dof_names: tuple[str, ...] | ArrayLike
    root_position: ArrayLike
    root_rotation: ArrayLike
    dof_position: ArrayLike
    root_linear_velocity: ArrayLike
    root_angular_velocity: ArrayLike
    dof_velocity: ArrayLike
    foot_contact_probability: ArrayLike
    frame_valid: ArrayLike
    solver_status: ArrayLike
    solver_residual: ArrayLike
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        timestamps = _array(self.timestamps, np.float64, field="timestamps", ndim=1)
        names = _names(self.dof_names, field="dof_names")
        root_position = _array(
            self.root_position, np.float32, field="root_position", ndim=2
        )
        root_rotation = _array(
            self.root_rotation, np.float32, field="root_rotation", ndim=2
        )
        dof_position = _array(
            self.dof_position, np.float32, field="dof_position", ndim=2
        )
        root_linear_velocity = _array(
            self.root_linear_velocity,
            np.float32,
            field="root_linear_velocity",
            ndim=2,
        )
        root_angular_velocity = _array(
            self.root_angular_velocity,
            np.float32,
            field="root_angular_velocity",
            ndim=2,
        )
        dof_velocity = _array(
            self.dof_velocity, np.float32, field="dof_velocity", ndim=2
        )
        contact = _array(
            self.foot_contact_probability,
            np.float32,
            field="foot_contact_probability",
            ndim=2,
        )
        frame_valid = _array(self.frame_valid, np.bool_, field="frame_valid", ndim=1)
        solver_status = _array(
            self.solver_status, np.int16, field="solver_status", ndim=1
        )
        solver_residual = _array(
            self.solver_residual, np.float32, field="solver_residual", ndim=1
        )
        metadata = _require_metadata(
            self.metadata,
            {
                "coordinate_frame": COORDINATE_FRAME,
                "quaternion_order": "wxyz",
                "root_velocity_frame": "world",
                "model_id": None,
                "model_source_commit": None,
                "model_sha256": None,
                "robot_config_sha256": None,
                "contact_order": list(CONTACT_ORDER),
                "source_motion_sha256": None,
                "retarget_config": None,
                "created_by": None,
            },
        )
        _require_nonempty_text(metadata, "model_id")
        _require_hex(metadata, "model_source_commit", 40)
        _require_hex(metadata, "model_sha256", 64)
        _require_hex(metadata, "robot_config_sha256", 64)
        _require_hex(metadata, "source_motion_sha256", 64)
        if not isinstance(metadata["retarget_config"], Mapping):
            raise MotionValidationError(
                "field 'retarget_config' must be a JSON object",
                field="metadata_json",
            )
        _validate_timestamps(timestamps)
        frames, dofs = len(timestamps), len(names)
        expected = {
            "root_position": (frames, 3),
            "root_rotation": (frames, 4),
            "dof_position": (frames, dofs),
            "root_linear_velocity": (frames, 3),
            "root_angular_velocity": (frames, 3),
            "dof_velocity": (frames, dofs),
            "foot_contact_probability": (frames, 4),
            "frame_valid": (frames,),
            "solver_status": (frames,),
            "solver_residual": (frames,),
        }
        actual = {
            "root_position": root_position.shape,
            "root_rotation": root_rotation.shape,
            "dof_position": dof_position.shape,
            "root_linear_velocity": root_linear_velocity.shape,
            "root_angular_velocity": root_angular_velocity.shape,
            "dof_velocity": dof_velocity.shape,
            "foot_contact_probability": contact.shape,
            "frame_valid": frame_valid.shape,
            "solver_status": solver_status.shape,
            "solver_residual": solver_residual.shape,
        }
        for field, shape in expected.items():
            if actual[field] != shape:
                raise MotionValidationError(
                    f"expected shape {shape}, got {actual[field]}", field=field
                )
        _validate_probability(contact, field="foot_contact_probability", allow_nan=True)
        known_statuses = np.array([status.value for status in SolverStatus], dtype=np.int16)
        if not np.all(np.isin(solver_status, known_statuses)):
            raise MotionValidationError("contains an unknown status code", field="solver_status")
        forbidden_valid = (solver_status >= SolverStatus.UNREACHABLE) & (
            solver_status != SolverStatus.INTERPOLATED
        )
        if np.any(frame_valid & forbidden_valid):
            raise MotionValidationError(
                "status UNREACHABLE/MISSING_INPUT/NUMERICAL_ERROR cannot be valid",
                field="frame_valid",
            )
        finite_fields = (
            root_position,
            root_rotation,
            dof_position,
            root_linear_velocity,
            root_angular_velocity,
            dof_velocity,
        )
        for values in finite_fields:
            row_finite = np.all(np.isfinite(values.reshape(frames, -1)), axis=1)
            if np.any(frame_valid & ~row_finite):
                raise MotionValidationError(
                    "valid frames must have finite kinematic fields", field="frame_valid"
                )
        finite_quaternions = np.all(np.isfinite(root_rotation), axis=1)
        norms = np.linalg.norm(root_rotation[finite_quaternions], axis=1)
        if np.any(np.abs(norms - 1.0) >= 1e-5):
            raise MotionValidationError(
                "finite quaternions must have unit norm within 1e-5",
                field="root_rotation",
            )
        finite_residual = np.isfinite(solver_residual)
        if np.any(frame_valid & ~finite_residual) or np.any(
            finite_residual & (solver_residual < 0.0)
        ):
            raise MotionValidationError(
                "valid residuals must be finite and non-negative",
                field="solver_residual",
            )
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "dof_names", names)
        object.__setattr__(self, "root_position", root_position)
        object.__setattr__(self, "root_rotation", root_rotation)
        object.__setattr__(self, "dof_position", dof_position)
        object.__setattr__(self, "root_linear_velocity", root_linear_velocity)
        object.__setattr__(self, "root_angular_velocity", root_angular_velocity)
        object.__setattr__(self, "dof_velocity", dof_velocity)
        object.__setattr__(self, "foot_contact_probability", contact)
        object.__setattr__(self, "frame_valid", frame_valid)
        object.__setattr__(self, "solver_status", solver_status)
        object.__setattr__(self, "solver_residual", solver_residual)
        object.__setattr__(self, "metadata", metadata)

    @property
    def frame_count(self) -> int:
        return len(self.timestamps)

    @property
    def duration(self) -> float:
        return float(self.timestamps[-1])
