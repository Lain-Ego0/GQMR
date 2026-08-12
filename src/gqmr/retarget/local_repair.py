"""Serializable local-repair contracts and deterministic solver execution."""

from __future__ import annotations

import uuid
from copy import deepcopy
from dataclasses import dataclass, replace
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

from gqmr.core.errors import GQMRError
from gqmr.core.io import motion_sha256
from gqmr.core.motion import RobotMotion

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
