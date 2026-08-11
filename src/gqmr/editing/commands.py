"""Non-destructive editing commands keyed by timestamps and stable UUIDs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TypeAlias

import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation

from gqmr.core.coordinates import slerp_wxyz, wxyz_to_xyzw, xyzw_to_wxyz
from gqmr.core.derivatives import angular_velocity_world, linear_velocity
from gqmr.core.errors import GQMRError
from gqmr.core.motion import AnimalMotion, RobotMotion
from gqmr.project.model import EditCommand

Motion: TypeAlias = AnimalMotion | RobotMotion


class EditingError(GQMRError, ValueError):
    """Raised when an edit cannot preserve Motion Schema v1 invariants."""


def _metadata(motion: Motion, command: EditCommand) -> dict:
    metadata = dict(motion.metadata)
    history = list(metadata.get("edit_history", []))
    history.append(command.command_id)
    metadata["edit_history"] = history
    return metadata


def _time_mask(motion: Motion, start: float, end: float) -> np.ndarray:
    if not np.isfinite(start) or not np.isfinite(end) or start < 0.0 or end < start:
        raise EditingError("edit time range is invalid")
    mask = (motion.timestamps >= start) & (motion.timestamps <= end)
    if not np.any(mask):
        raise EditingError("edit time range contains no frames")
    return mask


def _trim(motion: Motion, command: EditCommand) -> Motion:
    start = float(command.parameters["start"])
    end = float(command.parameters["end"])
    mask = _time_mask(motion, start, end)
    indices = np.flatnonzero(mask)
    timeline = motion.timestamps[indices] - motion.timestamps[indices[0]]
    if isinstance(motion, AnimalMotion):
        return replace(
            motion,
            timestamps=timeline,
            positions=motion.positions[indices],
            confidence=motion.confidence[indices],
            valid_mask=motion.valid_mask[indices],
            contact_probability=motion.contact_probability[indices],
            frame_valid=motion.frame_valid[indices],
            metadata=_metadata(motion, command),
        )
    return replace(
        motion,
        timestamps=timeline,
        root_position=motion.root_position[indices],
        root_rotation=motion.root_rotation[indices],
        dof_position=motion.dof_position[indices],
        root_linear_velocity=motion.root_linear_velocity[indices],
        root_angular_velocity=motion.root_angular_velocity[indices],
        dof_velocity=motion.dof_velocity[indices],
        foot_contact_probability=motion.foot_contact_probability[indices],
        frame_valid=motion.frame_valid[indices],
        solver_status=motion.solver_status[indices],
        solver_residual=motion.solver_residual[indices],
        metadata=_metadata(motion, command),
    )


def _time_scale(motion: Motion, command: EditCommand) -> Motion:
    speed = float(command.parameters["speed"])
    if not np.isfinite(speed) or speed <= 0.0:
        raise EditingError("time scale speed must be finite and positive")
    if isinstance(motion, AnimalMotion):
        return replace(
            motion,
            timestamps=motion.timestamps / speed,
            metadata=_metadata(motion, command),
        )
    return replace(
        motion,
        timestamps=motion.timestamps / speed,
        root_linear_velocity=motion.root_linear_velocity * speed,
        root_angular_velocity=motion.root_angular_velocity * speed,
        dof_velocity=motion.dof_velocity * speed,
        metadata=_metadata(motion, command),
    )


def _root_transform(motion: Motion, command: EditCommand) -> Motion:
    translation = np.asarray(command.parameters["translation"], dtype=np.float64)
    yaw = float(command.parameters["yaw"])
    if translation.shape != (3,) or not np.all(np.isfinite(translation)) or not np.isfinite(yaw):
        raise EditingError("root transform requires finite translation[3] and yaw")
    rotation = Rotation.from_euler("z", yaw)
    if isinstance(motion, AnimalMotion):
        positions = rotation.apply(motion.positions.reshape(-1, 3)).reshape(
            motion.positions.shape
        ) + translation
        return replace(
            motion,
            positions=positions,
            metadata=_metadata(motion, command),
        )
    root_position = rotation.apply(motion.root_position) + translation
    source_rotation = Rotation.from_quat(wxyz_to_xyzw(motion.root_rotation))
    root_rotation = xyzw_to_wxyz((rotation * source_rotation).as_quat())
    return replace(
        motion,
        root_position=root_position,
        root_rotation=root_rotation,
        root_linear_velocity=rotation.apply(motion.root_linear_velocity),
        root_angular_velocity=rotation.apply(motion.root_angular_velocity),
        metadata=_metadata(motion, command),
    )


def _contact_override(motion: Motion, command: EditCommand) -> Motion:
    start = float(command.parameters["start"])
    end = float(command.parameters["end"])
    leg = str(command.parameters["leg"])
    probability = float(command.parameters["probability"])
    if leg not in {"FL", "FR", "RL", "RR"} or not 0.0 <= probability <= 1.0:
        raise EditingError("contact override leg/probability is invalid")
    mask = _time_mask(motion, start, end)
    leg_index = ("FL", "FR", "RL", "RR").index(leg)
    if isinstance(motion, AnimalMotion):
        contact = motion.contact_probability.copy()
        contact[mask, leg_index] = probability
        metadata = _metadata(motion, command)
        metadata["contact_source"] = "mixed"
        return replace(motion, contact_probability=contact, metadata=metadata)
    contact = motion.foot_contact_probability.copy()
    contact[mask, leg_index] = probability
    return replace(
        motion,
        foot_contact_probability=contact,
        metadata=_metadata(motion, command),
    )


def _resample(motion: Motion, command: EditCommand) -> Motion:
    fps = int(command.parameters["fps"])
    if fps <= 0 or motion.frame_count < 3:
        raise EditingError("resample requires positive fps and at least three frames")
    count = int(np.floor(motion.duration * fps + 1e-12)) + 1
    if count < 3:
        raise EditingError("resample output would contain fewer than three frames")
    target = np.arange(count, dtype=np.float64) / fps

    def hermite(values: np.ndarray) -> np.ndarray:
        if not np.all(np.isfinite(values)):
            raise EditingError("resample does not interpolate non-finite values")
        derivative = linear_velocity(motion.timestamps, values)
        return CubicHermiteSpline(
            motion.timestamps, values, derivative, axis=0
        )(target)

    nearest = np.minimum(
        np.searchsorted(motion.timestamps, target, side="left"),
        motion.frame_count - 1,
    )
    if isinstance(motion, AnimalMotion):
        if not np.all(motion.frame_valid) or not np.all(motion.valid_mask):
            raise EditingError("resample requires fully valid AnimalMotion")
        positions = hermite(motion.positions)
        confidence = np.clip(
            np.stack(
                [np.interp(target, motion.timestamps, motion.confidence[:, index]) for index in range(len(motion.keypoint_names))],
                axis=1,
            ),
            0.0,
            1.0,
        )
        contact = np.stack(
            [np.interp(target, motion.timestamps, motion.contact_probability[:, index]) for index in range(4)],
            axis=1,
        )
        return replace(
            motion,
            timestamps=target,
            positions=positions,
            confidence=confidence,
            valid_mask=np.ones((count, len(motion.keypoint_names)), dtype=bool),
            contact_probability=contact,
            frame_valid=np.ones(count, dtype=bool),
            metadata=_metadata(motion, command),
        )
    if not np.all(motion.frame_valid):
        raise EditingError("resample requires fully valid RobotMotion")
    root_position = hermite(motion.root_position)
    root_rotation = slerp_wxyz(motion.timestamps, motion.root_rotation, target)
    dof_position = hermite(motion.dof_position)
    contact = np.stack(
        [np.interp(target, motion.timestamps, motion.foot_contact_probability[:, index]) for index in range(4)],
        axis=1,
    )
    residual = np.interp(target, motion.timestamps, motion.solver_residual)
    return replace(
        motion,
        timestamps=target,
        root_position=root_position,
        root_rotation=root_rotation,
        dof_position=dof_position,
        root_linear_velocity=linear_velocity(target, root_position),
        root_angular_velocity=angular_velocity_world(target, root_rotation),
        dof_velocity=linear_velocity(target, dof_position),
        foot_contact_probability=np.clip(contact, 0.0, 1.0),
        frame_valid=np.ones(count, dtype=bool),
        solver_status=motion.solver_status[nearest],
        solver_residual=residual,
        metadata=_metadata(motion, command),
    )


def apply_edit(motion: Motion, command: EditCommand) -> Motion:
    operations = {
        "trim": _trim,
        "time_scale": _time_scale,
        "root_transform": _root_transform,
        "contact_override": _contact_override,
        "resample": _resample,
    }
    try:
        return operations[command.kind](motion, command)
    except KeyError as error:
        raise EditingError(f"edit parameters are incomplete: {error}") from error


@dataclass(slots=True)
class EditStack:
    base_motion: Motion
    commands: list[EditCommand]
    cursor: int = 0

    def __init__(self, base_motion: Motion) -> None:
        self.base_motion = base_motion
        self.commands = []
        self.cursor = 0

    def push(self, command: EditCommand) -> Motion:
        del self.commands[self.cursor :]
        self.commands.append(command)
        self.cursor += 1
        return self.current()

    def undo(self) -> Motion:
        if self.cursor > 0:
            self.cursor -= 1
        return self.current()

    def redo(self) -> Motion:
        if self.cursor < len(self.commands):
            self.cursor += 1
        return self.current()

    def current(self) -> Motion:
        motion = self.base_motion
        for command in self.commands[: self.cursor]:
            motion = apply_edit(motion, command)
        return motion
