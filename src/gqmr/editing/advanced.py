"""Filtering and aligned concatenation for RobotMotion."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from gqmr.core.coordinates import (
    canonicalize_quaternion_sequence,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)
from gqmr.core.derivatives import angular_velocity_world, linear_velocity
from gqmr.core.motion import RobotMotion
from gqmr.editing.commands import EditingError


def filter_robot_motion(
    motion: RobotMotion, *, window_frames: int = 9, polynomial_order: int = 3
) -> RobotMotion:
    """Apply Savitzky-Golay filtering, including SO(3) tangent-space root filtering."""

    if window_frames < 3 or window_frames % 2 == 0 or window_frames > motion.frame_count:
        raise EditingError("filter window must be odd, >=3, and no longer than the clip")
    if polynomial_order < 1 or polynomial_order >= window_frames:
        raise EditingError("filter polynomial order is invalid")
    if not np.all(motion.frame_valid):
        raise EditingError("filtering requires fully valid RobotMotion")
    root_position = savgol_filter(
        motion.root_position, window_frames, polynomial_order, axis=0, mode="interp"
    )
    dof_position = savgol_filter(
        motion.dof_position, window_frames, polynomial_order, axis=0, mode="interp"
    )
    rotations = Rotation.from_quat(
        wxyz_to_xyzw(canonicalize_quaternion_sequence(motion.root_rotation))
    )
    rotation_vectors = rotations.as_rotvec()
    root_rotation = xyzw_to_wxyz(
        Rotation.from_rotvec(
            savgol_filter(
                rotation_vectors,
                window_frames,
                polynomial_order,
                axis=0,
                mode="interp",
            )
        ).as_quat()
    )
    contact = motion.foot_contact_probability.copy()
    for leg in range(4):
        if np.all(np.isfinite(contact[:, leg])):
            contact[:, leg] = np.clip(
                savgol_filter(
                    contact[:, leg], window_frames, polynomial_order, mode="interp"
                ),
                0.0,
                1.0,
            )
    metadata = dict(motion.metadata)
    history = list(metadata.get("edit_history", []))
    history.append(
        {"kind": "savgol_so3", "window_frames": window_frames, "polynomial_order": polynomial_order}
    )
    metadata["edit_history"] = history
    return replace(
        motion,
        root_position=root_position,
        root_rotation=root_rotation,
        dof_position=dof_position,
        root_linear_velocity=linear_velocity(motion.timestamps, root_position),
        root_angular_velocity=angular_velocity_world(motion.timestamps, root_rotation),
        dof_velocity=linear_velocity(motion.timestamps, dof_position),
        foot_contact_probability=contact,
        metadata=metadata,
    )


def concatenate_robot_motions(
    motions: list[RobotMotion], *, blend_seconds: float = 0.15
) -> RobotMotion:
    """Align roots and concatenate clips with a short scalar-DOF seam blend."""

    if not motions:
        raise EditingError("concatenation requires at least one motion")
    reference = motions[0]
    if any(
        motion.dof_names != reference.dof_names
        or motion.metadata["model_sha256"] != reference.metadata["model_sha256"]
        or not np.all(motion.frame_valid)
        for motion in motions
    ):
        raise EditingError("concatenated motions must be valid and use one model/DOF order")
    if not np.isfinite(blend_seconds) or blend_seconds < 0.0:
        raise EditingError("blend_seconds must be finite and non-negative")
    fields: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "root_position",
            "root_rotation",
            "dof_position",
            "foot_contact_probability",
            "frame_valid",
            "solver_status",
            "solver_residual",
        )
    }
    timestamps: list[np.ndarray] = []
    elapsed = 0.0
    previous_end_position: np.ndarray | None = None
    previous_end_rotation: Rotation | None = None
    previous_end_dof: np.ndarray | None = None
    for clip_index, motion in enumerate(motions):
        root_position = motion.root_position.astype(np.float64).copy()
        root_rotations = Rotation.from_quat(wxyz_to_xyzw(motion.root_rotation))
        dof_position = motion.dof_position.astype(np.float64).copy()
        if clip_index:
            assert previous_end_position is not None
            assert previous_end_rotation is not None
            assert previous_end_dof is not None
            alignment = previous_end_rotation * root_rotations[0].inv()
            root_position = previous_end_position + alignment.apply(
                root_position - root_position[0]
            )
            root_rotations = alignment * root_rotations
            if blend_seconds > 0.0:
                count = min(
                    motion.frame_count,
                    max(1, int(round(blend_seconds / np.median(np.diff(motion.timestamps))))),
                )
                weights = np.linspace(1.0, 0.0, count, endpoint=True)[:, None]
                dof_position[:count] += weights * (previous_end_dof - dof_position[0])
            keep = slice(1, None)
            clip_timestamps = elapsed + motion.timestamps[keep]
        else:
            keep = slice(None)
            clip_timestamps = motion.timestamps.copy()
        timestamps.append(clip_timestamps)
        fields["root_position"].append(root_position[keep])
        fields["root_rotation"].append(xyzw_to_wxyz(root_rotations.as_quat())[keep])
        fields["dof_position"].append(dof_position[keep])
        for name in (
            "foot_contact_probability",
            "frame_valid",
            "solver_status",
            "solver_residual",
        ):
            fields[name].append(getattr(motion, name)[keep])
        previous_end_position = root_position[-1]
        previous_end_rotation = root_rotations[-1]
        previous_end_dof = dof_position[-1]
        elapsed = float(clip_timestamps[-1])
    time = np.concatenate(timestamps)
    root_position = np.concatenate(fields["root_position"])
    root_rotation = np.concatenate(fields["root_rotation"])
    dof_position = np.concatenate(fields["dof_position"])
    metadata = dict(reference.metadata)
    history = list(metadata.get("edit_history", []))
    history.append({"kind": "concatenate", "clips": len(motions), "blend_seconds": blend_seconds})
    metadata["edit_history"] = history
    return replace(
        reference,
        timestamps=time,
        root_position=root_position,
        root_rotation=root_rotation,
        dof_position=dof_position,
        root_linear_velocity=linear_velocity(time, root_position),
        root_angular_velocity=angular_velocity_world(time, root_rotation),
        dof_velocity=linear_velocity(time, dof_position),
        foot_contact_probability=np.concatenate(fields["foot_contact_probability"]),
        frame_valid=np.concatenate(fields["frame_valid"]),
        solver_status=np.concatenate(fields["solver_status"]),
        solver_residual=np.concatenate(fields["solver_residual"]),
        metadata=metadata,
    )
