"""Loop closure for canonical RobotMotion."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from gqmr.core.coordinates import wxyz_to_xyzw, xyzw_to_wxyz
from gqmr.core.derivatives import angular_velocity_world, linear_velocity
from gqmr.core.motion import RobotMotion
from gqmr.editing.commands import EditingError


def make_robot_loop(
    motion: RobotMotion,
    *,
    preserve_forward_displacement: bool = True,
) -> RobotMotion:
    """Distribute endpoint pose error over a clip so it loops without a seam."""

    if motion.frame_count < 3 or not np.all(motion.frame_valid):
        raise EditingError("loop closure requires at least three fully valid frames")
    phase = motion.timestamps / motion.duration
    smooth = phase * phase * (3.0 - 2.0 * phase)
    dof_delta = motion.dof_position[-1] - motion.dof_position[0]
    dof_position = motion.dof_position - smooth[:, None] * dof_delta
    root_position = motion.root_position.copy()
    position_delta = root_position[-1] - root_position[0]
    if preserve_forward_displacement:
        position_delta[0] = 0.0
    root_position -= smooth[:, None] * position_delta

    rotations = Rotation.from_quat(wxyz_to_xyzw(motion.root_rotation))
    endpoint_error = rotations[0] * rotations[-1].inv()
    correction = Slerp(
        [0.0, 1.0],
        Rotation.from_quat(
            np.stack(
                (
                    Rotation.identity().as_quat(),
                    endpoint_error.as_quat(),
                )
            )
        ),
    )(smooth)
    root_rotation = xyzw_to_wxyz((correction * rotations).as_quat())
    contact = motion.foot_contact_probability.copy()
    endpoint_contact = np.nanmean(
        np.stack((contact[0], contact[-1])), axis=0
    )
    contact[0] = endpoint_contact
    contact[-1] = endpoint_contact
    metadata = dict(motion.metadata)
    edit_history = list(metadata.get("edit_history", []))
    edit_history.append("loop_closure_v1")
    metadata["edit_history"] = edit_history
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
