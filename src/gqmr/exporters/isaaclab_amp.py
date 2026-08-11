"""Isaac Lab v2.3.2 AMP NPZ export and lightweight compatibility loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicHermiteSpline

from gqmr.core.coordinates import canonicalize_quaternion_sequence, slerp_wxyz
from gqmr.core.derivatives import angular_velocity_world, linear_velocity
from gqmr.core.motion import RobotMotion
from gqmr.exporters.common import ExportError, atomic_write, require_exportable
from gqmr.robots import RobotModel

_KEYS = {
    "fps",
    "dof_names",
    "body_names",
    "dof_positions",
    "dof_velocities",
    "body_positions",
    "body_rotations",
    "body_linear_velocities",
    "body_angular_velocities",
}


@dataclass(frozen=True, slots=True)
class IsaacLabAMPClip:
    fps: int
    dof_names: tuple[str, ...]
    body_names: tuple[str, ...]
    dof_positions: NDArray[np.float32]
    dof_velocities: NDArray[np.float32]
    body_positions: NDArray[np.float32]
    body_rotations: NDArray[np.float32]
    body_linear_velocities: NDArray[np.float32]
    body_angular_velocities: NDArray[np.float32]

    @property
    def frame_count(self) -> int:
        return self.dof_positions.shape[0]

    @property
    def duration(self) -> float:
        return (self.frame_count - 1) / self.fps

    def sample(self, times: np.ndarray) -> dict[str, np.ndarray]:
        """Linearly sample AMP fields at times, with normalized quaternion lerp."""

        sample_times = np.asarray(times, dtype=np.float64)
        if sample_times.ndim != 1 or not np.all(np.isfinite(sample_times)):
            raise ExportError("sample times must be a finite 1D array")
        if np.any(sample_times < 0.0) or np.any(sample_times > self.duration):
            raise ExportError("sample times are outside the AMP clip")
        phase = sample_times * self.fps
        left = np.floor(phase).astype(np.int64)
        right = np.minimum(left + 1, self.frame_count - 1)
        alpha = (phase - left).astype(np.float32)

        def interpolate(values: np.ndarray) -> np.ndarray:
            shape = (len(alpha),) + (1,) * (values.ndim - 1)
            weight = alpha.reshape(shape)
            return values[left] * (1.0 - weight) + values[right] * weight

        rotations = interpolate(self.body_rotations)
        rotations /= np.linalg.norm(rotations, axis=-1, keepdims=True)
        return {
            "dof_positions": interpolate(self.dof_positions),
            "dof_velocities": interpolate(self.dof_velocities),
            "body_positions": interpolate(self.body_positions),
            "body_rotations": rotations,
            "body_linear_velocities": interpolate(self.body_linear_velocities),
            "body_angular_velocities": interpolate(self.body_angular_velocities),
        }


def _target_times(motion: RobotMotion, fps: int) -> np.ndarray:
    if not isinstance(fps, int) or fps <= 0:
        raise ExportError("fps must be a positive integer")
    count = int(np.floor(motion.duration * fps + 1e-12)) + 1
    if count < 3:
        raise ExportError("selected fps produces fewer than three output frames")
    return np.arange(count, dtype=np.float64) / fps


def _hermite(times: np.ndarray, values: np.ndarray, target: np.ndarray) -> np.ndarray:
    derivative = linear_velocity(times, values)
    return CubicHermiteSpline(times, values, derivative, axis=0)(target)


def export_isaaclab_amp_v232(
    motion: RobotMotion,
    robot: RobotModel,
    destination: str | Path,
    *,
    fps: int = 60,
) -> Path:
    require_exportable(motion)
    if motion.metadata["model_sha256"] != robot.config.model_sha256:
        raise ExportError("RobotMotion model hash does not match export robot")
    if motion.dof_names != robot.config.dof_order:
        raise ExportError("RobotMotion DOF order does not match export robot")
    target = _target_times(motion, fps)
    root_position = _hermite(motion.timestamps, motion.root_position, target)
    root_rotation = slerp_wxyz(motion.timestamps, motion.root_rotation, target)
    dof_position = _hermite(motion.timestamps, motion.dof_position, target)
    dof_position = np.clip(
        dof_position, robot.joint_ranges[:, 0], robot.joint_ranges[:, 1]
    )
    dof_velocity = linear_velocity(target, dof_position)
    body_names = robot.named_bodies()
    body_positions = np.empty((len(target), len(body_names), 3), dtype=np.float64)
    body_rotations = np.empty((len(target), len(body_names), 4), dtype=np.float64)
    for frame in range(len(target)):
        robot.set_pose(root_position[frame], root_rotation[frame], dof_position[frame])
        for body_index, body_name in enumerate(body_names):
            position, rotation = robot.body_pose(body_name)
            body_positions[frame, body_index] = position
            body_rotations[frame, body_index] = rotation
    for body_index in range(len(body_names)):
        body_rotations[:, body_index] = canonicalize_quaternion_sequence(
            body_rotations[:, body_index]
        )
    body_linear_velocity = linear_velocity(target, body_positions)
    body_angular_velocity = np.stack(
        [angular_velocity_world(target, body_rotations[:, index]) for index in range(len(body_names))],
        axis=1,
    )
    arrays = {
        "fps": np.asarray(fps, dtype=np.int64),
        "dof_names": np.asarray(motion.dof_names, dtype=np.str_),
        "body_names": np.asarray(body_names, dtype=np.str_),
        "dof_positions": np.ascontiguousarray(dof_position, dtype="<f4"),
        "dof_velocities": np.ascontiguousarray(dof_velocity, dtype="<f4"),
        "body_positions": np.ascontiguousarray(body_positions, dtype="<f4"),
        "body_rotations": np.ascontiguousarray(body_rotations, dtype="<f4"),
        "body_linear_velocities": np.ascontiguousarray(body_linear_velocity, dtype="<f4"),
        "body_angular_velocities": np.ascontiguousarray(body_angular_velocity, dtype="<f4"),
    }
    return atomic_write(
        destination, lambda stream: np.savez_compressed(stream, **arrays)
    )


def load_isaaclab_amp_v232(path: str | Path) -> IsaacLabAMPClip:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != _KEYS:
                raise ExportError("AMP NPZ field set does not match Isaac Lab v2.3.2")
            arrays = {name: archive[name] for name in archive.files}
    except (OSError, ValueError) as error:
        if isinstance(error, ExportError):
            raise
        raise ExportError(f"cannot load AMP NPZ: {error}") from error
    fps_array = arrays["fps"]
    if fps_array.shape != () or fps_array.dtype != np.dtype(np.int64):
        raise ExportError("AMP fps must be an int64 scalar")
    fps = int(fps_array)
    dof_names = tuple(str(value) for value in arrays["dof_names"].tolist())
    body_names = tuple(str(value) for value in arrays["body_names"].tolist())
    frames, dofs, bodies = arrays["dof_positions"].shape[0], len(dof_names), len(body_names)
    expected = {
        "dof_positions": (frames, dofs),
        "dof_velocities": (frames, dofs),
        "body_positions": (frames, bodies, 3),
        "body_rotations": (frames, bodies, 4),
        "body_linear_velocities": (frames, bodies, 3),
        "body_angular_velocities": (frames, bodies, 3),
    }
    for name, shape in expected.items():
        value = arrays[name]
        if value.shape != shape or value.dtype != np.dtype("<f4"):
            raise ExportError(f"AMP field {name} has invalid shape or dtype")
        if not np.all(np.isfinite(value)):
            raise ExportError(f"AMP field {name} contains non-finite values")
    norms = np.linalg.norm(arrays["body_rotations"], axis=-1)
    if np.any(np.abs(norms - 1.0) >= 1e-5):
        raise ExportError("AMP body rotations are not unit wxyz quaternions")
    return IsaacLabAMPClip(
        fps=fps,
        dof_names=dof_names,
        body_names=body_names,
        **{name: arrays[name] for name in expected},
    )
