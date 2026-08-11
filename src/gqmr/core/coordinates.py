"""Quaternion and rotation helpers for GQMR's wxyz convention."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation, Slerp

from gqmr.core.errors import MotionValidationError

_EPS = 1e-12


def _quaternion_array(value: ArrayLike, *, name: str) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape[-1:] != (4,):
        raise MotionValidationError("expected shape (..., 4)", field=name)
    if not np.all(np.isfinite(array)):
        raise MotionValidationError("contains non-finite values", field=name)
    return array


def wxyz_to_xyzw(quaternion: ArrayLike) -> NDArray[np.float64]:
    """Convert quaternion components without changing the represented rotation."""

    q = _quaternion_array(quaternion, name="quaternion")
    return np.ascontiguousarray(q[..., [1, 2, 3, 0]])


def xyzw_to_wxyz(quaternion: ArrayLike) -> NDArray[np.float64]:
    """Convert SciPy-style xyzw quaternions to GQMR's wxyz order."""

    q = _quaternion_array(quaternion, name="quaternion")
    return np.ascontiguousarray(q[..., [3, 0, 1, 2]])


def normalize_quaternions(quaternion: ArrayLike) -> NDArray[np.float64]:
    """Return unit wxyz quaternions, rejecting zero-length inputs."""

    q = _quaternion_array(quaternion, name="quaternion")
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < _EPS):
        raise MotionValidationError("has zero norm", field="quaternion")
    return np.ascontiguousarray(q / norm)


def canonicalize_quaternion_sequence(quaternion: ArrayLike) -> NDArray[np.float64]:
    """Normalize a sequence and flip signs to keep adjacent samples continuous."""

    q = normalize_quaternions(quaternion).copy()
    if q.ndim != 2:
        raise MotionValidationError("expected shape (T, 4)", field="quaternion")
    for index in range(1, len(q)):
        if np.dot(q[index - 1], q[index]) < 0.0:
            q[index] *= -1.0
    return np.ascontiguousarray(q)


def quaternion_to_matrix(quaternion: ArrayLike) -> NDArray[np.float64]:
    """Convert unit wxyz quaternions to rotation matrices."""

    q = normalize_quaternions(quaternion)
    return Rotation.from_quat(wxyz_to_xyzw(q)).as_matrix()


def matrix_to_quaternion(matrix: ArrayLike) -> NDArray[np.float64]:
    """Convert rotation matrices to normalized wxyz quaternions."""

    matrices = np.asarray(matrix, dtype=np.float64)
    if matrices.shape[-2:] != (3, 3):
        raise MotionValidationError("expected shape (..., 3, 3)", field="matrix")
    if not np.all(np.isfinite(matrices)):
        raise MotionValidationError("contains non-finite values", field="matrix")
    return normalize_quaternions(xyzw_to_wxyz(Rotation.from_matrix(matrices).as_quat()))


def slerp_wxyz(
    timestamps: ArrayLike,
    quaternions: ArrayLike,
    target_timestamps: ArrayLike,
) -> NDArray[np.float64]:
    """Shortest-arc SLERP on a strictly increasing time axis."""

    times = np.asarray(timestamps, dtype=np.float64)
    target = np.asarray(target_timestamps, dtype=np.float64)
    q = canonicalize_quaternion_sequence(quaternions)
    if times.ndim != 1 or len(times) != len(q) or len(times) < 2:
        raise MotionValidationError(
            "timestamps and quaternions must contain at least two matching samples"
        )
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
        raise MotionValidationError(
            "must be finite and strictly increasing", field="timestamps"
        )
    if target.ndim != 1 or not np.all(np.isfinite(target)):
        raise MotionValidationError("must be a finite 1D array", field="target_timestamps")
    tolerance = np.finfo(np.float64).eps * max(1.0, abs(times[-1])) * 8.0
    if np.any(target < times[0] - tolerance) or np.any(target > times[-1] + tolerance):
        raise MotionValidationError(
            "SLERP does not extrapolate outside the source time range",
            field="target_timestamps",
        )
    interpolator = Slerp(times, Rotation.from_quat(wxyz_to_xyzw(q)))
    result = xyzw_to_wxyz(interpolator(np.clip(target, times[0], times[-1])).as_quat())
    return canonicalize_quaternion_sequence(result)


def quaternion_geodesic_distance(a: ArrayLike, b: ArrayLike) -> NDArray[np.float64]:
    """Return the unsigned SO(3) geodesic distance in radians."""

    qa = normalize_quaternions(a)
    qb = normalize_quaternions(b)
    if qa.shape != qb.shape:
        raise MotionValidationError("quaternion shapes do not match")
    dot = np.abs(np.sum(qa * qb, axis=-1))
    return 2.0 * np.arccos(np.clip(dot, -1.0, 1.0))

