"""Derivative estimation on canonical motion time axes."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation

from gqmr.core.coordinates import canonicalize_quaternion_sequence, wxyz_to_xyzw
from gqmr.core.errors import MotionValidationError


def validate_timestamps(timestamps: ArrayLike, *, minimum_samples: int = 1) -> NDArray[np.float64]:
    """Validate and return a finite, normalized, strictly increasing timeline."""

    times = np.asarray(timestamps, dtype=np.float64)
    if times.ndim != 1:
        raise MotionValidationError("expected a 1D array", field="timestamps")
    if len(times) < minimum_samples:
        raise MotionValidationError(
            f"requires at least {minimum_samples} samples", field="timestamps"
        )
    if not np.all(np.isfinite(times)):
        raise MotionValidationError("contains non-finite values", field="timestamps")
    if len(times) and times[0] != 0.0:
        raise MotionValidationError("must start at 0.0", field="timestamps")
    if len(times) > 1 and np.any(np.diff(times) <= 0.0):
        raise MotionValidationError("must be strictly increasing", field="timestamps")
    return times


def linear_velocity(timestamps: ArrayLike, values: ArrayLike) -> NDArray[np.float64]:
    """Differentiate scalar/vector samples on a possibly non-uniform timeline."""

    times = validate_timestamps(timestamps, minimum_samples=3)
    samples = np.asarray(values, dtype=np.float64)
    if samples.ndim < 1 or samples.shape[0] != len(times):
        raise MotionValidationError(
            "first dimension must match timestamps", field="values"
        )
    if not np.all(np.isfinite(samples)):
        raise MotionValidationError("contains non-finite values", field="values")
    return np.ascontiguousarray(np.gradient(samples, times, axis=0, edge_order=2))


def angular_velocity_world(
    timestamps: ArrayLike, quaternions_wxyz: ArrayLike
) -> NDArray[np.float64]:
    """Estimate world-frame angular velocity using SO(3) logarithms."""

    times = validate_timestamps(timestamps, minimum_samples=3)
    quaternions = canonicalize_quaternion_sequence(quaternions_wxyz)
    if len(quaternions) != len(times):
        raise MotionValidationError(
            "first dimension must match timestamps", field="root_rotation"
        )
    rotations = Rotation.from_quat(wxyz_to_xyzw(quaternions))
    result = np.empty((len(times), 3), dtype=np.float64)

    def interval_velocity(left: int, right: int) -> NDArray[np.float64]:
        # R_right * R_left^-1 expresses the increment in world coordinates.
        delta = rotations[right] * rotations[left].inv()
        return delta.as_rotvec() / (times[right] - times[left])

    result[0] = interval_velocity(0, 1)
    result[-1] = interval_velocity(len(times) - 2, len(times) - 1)
    for index in range(1, len(times) - 1):
        result[index] = interval_velocity(index - 1, index + 1)
    return np.ascontiguousarray(result)

