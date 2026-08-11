from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from gqmr.core.coordinates import xyzw_to_wxyz
from gqmr.core.derivatives import angular_velocity_world, linear_velocity


def test_linear_velocity_sine_rmse_below_one_percent() -> None:
    timestamps = np.linspace(0.0, 2.0, 1001) ** 1.1
    values = np.sin(2.0 * np.pi * timestamps)
    expected = 2.0 * np.pi * np.cos(2.0 * np.pi * timestamps)
    actual = linear_velocity(timestamps, values)
    relative_rmse = np.sqrt(np.mean((actual - expected) ** 2)) / np.sqrt(
        np.mean(expected**2)
    )
    assert relative_rmse < 0.01


def test_world_angular_velocity_constant_rotation_rmse_below_one_percent() -> None:
    timestamps = np.linspace(0.0, 2.0, 501)
    expected = np.array([0.2, -0.1, 0.4])
    rotations = Rotation.from_rotvec(timestamps[:, None] * expected)
    quaternions = xyzw_to_wxyz(rotations.as_quat())
    actual = angular_velocity_world(timestamps, quaternions)
    relative_rmse = np.sqrt(np.mean((actual - expected) ** 2)) / np.linalg.norm(expected)
    assert relative_rmse < 0.01

