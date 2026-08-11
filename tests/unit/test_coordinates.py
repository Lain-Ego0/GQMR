from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from gqmr.core.coordinates import (
    matrix_to_quaternion,
    quaternion_geodesic_distance,
    quaternion_to_matrix,
    slerp_wxyz,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)


def test_wxyz_xyzw_round_trip_100k() -> None:
    rng = np.random.default_rng(20260811)
    quaternions = rng.normal(size=(100_000, 4))
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    restored = xyzw_to_wxyz(wxyz_to_xyzw(quaternions))
    assert np.max(np.abs(restored - quaternions)) < 1e-6


def test_matrix_quaternion_round_trip_geodesic_error() -> None:
    rng = np.random.default_rng(5)
    original = xyzw_to_wxyz(Rotation.random(10_000, random_state=rng).as_quat())
    restored = matrix_to_quaternion(quaternion_to_matrix(original))
    assert np.max(quaternion_geodesic_distance(original, restored)) < 1e-6


def test_slerp_uses_shortest_arc_and_unit_quaternions() -> None:
    # The second sample has the opposite sign but represents +10 degrees.
    end_xyzw = Rotation.from_euler("z", 10.0, degrees=True).as_quat()
    end_wxyz = -xyzw_to_wxyz(end_xyzw)
    result = slerp_wxyz(
        np.array([0.0, 1.0]),
        np.stack(([1.0, 0.0, 0.0, 0.0], end_wxyz)),
        np.array([0.0, 0.5, 1.0]),
    )
    midpoint = Rotation.from_quat(wxyz_to_xyzw(result[1])).as_euler("xyz", degrees=True)
    assert np.max(np.abs(np.linalg.norm(result, axis=1) - 1.0)) < 1e-6
    assert abs(midpoint[2] - 5.0) < 1e-8
    assert np.all(np.sum(result[:-1] * result[1:], axis=1) >= 0.0)

