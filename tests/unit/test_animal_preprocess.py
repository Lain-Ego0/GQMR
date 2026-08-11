from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from gqmr.core.coordinates import quaternion_geodesic_distance
from gqmr.core.motion import SolverStatus
from gqmr.retarget.animal_preprocess import (
    estimate_body_scale,
    estimate_contact_probability,
    estimate_root_motion,
)
from gqmr.synthetic import generate_dog27_motion, generate_dog27_suite


def test_all_mit_synthetic_gaits_are_valid_and_deterministic() -> None:
    suite = generate_dog27_suite(duration=1.0, fps=30.0)

    assert set(suite) == {"walk", "trot", "pace", "turn"}
    for gait, motion in suite.items():
        repeated = generate_dog27_motion(gait, duration=1.0, fps=30.0)
        assert motion.frame_count == 31
        assert motion.positions.shape == (31, 27, 3)
        assert np.array_equal(motion.positions, repeated.positions)
        assert np.all((motion.contact_probability >= 0.0) & (motion.contact_probability <= 1.0))
        assert motion.metadata["source"]["license"] == "MIT"
        assert motion.metadata["source"]["gait"] == gait


def test_root_estimator_tracks_turn_and_degrades_to_previous_orientation() -> None:
    motion = generate_dog27_motion("turn", duration=1.0, fps=30.0)
    estimate = estimate_root_motion(motion)

    assert np.all(estimate.valid)
    assert np.all(estimate.status == SolverStatus.OK)
    turn_angle = float(quaternion_geodesic_distance(estimate.rotation[0], estimate.rotation[-1]))
    assert turn_angle == pytest.approx(0.65, abs=2e-3)

    valid_mask = motion.valid_mask.copy()
    valid_mask[5, 6] = False
    degraded = estimate_root_motion(replace(motion, valid_mask=valid_mask))
    assert degraded.status[5] == SolverStatus.MISSING_INPUT
    assert not degraded.valid[5]
    assert np.allclose(degraded.rotation[5], degraded.rotation[4])


def test_root_estimator_marks_degenerate_first_frame_invalid() -> None:
    motion = generate_dog27_motion("walk", duration=0.2, fps=20.0)
    positions = motion.positions.copy()
    positions[0, 3] = positions[0, 0]
    estimate = estimate_root_motion(replace(motion, positions=positions))

    assert estimate.status[0] == SolverStatus.DEGRADED_ROOT
    assert not estimate.valid[0]
    assert np.array_equal(estimate.rotation[0], np.array([1.0, 0.0, 0.0, 0.0]))


def test_body_scale_and_contact_probability() -> None:
    motion = generate_dog27_motion("walk", duration=1.0, fps=30.0)
    scale = estimate_body_scale(motion)
    probability = estimate_contact_probability(motion)

    assert scale.torso_length == pytest.approx(0.426, abs=0.01)
    assert scale.shoulder_width == pytest.approx(0.28, abs=1e-5)
    assert scale.hip_width == pytest.approx(0.24, abs=1e-5)
    assert set(scale.leg_lengths) == {"FL", "FR", "RL", "RR"}
    assert all(0.45 < length < 0.56 for length in scale.leg_lengths.values())
    assert probability.shape == (31, 4)
    assert np.all(np.isfinite(probability))
    assert np.all((probability >= 0.0) & (probability <= 1.0))


def test_body_scale_rejects_missing_required_samples() -> None:
    motion = generate_dog27_motion("walk", duration=0.2, fps=20.0)
    valid_mask = motion.valid_mask.copy()
    valid_mask[:, 0] = False

    with pytest.raises(ValueError, match="no valid samples"):
        estimate_body_scale(replace(motion, valid_mask=valid_mask))
