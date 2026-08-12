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
from gqmr.retarget.preprocess import (
    estimate_ground_plane,
    preprocess_animal_motion,
    reestimate_contact_and_ground,
)
from gqmr.synthetic import (
    available_motion_presets,
    generate_dog27_motion,
    generate_dog27_preset,
    generate_dog27_preset_suite,
    generate_dog27_suite,
)
from gqmr.skeletons import get_skeleton


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
        assert motion.metadata["source"]["generator_version"] == 3


def test_motion_presets_are_deterministic_and_have_stable_metadata() -> None:
    presets = available_motion_presets()
    suite = generate_dog27_preset_suite(duration=0.5, fps=30.0)

    assert [preset.id for preset in presets] == [
        "walk_slow",
        "walk_standard",
        "trot_slow",
        "trot_standard",
        "trot_fast",
        "pace_standard",
        "turn_left",
        "turn_right",
    ]
    assert set(suite) == {preset.id for preset in presets}
    for preset in presets:
        motion = suite[preset.id]
        repeated = generate_dog27_preset(preset.id, duration=0.5, fps=30.0)
        assert np.array_equal(motion.positions, repeated.positions)
        assert motion.metadata["source"]["preset_id"] == preset.id
        assert motion.metadata["source"]["preset_label"] == preset.label


def test_speed_presets_and_turn_directions_are_distinct() -> None:
    slow = generate_dog27_preset("trot_slow", duration=1.0, fps=30.0)
    standard = generate_dog27_preset("trot_standard", duration=1.0, fps=30.0)
    fast = generate_dog27_preset("trot_fast", duration=1.0, fps=30.0)
    root_id = slow.keypoint_names.index("pelvis")
    displacement = [
        motion.positions[-1, root_id, 0] - motion.positions[0, root_id, 0]
        for motion in (slow, standard, fast)
    ]
    assert displacement[0] < displacement[1] < displacement[2]

    left = estimate_root_motion(
        generate_dog27_preset("turn_left", duration=1.0, fps=30.0)
    )
    right = estimate_root_motion(
        generate_dog27_preset("turn_right", duration=1.0, fps=30.0)
    )
    left_yaw = 2.0 * np.arctan2(left.rotation[-1, 3], left.rotation[-1, 0])
    right_yaw = 2.0 * np.arctan2(right.rotation[-1, 3], right.rotation[-1, 0])
    assert left_yaw == pytest.approx(0.65, abs=2e-3)
    assert right_yaw == pytest.approx(-0.65, abs=2e-3)


def test_synthetic_trot_has_physical_contact_and_fixed_bone_lengths() -> None:
    motion = generate_dog27_motion("trot", duration=1.0, fps=60.0)
    skeleton = get_skeleton("dog-27")
    name_to_index = {name: index for index, name in enumerate(motion.keypoint_names)}

    assert np.array_equal(motion.contact_probability[:, 0], motion.contact_probability[:, 3])
    assert np.array_equal(motion.contact_probability[:, 1], motion.contact_probability[:, 2])
    assert np.array_equal(
        motion.contact_probability[:, 0], 1.0 - motion.contact_probability[:, 1]
    )

    toe_ids = np.array(
        [
            name_to_index[skeleton.limb_chains[leg][-1]]
            for leg in ("FL", "FR", "RL", "RR")
        ]
    )
    toe_velocity = np.diff(motion.positions[:, toe_ids], axis=0) / np.diff(
        motion.timestamps
    )[:, None, None]
    stable_contact = (motion.contact_probability[:-1] >= 0.5) & (
        motion.contact_probability[1:] >= 0.5
    )
    assert np.max(np.linalg.norm(toe_velocity, axis=2)[stable_contact]) < 1e-4

    front_left_swing = motion.contact_probability[:, 0] < 0.5
    swing_x = motion.positions[front_left_swing, toe_ids[0], 0]
    assert swing_x[-1] > swing_x[0] + 0.15

    for leg in ("FL", "FR", "RL", "RR"):
        chain = np.array(
            [name_to_index[name] for name in skeleton.limb_chains[leg]]
        )
        lengths = np.linalg.norm(
            np.diff(motion.positions[:, chain], axis=1), axis=2
        )
        relative_change = np.abs(lengths - lengths[0]) / lengths[0]
        assert np.max(relative_change) < 2e-5

    root = estimate_root_motion(motion)
    assert root.position[-1, 2] == pytest.approx(root.position[0, 2], abs=1e-6)


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


def test_adaptive_preprocess_detects_spikes_and_smooths_keypoints() -> None:
    motion = generate_dog27_motion("walk", duration=1.0, fps=60.0)
    positions = motion.positions.copy()
    alternating = (np.arange(motion.frame_count) % 2 * 2 - 1)[:, None, None]
    positions += alternating * np.array([0.004, 0.0, 0.002])
    positions[25, 10] += np.array([0.35, 0.0, 0.2])
    noisy = replace(
        motion,
        positions=positions,
        contact_probability=np.full_like(motion.contact_probability, np.nan),
    )

    processed, report = preprocess_animal_motion(noisy)

    assert report.bone_anomaly[25, 10] or report.velocity_anomaly[25, 10]
    assert report.frame_abnormal[25]
    assert np.all(np.isfinite(processed.positions))
    assert np.all(np.isfinite(processed.contact_probability))
    raw_acceleration = np.diff(noisy.positions[:, 3], n=2, axis=0)
    filtered_acceleration = np.diff(processed.positions[:, 3], n=2, axis=0)
    assert np.std(filtered_acceleration) < 0.4 * np.std(raw_acceleration)
    assert processed.metadata["preprocess"]["mode"] == "adaptive_real_motion_v1"
    start, stop = report.neutral_frame_range
    assert 0 <= start <= stop < motion.frame_count


def test_ground_plane_estimation_tracks_a_sloped_floor() -> None:
    motion = generate_dog27_motion("walk", duration=1.0, fps=60.0)
    positions = motion.positions.copy()
    positions[..., 2] += 0.08 * positions[..., 0] - 0.04 * positions[..., 1]
    sloped = replace(motion, positions=positions)

    ground = estimate_ground_plane(sloped)
    expected = np.array([-0.08, 0.04, 1.0])
    expected /= np.linalg.norm(expected)

    assert np.dot(ground.normal, expected) > 0.995
    assert ground.candidate_count >= 6
    assert ground.rmse < 0.01


def test_local_contact_ground_reestimate_preserves_outside_interval() -> None:
    motion = generate_dog27_motion("walk", duration=1.0, fps=30.0)
    unknown = replace(
        motion,
        contact_probability=np.zeros_like(motion.contact_probability),
    )

    repaired, ground = reestimate_contact_and_ground(unknown, (8, 20))

    assert np.array_equal(repaired.contact_probability[:8], unknown.contact_probability[:8])
    assert np.array_equal(repaired.contact_probability[21:], unknown.contact_probability[21:])
    assert np.any(repaired.contact_probability[8:21] > 0.0)
    assert ground.normal[2] > 0.99
    assert repaired.metadata["contact_source"] == "mixed"
