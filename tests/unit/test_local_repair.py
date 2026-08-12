from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from gqmr.core.coordinates import quaternion_geodesic_distance, xyzw_to_wxyz
from gqmr.core.derivatives import angular_velocity_world, linear_velocity
from gqmr.core.motion import RobotMotion, SolverStatus
from gqmr.retarget import (
    LocalRepairConfig,
    LocalRepairError,
    LocalRepairIntervalSolver,
    LocalRepairSolveConfig,
    build_local_repair_targets,
    run_local_repair,
    solve_local_repair,
)
from gqmr.robots.model import RobotModel
from test_robot_model import _synthetic_config, _synthetic_xml


def _robot_and_motion(tmp_path: Path, *, frames: int = 9) -> tuple[RobotModel, RobotMotion]:
    path = tmp_path / "synthetic.xml"
    path.write_text(_synthetic_xml(), encoding="utf-8")
    robot = RobotModel.from_xml_path(path, _synthetic_config())
    timestamps = np.arange(frames, dtype=np.float64) * 0.05
    root_position = np.zeros((frames, 3), dtype=np.float64)
    root_position[:, 0] = np.linspace(0.0, 0.4, frames)
    root_position[:, 2] = 0.3 + 0.015 * np.sin(np.linspace(0.0, 2.0 * np.pi, frames))
    euler = np.zeros((frames, 3), dtype=np.float64)
    euler[:, 0] = 0.20 * np.sin(np.linspace(0.0, np.pi, frames))
    euler[:, 1] = -0.15 * np.sin(np.linspace(0.0, np.pi, frames))
    euler[:, 2] = np.linspace(0.0, 0.4, frames)
    root_rotation = xyzw_to_wxyz(Rotation.from_euler("xyz", euler).as_quat())
    dof_position = np.tile(np.asarray(robot.config.default_dof_position), (frames, 1))
    dof_position[:, 1::3] += 0.08 * np.sin(np.linspace(0.0, 3.0 * np.pi, frames))[:, None]
    contact = np.tile(np.array([0.2, 0.4, 0.6, 0.8]), (frames, 1))
    motion = RobotMotion(
        timestamps=timestamps,
        dof_names=robot.config.dof_order,
        root_position=root_position,
        root_rotation=root_rotation,
        dof_position=dof_position,
        root_linear_velocity=np.zeros((frames, 3)),
        root_angular_velocity=np.zeros((frames, 3)),
        dof_velocity=np.zeros((frames, 12)),
        foot_contact_probability=contact,
        frame_valid=np.ones(frames, dtype=bool),
        solver_status=np.full(frames, SolverStatus.OK),
        solver_residual=np.zeros(frames),
        metadata={
            "coordinate_frame": "gqmr_world_x_forward_y_left_z_up",
            "quaternion_order": "wxyz",
            "root_velocity_frame": "world",
            "model_id": robot.config.id,
            "model_source_commit": "a" * 40,
            "model_sha256": robot.config.model_sha256,
            "robot_config_sha256": robot.config.sha256,
            "contact_order": ["FL", "FR", "RL", "RR"],
            "source_motion_sha256": "b" * 64,
            "retarget_config": {},
            "created_by": {"gqmr_version": "0.0.1"},
        },
    )
    return robot, motion


def _outside(frame_count: int, start: int, stop: int) -> np.ndarray:
    mask = np.ones(frame_count, dtype=bool)
    mask[start : stop + 1] = False
    return mask


def test_a2_root_height_translation_and_tilt_are_grouped(tmp_path: Path) -> None:
    robot, motion = _robot_and_motion(tmp_path)
    start, stop = 2, 6
    config = LocalRepairConfig(
        root_height_offset_m=0.04,
        root_translation_scale=0.5,
        root_tilt_scale=0.0,
    )
    targets = build_local_repair_targets(motion, robot, (start, stop), config)
    outside = _outside(motion.frame_count, start, stop)

    expected_delta = 0.5 * (
        motion.root_position[stop] - motion.root_position[start]
    )
    expected_delta[2] += 0.04 - 0.04
    assert targets.root_position[stop] - targets.root_position[start] == pytest.approx(
        expected_delta
    )
    assert targets.root_position[start, 2] == pytest.approx(
        motion.root_position[start, 2] + 0.04
    )
    heading = Rotation.from_quat(
        targets.root_rotation[start : stop + 1][:, [1, 2, 3, 0]]
    ).apply([1.0, 0.0, 0.0])
    assert np.max(np.abs(heading[:, 2])) < 1e-7
    assert np.array_equal(targets.root_position[outside], motion.root_position[outside])
    assert np.array_equal(targets.root_rotation[outside], motion.root_rotation[outside])


def test_a2_limb_amplitude_and_contact_modes_are_local(tmp_path: Path) -> None:
    robot, motion = _robot_and_motion(tmp_path)
    start, stop = 1, 7
    baseline = build_local_repair_targets(
        motion, robot, (start, stop), LocalRepairConfig()
    )
    targets = build_local_repair_targets(
        motion,
        robot,
        (start, stop),
        LocalRepairConfig(
            limb_target_scale=0.5,
            foot_modes={"FL": "lock", "RL": "unlock"},
        ),
    )
    outside = _outside(motion.frame_count, start, stop)
    baseline_center = np.median(baseline.foot_positions[start : stop + 1], axis=0)
    scaled_center = np.median(targets.foot_positions[start : stop + 1], axis=0)
    baseline_spread = np.linalg.norm(
        baseline.foot_positions[start : stop + 1] - baseline_center, axis=2
    )
    scaled_spread = np.linalg.norm(
        targets.foot_positions[start : stop + 1] - scaled_center, axis=2
    )

    assert np.mean(scaled_spread) < np.mean(baseline_spread)
    assert np.all(targets.contact_probability[start : stop + 1, 0] == 1.0)
    assert np.all(targets.contact_probability[start : stop + 1, 2] == 0.0)
    assert np.array_equal(
        targets.contact_probability[start : stop + 1, 1],
        motion.foot_contact_probability[start : stop + 1, 1],
    )
    assert np.array_equal(
        targets.contact_probability[outside], motion.foot_contact_probability[outside]
    )
    assert np.array_equal(
        targets.foot_positions[outside], baseline.foot_positions[outside]
    )


def test_a2_foot_targets_follow_modified_root_without_changing_local_limb_shape(
    tmp_path: Path,
) -> None:
    robot, motion = _robot_and_motion(tmp_path)
    start, stop = 2, 6
    baseline = build_local_repair_targets(
        motion, robot, (start, stop), LocalRepairConfig()
    )
    shifted = build_local_repair_targets(
        motion,
        robot,
        (start, stop),
        LocalRepairConfig(root_height_offset_m=0.05, root_translation_scale=0.7),
    )
    baseline_rotation = Rotation.from_quat(
        baseline.root_rotation[start : stop + 1][:, [1, 2, 3, 0]]
    )
    shifted_rotation = Rotation.from_quat(
        shifted.root_rotation[start : stop + 1][:, [1, 2, 3, 0]]
    )
    baseline_local = np.empty((stop - start + 1, 4, 3))
    shifted_local = np.empty_like(baseline_local)
    for offset in range(stop - start + 1):
        baseline_local[offset] = baseline_rotation[offset].inv().apply(
            baseline.foot_positions[start + offset]
            - baseline.root_position[start + offset]
        )
        shifted_local[offset] = shifted_rotation[offset].inv().apply(
            shifted.foot_positions[start + offset]
            - shifted.root_position[start + offset]
        )
    assert shifted_local == pytest.approx(baseline_local)


def test_a2_smoothing_reduces_target_roughness_and_is_deterministic(tmp_path: Path) -> None:
    robot, motion = _robot_and_motion(tmp_path)
    root_position = motion.root_position.copy()
    root_position[4, 1] += 0.2
    noisy = replace(motion, root_position=root_position)
    raw = build_local_repair_targets(noisy, robot, (1, 7), LocalRepairConfig())
    config = LocalRepairConfig(smoothing_strength=1.0)
    first = build_local_repair_targets(noisy, robot, (1, 7), config)
    second = build_local_repair_targets(noisy, robot, (1, 7), config)

    raw_roughness = np.linalg.norm(np.diff(raw.root_position[1:8], n=2, axis=0))
    smooth_roughness = np.linalg.norm(np.diff(first.root_position[1:8], n=2, axis=0))
    assert smooth_roughness < raw_roughness
    assert np.array_equal(first.root_position, second.root_position)
    assert np.array_equal(first.root_rotation, second.root_rotation)
    assert np.array_equal(first.foot_positions, second.foot_positions)
    assert np.array_equal(first.contact_probability, second.contact_probability)


def test_a2_contact_and_ground_reestimate_are_interval_scoped(tmp_path: Path) -> None:
    robot, motion = _robot_and_motion(tmp_path)
    motion = replace(
        motion,
        foot_contact_probability=np.full_like(
            motion.foot_contact_probability, 0.25
        ),
    )
    targets = build_local_repair_targets(
        motion,
        robot,
        (2, 6),
        LocalRepairConfig(reestimate_contact=True, reestimate_ground=True),
    )

    assert targets.ground is not None
    assert targets.ground.normal[2] > 0.85
    assert np.all(np.isfinite(targets.contact_probability[2:7]))
    assert np.any(targets.contact_probability[2:7] != 0.25)
    assert np.array_equal(
        targets.contact_probability[:2], motion.foot_contact_probability[:2]
    )
    assert np.array_equal(
        targets.contact_probability[7:], motion.foot_contact_probability[7:]
    )


def test_a2_rejects_incompatible_robot_and_preserves_input(tmp_path: Path) -> None:
    robot, motion = _robot_and_motion(tmp_path)
    root_before = motion.root_position.copy()
    with pytest.raises(LocalRepairError, match="model hash"):
        build_local_repair_targets(
            replace(motion, metadata={**motion.metadata, "model_sha256": "f" * 64}),
            robot,
            (1, 3),
            LocalRepairConfig(),
        )
    build_local_repair_targets(motion, robot, (1, 3), LocalRepairConfig())
    assert np.array_equal(motion.root_position, root_before)


def test_a2_tilt_scale_one_preserves_selected_rotation(tmp_path: Path) -> None:
    robot, motion = _robot_and_motion(tmp_path)
    targets = build_local_repair_targets(
        motion, robot, (2, 6), LocalRepairConfig(root_tilt_scale=1.0)
    )
    distance = quaternion_geodesic_distance(
        targets.root_rotation[2:7], motion.root_rotation[2:7]
    )
    assert np.max(distance) < 1e-7


def test_a3_joint_root_solver_is_local_and_uses_so3_boundary_blend(
    tmp_path: Path,
) -> None:
    robot, motion = _robot_and_motion(tmp_path, frames=13)
    start, stop = 2, 10
    output = solve_local_repair(
        motion,
        robot,
        (start, stop),
        LocalRepairConfig(
            root_height_offset_m=0.04,
            root_tilt_scale=0.25,
            limb_target_scale=0.8,
        ),
    )
    repaired = output.motion
    outside = _outside(motion.frame_count, start, stop)

    for field in (
        "root_position",
        "root_rotation",
        "dof_position",
        "root_linear_velocity",
        "root_angular_velocity",
        "dof_velocity",
        "foot_contact_probability",
        "frame_valid",
        "solver_status",
        "solver_residual",
    ):
        assert np.array_equal(
            getattr(repaired, field)[outside], getattr(motion, field)[outside]
        )
    assert np.array_equal(repaired.root_position[start], motion.root_position[start])
    assert np.array_equal(repaired.root_position[stop], motion.root_position[stop])
    assert quaternion_geodesic_distance(
        repaired.root_rotation[start], motion.root_rotation[start]
    ) == pytest.approx(0.0)
    assert quaternion_geodesic_distance(
        repaired.root_rotation[stop], motion.root_rotation[stop]
    ) == pytest.approx(0.0)
    midpoint = (start + stop) // 2
    assert repaired.root_position[midpoint, 2] > motion.root_position[midpoint, 2]
    assert quaternion_geodesic_distance(
        repaired.root_rotation[midpoint], motion.root_rotation[midpoint]
    ) > 1e-4
    weights = output.diagnostics.details["transition_weights"]
    assert weights[0] == pytest.approx(0.0)
    assert weights[-1] == pytest.approx(0.0)
    assert max(weights) == pytest.approx(1.0)
    assert output.diagnostics.details["buffer_frames"] >= 1


def test_a3_recomputes_only_selected_velocities_states_and_residuals(
    tmp_path: Path,
) -> None:
    robot, motion = _robot_and_motion(tmp_path, frames=13)
    start, stop = 2, 10
    output = solve_local_repair(
        motion,
        robot,
        (start, stop),
        LocalRepairConfig(root_height_offset_m=0.03, limb_target_scale=0.85),
    )
    repaired = output.motion
    selected = slice(start, stop + 1)

    assert repaired.root_linear_velocity[selected] == pytest.approx(
        linear_velocity(repaired.timestamps, repaired.root_position)[selected],
        abs=1e-6,
    )
    assert repaired.root_angular_velocity[selected] == pytest.approx(
        angular_velocity_world(repaired.timestamps, repaired.root_rotation)[selected],
        abs=1e-6,
    )
    assert repaired.dof_velocity[selected] == pytest.approx(
        linear_velocity(repaired.timestamps, repaired.dof_position)[selected],
        abs=1e-6,
    )
    assert np.all(np.isfinite(repaired.solver_residual[selected]))
    assert np.all(
        np.isin(
            repaired.solver_status[selected],
            [SolverStatus.OK, SolverStatus.MAX_ITER, SolverStatus.UNREACHABLE],
        )
    )
    assert output.diagnostics.frames_processed == stop - start + 1
    assert output.diagnostics.residual_rmse_after_m is not None


def test_a3_bound_solver_replays_deterministically_and_adapts_buffer(
    tmp_path: Path,
) -> None:
    robot, motion = _robot_and_motion(tmp_path, frames=17)
    solver = LocalRepairIntervalSolver(
        robot,
        LocalRepairSolveConfig(maximum_buffer_seconds=0.20),
    )
    config = LocalRepairConfig(root_height_offset_m=0.02, smoothing_strength=0.5)
    first = run_local_repair(motion, (2, 14), config, solver)
    second = run_local_repair(motion, (2, 14), config, solver)

    assert first.output_motion_sha256 == second.output_motion_sha256
    assert first.diagnostics == second.diagnostics
    assert first.diagnostics.details["buffer_frames"] > 1
    assert first.diagnostics.residual_rmse_after_m <= (
        first.diagnostics.residual_rmse_before_m + 1e-9
    )
