from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from gqmr.cli.main import main
from gqmr.core.coordinates import quaternion_geodesic_distance
from gqmr.core.io import load_motion, save_motion
from gqmr.core.motion import RobotMotion, SolverStatus
from gqmr.exporters import load_isaaclab_amp_v232
from gqmr.exporters.isaaclab_amp import export_isaaclab_amp_v232
from gqmr.retarget import replay_quality_report, retarget_fast
from gqmr.robots import LEG_ORDER, load_robot_model
from gqmr.synthetic import generate_dog27_motion


def _asset_cache() -> Path:
    value = os.environ.get("GQMR_TEST_ASSET_CACHE")
    if not value:
        pytest.skip("set GQMR_TEST_ASSET_CACHE to a verified Unitree asset cache")
    return Path(value)


def _rotation_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rotation_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _analytic_default_feet(robot_id: str) -> np.ndarray:
    if robot_id == "unitree-go2":
        root = np.array([0.0, 0.0, 0.27])
        hip_x, hip_y, thigh_y = 0.1934, 0.0465, 0.0955
        upper, lower = 0.213, 0.213
        pose = (0.0, 0.9, -1.8)
        calf_y = {leg: 0.0 for leg in LEG_ORDER}
    else:
        root = np.array([0.0, 0.0, 0.5])
        hip_x, hip_y, thigh_y = 0.3285, 0.072, 0.11973
        upper, lower = 0.35, 0.35
        pose = (0.0, 1.28, -2.8)
        calf_y = {
            "FL": -8.6984e-05,
            "FR": 8.6986e-05,
            "RL": -8.6984e-05,
            "RR": 8.6986e-05,
        }
    result = []
    for leg in LEG_ORDER:
        front = leg[0] == "F"
        left = leg[1] == "L"
        position = root + np.array(
            [hip_x if front else -hip_x, hip_y if left else -hip_y, 0.0]
        )
        rotation = _rotation_x(pose[0])
        position = position + rotation @ np.array(
            [0.0, thigh_y if left else -thigh_y, 0.0]
        )
        rotation = rotation @ _rotation_y(pose[1])
        position = position + rotation @ np.array([0.0, calf_y[leg], -upper])
        rotation = rotation @ _rotation_y(pose[2])
        position = position + rotation @ np.array([0.0, 0.0, -lower])
        result.append(position)
    return np.stack(result)


@pytest.mark.parametrize("robot_id", ["unitree-go2", "unitree-b2"])
def test_unitree_default_fk_matches_independent_chain(robot_id: str) -> None:
    robot = load_robot_model(robot_id, cache_dir=_asset_cache())
    assert (robot.model.nq, robot.model.nv, robot.model.nu) == (19, 18, 12)
    assert np.max(np.abs(robot.foot_positions() - _analytic_default_feet(robot_id))) < 1e-5
    assert np.max(
        np.abs(robot.body_position("base_link") - robot.config.default_root_position)
    ) < 1e-12


@pytest.mark.parametrize("robot_id", ["unitree-go2", "unitree-b2"])
def test_unitree_foot_jacobians_match_100_pose_central_difference(
    robot_id: str,
) -> None:
    robot = load_robot_model(robot_id, cache_dir=_asset_cache())
    rng = np.random.default_rng(20260811)
    lower = robot.joint_ranges[:, 0]
    upper = robot.joint_ranges[:, 1]
    margin = np.minimum((upper - lower) * 0.1, 0.05)
    step = 1e-6
    for _ in range(100):
        pose = rng.uniform(lower + margin, upper - margin)
        robot.set_pose(
            robot.config.default_root_position,
            robot.config.default_root_rotation,
            pose,
        )
        analytic = robot.foot_jacobians()
        finite = np.zeros_like(analytic)
        for dof_index in range(12):
            plus = pose.copy()
            minus = pose.copy()
            plus[dof_index] += step
            minus[dof_index] -= step
            robot.set_pose(
                robot.config.default_root_position,
                robot.config.default_root_rotation,
                plus,
            )
            plus_positions = robot.foot_positions()
            robot.set_pose(
                robot.config.default_root_position,
                robot.config.default_root_rotation,
                minus,
            )
            minus_positions = robot.foot_positions()
            finite[:, :, dof_index] = (plus_positions - minus_positions) / (2 * step)
        relative_error = np.linalg.norm(analytic - finite) / max(
            np.linalg.norm(finite), 1e-12
        )
        assert relative_error < 1e-4


def test_robot_inspect_and_motion_model_binding_cli(tmp_path: Path, capsys) -> None:
    cache = _asset_cache()
    result = main(
        ["robots", "inspect", "unitree-go2", "--cache-dir", str(cache)]
    )
    assert result == 0
    assert '"nq": 19' in capsys.readouterr().out

    robot = load_robot_model("unitree-go2", cache_dir=cache)
    frames = 3
    motion = RobotMotion(
        timestamps=[0.0, 0.01, 0.02],
        dof_names=robot.config.dof_order,
        root_position=np.tile(robot.config.default_root_position, (frames, 1)),
        root_rotation=np.tile(robot.config.default_root_rotation, (frames, 1)),
        dof_position=np.tile(robot.config.default_dof_position, (frames, 1)),
        root_linear_velocity=np.zeros((frames, 3)),
        root_angular_velocity=np.zeros((frames, 3)),
        dof_velocity=np.zeros((frames, 12)),
        foot_contact_probability=np.zeros((frames, 4)),
        frame_valid=np.ones(frames, dtype=bool),
        solver_status=np.full(frames, SolverStatus.OK),
        solver_residual=np.zeros(frames),
        metadata={
            "coordinate_frame": "gqmr_world_x_forward_y_left_z_up",
            "quaternion_order": "wxyz",
            "root_velocity_frame": "world",
            "model_id": robot.config.id,
            "model_source_commit": "ae6a8403e272733e9996ef59990880330496177f",
            "model_sha256": robot.config.model_sha256,
            "robot_config_sha256": robot.config.sha256,
            "contact_order": ["FL", "FR", "RL", "RR"],
            "source_motion_sha256": "e" * 64,
            "retarget_config": {},
            "created_by": {"gqmr_version": "0.0.1"},
        },
    )
    destination = tmp_path / "go2.robot.npz"
    save_motion(destination, motion)
    result = main(
        [
            "validate",
            str(destination),
            "--robot",
            "unitree-go2",
            "--cache-dir",
            str(cache),
        ]
    )
    assert result == 0
    assert '"valid": true' in capsys.readouterr().out


@pytest.mark.parametrize("robot_id", ["unitree-go2", "unitree-b2"])
@pytest.mark.parametrize("gait", ["walk", "trot", "pace", "turn"])
def test_fast_retarget_meets_synthetic_acceptance(robot_id: str, gait: str) -> None:
    robot = load_robot_model(robot_id, cache_dir=_asset_cache())
    animal = generate_dog27_motion(gait, duration=1.0, fps=60.0)

    motion, diagnostics = retarget_fast(animal, robot)
    report = replay_quality_report(motion, robot)

    assert np.mean(motion.frame_valid) >= 0.995
    assert np.sqrt(np.mean(motion.solver_residual[motion.frame_valid] ** 2)) <= 0.03
    assert report["joint_limit_violation_frames"] == 0
    assert report["replayed_frames"] == motion.frame_count
    assert diagnostics.target_foot_positions.shape == (61, 4, 3)
    assert np.max(
        np.abs(
            diagnostics.target_foot_positions[motion.frame_valid]
            - diagnostics.achieved_foot_positions[motion.frame_valid]
        )
    ) < 0.09


def test_synthetic_to_robot_cli_closed_loop(tmp_path: Path, capsys) -> None:
    cache = _asset_cache()
    animal_path = tmp_path / "pace.animal.npz"
    robot_path = tmp_path / "pace.go2.npz"

    assert main(
        [
            "synthetic",
            "pace",
            "--duration",
            "0.5",
            "--fps",
            "30",
            "--output",
            str(animal_path),
        ]
    ) == 0
    capsys.readouterr()
    assert main(
        [
            "retarget",
            str(animal_path),
            "--robot",
            "unitree-go2",
            "--cache-dir",
            str(cache),
            "--output",
            str(robot_path),
        ]
    ) == 0
    retarget_output = capsys.readouterr().out
    assert '"valid_frame_ratio": 1.0' in retarget_output
    assert isinstance(load_motion(robot_path), RobotMotion)

    assert main(
        [
            "play",
            str(robot_path),
            "--robot",
            "unitree-go2",
            "--cache-dir",
            str(cache),
        ]
    ) == 0
    play_output = capsys.readouterr().out
    assert '"joint_limit_violation_frames": 0' in play_output


@pytest.mark.parametrize("robot_id", ["unitree-go2", "unitree-b2"])
def test_isaaclab_amp_export_fk_roundtrip(tmp_path: Path, robot_id: str) -> None:
    robot = load_robot_model(robot_id, cache_dir=_asset_cache())
    animal = generate_dog27_motion("turn", duration=0.5, fps=60.0)
    motion, _ = retarget_fast(animal, robot)
    destination = tmp_path / f"{robot_id}.amp.npz"

    export_isaaclab_amp_v232(motion, robot, destination, fps=60)
    clip = load_isaaclab_amp_v232(destination)

    assert clip.dof_names == robot.config.dof_order
    assert robot.config.base_body in clip.body_names
    assert clip.frame_count == motion.frame_count
    rng = np.random.default_rng(20260811)
    samples = clip.sample(rng.uniform(0.0, clip.duration, size=1000))
    assert samples["dof_positions"].shape == (1000, 12)
    assert np.max(
        np.abs(np.linalg.norm(samples["body_rotations"], axis=-1) - 1.0)
    ) < 1e-5

    base_index = clip.body_names.index(robot.config.base_body)
    maximum_position_error = 0.0
    maximum_rotation_error = 0.0
    for frame in range(clip.frame_count):
        robot.set_pose(
            clip.body_positions[frame, base_index],
            clip.body_rotations[frame, base_index],
            clip.dof_positions[frame],
        )
        for body_index, body_name in enumerate(clip.body_names):
            position, rotation = robot.body_pose(body_name)
            maximum_position_error = max(
                maximum_position_error,
                float(np.max(np.abs(position - clip.body_positions[frame, body_index]))),
            )
            maximum_rotation_error = max(
                maximum_rotation_error,
                float(
                    quaternion_geodesic_distance(
                        rotation, clip.body_rotations[frame, body_index]
                    )
                ),
            )
    assert maximum_position_error < 1e-5
    assert maximum_rotation_error < 1e-5
