from __future__ import annotations

import uuid
from datetime import datetime, timezone

import numpy as np
import pytest
from pydantic import ValidationError

from gqmr.editing import (
    EditStack,
    EditingError,
    apply_edit,
    concatenate_robot_motions,
    filter_robot_motion,
    make_robot_loop,
)
from dataclasses import replace
from gqmr.core.coordinates import quaternion_geodesic_distance
from gqmr.core.io import motion_sha256
from test_motion_io import make_robot_motion
from gqmr.project.model import EditCommand
from gqmr.retarget import (
    LocalRepairCommand,
    LocalRepairConfig,
    LocalRepairDiagnostics,
    LocalRepairError,
    LocalRepairSolverOutput,
    run_local_repair,
)
from gqmr.synthetic import generate_dog27_motion


def _command(kind: str, parameters: dict) -> EditCommand:
    return EditCommand(
        command_id=str(uuid.uuid4()),
        kind=kind,
        resource_id=str(uuid.uuid4()),
        parameters=parameters,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def test_trim_time_scale_root_transform_and_contact_override() -> None:
    motion = generate_dog27_motion("walk", duration=1.0, fps=20.0)
    trimmed = apply_edit(motion, _command("trim", {"start": 0.25, "end": 0.75}))
    scaled = apply_edit(trimmed, _command("time_scale", {"speed": 2.0}))
    transformed = apply_edit(
        scaled,
        _command(
            "root_transform", {"translation": [1.0, 2.0, 0.0], "yaw": np.pi / 2}
        ),
    )
    contacted = apply_edit(
        transformed,
        _command(
            "contact_override",
            {"start": 0.0, "end": transformed.duration, "leg": "FL", "probability": 1.0},
        ),
    )

    assert trimmed.timestamps[0] == 0.0
    assert scaled.duration == pytest.approx(trimmed.duration / 2.0)
    assert np.allclose(
        transformed.positions[0, 0, :2],
        [-scaled.positions[0, 0, 1] + 1.0, scaled.positions[0, 0, 0] + 2.0],
        atol=1e-6,
    )
    assert np.all(contacted.contact_probability[:, 0] == 1.0)
    assert contacted.metadata["contact_source"] == "mixed"
    assert len(contacted.metadata["edit_history"]) == 4


def test_resample_and_undo_redo_are_deterministic() -> None:
    motion = generate_dog27_motion("turn", duration=1.0, fps=20.0)
    command = _command("resample", {"fps": 40})
    stack = EditStack(motion)

    edited = stack.push(command)
    assert edited.frame_count == 41
    assert stack.undo() is motion
    redone = stack.redo()
    assert np.array_equal(redone.positions, edited.positions)
    assert redone.metadata["edit_history"] == edited.metadata["edit_history"]


def test_robot_loop_closes_rotation_and_dofs() -> None:
    motion = make_robot_motion()
    root_position = motion.root_position.copy()
    root_position[-1] = [1.0, 0.2, 0.1]
    dof_position = motion.dof_position.copy()
    dof_position[-1] = [0.2, -0.1]
    motion = motion.__class__(
        timestamps=motion.timestamps,
        dof_names=motion.dof_names,
        root_position=root_position,
        root_rotation=motion.root_rotation,
        dof_position=dof_position,
        root_linear_velocity=motion.root_linear_velocity,
        root_angular_velocity=motion.root_angular_velocity,
        dof_velocity=motion.dof_velocity,
        foot_contact_probability=motion.foot_contact_probability,
        frame_valid=motion.frame_valid,
        solver_status=motion.solver_status,
        solver_residual=motion.solver_residual,
        metadata=motion.metadata,
    )
    loop = make_robot_loop(motion)

    assert float(quaternion_geodesic_distance(loop.root_rotation[0], loop.root_rotation[-1])) < np.deg2rad(1.0)
    assert np.max(np.abs(loop.dof_position[-1] - loop.dof_position[0])) < 0.03
    assert loop.root_position[-1, 0] == 1.0
    assert np.allclose(loop.root_position[-1, 1:], loop.root_position[0, 1:])


def test_filter_and_concatenate_robot_motion() -> None:
    base = make_robot_motion()
    timestamps = np.arange(11, dtype=np.float64) * 0.01
    noise = np.array([(-1.0) ** index for index in range(11)])[:, None] * 0.05
    motion = replace(
        base,
        timestamps=timestamps,
        root_position=np.zeros((11, 3)),
        root_rotation=np.tile([1.0, 0.0, 0.0, 0.0], (11, 1)),
        dof_position=np.tile([0.2, -0.1], (11, 1)) + noise,
        root_linear_velocity=np.zeros((11, 3)),
        root_angular_velocity=np.zeros((11, 3)),
        dof_velocity=np.zeros((11, 2)),
        foot_contact_probability=np.zeros((11, 4)),
        frame_valid=np.ones(11, dtype=bool),
        solver_status=np.zeros(11, dtype=np.int16),
        solver_residual=np.zeros(11),
    )
    filtered = filter_robot_motion(motion, window_frames=5, polynomial_order=2)
    combined = concatenate_robot_motions([filtered, filtered], blend_seconds=0.03)

    assert np.std(filtered.dof_position[:, 0]) < np.std(motion.dof_position[:, 0])
    assert combined.frame_count == 21
    assert np.all(np.diff(combined.timestamps) > 0.0)
    assert combined.timestamps[-1] == pytest.approx(0.2)


def test_one_hundred_undo_redo_cycles_preserve_hash() -> None:
    motion = generate_dog27_motion("walk", duration=0.2, fps=20.0)
    stack = EditStack(motion)
    for _ in range(100):
        stack.push(_command("time_scale", {"speed": 1.0}))
    edited_hash = motion_sha256(stack.current())
    for _ in range(100):
        stack.undo()
    assert motion_sha256(stack.current()) == motion_sha256(motion)
    for _ in range(100):
        stack.redo()
    assert motion_sha256(stack.current()) == edited_hash


def _local_repair_solver(
    motion, frame_range: tuple[int, int], config: LocalRepairConfig
) -> LocalRepairSolverOutput:
    start, stop = frame_range
    root_position = motion.root_position.copy()
    root_position[start : stop + 1, 2] += config.root_height_offset_m
    repaired = replace(motion, root_position=root_position)
    diagnostics = LocalRepairDiagnostics(
        solver="fixed-test-solver",
        solver_version="1.0",
        frames_processed=stop - start + 1,
        iterations=3,
        converged=True,
        residual_rmse_before_m=0.02,
        residual_rmse_after_m=0.01,
        status_counts={"ok": stop - start + 1},
    )
    return LocalRepairSolverOutput(
        motion=repaired,
        applied_config=config,
        diagnostics=diagnostics,
    )


def test_local_repair_config_validation_and_json_round_trip() -> None:
    config = LocalRepairConfig(
        root_height_offset_m=0.04,
        root_translation_scale=0.8,
        root_tilt_scale=0.7,
        limb_target_scale=0.9,
        smoothing_strength=0.5,
        foot_modes={"FL": "lock", "FR": "auto", "RL": "unlock", "RR": "auto"},
        reestimate_contact=True,
        reestimate_ground=True,
    )

    assert LocalRepairConfig.model_validate_json(config.model_dump_json()) == config
    with pytest.raises(ValidationError, match="less than or equal to 0.25"):
        LocalRepairConfig(root_height_offset_m=0.3)
    assert LocalRepairConfig(foot_modes={"FL": "lock"}).foot_modes.FR == "auto"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        LocalRepairConfig(foot_modes={"FL": "lock", "middle": "auto"})
    with pytest.raises(ValidationError, match="finite number"):
        LocalRepairConfig(smoothing_strength=float("nan"))


def test_local_repair_result_command_round_trip_and_replay() -> None:
    motion = make_robot_motion()
    config = LocalRepairConfig(root_height_offset_m=0.02)
    result = run_local_repair(motion, (0, 1), config, _local_repair_solver)
    command = LocalRepairCommand.from_result(
        result,
        resource_id=str(uuid.uuid4()),
        command_id=str(uuid.uuid4()),
        created_at="2026-08-12T00:00:00Z",
    )
    edit_command = command.to_edit_command()
    restored = LocalRepairCommand.from_edit_command(edit_command)
    stack = EditStack(motion, local_repair_solver=_local_repair_solver)

    edited = stack.push(edit_command)
    assert restored == command
    assert motion_sha256(edited) == result.output_motion_sha256
    assert edited.metadata["local_repair_history"][-1]["applied_config"] == (
        config.model_dump(mode="json")
    )
    for _ in range(100):
        assert stack.undo() is motion
        assert motion_sha256(stack.redo()) == result.output_motion_sha256


def test_local_repair_rejects_outside_changes_and_nondeterministic_replay() -> None:
    motion = make_robot_motion()
    config = LocalRepairConfig(root_height_offset_m=0.02)

    def leaking_solver(motion, frame_range, config):
        output = _local_repair_solver(motion, frame_range, config)
        root_position = output.motion.root_position.copy()
        root_position[-1, 0] += 0.1
        return replace(output, motion=replace(output.motion, root_position=root_position))

    with pytest.raises(LocalRepairError, match="outside frame_range"):
        run_local_repair(motion, (0, 0), config, leaking_solver)

    result = run_local_repair(motion, (0, 1), config, _local_repair_solver)
    command = LocalRepairCommand.from_result(
        result, resource_id=str(uuid.uuid4())
    ).to_edit_command()

    def changed_solver(motion, frame_range, config):
        output = _local_repair_solver(motion, frame_range, config)
        root_position = output.motion.root_position.copy()
        root_position[frame_range[0], 2] += 0.001
        return replace(output, motion=replace(output.motion, root_position=root_position))

    with pytest.raises(EditingError, match="different content hash"):
        apply_edit(motion, command, local_repair_solver=changed_solver)


def test_local_repair_solver_cannot_mutate_the_base_motion() -> None:
    motion = make_robot_motion()
    original = motion.root_position.copy()

    def mutating_solver(motion, frame_range, config):
        motion.root_position[:] = 42.0
        return _local_repair_solver(motion, frame_range, config)

    with pytest.raises(LocalRepairError, match="outside frame_range"):
        run_local_repair(
            motion,
            (0, 0),
            LocalRepairConfig(root_height_offset_m=0.01),
            mutating_solver,
        )
    assert np.array_equal(motion.root_position, original)
