from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from gqmr.core.errors import MotionValidationError, UnsafeMotionFileError
from gqmr.core.io import load_motion, save_motion
from gqmr.core.motion import AnimalMotion, RobotMotion, SolverStatus


def animal_metadata() -> dict[str, object]:
    return {
        "coordinate_frame": "gqmr_world_x_forward_y_left_z_up",
        "length_unit": "m",
        "time_unit": "s",
        "skeleton_id": "test-2",
        "skeleton_sha256": "a" * 64,
        "contact_order": ["FL", "FR", "RL", "RR"],
        "contact_source": "unknown",
        "source": {"kind": "synthetic"},
        "created_by": {"gqmr_version": "0.0.1"},
    }


def robot_metadata() -> dict[str, object]:
    return {
        "coordinate_frame": "gqmr_world_x_forward_y_left_z_up",
        "quaternion_order": "wxyz",
        "root_velocity_frame": "world",
        "model_id": "test-robot",
        "model_source_commit": "b" * 40,
        "model_sha256": "c" * 64,
        "robot_config_sha256": "d" * 64,
        "contact_order": ["FL", "FR", "RL", "RR"],
        "source_motion_sha256": "e" * 64,
        "retarget_config": {},
        "created_by": {"gqmr_version": "0.0.1"},
    }


def make_animal_motion() -> AnimalMotion:
    return AnimalMotion(
        timestamps=[0.0, 0.01, 0.02],
        keypoint_names=("hip", "foot"),
        positions=np.zeros((3, 2, 3)),
        confidence=np.ones((3, 2)),
        valid_mask=np.ones((3, 2), dtype=bool),
        contact_probability=np.full((3, 4), np.nan),
        frame_valid=np.ones(3, dtype=bool),
        metadata=animal_metadata(),
    )


def make_robot_motion() -> RobotMotion:
    return RobotMotion(
        timestamps=[0.0, 0.01, 0.02],
        dof_names=("hip", "knee"),
        root_position=np.zeros((3, 3)),
        root_rotation=np.array(
            [[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]]
        ),
        dof_position=np.zeros((3, 2)),
        root_linear_velocity=np.zeros((3, 3)),
        root_angular_velocity=np.zeros((3, 3)),
        dof_velocity=np.zeros((3, 2)),
        foot_contact_probability=np.zeros((3, 4)),
        frame_valid=np.ones(3, dtype=bool),
        solver_status=np.full(3, SolverStatus.OK),
        solver_residual=np.zeros(3),
        metadata=robot_metadata(),
    )


def test_animal_motion_round_trip(tmp_path: Path) -> None:
    destination = tmp_path / "sample.animal.npz"
    save_motion(destination, make_animal_motion())
    loaded = load_motion(destination)
    assert isinstance(loaded, AnimalMotion)
    assert loaded.keypoint_names == ("hip", "foot")
    assert loaded.timestamps.dtype == np.dtype("<f8")
    assert loaded.positions.dtype == np.dtype("<f4")


def test_robot_motion_round_trip_flips_quaternion_signs(tmp_path: Path) -> None:
    destination = tmp_path / "sample.robot.npz"
    save_motion(destination, make_robot_motion())
    loaded = load_motion(destination, expected_model_sha256="c" * 64)
    assert isinstance(loaded, RobotMotion)
    assert np.all(np.sum(loaded.root_rotation[:-1] * loaded.root_rotation[1:], axis=1) >= 0)


def test_model_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    destination = tmp_path / "sample.robot.npz"
    save_motion(destination, make_robot_motion())
    with pytest.raises(MotionValidationError, match="requested robot model"):
        load_motion(destination, expected_model_sha256="f" * 64)


@pytest.mark.parametrize(
    "timestamps",
    ([0.0, 0.0, 0.1], [0.0, 0.2, 0.1], [0.0, np.nan, 0.2], [1.0, 2.0, 3.0]),
)
def test_invalid_timestamps_are_rejected(timestamps: list[float]) -> None:
    motion = make_animal_motion()
    with pytest.raises(MotionValidationError, match="timestamps"):
        AnimalMotion(
            timestamps=timestamps,
            keypoint_names=motion.keypoint_names,
            positions=motion.positions,
            confidence=motion.confidence,
            valid_mask=motion.valid_mask,
            contact_probability=motion.contact_probability,
            frame_valid=motion.frame_valid,
            metadata=motion.metadata,
        )


def test_duplicate_names_are_rejected() -> None:
    motion = make_animal_motion()
    with pytest.raises(MotionValidationError, match="unique"):
        AnimalMotion(
            timestamps=motion.timestamps,
            keypoint_names=("foot", "foot"),
            positions=motion.positions,
            confidence=motion.confidence,
            valid_mask=motion.valid_mask,
            contact_probability=motion.contact_probability,
            frame_valid=motion.frame_valid,
            metadata=motion.metadata,
        )


def test_invalid_solver_status_cannot_be_exportable() -> None:
    motion = make_robot_motion()
    with pytest.raises(MotionValidationError, match="cannot be valid"):
        RobotMotion(
            timestamps=motion.timestamps,
            dof_names=motion.dof_names,
            root_position=motion.root_position,
            root_rotation=motion.root_rotation,
            dof_position=motion.dof_position,
            root_linear_velocity=motion.root_linear_velocity,
            root_angular_velocity=motion.root_angular_velocity,
            dof_velocity=motion.dof_velocity,
            foot_contact_probability=motion.foot_contact_probability,
            frame_valid=[True, True, True],
            solver_status=[SolverStatus.OK, SolverStatus.UNREACHABLE, SolverStatus.OK],
            solver_residual=motion.solver_residual,
            metadata=motion.metadata,
        )


def _npy_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, array, allow_pickle=True)
    return stream.getvalue()


def test_object_dtype_is_rejected(tmp_path: Path) -> None:
    destination = tmp_path / "object.npz"
    np.savez(destination, schema_id=np.array([{"bad": True}], dtype=object))
    with pytest.raises(UnsafeMotionFileError):
        load_motion(destination)


def test_invalid_json_is_rejected(tmp_path: Path) -> None:
    destination = tmp_path / "bad-json.npz"
    motion = make_animal_motion()
    save_motion(destination, motion)
    with np.load(destination, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    arrays["metadata_json"] = np.frombuffer(b'{"bad":NaN}', dtype=np.uint8)
    np.savez(destination, **arrays)
    with pytest.raises(UnsafeMotionFileError, match="invalid metadata_json"):
        load_motion(destination)


def test_duplicate_npz_member_is_rejected(tmp_path: Path) -> None:
    destination = tmp_path / "duplicate.npz"
    payload = _npy_bytes(np.asarray("gqmr.animal_motion"))
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(destination, "w") as archive:
            archive.writestr("schema_id.npy", payload)
            archive.writestr("schema_id.npy", payload)
    with pytest.raises(UnsafeMotionFileError, match="duplicate member"):
        load_motion(destination)


def test_duplicate_metadata_key_is_rejected(tmp_path: Path) -> None:
    destination = tmp_path / "duplicate-json-key.npz"
    motion = make_animal_motion()
    save_motion(destination, motion)
    with np.load(destination, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    text = json.dumps(animal_metadata())[:-1] + ',"length_unit":"cm"}'
    arrays["metadata_json"] = np.frombuffer(text.encode(), dtype=np.uint8)
    np.savez(destination, **arrays)
    with pytest.raises(UnsafeMotionFileError, match="duplicate key"):
        load_motion(destination)


def test_metadata_hash_format_is_rejected() -> None:
    motion = make_robot_motion()
    metadata = dict(motion.metadata)
    metadata["model_sha256"] = "not-a-sha256"
    with pytest.raises(MotionValidationError, match="model_sha256"):
        RobotMotion(
            timestamps=motion.timestamps,
            dof_names=motion.dof_names,
            root_position=motion.root_position,
            root_rotation=motion.root_rotation,
            dof_position=motion.dof_position,
            root_linear_velocity=motion.root_linear_velocity,
            root_angular_velocity=motion.root_angular_velocity,
            dof_velocity=motion.dof_velocity,
            foot_contact_probability=motion.foot_contact_probability,
            frame_valid=motion.frame_valid,
            solver_status=motion.solver_status,
            solver_residual=motion.solver_residual,
            metadata=metadata,
        )
