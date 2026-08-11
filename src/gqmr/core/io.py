"""Safe, strict NPZ serialization for GQMR Motion Schema v1."""

from __future__ import annotations

import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Mapping, TypeAlias

import numpy as np

from gqmr.core.errors import MotionValidationError, UnsafeMotionFileError
from gqmr.core.motion import AnimalMotion, RobotMotion

Motion: TypeAlias = AnimalMotion | RobotMotion

_COMMON_KEYS = {"schema_id", "schema_version", "timestamps", "metadata_json"}
_ANIMAL_KEYS = _COMMON_KEYS | {
    "keypoint_names",
    "positions",
    "confidence",
    "valid_mask",
    "contact_probability",
    "frame_valid",
}
_ROBOT_KEYS = _COMMON_KEYS | {
    "dof_names",
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
}
_MAX_MEMBERS = 128
_MAX_UNCOMPRESSED_BYTES = 8 * 1024 * 1024 * 1024
_MAX_COMPRESSION_RATIO = 10_000


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise UnsafeMotionFileError(f"metadata_json has duplicate key {key!r}")
        result[key] = value
    return result


def _parse_metadata(raw: np.ndarray) -> dict[str, Any]:
    if raw.dtype != np.dtype(np.uint8) or raw.ndim != 1:
        raise MotionValidationError("must be uint8[M]", field="metadata_json")
    try:
        text = raw.tobytes().decode("utf-8")
        value = json.loads(
            text,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {constant}")
            ),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise UnsafeMotionFileError(f"invalid metadata_json: {error}") from error
    if not isinstance(value, dict):
        raise MotionValidationError("must encode a JSON object", field="metadata_json")
    return value


def _metadata_array(metadata: Mapping[str, Any]) -> np.ndarray:
    try:
        encoded = json.dumps(
            metadata,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise MotionValidationError(
            f"is not strict JSON: {error}", field="metadata_json"
        ) from error
    return np.frombuffer(encoded, dtype=np.uint8).copy()


def _inspect_npz(path: Path) -> None:
    if not path.is_file():
        raise UnsafeMotionFileError(f"motion file does not exist: {path}")
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if len(members) > _MAX_MEMBERS:
                raise UnsafeMotionFileError("NPZ contains too many members")
            filenames = [member.filename for member in members]
            if len(filenames) != len(set(filenames)):
                raise UnsafeMotionFileError("NPZ contains duplicate member names")
            total_size = 0
            logical_names: set[str] = set()
            for member in members:
                name = member.filename
                if member.flag_bits & 0x1:
                    raise UnsafeMotionFileError("encrypted NPZ members are not supported")
                if member.is_dir() or "/" in name or "\\" in name or not name.endswith(".npy"):
                    raise UnsafeMotionFileError(f"unexpected NPZ member path: {name!r}")
                logical_name = name[:-4]
                if logical_name in logical_names:
                    raise UnsafeMotionFileError(
                        f"NPZ contains duplicate array name {logical_name!r}"
                    )
                logical_names.add(logical_name)
                total_size += member.file_size
                if total_size > _MAX_UNCOMPRESSED_BYTES:
                    raise UnsafeMotionFileError("NPZ uncompressed size exceeds safety limit")
                if (
                    member.file_size > 1024 * 1024
                    and member.file_size
                    > max(member.compress_size, 1) * _MAX_COMPRESSION_RATIO
                ):
                    raise UnsafeMotionFileError("NPZ compression ratio exceeds safety limit")
    except zipfile.BadZipFile as error:
        raise UnsafeMotionFileError("file is not a valid NPZ/ZIP container") from error


def _read_arrays(path: Path) -> dict[str, np.ndarray]:
    _inspect_npz(path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as error:
        raise UnsafeMotionFileError(f"cannot safely load NPZ: {error}") from error
    for key, value in arrays.items():
        if value.dtype.hasobject:
            raise UnsafeMotionFileError(f"array {key!r} uses object dtype")
        if not value.flags.c_contiguous:
            raise MotionValidationError("must be C contiguous", field=key)
    return arrays


def _scalar_text(arrays: Mapping[str, np.ndarray], key: str) -> str:
    if key not in arrays:
        raise MotionValidationError("missing required array", field=key)
    value = arrays[key]
    if value.ndim != 0 or value.dtype.kind != "U":
        raise MotionValidationError("must be a Unicode scalar", field=key)
    return str(value.item())


def _require_exact_keys(arrays: Mapping[str, np.ndarray], expected: set[str]) -> None:
    actual = set(arrays)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        raise MotionValidationError(f"missing arrays: {', '.join(missing)}")
    if extra:
        raise MotionValidationError(f"unexpected arrays: {', '.join(extra)}")


def _require_dtype(array: np.ndarray, dtype: str | np.dtype[Any], field: str) -> None:
    expected = np.dtype(dtype)
    if array.dtype != expected:
        raise MotionValidationError(
            f"expected dtype {expected.str}, got {array.dtype.str}", field=field
        )


def _require_unicode_names(array: np.ndarray, field: str) -> None:
    if array.ndim != 1 or array.dtype.kind != "U":
        raise MotionValidationError("must be a Unicode 1D array", field=field)


def _validate_disk_dtypes(arrays: Mapping[str, np.ndarray], schema_id: str) -> None:
    _require_dtype(arrays["timestamps"], "<f8", "timestamps")
    _require_dtype(arrays["metadata_json"], np.uint8, "metadata_json")
    if schema_id == AnimalMotion.schema_id:
        _require_unicode_names(arrays["keypoint_names"], "keypoint_names")
        for key in ("positions", "confidence", "contact_probability"):
            _require_dtype(arrays[key], "<f4", key)
        for key in ("valid_mask", "frame_valid"):
            _require_dtype(arrays[key], np.bool_, key)
    else:
        _require_unicode_names(arrays["dof_names"], "dof_names")
        for key in (
            "root_position",
            "root_rotation",
            "dof_position",
            "root_linear_velocity",
            "root_angular_velocity",
            "dof_velocity",
            "foot_contact_probability",
            "solver_residual",
        ):
            _require_dtype(arrays[key], "<f4", key)
        _require_dtype(arrays["frame_valid"], np.bool_, "frame_valid")
        _require_dtype(arrays["solver_status"], "<i2", "solver_status")


def load_motion(
    path: str | os.PathLike[str], *, expected_model_sha256: str | None = None
) -> Motion:
    """Safely load and fully validate an AnimalMotion or RobotMotion NPZ."""

    motion_path = Path(path)
    arrays = _read_arrays(motion_path)
    schema_id = _scalar_text(arrays, "schema_id")
    schema_version = _scalar_text(arrays, "schema_version")
    if schema_version != "1.0":
        raise MotionValidationError(
            f"unsupported schema version {schema_version!r}", field="schema_version"
        )
    if schema_id == AnimalMotion.schema_id:
        _require_exact_keys(arrays, _ANIMAL_KEYS)
    elif schema_id == RobotMotion.schema_id:
        _require_exact_keys(arrays, _ROBOT_KEYS)
    else:
        raise MotionValidationError(
            f"unsupported schema ID {schema_id!r}", field="schema_id"
        )
    _validate_disk_dtypes(arrays, schema_id)
    metadata = _parse_metadata(arrays["metadata_json"])
    if schema_id == AnimalMotion.schema_id:
        return AnimalMotion(
            timestamps=arrays["timestamps"],
            keypoint_names=arrays["keypoint_names"],
            positions=arrays["positions"],
            confidence=arrays["confidence"],
            valid_mask=arrays["valid_mask"],
            contact_probability=arrays["contact_probability"],
            frame_valid=arrays["frame_valid"],
            metadata=metadata,
        )
    if expected_model_sha256 is not None and metadata.get("model_sha256") != expected_model_sha256:
        raise MotionValidationError(
            "does not match the requested robot model", field="metadata_json.model_sha256"
        )
    return RobotMotion(
        timestamps=arrays["timestamps"],
        dof_names=arrays["dof_names"],
        root_position=arrays["root_position"],
        root_rotation=arrays["root_rotation"],
        dof_position=arrays["dof_position"],
        root_linear_velocity=arrays["root_linear_velocity"],
        root_angular_velocity=arrays["root_angular_velocity"],
        dof_velocity=arrays["dof_velocity"],
        foot_contact_probability=arrays["foot_contact_probability"],
        frame_valid=arrays["frame_valid"],
        solver_status=arrays["solver_status"],
        solver_residual=arrays["solver_residual"],
        metadata=metadata,
    )


def _continuous_root_rotation(rotation: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(rotation, dtype="<f4").copy()
    previous: np.ndarray | None = None
    for index in range(len(result)):
        current = result[index]
        if not np.all(np.isfinite(current)):
            previous = None
            continue
        if previous is not None and np.dot(previous, current) < 0.0:
            current *= -1.0
        previous = current
    return result


def _motion_arrays(motion: Motion) -> dict[str, np.ndarray]:
    common = {
        "schema_id": np.asarray(motion.schema_id),
        "schema_version": np.asarray(motion.schema_version),
        "timestamps": np.ascontiguousarray(motion.timestamps, dtype="<f8"),
        "metadata_json": _metadata_array(motion.metadata),
    }
    if isinstance(motion, AnimalMotion):
        return {
            **common,
            "keypoint_names": np.asarray(motion.keypoint_names, dtype=np.str_),
            "positions": np.ascontiguousarray(motion.positions, dtype="<f4"),
            "confidence": np.ascontiguousarray(motion.confidence, dtype="<f4"),
            "valid_mask": np.ascontiguousarray(motion.valid_mask, dtype=np.bool_),
            "contact_probability": np.ascontiguousarray(
                motion.contact_probability, dtype="<f4"
            ),
            "frame_valid": np.ascontiguousarray(motion.frame_valid, dtype=np.bool_),
        }
    return {
        **common,
        "dof_names": np.asarray(motion.dof_names, dtype=np.str_),
        "root_position": np.ascontiguousarray(motion.root_position, dtype="<f4"),
        "root_rotation": _continuous_root_rotation(motion.root_rotation),
        "dof_position": np.ascontiguousarray(motion.dof_position, dtype="<f4"),
        "root_linear_velocity": np.ascontiguousarray(
            motion.root_linear_velocity, dtype="<f4"
        ),
        "root_angular_velocity": np.ascontiguousarray(
            motion.root_angular_velocity, dtype="<f4"
        ),
        "dof_velocity": np.ascontiguousarray(motion.dof_velocity, dtype="<f4"),
        "foot_contact_probability": np.ascontiguousarray(
            motion.foot_contact_probability, dtype="<f4"
        ),
        "frame_valid": np.ascontiguousarray(motion.frame_valid, dtype=np.bool_),
        "solver_status": np.ascontiguousarray(motion.solver_status, dtype="<i2"),
        "solver_residual": np.ascontiguousarray(motion.solver_residual, dtype="<f4"),
    }


def save_motion(path: str | os.PathLike[str], motion: Motion) -> None:
    """Atomically save canonical motion; a failed write never replaces the target."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays = _motion_arrays(motion)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(file_descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise

