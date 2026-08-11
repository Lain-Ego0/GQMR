"""Generic, DeepLabCut, and SLEAP keypoint-result readers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from gqmr.pose.api import KeypointBatch, PoseDataError
from gqmr.exporters.common import atomic_write
from gqmr.core.json import StrictJSONError, loads_strict_json


def _timestamps(frames: int, fps: float) -> np.ndarray:
    if not np.isfinite(fps) or fps <= 0.0:
        raise PoseDataError("fps must be finite and positive")
    return np.arange(frames, dtype=np.float64) / fps


def load_generic_keypoints_json(path: str | Path) -> KeypointBatch:
    try:
        document = loads_strict_json(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, StrictJSONError) as error:
        raise PoseDataError(f"cannot read keypoint JSON: {error}") from error
    if not isinstance(document, dict):
        raise PoseDataError("keypoint JSON must be an object")
    required = {
        "timestamps",
        "keypoint_names",
        "instance_ids",
        "positions",
        "confidence",
        "valid_mask",
        "coordinate_frame",
    }
    if not required.issubset(document):
        raise PoseDataError(f"keypoint JSON missing fields: {sorted(required - set(document))}")
    return KeypointBatch(
        timestamps=document["timestamps"],
        keypoint_names=tuple(document["keypoint_names"]),
        instance_ids=tuple(document["instance_ids"]),
        positions=document["positions"],
        confidence=document["confidence"],
        valid_mask=document["valid_mask"],
        coordinate_frame=document["coordinate_frame"],
        metadata=dict(document.get("metadata", {})),
    )


def load_generic_keypoints_csv(path: str | Path) -> KeypointBatch:
    """Load long-form timestamp,instance,keypoint,x,y[,z],confidence CSV."""

    try:
        with Path(path).open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"timestamp", "instance", "keypoint", "x", "y", "confidence"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise PoseDataError(
                    "generic CSV must contain timestamp,instance,keypoint,x,y,confidence"
                )
            dimensions = 3 if "z" in reader.fieldnames else 2
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise PoseDataError(f"cannot read generic keypoint CSV: {error}") from error
    if not rows:
        raise PoseDataError("generic keypoint CSV has no observations")
    try:
        times = sorted({float(row["timestamp"]) for row in rows})
    except ValueError as error:
        raise PoseDataError("generic keypoint CSV timestamp must be numeric") from error
    instances = list(dict.fromkeys(row["instance"] for row in rows))
    names = list(dict.fromkeys(row["keypoint"] for row in rows))
    if any(not value for value in (*instances, *names)):
        raise PoseDataError("generic keypoint CSV instance/keypoint names are required")
    time_index = {value: index for index, value in enumerate(times)}
    instance_index = {value: index for index, value in enumerate(instances)}
    name_index = {value: index for index, value in enumerate(names)}
    shape = (len(times), len(instances), len(names))
    positions = np.full((*shape, dimensions), np.nan, dtype=np.float32)
    confidence = np.zeros(shape, dtype=np.float32)
    seen: set[tuple[int, int, int]] = set()
    for row in rows:
        index = (
            time_index[float(row["timestamp"])],
            instance_index[row["instance"]],
            name_index[row["keypoint"]],
        )
        if index in seen:
            raise PoseDataError("generic keypoint CSV has a duplicate observation")
        seen.add(index)
        try:
            coordinate = [float(row["x"]), float(row["y"])]
            if dimensions == 3:
                coordinate.append(float(row["z"]))
            positions[index] = coordinate
            confidence[index] = float(row["confidence"])
        except ValueError as error:
            raise PoseDataError("generic keypoint CSV contains a non-numeric value") from error
    valid = np.all(np.isfinite(positions), axis=-1)
    coordinate_frame = (
        "gqmr_world_x_forward_y_left_z_up"
        if dimensions == 3
        else "image_pixels_x_right_y_down"
    )
    return KeypointBatch(
        timestamps=times,
        keypoint_names=tuple(names),
        instance_ids=tuple(instances),
        positions=positions,
        confidence=confidence,
        valid_mask=valid,
        coordinate_frame=coordinate_frame,
        metadata={"format": "generic_long_csv", "path": str(path)},
    )


def load_generic_keypoints_npz(path: str | Path) -> KeypointBatch:
    required = {
        "timestamps",
        "keypoint_names",
        "instance_ids",
        "positions",
        "confidence",
        "valid_mask",
        "coordinate_frame",
    }
    try:
        with np.load(path, allow_pickle=False) as archive:
            fields = set(archive.files)
            if not required.issubset(fields) or fields - required - {"metadata_json"}:
                raise PoseDataError("generic keypoint NPZ field set is invalid")
            arrays = {key: archive[key] for key in archive.files}
    except (OSError, ValueError) as error:
        if isinstance(error, PoseDataError):
            raise
        raise PoseDataError(f"cannot load generic keypoint NPZ: {error}") from error
    coordinate = arrays["coordinate_frame"]
    if coordinate.shape != () or coordinate.dtype.kind != "U":
        raise PoseDataError("coordinate_frame must be a Unicode scalar")
    metadata: dict[str, Any] = {"format": "generic_npz"}
    if "metadata_json" in arrays:
        encoded = arrays["metadata_json"]
        if encoded.shape != () or encoded.dtype.kind != "U":
            raise PoseDataError("metadata_json must be a Unicode scalar")
        try:
            decoded = loads_strict_json(str(encoded.item()))
        except (json.JSONDecodeError, StrictJSONError) as error:
            raise PoseDataError(f"invalid metadata_json: {error}") from error
        if not isinstance(decoded, dict):
            raise PoseDataError("metadata_json must contain a JSON object")
        metadata = decoded
    return KeypointBatch(
        timestamps=arrays["timestamps"],
        keypoint_names=tuple(str(value) for value in arrays["keypoint_names"].tolist()),
        instance_ids=tuple(str(value) for value in arrays["instance_ids"].tolist()),
        positions=arrays["positions"],
        confidence=arrays["confidence"],
        valid_mask=arrays["valid_mask"],
        coordinate_frame=str(coordinate.item()),
        metadata=metadata,
    )


def save_generic_keypoints_npz(path: str | Path, batch: KeypointBatch) -> Path:
    try:
        metadata_json = json.dumps(
            batch.metadata,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise PoseDataError(f"keypoint metadata is not strict JSON: {error}") from error
    arrays = {
        "timestamps": np.ascontiguousarray(batch.timestamps, dtype="<f8"),
        "keypoint_names": np.asarray(batch.keypoint_names, dtype=np.str_),
        "instance_ids": np.asarray(batch.instance_ids, dtype=np.str_),
        "positions": np.ascontiguousarray(batch.positions, dtype="<f4"),
        "confidence": np.ascontiguousarray(batch.confidence, dtype="<f4"),
        "valid_mask": np.ascontiguousarray(batch.valid_mask, dtype=np.bool_),
        "coordinate_frame": np.asarray(batch.coordinate_frame),
        "metadata_json": np.asarray(metadata_json),
    }
    return atomic_write(path, lambda stream: np.savez_compressed(stream, **arrays))


def load_deeplabcut_csv(path: str | Path, *, fps: float) -> KeypointBatch:
    try:
        with Path(path).open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream))
    except (OSError, UnicodeError, csv.Error) as error:
        raise PoseDataError(f"cannot read DeepLabCut CSV: {error}") from error
    if len(rows) < 4 or len(rows[0]) < 4:
        raise PoseDataError("DeepLabCut CSV is too short")
    bodyparts = rows[1][1:]
    coordinates = rows[2][1:]
    names: list[str] = []
    columns: dict[str, dict[str, int]] = {}
    for column, (name, coordinate) in enumerate(zip(bodyparts, coordinates), start=1):
        if name not in columns:
            names.append(name)
            columns[name] = {}
        columns[name][coordinate.lower()] = column
    if any(not {"x", "y"}.issubset(columns[name]) for name in names):
        raise PoseDataError("DeepLabCut CSV lacks x/y columns")
    frames = len(rows) - 3
    positions = np.full((frames, 1, len(names), 2), np.nan, dtype=np.float32)
    confidence = np.ones((frames, 1, len(names)), dtype=np.float32)
    for frame, row in enumerate(rows[3:]):
        for index, name in enumerate(names):
            try:
                positions[frame, 0, index] = [
                    float(row[columns[name]["x"]]),
                    float(row[columns[name]["y"]]),
                ]
                if "likelihood" in columns[name]:
                    confidence[frame, 0, index] = float(row[columns[name]["likelihood"]])
            except (IndexError, ValueError) as error:
                raise PoseDataError(f"invalid DeepLabCut value at data row {frame + 1}") from error
    valid = np.all(np.isfinite(positions), axis=-1) & np.isfinite(confidence)
    return KeypointBatch(
        timestamps=_timestamps(frames, fps),
        keypoint_names=tuple(names),
        instance_ids=("animal-0",),
        positions=positions,
        confidence=confidence,
        valid_mask=valid,
        coordinate_frame="image_pixels_x_right_y_down",
        metadata={"format": "deeplabcut_csv", "path": str(path)},
    )


def load_sleap_csv(path: str | Path, *, fps: float) -> KeypointBatch:
    try:
        with Path(path).open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"frame_idx", "track", "node", "x", "y", "score"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise PoseDataError("SLEAP CSV must contain frame_idx,track,node,x,y,score")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise PoseDataError(f"cannot read SLEAP CSV: {error}") from error
    if not rows:
        raise PoseDataError("SLEAP CSV has no observations")
    try:
        frame_ids = sorted({int(row["frame_idx"]) for row in rows})
        tracks = sorted({row["track"] or "animal-0" for row in rows})
        nodes = list(dict.fromkeys(row["node"] for row in rows))
    except ValueError as error:
        raise PoseDataError("SLEAP frame_idx must be an integer") from error
    if frame_ids != list(range(frame_ids[0], frame_ids[-1] + 1)):
        raise PoseDataError("SLEAP CSV frame indices must be contiguous")
    frame_lookup = {value: index for index, value in enumerate(frame_ids)}
    track_lookup = {value: index for index, value in enumerate(tracks)}
    node_lookup = {value: index for index, value in enumerate(nodes)}
    positions = np.full((len(frame_ids), len(tracks), len(nodes), 2), np.nan, dtype=np.float32)
    confidence = np.zeros((len(frame_ids), len(tracks), len(nodes)), dtype=np.float32)
    for row in rows:
        frame = frame_lookup[int(row["frame_idx"])]
        track = track_lookup[row["track"] or "animal-0"]
        node = node_lookup[row["node"]]
        try:
            positions[frame, track, node] = [float(row["x"]), float(row["y"])]
            confidence[frame, track, node] = float(row["score"])
        except ValueError as error:
            raise PoseDataError("SLEAP CSV contains a non-numeric coordinate/score") from error
    valid = np.all(np.isfinite(positions), axis=-1)
    return KeypointBatch(
        timestamps=_timestamps(len(frame_ids), fps),
        keypoint_names=tuple(nodes),
        instance_ids=tuple(tracks),
        positions=positions,
        confidence=confidence,
        valid_mask=valid,
        coordinate_frame="image_pixels_x_right_y_down",
        metadata={"format": "sleap_csv", "first_frame_idx": frame_ids[0], "path": str(path)},
    )
