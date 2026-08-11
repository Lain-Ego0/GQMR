"""Convert calibrated 3D plugin output to canonical AnimalMotion."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from gqmr import __version__
from gqmr.core.motion import AnimalMotion, COORDINATE_FRAME
from gqmr.pose.api import KeypointBatch, PoseDataError
from gqmr.skeletons import AnimalSkeleton, get_skeleton


def keypoint_batch_to_animal_motion(
    batch: KeypointBatch,
    *,
    skeleton: AnimalSkeleton | None = None,
    instance_id: str | None = None,
) -> AnimalMotion:
    if batch.dimensions != 3:
        raise PoseDataError("AnimalMotion conversion requires calibrated 3D keypoints")
    if batch.coordinate_frame != COORDINATE_FRAME:
        raise PoseDataError(
            f"3D keypoints must be transformed to {COORDINATE_FRAME} before conversion"
        )
    skeleton = skeleton or get_skeleton("dog-27")
    selected_instance = instance_id or batch.instance_ids[0]
    try:
        instance_index = batch.instance_ids.index(selected_instance)
    except ValueError as error:
        raise PoseDataError(f"unknown instance ID {selected_instance!r}") from error
    source_index = {name: index for index, name in enumerate(batch.keypoint_names)}
    missing = sorted(set(skeleton.names) - set(source_index))
    if missing:
        raise PoseDataError(f"keypoint batch is missing skeleton points: {missing}")
    order = [source_index[name] for name in skeleton.names]
    positions = batch.positions[:, instance_index, order]
    confidence = batch.confidence[:, instance_index, order]
    valid = batch.valid_mask[:, instance_index, order]
    timestamps = batch.timestamps - batch.timestamps[0]
    source_payload = json.dumps(
        batch.metadata, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return AnimalMotion(
        timestamps=timestamps,
        keypoint_names=skeleton.names,
        positions=positions,
        confidence=confidence,
        valid_mask=valid,
        contact_probability=np.full((len(timestamps), 4), np.nan, dtype=np.float32),
        frame_valid=np.all(valid, axis=1),
        metadata={
            "coordinate_frame": COORDINATE_FRAME,
            "length_unit": "m",
            "time_unit": "s",
            "skeleton_id": skeleton.id,
            "skeleton_sha256": skeleton.sha256,
            "contact_order": ["FL", "FR", "RL", "RR"],
            "contact_source": "unknown",
            "source": {
                "format": "pose_backend_v1",
                "instance_id": selected_instance,
                "batch_metadata_sha256": hashlib.sha256(source_payload).hexdigest(),
                "batch_metadata": batch.metadata,
            },
            "created_by": {"gqmr_version": __version__},
        },
    )
