"""Calibrated multi-view linear triangulation with reprojection diagnostics."""

from __future__ import annotations

import numpy as np

from gqmr.pose.api import KeypointBatch, PoseDataError


def triangulate_keypoints(
    views: list[KeypointBatch], camera_matrices: np.ndarray
) -> KeypointBatch:
    if len(views) < 2:
        raise PoseDataError("triangulation requires at least two views")
    cameras = np.asarray(camera_matrices, dtype=np.float64)
    if cameras.shape != (len(views), 3, 4) or not np.all(np.isfinite(cameras)):
        raise PoseDataError("camera_matrices must have shape [V,3,4]")
    reference = views[0]
    if reference.dimensions != 2:
        raise PoseDataError("triangulation inputs must be 2D")
    for view in views[1:]:
        if (
            view.dimensions != 2
            or view.keypoint_names != reference.keypoint_names
            or view.instance_ids != reference.instance_ids
            or not np.array_equal(view.timestamps, reference.timestamps)
        ):
            raise PoseDataError("triangulation views have incompatible axes")
    shape = reference.positions.shape[:3]
    positions = np.full((*shape, 3), np.nan, dtype=np.float32)
    confidence = np.zeros(shape, dtype=np.float32)
    reprojection = np.full(shape, np.nan, dtype=np.float32)
    valid = np.zeros(shape, dtype=bool)
    for index in np.ndindex(shape):
        rows: list[np.ndarray] = []
        used: list[int] = []
        weights: list[float] = []
        for view_index, view in enumerate(views):
            if not view.valid_mask[index]:
                continue
            x, y = view.positions[index]
            weight = max(float(view.confidence[index]), 1e-3)
            matrix = cameras[view_index]
            rows.extend((weight * (x * matrix[2] - matrix[0]), weight * (y * matrix[2] - matrix[1])))
            used.append(view_index)
            weights.append(weight)
        if len(used) < 2:
            continue
        _, _, vh = np.linalg.svd(np.stack(rows), full_matrices=False)
        homogeneous = vh[-1]
        if abs(homogeneous[3]) < 1e-12:
            continue
        point = homogeneous[:3] / homogeneous[3]
        errors: list[float] = []
        for view_index in used:
            projected = cameras[view_index] @ np.append(point, 1.0)
            if abs(projected[2]) < 1e-12:
                errors = []
                break
            pixel = projected[:2] / projected[2]
            errors.append(float(np.linalg.norm(pixel - views[view_index].positions[index])))
        if not errors:
            continue
        positions[index] = point
        confidence[index] = min(weights)
        reprojection[index] = float(np.sqrt(np.mean(np.square(errors))))
        valid[index] = True
    return KeypointBatch(
        timestamps=reference.timestamps,
        keypoint_names=reference.keypoint_names,
        instance_ids=reference.instance_ids,
        positions=positions,
        confidence=confidence,
        valid_mask=valid,
        coordinate_frame="calibrated_world",
        metadata={"format": "multiview_dlt", "reprojection_error_pixels": reprojection.tolist()},
    )
