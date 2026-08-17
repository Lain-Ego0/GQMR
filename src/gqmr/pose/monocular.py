"""Experimental side-view lifting from AP-10K-style 2D dog poses."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

import numpy as np

from gqmr import __version__
from gqmr.core.motion import AnimalMotion, COORDINATE_FRAME
from gqmr.pose.api import KeypointBatch, PoseDataError
from gqmr.skeletons import get_skeleton

Facing = Literal["auto", "left", "right"]

_SOURCE_POINTS = {
    "left_eye": "L_Eye",
    "right_eye": "R_Eye",
    "nose": "Nose",
    "neck": "Neck",
    "tail_root": "Root of tail",
    "left_shoulder": "L_Shoulder",
    "left_elbow": "L_Elbow",
    "left_front_paw": "L_F_Paw",
    "right_shoulder": "R_Shoulder",
    "right_elbow": "R_Elbow",
    "right_front_paw": "R_F_Paw",
    "left_hip": "L_Hip",
    "left_knee": "L_Knee",
    "left_hind_paw": "L_B_Paw",
    "right_hip": "R_Hip",
    "right_knee": "R_Knee",
    "right_hind_paw": "R_B_Paw",
}


def _normalized_name(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum())


def _source_indices(names: tuple[str, ...]) -> dict[str, int]:
    available = {_normalized_name(name): index for index, name in enumerate(names)}
    resolved: dict[str, int] = {}
    missing: list[str] = []
    for semantic_name, expected_name in _SOURCE_POINTS.items():
        index = available.get(_normalized_name(expected_name))
        if index is None:
            missing.append(expected_name)
        else:
            resolved[semantic_name] = index
    if missing:
        raise PoseDataError(
            "experimental monocular lift requires AP-10K-style points; "
            f"missing: {sorted(missing)}"
        )
    return resolved


def _interpolate_tracks(
    timestamps: np.ndarray,
    positions: np.ndarray,
    usable: np.ndarray,
) -> np.ndarray:
    result = positions.astype(np.float64, copy=True)
    for point in range(positions.shape[1]):
        valid_indices = np.flatnonzero(usable[:, point])
        if not len(valid_indices):
            raise PoseDataError(
                f"keypoint {point} has no observations above the confidence threshold"
            )
        for axis in range(2):
            result[:, point, axis] = np.interp(
                timestamps,
                timestamps[valid_indices],
                positions[valid_indices, point, axis],
            )
    return result


def _smooth_tracks(positions: np.ndarray, window: int) -> np.ndarray:
    if window == 1:
        return positions
    radius = window // 2
    kernel = np.full(window, 1.0 / window, dtype=np.float64)
    result = np.empty_like(positions)
    for point in range(positions.shape[1]):
        for axis in range(2):
            padded = np.pad(positions[:, point, axis], radius, mode="edge")
            result[:, point, axis] = np.convolve(padded, kernel, mode="valid")
    return result


def _combine_confidence(
    confidence: np.ndarray,
    indices: tuple[int, ...],
    *,
    penalty: float = 1.0,
) -> np.ndarray:
    return np.min(confidence[:, indices], axis=1) * penalty


def _combine_valid(valid: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    return np.all(valid[:, indices], axis=1)


def lift_ap10k_monocular_to_dog27(
    batch: KeypointBatch,
    *,
    instance_id: str | None = None,
    facing: Facing = "auto",
    torso_length: float = 0.48,
    body_width: float = 0.28,
    smoothing_window: int = 5,
    confidence_threshold: float = 0.15,
) -> AnimalMotion:
    """Create an approximate dog-27 motion from a mostly side-view 2D clip.

    This is deliberately labeled experimental: depth is a fixed anatomical prior,
    not an estimate recovered from the source video.
    """

    if batch.dimensions != 2:
        raise PoseDataError("experimental monocular lift requires 2D keypoints")
    if batch.coordinate_frame != "image_pixels_x_right_y_down":
        raise PoseDataError(
            "experimental monocular lift requires image pixel coordinates"
        )
    if facing not in {"auto", "left", "right"}:
        raise PoseDataError("facing must be auto, left, or right")
    if not np.isfinite(torso_length) or torso_length <= 0.0:
        raise PoseDataError("torso length must be finite and positive")
    if not np.isfinite(body_width) or body_width <= 0.0:
        raise PoseDataError("body width must be finite and positive")
    if smoothing_window <= 0 or smoothing_window % 2 == 0:
        raise PoseDataError("smoothing window must be a positive odd integer")
    if (
        not np.isfinite(confidence_threshold)
        or not 0.0 <= confidence_threshold <= 1.0
    ):
        raise PoseDataError("confidence threshold must be finite in [0,1]")

    selected_instance = instance_id or batch.instance_ids[0]
    try:
        instance_index = batch.instance_ids.index(selected_instance)
    except ValueError as error:
        raise PoseDataError(f"unknown instance ID {selected_instance!r}") from error

    source = _source_indices(batch.keypoint_names)
    source_order = tuple(source.values())
    positions_2d = batch.positions[:, instance_index, source_order].astype(np.float64)
    confidence_2d = batch.confidence[:, instance_index, source_order].astype(np.float64)
    observed_valid = batch.valid_mask[:, instance_index, source_order].copy()
    observed_valid &= confidence_2d >= confidence_threshold
    observed_valid &= np.all(np.isfinite(positions_2d), axis=-1)
    positions_2d = _interpolate_tracks(
        batch.timestamps, positions_2d, observed_valid
    )
    positions_2d = _smooth_tracks(positions_2d, smoothing_window)

    compact_index = {
        semantic_name: compact for compact, semantic_name in enumerate(source)
    }

    def point(name: str) -> np.ndarray:
        return positions_2d[:, compact_index[name]]

    neck_tail_distance = np.linalg.norm(point("neck") - point("tail_root"), axis=1)
    neck_tail_distance = neck_tail_distance[
        np.isfinite(neck_tail_distance) & (neck_tail_distance > 1.0)
    ]
    if not len(neck_tail_distance):
        raise PoseDataError("cannot estimate image scale from neck and tail root")
    pixels_per_meter = float(np.median(neck_tail_distance) / torso_length)

    nose_tail_dx = float(
        np.median(point("nose")[:, 0] - point("tail_root")[:, 0])
    )
    if facing == "auto":
        if abs(nose_tail_dx) < 1.0:
            raise PoseDataError("cannot infer facing direction from nose and tail root")
        resolved_facing: Literal["left", "right"] = (
            "right" if nose_tail_dx > 0.0 else "left"
        )
    else:
        resolved_facing = facing
    horizontal_sign = 1.0 if resolved_facing == "right" else -1.0

    hip_center_2d = 0.5 * (point("left_hip") + point("right_hip"))
    reference_x = float(hip_center_2d[0, 0])
    paw_names = (
        "left_front_paw",
        "right_front_paw",
        "left_hind_paw",
        "right_hind_paw",
    )
    ground_y = float(
        np.percentile(
            np.concatenate([point(name)[:, 1] for name in paw_names]), 98.0
        )
    )

    depth = {
        "left_eye": 0.18,
        "right_eye": -0.18,
        "left_shoulder": 0.50,
        "right_shoulder": -0.50,
        "left_elbow": 0.61,
        "right_elbow": -0.61,
        "left_front_paw": 0.61,
        "right_front_paw": -0.61,
        "left_hip": 0.43,
        "right_hip": -0.43,
        "left_knee": 0.54,
        "right_knee": -0.54,
        "left_hind_paw": 0.54,
        "right_hind_paw": -0.54,
    }
    world: dict[str, np.ndarray] = {}
    for name in source:
        coordinate = point(name)
        lifted = np.zeros((len(batch.timestamps), 3), dtype=np.float64)
        lifted[:, 0] = (
            horizontal_sign * (coordinate[:, 0] - reference_x) / pixels_per_meter
        )
        lifted[:, 1] = depth.get(name, 0.0) * body_width
        lifted[:, 2] = (ground_y - coordinate[:, 1]) / pixels_per_meter
        world[name] = lifted

    skeleton = get_skeleton("dog-27")
    name_to_target = skeleton.name_to_index
    frames = len(batch.timestamps)
    output = np.empty((frames, len(skeleton.names), 3), dtype=np.float64)
    output_confidence = np.empty((frames, len(skeleton.names)), dtype=np.float64)
    output_valid = np.empty((frames, len(skeleton.names)), dtype=bool)

    def source_ids(*names: str) -> tuple[int, ...]:
        return tuple(compact_index[name] for name in names)

    def assign(
        target: str,
        value: np.ndarray,
        sources: tuple[str, ...],
        *,
        penalty: float = 1.0,
    ) -> None:
        indices = source_ids(*sources)
        output[:, name_to_target[target]] = value
        output_confidence[:, name_to_target[target]] = _combine_confidence(
            confidence_2d, indices, penalty=penalty
        )
        output_valid[:, name_to_target[target]] = _combine_valid(
            observed_valid, indices
        )

    pelvis = 0.5 * (world["left_hip"] + world["right_hip"])
    pelvis[:, 1] = 0.0
    neck = world["neck"]
    neck[:, 1] = 0.0
    eye_center = 0.5 * (world["left_eye"] + world["right_eye"])
    eye_center[:, 1] = 0.0
    tail_base = world["tail_root"]
    tail_base[:, 1] = 0.0

    assign("pelvis", pelvis, ("left_hip", "right_hip"))
    assign(
        "pelvis_duplicate",
        pelvis,
        ("left_hip", "right_hip"),
        penalty=0.95,
    )
    assign(
        "spine",
        0.5 * (pelvis + neck),
        ("left_hip", "right_hip", "neck"),
        penalty=0.9,
    )
    assign("neck", neck, ("neck",))
    assign("head", eye_center, ("left_eye", "right_eye"), penalty=0.95)
    assign("muzzle", world["nose"], ("nose",))

    for side in ("left", "right"):
        shoulder = world[f"{side}_shoulder"]
        elbow = world[f"{side}_elbow"]
        paw = world[f"{side}_front_paw"]
        assign(f"{side}_shoulder", shoulder, (f"{side}_shoulder",))
        assign(
            f"{side}_front_upper",
            0.45 * shoulder + 0.55 * elbow,
            (f"{side}_shoulder", f"{side}_elbow"),
            penalty=0.9,
        )
        assign(f"{side}_front_elbow", elbow, (f"{side}_elbow",))
        assign(
            f"{side}_front_wrist",
            0.30 * elbow + 0.70 * paw,
            (f"{side}_elbow", f"{side}_front_paw"),
            penalty=0.9,
        )
        assign(f"{side}_front_toe", paw, (f"{side}_front_paw",))

        hip = world[f"{side}_hip"]
        knee = world[f"{side}_knee"]
        hind_paw = world[f"{side}_hind_paw"]
        assign(f"{side}_hip", hip, (f"{side}_hip",))
        assign(f"{side}_hind_knee", knee, (f"{side}_knee",))
        assign(
            f"{side}_hind_ankle",
            0.28 * knee + 0.72 * hind_paw,
            (f"{side}_knee", f"{side}_hind_paw"),
            penalty=0.9,
        )
        assign(f"{side}_hind_toe", hind_paw, (f"{side}_hind_paw",))

    backward = tail_base - neck
    backward[:, 1] = 0.0
    backward_norm = np.linalg.norm(backward, axis=1, keepdims=True)
    backward /= np.maximum(backward_norm, 1e-8)
    tail_mid = tail_base + 0.16 * backward
    tail_mid[:, 2] -= 0.03
    tail_tip = tail_base + 0.32 * backward
    tail_tip[:, 2] -= 0.07
    assign("tail_base", tail_base, ("tail_root",))
    assign("tail_mid", tail_mid, ("tail_root", "neck"), penalty=0.65)
    assign("tail_tip", tail_tip, ("tail_root", "neck"), penalty=0.55)

    toe_indices = [
        name_to_target[name]
        for name in (
            "left_front_toe",
            "right_front_toe",
            "left_hind_toe",
            "right_hind_toe",
        )
    ]
    ground_offset = float(np.percentile(output[:, toe_indices, 2], 2.0) - 0.025)
    output[:, :, 2] -= ground_offset

    metadata_payload = json.dumps(
        batch.metadata,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    timestamps = batch.timestamps - batch.timestamps[0]
    return AnimalMotion(
        timestamps=timestamps,
        keypoint_names=skeleton.names,
        positions=np.ascontiguousarray(output, dtype=np.float32),
        confidence=np.ascontiguousarray(
            np.clip(output_confidence, 0.0, 1.0), dtype=np.float32
        ),
        valid_mask=np.ascontiguousarray(output_valid),
        contact_probability=np.full((len(timestamps), 4), np.nan, dtype=np.float32),
        frame_valid=np.all(output_valid, axis=1),
        metadata={
            "coordinate_frame": COORDINATE_FRAME,
            "length_unit": "m",
            "time_unit": "s",
            "skeleton_id": skeleton.id,
            "skeleton_sha256": skeleton.sha256,
            "contact_order": ["FL", "FR", "RL", "RR"],
            "contact_source": "unknown",
            "source": {
                "format": "experimental_monocular_ap10k_lift_v1",
                "warning": (
                    "Depth and missing dog-27 joints are anatomical priors, not "
                    "measured 3D observations."
                ),
                "instance_id": selected_instance,
                "input_metadata_sha256": hashlib.sha256(metadata_payload).hexdigest(),
                "input_metadata": batch.metadata,
                "parameters": {
                    "facing": resolved_facing,
                    "torso_length_m": torso_length,
                    "body_width_m": body_width,
                    "smoothing_window": smoothing_window,
                    "confidence_threshold": confidence_threshold,
                    "pixels_per_meter": pixels_per_meter,
                },
            },
            "created_by": {"gqmr_version": __version__},
        },
    )
