"""Root, scale, and contact preprocessing for semantic animal motion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from gqmr.core.coordinates import canonicalize_quaternion_sequence, matrix_to_quaternion
from gqmr.core.motion import AnimalMotion, SolverStatus
from gqmr.skeletons import AnimalSkeleton, get_skeleton

_EPSILON = 1e-6


@dataclass(frozen=True, slots=True)
class RootEstimate:
    position: NDArray[np.float32]
    rotation: NDArray[np.float32]
    status: NDArray[np.int16]
    valid: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class BodyScaleEstimate:
    torso_length: float
    shoulder_width: float
    hip_width: float
    leg_lengths: dict[str, float]


def _indices(motion: AnimalMotion, skeleton: AnimalSkeleton) -> dict[str, int]:
    motion_indices = {name: index for index, name in enumerate(motion.keypoint_names)}
    required = set(skeleton.names)
    missing = sorted(required - set(motion_indices))
    if missing:
        raise ValueError(f"AnimalMotion is missing skeleton keypoints: {missing}")
    return motion_indices


def _normalize(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < _EPSILON:
        return None
    return vector / norm


def estimate_root_motion(
    motion: AnimalMotion, skeleton: AnimalSkeleton | None = None
) -> RootEstimate:
    """Estimate world root pose with previous-reliable-orientation degradation."""

    skeleton = skeleton or get_skeleton(motion.metadata["skeleton_id"])
    indices = _indices(motion, skeleton)
    landmarks = {
        role: indices[name] for role, name in skeleton.root_landmarks.items()
    }
    frames = motion.frame_count
    positions = np.empty((frames, 3), dtype=np.float64)
    rotations = np.empty((frames, 4), dtype=np.float64)
    statuses = np.empty(frames, dtype=np.int16)
    valid = np.empty(frames, dtype=np.bool_)
    previous_rotation: np.ndarray | None = None
    previous_position = np.zeros(3, dtype=np.float64)

    for frame in range(frames):
        pelvis_id = landmarks["pelvis"]
        neck_id = landmarks["neck"]
        root_point_ids = tuple(landmarks.values())
        usable = all(motion.valid_mask[frame, point_id] for point_id in root_point_ids)
        pelvis_neck_usable = (
            motion.valid_mask[frame, pelvis_id] and motion.valid_mask[frame, neck_id]
        )
        if pelvis_neck_usable:
            root_position = 0.5 * (
                motion.positions[frame, pelvis_id] + motion.positions[frame, neck_id]
            )
            previous_position = np.asarray(root_position, dtype=np.float64)
        else:
            root_position = previous_position
        positions[frame] = root_position
        if not usable:
            rotations[frame] = (
                previous_rotation
                if previous_rotation is not None
                else np.array([1.0, 0.0, 0.0, 0.0])
            )
            statuses[frame] = SolverStatus.MISSING_INPUT
            valid[frame] = False
            continue

        points = motion.positions[frame]
        forward = _normalize(points[neck_id] - points[pelvis_id])
        shoulder_left = _normalize(
            points[landmarks["left_shoulder"]]
            - points[landmarks["right_shoulder"]]
        )
        hip_left = _normalize(
            points[landmarks["left_hip"]] - points[landmarks["right_hip"]]
        )
        left = (
            _normalize(shoulder_left + hip_left)
            if shoulder_left is not None and hip_left is not None
            else None
        )
        up = _normalize(np.cross(forward, left)) if forward is not None and left is not None else None
        if up is None or forward is None:
            rotations[frame] = (
                previous_rotation
                if previous_rotation is not None
                else np.array([1.0, 0.0, 0.0, 0.0])
            )
            statuses[frame] = SolverStatus.DEGRADED_ROOT
            valid[frame] = previous_rotation is not None
            continue
        if np.dot(up, np.array([0.0, 0.0, 1.0])) < 0.0:
            up = -up
        left = _normalize(np.cross(up, forward))
        if left is None:
            rotations[frame] = previous_rotation if previous_rotation is not None else [1, 0, 0, 0]
            statuses[frame] = SolverStatus.DEGRADED_ROOT
            valid[frame] = previous_rotation is not None
            continue
        forward = _normalize(np.cross(left, up))
        matrix = np.column_stack((forward, left, up))
        quaternion = matrix_to_quaternion(matrix)
        rotations[frame] = quaternion
        previous_rotation = quaternion
        statuses[frame] = SolverStatus.OK
        valid[frame] = True

    rotations = canonicalize_quaternion_sequence(rotations)
    return RootEstimate(
        position=np.ascontiguousarray(positions, dtype=np.float32),
        rotation=np.ascontiguousarray(rotations, dtype=np.float32),
        status=statuses,
        valid=valid,
    )


def estimate_body_scale(
    motion: AnimalMotion, skeleton: AnimalSkeleton | None = None
) -> BodyScaleEstimate:
    """Estimate robust median torso widths and semantic limb-chain lengths."""

    skeleton = skeleton or get_skeleton(motion.metadata["skeleton_id"])
    indices = _indices(motion, skeleton)

    def median_distance(a: str, b: str) -> float:
        left, right = indices[a], indices[b]
        usable = motion.valid_mask[:, left] & motion.valid_mask[:, right]
        if not np.any(usable):
            raise ValueError(f"no valid samples for scale segment {a}-{b}")
        distances = np.linalg.norm(
            motion.positions[usable, left] - motion.positions[usable, right], axis=1
        )
        distances = distances[np.isfinite(distances) & (distances >= _EPSILON)]
        if not len(distances):
            raise ValueError(f"degenerate scale segment {a}-{b}")
        return float(np.median(distances))

    landmarks = skeleton.root_landmarks
    leg_lengths: dict[str, float] = {}
    for leg, chain in skeleton.limb_chains.items():
        leg_lengths[leg] = sum(
            median_distance(parent, child) for parent, child in zip(chain[:-1], chain[1:])
        )
    return BodyScaleEstimate(
        torso_length=median_distance(landmarks["pelvis"], landmarks["neck"]),
        shoulder_width=median_distance(
            landmarks["left_shoulder"], landmarks["right_shoulder"]
        ),
        hip_width=median_distance(landmarks["left_hip"], landmarks["right_hip"]),
        leg_lengths=leg_lengths,
    )


def estimate_contact_probability(
    motion: AnimalMotion,
    skeleton: AnimalSkeleton | None = None,
    *,
    height_threshold: float = 0.035,
    speed_threshold: float = 0.35,
) -> NDArray[np.float32]:
    """Estimate contact probability from toe height above ground and world speed."""

    skeleton = skeleton or get_skeleton(motion.metadata["skeleton_id"])
    indices = _indices(motion, skeleton)
    toe_indices = [indices[skeleton.limb_chains[leg][-1]] for leg in _LEGS]
    toe_positions = motion.positions[:, toe_indices].astype(np.float64)
    toe_valid = motion.valid_mask[:, toe_indices]
    if motion.frame_count < 3:
        return np.full((motion.frame_count, 4), np.nan, dtype=np.float32)
    finite_heights = toe_positions[..., 2][toe_valid]
    if not len(finite_heights):
        return np.full((motion.frame_count, 4), np.nan, dtype=np.float32)
    ground_height = float(np.percentile(finite_heights, 5.0))
    velocity = np.gradient(toe_positions, motion.timestamps, axis=0, edge_order=2)
    speed = np.linalg.norm(velocity, axis=-1)
    height = toe_positions[..., 2] - ground_height
    height_score = 1.0 / (1.0 + np.exp((height - height_threshold) / 0.01))
    speed_score = 1.0 / (1.0 + np.exp((speed - speed_threshold) / 0.08))
    probability = height_score * speed_score
    probability[~toe_valid] = np.nan
    return np.ascontiguousarray(np.clip(probability, 0.0, 1.0), dtype=np.float32)


_LEGS = ("FL", "FR", "RL", "RR")
