"""Adaptive preprocessing for noisy real-world animal keypoint motion."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
from numpy.typing import NDArray
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from gqmr.core.coordinates import wxyz_to_xyzw, xyzw_to_wxyz
from gqmr.core.motion import AnimalMotion
from gqmr.retarget.animal_preprocess import (
    estimate_contact_probability,
    estimate_root_motion,
)
from gqmr.skeletons import AnimalSkeleton, get_skeleton

_LEGS = ("FL", "FR", "RL", "RR")


@dataclass(frozen=True, slots=True)
class AnimalPreprocessConfig:
    filter_window_seconds: float = 0.15
    toe_filter_window_seconds: float = 0.07
    polynomial_order: int = 3
    root_rotation_window_seconds: float = 0.19
    bone_relative_tolerance: float = 0.12
    bone_outlier_z: float = 6.0
    velocity_outlier_z: float = 7.0
    minimum_neutral_seconds: float = 0.30

    def __post_init__(self) -> None:
        positive = (
            self.filter_window_seconds,
            self.toe_filter_window_seconds,
            self.root_rotation_window_seconds,
            self.bone_relative_tolerance,
            self.bone_outlier_z,
            self.velocity_outlier_z,
            self.minimum_neutral_seconds,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("animal preprocessing values must be finite and positive")
        if self.polynomial_order < 1:
            raise ValueError("polynomial_order must be positive")


@dataclass(frozen=True, slots=True)
class GroundEstimate:
    point: NDArray[np.float32]
    normal: NDArray[np.float32]
    rmse: float
    candidate_count: int


@dataclass(frozen=True, slots=True)
class AnimalPreprocessReport:
    observation_anomaly: NDArray[np.bool_]
    bone_anomaly: NDArray[np.bool_]
    velocity_anomaly: NDArray[np.bool_]
    frame_abnormal: NDArray[np.bool_]
    action_labels: tuple[str, ...]
    ground: GroundEstimate
    neutral_frame_range: tuple[int, int]
    root_rotation_raw: NDArray[np.float32]
    root_rotation_smoothed: NDArray[np.float32]

    @property
    def problem_frames(self) -> NDArray[np.int64]:
        return np.flatnonzero(self.frame_abnormal)

    def frame_message(self, frame: int) -> str:
        translations = {
            "stand": "站立",
            "walk": "行走",
            "run": "奔跑",
            "turn": "转弯",
            "flight": "腾空",
            "abnormal": "异常片段",
        }
        messages = [translations.get(self.action_labels[frame], self.action_labels[frame])]
        if self.bone_anomaly[frame].any():
            messages.append("骨长异常")
        if self.velocity_anomaly[frame].any():
            messages.append("速度突变")
        return "、".join(messages)

    def summary(self) -> dict[str, object]:
        labels, counts = np.unique(self.action_labels, return_counts=True)
        return {
            "problem_frame_count": int(np.count_nonzero(self.frame_abnormal)),
            "bone_anomaly_observations": int(np.count_nonzero(self.bone_anomaly)),
            "velocity_anomaly_observations": int(
                np.count_nonzero(self.velocity_anomaly)
            ),
            "ground": {
                "point": self.ground.point.tolist(),
                "normal": self.ground.normal.tolist(),
                "rmse_m": self.ground.rmse,
                "candidate_count": self.ground.candidate_count,
            },
            "neutral_frame_range": list(self.neutral_frame_range),
            "action_frame_counts": {
                str(label): int(count) for label, count in zip(labels, counts)
            },
        }


def _odd_window(frames: int, seconds: float, median_dt: float, order: int) -> int:
    requested = max(order + 2, int(round(seconds / median_dt)))
    if requested % 2 == 0:
        requested += 1
    maximum = frames if frames % 2 == 1 else frames - 1
    return min(requested, maximum) if maximum > order else 0


def _robust_scale(values: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    median = np.nanmedian(values, axis=axis)
    deviation = np.nanmedian(np.abs(values - np.expand_dims(median, axis)), axis=axis)
    return median, np.maximum(1.4826 * deviation, 1e-9)


def _detect_anomalies(
    motion: AnimalMotion,
    skeleton: AnimalSkeleton,
    config: AnimalPreprocessConfig,
) -> tuple[np.ndarray, np.ndarray]:
    frames, points = motion.valid_mask.shape
    bone = np.zeros((frames, points), dtype=np.bool_)
    index = {name: i for i, name in enumerate(motion.keypoint_names)}
    for definition in skeleton.keypoints:
        if definition.parent is None or definition.role == "source_duplicate":
            continue
        child, parent = index[definition.name], index[definition.parent]
        usable = motion.valid_mask[:, child] & motion.valid_mask[:, parent]
        lengths = np.full(frames, np.nan, dtype=np.float64)
        lengths[usable] = np.linalg.norm(
            motion.positions[usable, child] - motion.positions[usable, parent], axis=1
        )
        median, scale = _robust_scale(lengths)
        if not np.isfinite(median) or median < 1e-8:
            continue
        relative = np.abs(lengths - median) / median
        robust = np.abs(lengths - median) / scale
        bad = usable & (relative > config.bone_relative_tolerance) & (
            robust > config.bone_outlier_z
        )
        bone[bad, child] = True

    velocity = np.zeros((frames, points), dtype=np.bool_)
    if frames >= 3:
        samples = motion.positions.astype(np.float64)
        point_velocity = np.gradient(samples, motion.timestamps, axis=0, edge_order=2)
        acceleration = np.linalg.norm(
            np.gradient(
                point_velocity, motion.timestamps, axis=0, edge_order=2
            ),
            axis=2,
        )
        acceleration[~motion.valid_mask] = np.nan
        median, scale = _robust_scale(acceleration, axis=0)
        robust = (acceleration - median[None, :]) / scale[None, :]
        velocity = motion.valid_mask & (robust > config.velocity_outlier_z)
    return bone, velocity


def _interpolate_and_filter(
    motion: AnimalMotion,
    anomaly: np.ndarray,
    skeleton: AnimalSkeleton,
    config: AnimalPreprocessConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = motion.positions.astype(np.float64).copy()
    confidence = motion.confidence.astype(np.float64).copy()
    usable_output = motion.valid_mask.copy()
    median_dt = float(np.median(np.diff(motion.timestamps)))
    body_window = _odd_window(
        motion.frame_count,
        config.filter_window_seconds,
        median_dt,
        config.polynomial_order,
    )
    toe_window = _odd_window(
        motion.frame_count,
        config.toe_filter_window_seconds,
        median_dt,
        min(2, config.polynomial_order),
    )
    toe_names = {skeleton.limb_chains[leg][-1] for leg in _LEGS}
    for point, name in enumerate(motion.keypoint_names):
        usable = motion.valid_mask[:, point] & ~anomaly[:, point]
        if np.count_nonzero(usable) < 2:
            usable_output[:, point] = False
            continue
        repaired = ~usable
        for axis in range(3):
            positions[:, point, axis] = np.interp(
                motion.timestamps,
                motion.timestamps[usable],
                positions[usable, point, axis],
            )
        confidence[repaired, point] *= 0.5
        usable_output[:, point] = True
        window = toe_window if name in toe_names else body_window
        order = (
            min(2, config.polynomial_order)
            if name in toe_names
            else config.polynomial_order
        )
        if window > order:
            positions[:, point] = savgol_filter(
                positions[:, point], window, order, axis=0, mode="interp"
            )
    return positions, confidence, usable_output


def _smooth_root_orientation(
    motion: AnimalMotion,
    positions: np.ndarray,
    valid_mask: np.ndarray,
    skeleton: AnimalSkeleton,
    config: AnimalPreprocessConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    working = replace(
        motion,
        positions=positions,
        valid_mask=valid_mask,
        frame_valid=np.all(valid_mask, axis=1),
    )
    root = estimate_root_motion(working, skeleton)
    raw = Rotation.from_quat(wxyz_to_xyzw(root.rotation))
    relative = raw * raw[0].inv()
    rotation_vectors = relative.as_rotvec()
    median_dt = float(np.median(np.diff(motion.timestamps)))
    window = _odd_window(
        motion.frame_count,
        config.root_rotation_window_seconds,
        median_dt,
        config.polynomial_order,
    )
    if window > config.polynomial_order:
        rotation_vectors = savgol_filter(
            rotation_vectors,
            window,
            config.polynomial_order,
            axis=0,
            mode="interp",
        )
    smoothed = Rotation.from_rotvec(rotation_vectors) * raw[0]
    correction = smoothed * raw.inv()
    result = positions.copy()
    for frame in range(motion.frame_count):
        result[frame] = root.position[frame] + correction[frame].apply(
            positions[frame] - root.position[frame]
        )
    return (
        result,
        np.ascontiguousarray(root.rotation, dtype=np.float32),
        np.ascontiguousarray(xyzw_to_wxyz(smoothed.as_quat()), dtype=np.float32),
    )


def estimate_ground_plane(
    motion: AnimalMotion,
    skeleton: AnimalSkeleton | None = None,
    *,
    frame_range: tuple[int, int] | None = None,
) -> GroundEstimate:
    """Robustly fit a local ground plane to low, slow toe observations."""

    skeleton = skeleton or get_skeleton(motion.metadata["skeleton_id"])
    index = {name: i for i, name in enumerate(motion.keypoint_names)}
    toes = np.array([index[skeleton.limb_chains[leg][-1]] for leg in _LEGS])
    positions = motion.positions[:, toes].astype(np.float64)
    valid = motion.valid_mask[:, toes]
    if frame_range is not None:
        start, stop = frame_range
        selection = np.zeros(motion.frame_count, dtype=np.bool_)
        selection[start : stop + 1] = True
        valid &= selection[:, None]
    if motion.frame_count >= 3:
        speed = np.linalg.norm(
            np.gradient(positions, motion.timestamps, axis=0, edge_order=2), axis=2
        )
    else:
        speed = np.zeros(valid.shape, dtype=np.float64)
    heights = positions[..., 2][valid]
    if not len(heights):
        return GroundEstimate(
            point=np.zeros(3, dtype=np.float32),
            normal=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            rmse=float("nan"),
            candidate_count=0,
        )
    height_limit = float(np.percentile(heights, 30.0))
    speeds = speed[valid]
    speed_limit = float(np.percentile(speeds, 45.0)) if len(speeds) else np.inf
    candidates = valid & (positions[..., 2] <= height_limit) & (speed <= speed_limit)
    samples = positions[candidates]
    if len(samples) < 6:
        samples = positions[valid & (positions[..., 2] <= height_limit)]
    if len(samples) < 3:
        height = float(np.percentile(heights, 5.0))
        return GroundEstimate(
            point=np.array([0.0, 0.0, height], dtype=np.float32),
            normal=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            rmse=0.0,
            candidate_count=int(len(samples)),
        )
    center = np.mean(samples[:, :2], axis=0)
    design = np.column_stack(
        (
            samples[:, 0] - center[0],
            samples[:, 1] - center[1],
            np.ones(len(samples)),
        )
    )
    keep = np.ones(len(samples), dtype=np.bool_)
    coefficients = np.zeros(3, dtype=np.float64)
    for _ in range(4):
        coefficients = np.linalg.lstsq(design[keep], samples[keep, 2], rcond=None)[0]
        residual = samples[:, 2] - design @ coefficients
        median, scale = _robust_scale(residual)
        keep = np.abs(residual - median) <= 3.5 * scale
        if np.count_nonzero(keep) < 3:
            keep[:] = True
            break
    normal = np.array([-coefficients[0], -coefficients[1], 1.0])
    normal /= np.linalg.norm(normal)
    point = np.array([center[0], center[1], coefficients[2]])
    rmse = float(
        np.sqrt(np.mean((samples[keep, 2] - design[keep] @ coefficients) ** 2))
    )
    return GroundEstimate(
        point=np.ascontiguousarray(point, dtype=np.float32),
        normal=np.ascontiguousarray(normal, dtype=np.float32),
        rmse=rmse,
        candidate_count=int(np.count_nonzero(keep)),
    )


def _classify_action(
    motion: AnimalMotion,
    contact: np.ndarray,
    frame_abnormal: np.ndarray,
    config: AnimalPreprocessConfig,
) -> tuple[tuple[str, ...], tuple[int, int]]:
    root = estimate_root_motion(motion)
    velocity = np.gradient(root.position, motion.timestamps, axis=0, edge_order=2)
    horizontal_speed = np.linalg.norm(velocity[:, :2], axis=1)
    rotations = Rotation.from_quat(wxyz_to_xyzw(root.rotation))
    forward = rotations.apply(np.array([1.0, 0.0, 0.0]))
    yaw = np.unwrap(np.arctan2(forward[:, 1], forward[:, 0]))
    yaw_rate = np.abs(np.gradient(yaw, motion.timestamps, edge_order=2))
    support = np.sum(np.nan_to_num(contact, nan=0.0) >= 0.5, axis=1)
    labels = np.full(motion.frame_count, "walk", dtype=object)
    labels[(horizontal_speed < 0.08) & (yaw_rate < 0.25) & (support >= 2)] = "stand"
    labels[(horizontal_speed > 1.2) | ((horizontal_speed > 0.65) & (support <= 1))] = "run"
    labels[yaw_rate > 0.55] = "turn"
    labels[support == 0] = "flight"
    labels[frame_abnormal] = "abnormal"
    for start in range(1, motion.frame_count - 1):
        if labels[start - 1] == labels[start + 1] != labels[start]:
            labels[start] = labels[start - 1]
    score = (
        horizontal_speed
        + 0.35 * yaw_rate
        + 0.12 * (4.0 - support)
        + frame_abnormal.astype(np.float64) * 100.0
    )
    median_dt = float(np.median(np.diff(motion.timestamps)))
    count = min(
        motion.frame_count,
        max(1, int(round(config.minimum_neutral_seconds / median_dt))),
    )
    kernel = np.ones(count) / count
    average = np.convolve(score, kernel, mode="valid")
    neutral_start = int(np.argmin(average))
    neutral_stop = neutral_start + count - 1
    return tuple(str(label) for label in labels), (neutral_start, neutral_stop)


def preprocess_animal_motion(
    motion: AnimalMotion,
    *,
    skeleton: AnimalSkeleton | None = None,
    config: AnimalPreprocessConfig | None = None,
) -> tuple[AnimalMotion, AnimalPreprocessReport]:
    """Repair, smooth, classify, and estimate the environment for real motion."""

    if motion.frame_count < 3:
        raise ValueError("animal preprocessing requires at least three frames")
    skeleton = skeleton or get_skeleton(motion.metadata["skeleton_id"])
    config = config or AnimalPreprocessConfig()
    bone, velocity = _detect_anomalies(motion, skeleton, config)
    observation = ~motion.valid_mask | bone | velocity
    positions, confidence, valid_mask = _interpolate_and_filter(
        motion, observation, skeleton, config
    )
    positions, root_raw, root_smoothed = _smooth_root_orientation(
        motion, positions, valid_mask, skeleton, config
    )
    provisional = replace(
        motion,
        positions=positions,
        confidence=confidence,
        valid_mask=valid_mask,
        frame_valid=np.all(valid_mask, axis=1),
    )
    ground = estimate_ground_plane(provisional, skeleton)
    contact = estimate_contact_probability(provisional, skeleton, ground=ground)
    frame_abnormal = (
        np.mean(observation, axis=1) >= 0.15
    ) | ~np.all(valid_mask, axis=1)
    root_ids = [
        provisional.keypoint_names.index(name)
        for name in skeleton.root_landmarks.values()
    ]
    frame_abnormal |= np.any(observation[:, root_ids], axis=1)
    frame_abnormal |= np.any(bone, axis=1)
    frame_abnormal |= np.sum(velocity, axis=1) >= max(
        3, int(round(0.10 * len(motion.keypoint_names)))
    )
    labels, neutral = _classify_action(
        provisional, contact, frame_abnormal, config
    )
    report = AnimalPreprocessReport(
        observation_anomaly=np.ascontiguousarray(observation),
        bone_anomaly=np.ascontiguousarray(bone),
        velocity_anomaly=np.ascontiguousarray(velocity),
        frame_abnormal=np.ascontiguousarray(frame_abnormal),
        action_labels=labels,
        ground=ground,
        neutral_frame_range=neutral,
        root_rotation_raw=root_raw,
        root_rotation_smoothed=root_smoothed,
    )
    metadata = dict(motion.metadata)
    metadata["contact_source"] = "heuristic"
    metadata["preprocess"] = {
        "mode": "adaptive_real_motion_v1",
        "config": asdict(config),
        **report.summary(),
    }
    result = replace(
        provisional,
        positions=np.ascontiguousarray(positions, dtype=np.float32),
        confidence=np.ascontiguousarray(confidence, dtype=np.float32),
        contact_probability=contact,
        metadata=metadata,
    )
    return result, report


def reestimate_contact_and_ground(
    motion: AnimalMotion,
    frame_range: tuple[int, int],
    *,
    skeleton: AnimalSkeleton | None = None,
) -> tuple[AnimalMotion, GroundEstimate]:
    """Re-estimate environment/contact in one inclusive frame interval."""

    start, stop = frame_range
    if not 0 <= start <= stop < motion.frame_count:
        raise ValueError("frame_range must be inside the motion")
    skeleton = skeleton or get_skeleton(motion.metadata["skeleton_id"])
    ground = estimate_ground_plane(motion, skeleton, frame_range=frame_range)
    estimated = estimate_contact_probability(motion, skeleton, ground=ground)
    contact = motion.contact_probability.copy()
    contact[start : stop + 1] = estimated[start : stop + 1]
    metadata = dict(motion.metadata)
    metadata["contact_source"] = "mixed"
    history = list(metadata.get("environment_edit_history", []))
    history.append(
        {
            "kind": "reestimate_contact_ground",
            "frame_range": [start, stop],
            "ground_point": ground.point.tolist(),
            "ground_normal": ground.normal.tolist(),
            "ground_rmse_m": ground.rmse,
        }
    )
    metadata["environment_edit_history"] = history
    return replace(motion, contact_probability=contact, metadata=metadata), ground
