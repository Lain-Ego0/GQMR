"""Generate deterministic dog-27 gait clips without restricted source data."""

from __future__ import annotations

from typing import Literal

import numpy as np

from gqmr import __version__
from gqmr.core.motion import AnimalMotion
from gqmr.skeletons import get_skeleton

Gait = Literal["walk", "trot", "pace", "turn"]
_LEGS = ("FL", "FR", "RL", "RR")


def _body_pose(gait: Gait, time: float, speed: float) -> tuple[np.ndarray, np.ndarray]:
    yaw = 0.0 if gait != "turn" else 0.65 * time
    cosine, sine = np.cos(yaw), np.sin(yaw)
    rotation = np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    translation = np.array(
        [
            speed * time if gait != "turn" else 0.25 * np.sin(yaw),
            0.0 if gait != "turn" else 0.25 * (1.0 - np.cos(yaw)),
            0.0,
        ],
        dtype=np.float64,
    )
    return rotation, translation


def _smoothstep(value: float) -> float:
    return value * value * (3.0 - 2.0 * value)


def _foot_target(
    rest_toe: np.ndarray,
    *,
    gait: Gait,
    time: float,
    phase_offset: float,
    cycle_hz: float,
    speed: float,
    duty_factor: float,
    swing_height: float,
) -> tuple[np.ndarray, bool]:
    cycle = cycle_hz * time + phase_offset
    phase = cycle - np.floor(cycle)
    stance_duration = duty_factor / cycle_hz
    stride = speed * stance_duration
    front_toe = rest_toe + np.array([0.5 * stride, 0.0, 0.0])
    if phase < duty_factor:
        stance_start = time - phase / cycle_hz
        rotation, translation = _body_pose(gait, stance_start, speed)
        return rotation @ front_toe + translation, True

    swing_progress = (phase - duty_factor) / (1.0 - duty_factor)
    lift_time = time - (phase - duty_factor) / cycle_hz
    stance_start = lift_time - stance_duration
    landing_time = time + (1.0 - phase) / cycle_hz
    lift_rotation, lift_translation = _body_pose(gait, stance_start, speed)
    land_rotation, land_translation = _body_pose(gait, landing_time, speed)
    lift_position = lift_rotation @ front_toe + lift_translation
    land_position = land_rotation @ front_toe + land_translation
    blend = _smoothstep(swing_progress)
    target = (1.0 - blend) * lift_position + blend * land_position
    target[2] += swing_height * np.sin(np.pi * swing_progress)
    return target, False


def _solve_fixed_length_chain(
    initial: np.ndarray,
    target: np.ndarray,
    lengths: np.ndarray,
) -> np.ndarray:
    points = initial.astype(np.float64, copy=True)
    anchor = points[0].copy()
    total_length = float(np.sum(lengths))
    direction = target - anchor
    distance = float(np.linalg.norm(direction))
    if distance >= total_length - 1e-10:
        unit = direction / max(distance, 1e-12)
        for index, length in enumerate(lengths):
            points[index + 1] = points[index] + unit * length
        return points
    for _ in range(48):
        points[-1] = target
        for index in range(len(points) - 2, -1, -1):
            delta = points[index] - points[index + 1]
            norm = float(np.linalg.norm(delta))
            if norm < 1e-12:
                delta = initial[index] - initial[index + 1]
                norm = float(np.linalg.norm(delta))
            points[index] = points[index + 1] + delta * (lengths[index] / norm)
        points[0] = anchor
        for index in range(len(points) - 1):
            delta = points[index + 1] - points[index]
            norm = float(np.linalg.norm(delta))
            if norm < 1e-12:
                delta = initial[index + 1] - initial[index]
                norm = float(np.linalg.norm(delta))
            points[index + 1] = points[index] + delta * (lengths[index] / norm)
        if np.linalg.norm(points[-1] - target) < 1e-9:
            break
    return points


def _rest_pose(names: tuple[str, ...]) -> np.ndarray:
    points = {
        "pelvis": [-0.12, 0.0, 0.46],
        "pelvis_duplicate": [-0.12, 0.0, 0.46],
        "spine": [0.08, 0.0, 0.50],
        "neck": [0.30, 0.0, 0.53],
        "head": [0.42, 0.0, 0.57],
        "muzzle": [0.54, 0.0, 0.54],
        "left_shoulder": [0.24, 0.14, 0.48],
        "left_front_upper": [0.20, 0.17, 0.34],
        "left_front_elbow": [0.27, 0.17, 0.21],
        "left_front_wrist": [0.31, 0.17, 0.08],
        "left_front_toe": [0.39, 0.17, 0.025],
        "right_shoulder": [0.24, -0.14, 0.48],
        "right_front_upper": [0.20, -0.17, 0.34],
        "right_front_elbow": [0.27, -0.17, 0.21],
        "right_front_wrist": [0.31, -0.17, 0.08],
        "right_front_toe": [0.39, -0.17, 0.025],
        "left_hip": [-0.20, 0.12, 0.43],
        "left_hind_knee": [-0.12, 0.15, 0.28],
        "left_hind_ankle": [-0.25, 0.15, 0.11],
        "left_hind_toe": [-0.15, 0.15, 0.025],
        "right_hip": [-0.20, -0.12, 0.43],
        "right_hind_knee": [-0.12, -0.15, 0.28],
        "right_hind_ankle": [-0.25, -0.15, 0.11],
        "right_hind_toe": [-0.15, -0.15, 0.025],
        "tail_base": [-0.25, 0.0, 0.44],
        "tail_mid": [-0.40, 0.0, 0.40],
        "tail_tip": [-0.54, 0.0, 0.35],
    }
    return np.asarray([points[name] for name in names], dtype=np.float64)


def generate_dog27_motion(
    gait: Gait,
    *,
    duration: float = 2.0,
    fps: float = 60.0,
) -> AnimalMotion:
    if gait not in {"walk", "trot", "pace", "turn"}:
        raise ValueError(f"unsupported synthetic gait {gait!r}")
    if not np.isfinite(duration) or duration <= 0.0 or not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("duration and fps must be finite and positive")
    frames = max(3, int(round(duration * fps)) + 1)
    timestamps = np.arange(frames, dtype=np.float64) / fps
    skeleton = get_skeleton("dog-27")
    names = skeleton.names
    index = skeleton.name_to_index
    rest = _rest_pose(names)
    positions = np.empty((frames, len(names), 3), dtype=np.float64)
    contact = np.empty((frames, 4), dtype=np.float32)
    phase_offsets = {
        "walk": [0.0, 0.5, 0.75, 0.25],
        "trot": [0.0, 0.5, 0.5, 0.0],
        "pace": [0.0, 0.5, 0.0, 0.5],
        "turn": [0.0, 0.5, 0.5, 0.0],
    }[gait]
    cycle_hz = 1.5 if gait == "walk" else 2.0
    speed = 0.35 if gait == "walk" else 0.55
    if gait == "turn":
        speed = 0.18
    duty_factor = 0.62 if gait == "walk" else 0.5
    swing_height = 0.075
    chain_ids = {
        leg: np.array([index[name] for name in skeleton.limb_chains[leg]], dtype=np.int32)
        for leg in _LEGS
    }
    chain_lengths = {
        leg: np.linalg.norm(
            np.diff(rest[chain_ids[leg]], axis=0), axis=1
        )
        for leg in _LEGS
    }
    for frame, time in enumerate(timestamps):
        local = rest.copy()
        bob = 0.012 * np.sin(4.0 * np.pi * cycle_hz * time)
        local[:, 2] += bob
        rotation, translation = _body_pose(gait, float(time), speed)
        world = local @ rotation.T + translation
        for leg_index, leg in enumerate(_LEGS):
            target, in_contact = _foot_target(
                rest[chain_ids[leg][-1]],
                gait=gait,
                time=float(time),
                phase_offset=phase_offsets[leg_index],
                cycle_hz=cycle_hz,
                speed=speed,
                duty_factor=duty_factor,
                swing_height=swing_height,
            )
            world[chain_ids[leg]] = _solve_fixed_length_chain(
                world[chain_ids[leg]], target, chain_lengths[leg]
            )
            contact[frame, leg_index] = 1.0 if in_contact else 0.0
        positions[frame] = world
    return AnimalMotion(
        timestamps=timestamps,
        keypoint_names=names,
        positions=positions.astype(np.float32),
        confidence=np.ones((frames, len(names)), dtype=np.float32),
        valid_mask=np.ones((frames, len(names)), dtype=np.bool_),
        contact_probability=contact,
        frame_valid=np.ones(frames, dtype=np.bool_),
        metadata={
            "coordinate_frame": "gqmr_world_x_forward_y_left_z_up",
            "length_unit": "m",
            "time_unit": "s",
            "skeleton_id": skeleton.id,
            "skeleton_sha256": skeleton.sha256,
            "contact_order": ["FL", "FR", "RL", "RR"],
            "contact_source": "heuristic",
            "source": {
                "format": "gqmr_mit_synthetic",
                "gait": gait,
                "duration": duration,
                "fps": fps,
                "generator_version": 2,
                "license": "MIT",
            },
            "created_by": {"gqmr_version": __version__},
        },
    )


def generate_dog27_suite(
    *, duration: float = 2.0, fps: float = 60.0
) -> dict[str, AnimalMotion]:
    return {
        gait: generate_dog27_motion(gait, duration=duration, fps=fps)
        for gait in ("walk", "trot", "pace", "turn")
    }
