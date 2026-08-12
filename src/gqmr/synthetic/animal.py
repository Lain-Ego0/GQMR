"""Generate deterministic dog-27 gait clips without restricted source data."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from gqmr import __version__
from gqmr.core.motion import AnimalMotion
from gqmr.skeletons import get_skeleton

Gait = Literal["walk", "trot", "pace", "turn"]
_LEGS = ("FL", "FR", "RL", "RR")


@dataclass(frozen=True, slots=True)
class MotionPreset:
    id: str
    label: str
    gait: Gait
    speed: float
    cycle_hz: float
    duty_factor: float
    swing_height: float
    turn_rate: float = 0.0
    duration: float = 2.0
    fps: float = 60.0


MOTION_PRESETS = (
    MotionPreset("walk_slow", "慢速行走", "walk", 0.20, 1.10, 0.68, 0.060),
    MotionPreset("walk_standard", "标准行走", "walk", 0.35, 1.50, 0.62, 0.075),
    MotionPreset("trot_slow", "慢速小跑", "trot", 0.35, 1.55, 0.56, 0.065),
    MotionPreset("trot_standard", "标准小跑", "trot", 0.55, 2.00, 0.50, 0.075),
    MotionPreset("trot_fast", "快速小跑", "trot", 0.80, 2.70, 0.46, 0.095),
    MotionPreset("pace_standard", "侧对步", "pace", 0.45, 1.80, 0.52, 0.075),
    MotionPreset("turn_left", "左转", "turn", 0.18, 2.00, 0.50, 0.075, 0.65),
    MotionPreset("turn_right", "右转", "turn", 0.18, 2.00, 0.50, 0.075, -0.65),
)
_PRESET_BY_ID = {preset.id: preset for preset in MOTION_PRESETS}


def available_motion_presets() -> tuple[MotionPreset, ...]:
    return MOTION_PRESETS


def get_motion_preset(preset_id: str) -> MotionPreset:
    try:
        return _PRESET_BY_ID[preset_id]
    except KeyError as error:
        raise ValueError(f"unsupported synthetic motion preset {preset_id!r}") from error


def _body_pose(
    gait: Gait,
    time: float,
    speed: float,
    turn_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    yaw = 0.0 if gait != "turn" else turn_rate * time
    cosine, sine = np.cos(yaw), np.sin(yaw)
    rotation = np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    if gait == "turn":
        direction = 1.0 if turn_rate >= 0.0 else -1.0
        translation = np.array(
            [
                0.25 * np.sin(abs(yaw)),
                direction * 0.25 * (1.0 - np.cos(yaw)),
                0.0,
            ],
            dtype=np.float64,
        )
    else:
        translation = np.array([speed * time, 0.0, 0.0], dtype=np.float64)
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
    turn_rate: float,
) -> tuple[np.ndarray, bool]:
    cycle = cycle_hz * time + phase_offset
    phase = cycle - np.floor(cycle)
    stance_duration = duty_factor / cycle_hz
    stride = speed * stance_duration
    front_toe = rest_toe + np.array([0.5 * stride, 0.0, 0.0])
    if phase < duty_factor:
        stance_start = time - phase / cycle_hz
        rotation, translation = _body_pose(gait, stance_start, speed, turn_rate)
        return rotation @ front_toe + translation, True

    swing_progress = (phase - duty_factor) / (1.0 - duty_factor)
    lift_time = time - (phase - duty_factor) / cycle_hz
    stance_start = lift_time - stance_duration
    landing_time = time + (1.0 - phase) / cycle_hz
    lift_rotation, lift_translation = _body_pose(gait, stance_start, speed, turn_rate)
    land_rotation, land_translation = _body_pose(gait, landing_time, speed, turn_rate)
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
    speed: float | None = None,
    cycle_hz: float | None = None,
    duty_factor: float | None = None,
    swing_height: float | None = None,
    turn_rate: float | None = None,
) -> AnimalMotion:
    if gait not in {"walk", "trot", "pace", "turn"}:
        raise ValueError(f"unsupported synthetic gait {gait!r}")
    if not np.isfinite(duration) or duration <= 0.0 or not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("duration and fps must be finite and positive")
    default_cycle_hz = 1.5 if gait == "walk" else 2.0
    default_speed = 0.35 if gait == "walk" else 0.55
    if gait == "turn":
        default_speed = 0.18
    default_duty_factor = 0.62 if gait == "walk" else 0.5
    speed = default_speed if speed is None else speed
    cycle_hz = default_cycle_hz if cycle_hz is None else cycle_hz
    duty_factor = default_duty_factor if duty_factor is None else duty_factor
    swing_height = 0.075 if swing_height is None else swing_height
    turn_rate = (0.65 if gait == "turn" else 0.0) if turn_rate is None else turn_rate
    parameters = (speed, cycle_hz, duty_factor, swing_height, turn_rate)
    if not all(np.isfinite(value) for value in parameters):
        raise ValueError("synthetic gait parameters must be finite")
    if speed < 0.0 or cycle_hz <= 0.0 or not 0.0 < duty_factor < 1.0:
        raise ValueError("speed must be non-negative, cycle_hz positive, and duty_factor between 0 and 1")
    if swing_height < 0.0:
        raise ValueError("swing_height must be non-negative")
    if gait == "turn" and abs(turn_rate) < 1e-12:
        raise ValueError("turn_rate must be non-zero for turn motions")
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
        rotation, translation = _body_pose(gait, float(time), speed, turn_rate)
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
                turn_rate=turn_rate,
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
                "speed_mps": speed,
                "cycle_hz": cycle_hz,
                "duty_factor": duty_factor,
                "swing_height_m": swing_height,
                "turn_rate_rad_s": turn_rate,
                "generator_version": 3,
                "license": "MIT",
            },
            "created_by": {"gqmr_version": __version__},
        },
    )


def generate_dog27_preset(
    preset_id: str,
    *,
    duration: float | None = None,
    fps: float | None = None,
) -> AnimalMotion:
    preset = get_motion_preset(preset_id)
    motion = generate_dog27_motion(
        preset.gait,
        duration=preset.duration if duration is None else duration,
        fps=preset.fps if fps is None else fps,
        speed=preset.speed,
        cycle_hz=preset.cycle_hz,
        duty_factor=preset.duty_factor,
        swing_height=preset.swing_height,
        turn_rate=preset.turn_rate,
    )
    metadata = dict(motion.metadata)
    source = dict(metadata["source"])
    source.update({"preset_id": preset.id, "preset_label": preset.label})
    metadata["source"] = source
    return replace(motion, metadata=metadata)


def generate_dog27_suite(
    *, duration: float = 2.0, fps: float = 60.0
) -> dict[str, AnimalMotion]:
    return {
        gait: generate_dog27_motion(gait, duration=duration, fps=fps)
        for gait in ("walk", "trot", "pace", "turn")
    }


def generate_dog27_preset_suite(
    *, duration: float | None = None, fps: float | None = None
) -> dict[str, AnimalMotion]:
    return {
        preset.id: generate_dog27_preset(preset.id, duration=duration, fps=fps)
        for preset in MOTION_PRESETS
    }
