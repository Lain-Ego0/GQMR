"""Generate deterministic dog-27 gait clips without restricted source data."""

from __future__ import annotations

from typing import Literal

import numpy as np

from gqmr import __version__
from gqmr.core.motion import AnimalMotion
from gqmr.skeletons import get_skeleton

Gait = Literal["walk", "trot", "pace", "turn"]
_LEGS = ("FL", "FR", "RL", "RR")


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
    for frame, time in enumerate(timestamps):
        local = rest.copy()
        bob = 0.012 * np.sin(4.0 * np.pi * cycle_hz * time)
        local[:, 2] += bob
        for leg_index, leg in enumerate(_LEGS):
            phase = 2.0 * np.pi * (cycle_hz * time + phase_offsets[leg_index])
            swing = max(float(np.sin(phase)), 0.0)
            fore_aft = 0.07 * float(np.cos(phase))
            chain = skeleton.limb_chains[leg]
            for chain_index, name in enumerate(chain[1:], start=1):
                weight = chain_index / (len(chain) - 1)
                point_id = index[name]
                local[point_id, 0] += weight * fore_aft
                local[point_id, 2] += weight * 0.075 * swing
            contact[frame, leg_index] = np.clip(0.5 - 5.0 * np.sin(phase), 0.0, 1.0)
        yaw = 0.0 if gait != "turn" else 0.65 * time
        cosine, sine = np.cos(yaw), np.sin(yaw)
        rotation = np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])
        translation = np.array(
            [speed * time if gait != "turn" else 0.25 * np.sin(yaw),
             0.0 if gait != "turn" else 0.25 * (1.0 - np.cos(yaw)),
             0.0]
        )
        positions[frame] = local @ rotation.T + translation
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
