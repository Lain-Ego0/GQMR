"""Historical DeepMimic-style JSON compatibility export."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gqmr.core.coordinates import wxyz_to_xyzw
from gqmr.core.motion import RobotMotion
from gqmr.exporters.common import ExportError, atomic_write, require_exportable


def export_deepmimic_json(
    motion: RobotMotion,
    destination: str | Path,
    *,
    loop_mode: str = "Wrap",
) -> Path:
    """Export root xyz + root xyzw + scalar DOFs in historical frame layout."""

    require_exportable(motion)
    delta = np.diff(motion.timestamps)
    frame_duration = float(np.median(delta))
    if np.max(np.abs(delta - frame_duration)) > 1e-6:
        raise ExportError("DeepMimic compatibility export requires uniform timestamps")
    frames = np.concatenate(
        (
            motion.root_position,
            wxyz_to_xyzw(motion.root_rotation),
            motion.dof_position,
        ),
        axis=1,
    )
    document = {
        "LoopMode": loop_mode,
        "FrameDuration": frame_duration,
        "EnableCycleOffsetPosition": True,
        "EnableCycleOffsetRotation": True,
        "QuaternionOrder": "xyzw",
        "DOFNames": list(motion.dof_names),
        "Frames": frames.tolist(),
    }
    encoded = json.dumps(
        document, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")
    return atomic_write(destination, lambda stream: stream.write(encoded))
