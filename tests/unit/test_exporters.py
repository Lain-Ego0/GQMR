from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gqmr.exporters import export_deepmimic_json
from gqmr.exporters.common import ExportError
from test_motion_io import make_robot_motion


def test_deepmimic_export_has_historical_frame_layout(tmp_path: Path) -> None:
    motion = make_robot_motion()
    destination = tmp_path / "motion.json"

    export_deepmimic_json(motion, destination)
    document = json.loads(destination.read_text(encoding="utf-8"))

    assert document["LoopMode"] == "Wrap"
    assert document["FrameDuration"] == pytest.approx(0.01)
    assert document["QuaternionOrder"] == "xyzw"
    assert document["DOFNames"] == list(motion.dof_names)
    assert np.asarray(document["Frames"]).shape == (3, 7 + len(motion.dof_names))
    assert document["Frames"][0][3:7] == [0.0, 0.0, 0.0, 1.0]


def test_export_rejects_invalid_frames(tmp_path: Path) -> None:
    motion = make_robot_motion()
    invalid = motion.frame_valid.copy()
    invalid[1] = False
    status = motion.solver_status.copy()
    status[1] = 3
    residual = motion.solver_residual.copy()
    residual[1] = np.nan
    motion = replace(
        motion,
        frame_valid=invalid,
        solver_status=status,
        solver_residual=residual,
    )

    with pytest.raises(ExportError, match="invalid frames"):
        export_deepmimic_json(motion, tmp_path / "invalid.json")
