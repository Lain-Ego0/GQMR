from __future__ import annotations

import uuid
from datetime import datetime, timezone

import numpy as np
import pytest

from gqmr.editing import EditStack, apply_edit
from gqmr.project.model import EditCommand
from gqmr.synthetic import generate_dog27_motion


def _command(kind: str, parameters: dict) -> EditCommand:
    return EditCommand(
        command_id=str(uuid.uuid4()),
        kind=kind,
        resource_id=str(uuid.uuid4()),
        parameters=parameters,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def test_trim_time_scale_root_transform_and_contact_override() -> None:
    motion = generate_dog27_motion("walk", duration=1.0, fps=20.0)
    trimmed = apply_edit(motion, _command("trim", {"start": 0.25, "end": 0.75}))
    scaled = apply_edit(trimmed, _command("time_scale", {"speed": 2.0}))
    transformed = apply_edit(
        scaled,
        _command(
            "root_transform", {"translation": [1.0, 2.0, 0.0], "yaw": np.pi / 2}
        ),
    )
    contacted = apply_edit(
        transformed,
        _command(
            "contact_override",
            {"start": 0.0, "end": transformed.duration, "leg": "FL", "probability": 1.0},
        ),
    )

    assert trimmed.timestamps[0] == 0.0
    assert scaled.duration == pytest.approx(trimmed.duration / 2.0)
    assert np.allclose(
        transformed.positions[0, 0, :2],
        [-scaled.positions[0, 0, 1] + 1.0, scaled.positions[0, 0, 0] + 2.0],
        atol=1e-6,
    )
    assert np.all(contacted.contact_probability[:, 0] == 1.0)
    assert contacted.metadata["contact_source"] == "mixed"
    assert len(contacted.metadata["edit_history"]) == 4


def test_resample_and_undo_redo_are_deterministic() -> None:
    motion = generate_dog27_motion("turn", duration=1.0, fps=20.0)
    command = _command("resample", {"fps": 40})
    stack = EditStack(motion)

    edited = stack.push(command)
    assert edited.frame_count == 41
    assert stack.undo() is motion
    redone = stack.redo()
    assert np.array_equal(redone.positions, edited.positions)
    assert redone.metadata["edit_history"] == edited.metadata["edit_history"]
