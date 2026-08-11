from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from gqmr.sources.files.legacy_dog27 import (
    LegacyDog27Error,
    inspect_legacy_dog27,
    load_legacy_dog27,
)


def _write_legacy(path: Path, frames: int = 3) -> np.ndarray:
    values = np.arange(frames * 27 * 3, dtype=np.float64).reshape(frames, 27, 3)
    values[:, 1] = values[:, 0]
    path.write_text(
        "\n".join(",".join(str(value) for value in frame.ravel()) for frame in values),
        encoding="utf-8",
    )
    return values


def test_inspect_and_load_legacy_dog27(tmp_path: Path) -> None:
    source = tmp_path / "dog.txt"
    values = _write_legacy(source)

    summary = inspect_legacy_dog27(source, fps=50.0)
    motion = load_legacy_dog27(source, fps=50.0, start_frame=1, end_frame=3)

    rotation = Rotation.from_euler("z", 0.47 * np.pi) * Rotation.from_euler(
        "x", 0.5 * np.pi
    )
    expected = rotation.apply(values[1].reshape(-1, 3)).reshape(27, 3)
    assert summary["frames"] == 3
    assert summary["duration_seconds"] == pytest.approx(0.04)
    assert summary["pelvis_duplicate_max_error"] == 0.0
    assert summary["license"] == "CC-BY-NC-4.0"
    assert motion.frame_count == 2
    assert motion.timestamps.tolist() == [0.0, 0.02]
    assert np.allclose(motion.positions[0], expected, atol=1e-5)
    assert motion.metadata["source"]["license"] == "CC-BY-NC-4.0"
    assert motion.metadata["source"]["coordinate_transform"] == "Rz(0.47*pi) @ Rx(0.5*pi)"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (",".join(["0"] * 80), "exactly 81"),
        (",".join(["0"] * 80 + ["nan"]), "NaN or infinity"),
        (",".join(["0"] * 80 + ["not-a-number"]), "non-numeric"),
    ],
)
def test_legacy_reader_rejects_malformed_rows(
    tmp_path: Path, text: str, message: str
) -> None:
    source = tmp_path / "bad.txt"
    source.write_text(text, encoding="utf-8")

    with pytest.raises(LegacyDog27Error, match=message):
        load_legacy_dog27(source)


@pytest.mark.parametrize(
    ("start", "end"), [(-1, None), (2, 2), (0, 4), (3, None)]
)
def test_legacy_reader_rejects_invalid_frame_ranges(
    tmp_path: Path, start: int, end: int | None
) -> None:
    source = tmp_path / "dog.txt"
    _write_legacy(source)

    with pytest.raises(LegacyDog27Error, match="invalid start/end"):
        load_legacy_dog27(source, start_frame=start, end_frame=end)
