from __future__ import annotations

import json
from pathlib import Path

from gqmr.cli.main import main
from gqmr.core.io import load_motion, save_motion
from gqmr.core.motion import AnimalMotion
from test_motion_io import make_robot_motion


def test_validate_cli_reports_summary(tmp_path: Path, capsys) -> None:
    destination = tmp_path / "robot.npz"
    save_motion(destination, make_robot_motion())
    result = main(
        ["validate", str(destination), "--model-sha256", "c" * 64]
    )
    output = json.loads(capsys.readouterr().out)
    assert result == 0
    assert output["valid"] is True
    assert output["frames"] == 3


def test_validate_cli_reports_structured_error(tmp_path: Path, capsys) -> None:
    destination = tmp_path / "missing.npz"
    result = main(["validate", str(destination)])
    output = json.loads(capsys.readouterr().err)
    assert result == 2
    assert output["valid"] is False
    assert output["error_type"] == "UnsafeMotionFileError"


def test_assets_status_cli_reports_missing(tmp_path: Path, capsys) -> None:
    result = main(
        ["assets", "status", "unitree-go2", "--cache-dir", str(tmp_path)]
    )
    output = json.loads(capsys.readouterr().out)
    assert result == 1
    assert output["assets"][0]["state"] == "missing"


def test_inspect_and_convert_legacy_cli(tmp_path: Path, capsys) -> None:
    source = tmp_path / "legacy.txt"
    frame = ["0"] * 81
    source.write_text(",".join(frame) + "\n" + ",".join(frame), encoding="utf-8")

    inspect_result = main(["inspect", str(source)])
    inspected = json.loads(capsys.readouterr().out)
    destination = tmp_path / "animal.npz"
    convert_result = main(
        [
            "convert",
            str(source),
            "--skeleton",
            "dog-27",
            "--fps",
            "50",
            "--output",
            str(destination),
        ]
    )
    converted = json.loads(capsys.readouterr().out)
    motion = load_motion(destination)

    assert inspect_result == 0
    assert inspected["format"] == "legacy_ai4animation_dog27"
    assert inspected["frames"] == 2
    assert convert_result == 0
    assert converted["schema_id"] == "gqmr.animal_motion"
    assert isinstance(motion, AnimalMotion)
    assert motion.timestamps.tolist() == [0.0, 0.02]


def test_synthetic_cli_generates_mit_motion(tmp_path: Path, capsys) -> None:
    destination = tmp_path / "turn.npz"

    result = main(
        [
            "synthetic",
            "turn",
            "--duration",
            "0.5",
            "--fps",
            "20",
            "--output",
            str(destination),
        ]
    )
    output = json.loads(capsys.readouterr().out)
    motion = load_motion(destination)

    assert result == 0
    assert output["license"] == "MIT"
    assert output["frames"] == 11
    assert isinstance(motion, AnimalMotion)
    assert motion.metadata["source"]["gait"] == "turn"


def test_project_cli_new_add_info_and_pack(tmp_path: Path, capsys) -> None:
    project = tmp_path / "demo.gqmr"
    resource = tmp_path / "input.bin"
    resource.write_bytes(b"demo")

    assert main(["project", "new", str(project)]) == 0
    capsys.readouterr()
    assert main(
        ["project", "add", str(project), str(resource), "--kind", "animal"]
    ) == 0
    added = json.loads(capsys.readouterr().out)
    assert added["resources"] == 1
    assert added["active_animal_motion"] is not None

    assert main(["project", "info", str(project)]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["project_id"] == added["project_id"]

    portable = tmp_path / "portable.gqmr"
    assert main(["project", "pack", str(project), str(portable)]) == 0
    packed = json.loads(capsys.readouterr().out)
    assert packed["embedded_resources"] == 1


def test_edit_cli_trim_and_resample(tmp_path: Path, capsys) -> None:
    source = tmp_path / "animal.npz"
    trimmed = tmp_path / "trimmed.npz"
    resampled = tmp_path / "resampled.npz"
    assert main(
        [
            "synthetic",
            "walk",
            "--duration",
            "1",
            "--fps",
            "20",
            "--output",
            str(source),
        ]
    ) == 0
    capsys.readouterr()
    assert main(
        [
            "edit",
            "trim",
            str(source),
            "--start",
            "0.25",
            "--end",
            "0.75",
            "--output",
            str(trimmed),
        ]
    ) == 0
    capsys.readouterr()
    assert main(
        [
            "edit",
            "resample",
            str(trimmed),
            "--fps",
            "40",
            "--output",
            str(resampled),
        ]
    ) == 0
    output = json.loads(capsys.readouterr().out)
    motion = load_motion(resampled)

    assert output["frames"] == 21
    assert motion.duration == 0.5
