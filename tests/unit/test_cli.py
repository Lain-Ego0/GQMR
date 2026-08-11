from __future__ import annotations

import json
from pathlib import Path

from gqmr.cli.main import main
from gqmr.core.io import save_motion
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
