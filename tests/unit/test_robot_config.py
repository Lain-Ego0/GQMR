from __future__ import annotations

from pathlib import Path

import pytest

from gqmr.robots.config import (
    RobotConfigError,
    get_robot_config,
    load_robot_config,
)


def test_builtin_robot_configs_are_valid_and_bound_to_assets() -> None:
    go2 = get_robot_config("unitree-go2")
    b2 = get_robot_config("unitree-b2")
    assert go2.id == go2.asset_id == "unitree-go2"
    assert b2.id == b2.asset_id == "unitree-b2"
    assert len(go2.dof_order) == len(b2.dof_order) == 12
    assert tuple(go2.feet) == ("FL", "FR", "RL", "RR")
    assert len(go2.sha256) == len(b2.sha256) == 64
    assert go2.feet["FL"].contact_geoms == ("FL",)
    assert b2.feet["FL"].body == "FL_calf"
    assert b2.feet["FL"].local_position == (0.0, 0.0, -0.35)


def test_duplicate_yaml_keys_are_rejected(tmp_path: Path) -> None:
    config = tmp_path / "duplicate.yaml"
    config.write_text("schema_version: 1\nschema_version: 1\n", encoding="utf-8")
    with pytest.raises(RobotConfigError, match="duplicate YAML key"):
        load_robot_config(config)


def test_unknown_config_fields_are_rejected(tmp_path: Path) -> None:
    source = get_robot_config("unitree-go2")
    document = source.model_dump_json(indent=2)
    config = tmp_path / "invalid.yaml"
    config.write_text(document[:-2] + ',\n  "unknown": true\n}', encoding="utf-8")
    with pytest.raises(RobotConfigError, match="unknown"):
        load_robot_config(config)

