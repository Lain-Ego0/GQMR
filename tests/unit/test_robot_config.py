from __future__ import annotations

from pathlib import Path

import pytest

from gqmr.robots.config import (
    RobotConfigError,
    available_robot_configs,
    get_robot_config,
    load_robot_config,
)


def test_builtin_robot_configs_are_valid_and_bound_to_assets() -> None:
    configs = [get_robot_config(robot_id) for robot_id in available_robot_configs()]
    assert len(configs) == 7
    assert all(config.id == config.asset_id for config in configs)
    assert all(len(config.dof_order) == 12 for config in configs)
    assert all(tuple(config.feet) == ("FL", "FR", "RL", "RR") for config in configs)
    assert all(len(config.sha256) == 64 for config in configs)
    go2 = get_robot_config("unitree-go2")
    b2 = get_robot_config("unitree-b2")
    assert go2.id == go2.asset_id == "unitree-go2"
    assert b2.id == b2.asset_id == "unitree-b2"
    assert go2.feet["FL"].contact_geoms == ("FL",)
    assert b2.feet["FL"].body == "FL_calf"
    assert b2.feet["FL"].local_position == (0.0, 0.0, -0.35)
    lite3 = get_robot_config("deeprobotics-lite3")
    assert lite3.root_joint == "floating_base"
    assert lite3.feet["FL"].contact_geoms == ("FL_FOOT_collision",)


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
