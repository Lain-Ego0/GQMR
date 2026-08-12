"""Strict Pydantic schema for v1 robot YAML configuration."""

from __future__ import annotations

import hashlib
import json
from importlib import resources
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Mapping

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from gqmr.core.errors import GQMRError

Leg = Literal["FL", "FR", "RL", "RR"]
LEG_ORDER: tuple[Leg, ...] = ("FL", "FR", "RL", "RR")
_BUILTIN_ROBOTS = (
    "unitree-go2",
    "unitree-go1",
    "unitree-a1",
    "unitree-a2",
    "unitree-b2",
    "anybotics-anymal-c",
)
_HEX = frozenset("0123456789abcdef")


class RobotConfigError(GQMRError, ValueError):
    """Raised when robot configuration is invalid or ambiguous."""


class _UniqueKeySafeLoader(yaml.SafeLoader):
    pass


def _construct_mapping(
    loader: _UniqueKeySafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise RobotConfigError(f"duplicate YAML key {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


class FootConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    body: str = Field(min_length=1)
    local_position: tuple[float, float, float]
    contact_geoms: tuple[str, ...] = ()

    @field_validator("contact_geoms")
    @classmethod
    def unique_contact_geoms(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not name.strip() for name in value) or len(set(value)) != len(value):
            raise ValueError("contact geom names must be non-empty and unique")
        return value


class RobotConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    id: str = Field(min_length=1)
    asset_id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    model_sha256: str
    root_joint: str = Field(min_length=1)
    base_body: str = Field(min_length=1)
    dof_order: tuple[str, ...]
    feet: dict[Leg, FootConfig]
    default_root_position: tuple[float, float, float]
    default_root_rotation: tuple[float, float, float, float]
    default_dof_position: tuple[float, ...]

    @field_validator("model")
    @classmethod
    def safe_model_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or "." in path.parts or ".." in path.parts:
            raise ValueError("model must be a safe POSIX relative path")
        return value

    @field_validator("model_sha256")
    @classmethod
    def valid_model_hash(cls, value: str) -> str:
        if len(value) != 64 or any(character not in _HEX for character in value):
            raise ValueError("model_sha256 must be 64 lowercase hexadecimal characters")
        return value

    @model_validator(mode="after")
    def validate_v1_shape(self) -> "RobotConfig":
        if len(self.dof_order) != 12:
            raise ValueError("v1 robots must declare exactly 12 scalar DOFs")
        if len(set(self.dof_order)) != len(self.dof_order) or any(
            not name.strip() for name in self.dof_order
        ):
            raise ValueError("dof_order names must be non-empty and unique")
        if set(self.feet) != set(LEG_ORDER):
            raise ValueError("feet must contain exactly FL, FR, RL, RR")
        if len(self.default_dof_position) != len(self.dof_order):
            raise ValueError("default_dof_position must match dof_order")
        numeric_values = (
            *self.default_root_position,
            *self.default_root_rotation,
            *self.default_dof_position,
            *(coordinate for foot in self.feet.values() for coordinate in foot.local_position),
        )
        if not np.all(np.isfinite(numeric_values)):
            raise ValueError("default pose and foot offsets must be finite")
        quaternion_norm = float(np.linalg.norm(self.default_root_rotation))
        if abs(quaternion_norm - 1.0) >= 1e-8:
            raise ValueError("default_root_rotation must be a unit wxyz quaternion")
        return self

    @property
    def sha256(self) -> str:
        document = self.model_dump(mode="json")
        payload = json.dumps(
            document,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _parse_yaml(text: str, *, source: str) -> RobotConfig:
    try:
        document = yaml.load(text, Loader=_UniqueKeySafeLoader)
    except (yaml.YAMLError, RobotConfigError) as error:
        raise RobotConfigError(f"cannot parse robot config {source}: {error}") from error
    if not isinstance(document, Mapping):
        raise RobotConfigError(f"robot config {source} must be a YAML object")
    try:
        return RobotConfig.model_validate(document)
    except ValidationError as error:
        raise RobotConfigError(f"invalid robot config {source}: {error}") from error


def load_robot_config(path: str | Path) -> RobotConfig:
    config_path = Path(path)
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as error:
        raise RobotConfigError(f"cannot read robot config {config_path}: {error}") from error
    return _parse_yaml(text, source=str(config_path))


def available_robot_configs() -> tuple[str, ...]:
    return _BUILTIN_ROBOTS


def get_robot_config(robot_id: str) -> RobotConfig:
    if robot_id not in _BUILTIN_ROBOTS:
        raise RobotConfigError(
            f"unknown robot config {robot_id!r}; available: {', '.join(_BUILTIN_ROBOTS)}"
        )
    resource = resources.files("gqmr.configs.robots").joinpath(f"{robot_id}.yaml")
    try:
        text = resource.read_text(encoding="utf-8")
    except OSError as error:
        raise RobotConfigError(f"cannot read built-in robot config {robot_id}: {error}") from error
    config = _parse_yaml(text, source=f"built-in:{robot_id}")
    if config.id != robot_id:
        raise RobotConfigError(f"built-in robot config ID mismatch for {robot_id}")
    return config
