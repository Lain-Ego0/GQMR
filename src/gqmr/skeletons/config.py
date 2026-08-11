"""Strict animal skeleton configuration and semantic lookup."""

from __future__ import annotations

import hashlib
import json
from importlib import resources
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from gqmr.core.errors import GQMRError

_LEGS = ("FL", "FR", "RL", "RR")


class SkeletonConfigError(GQMRError, ValueError):
    """Raised when an animal skeleton declaration is invalid."""


class _UniqueKeySafeLoader(yaml.SafeLoader):
    pass


def _construct_mapping(
    loader: _UniqueKeySafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as error:
            raise SkeletonConfigError("YAML mapping keys must be scalar") from error
        if duplicate:
            raise SkeletonConfigError(f"duplicate YAML key {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


class KeypointDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    index: int = Field(ge=0)
    name: str = Field(min_length=1)
    parent: str | None
    side: Literal["left", "right", "center"]
    role: str = Field(min_length=1)


class AnimalSkeleton(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    id: str = Field(min_length=1)
    coordinate_frame: Literal["gqmr_world_x_forward_y_left_z_up"]
    keypoints: tuple[KeypointDefinition, ...]
    symmetry_pairs: tuple[tuple[str, str], ...]
    root_landmarks: dict[str, str]
    limb_chains: dict[str, tuple[str, ...]]
    source: dict[str, Any]

    @model_validator(mode="after")
    def validate_topology(self) -> "AnimalSkeleton":
        names = [point.name for point in self.keypoints]
        indices = [point.index for point in self.keypoints]
        if len(set(names)) != len(names):
            raise ValueError("keypoint names must be unique")
        if indices != list(range(len(indices))):
            raise ValueError("keypoint indices must be contiguous and declared in order")
        name_set = set(names)
        for point in self.keypoints:
            if point.parent is not None and point.parent not in name_set:
                raise ValueError(f"unknown parent {point.parent!r} for {point.name!r}")
            visited = {point.name}
            parent = point.parent
            while parent is not None:
                if parent in visited:
                    raise ValueError(f"parent cycle at {point.name!r}")
                visited.add(parent)
                parent = self.keypoints[names.index(parent)].parent
        paired: set[str] = set()
        for left, right in self.symmetry_pairs:
            if left not in name_set or right not in name_set or left in paired or right in paired:
                raise ValueError("symmetry pairs contain unknown or repeated names")
            paired.update((left, right))
        required_landmarks = {
            "pelvis",
            "neck",
            "left_shoulder",
            "right_shoulder",
            "left_hip",
            "right_hip",
        }
        if set(self.root_landmarks) != required_landmarks or any(
            name not in name_set for name in self.root_landmarks.values()
        ):
            raise ValueError("root_landmarks are incomplete or invalid")
        if set(self.limb_chains) != set(_LEGS):
            raise ValueError("limb_chains must contain FL, FR, RL, RR")
        if any(
            not chain or any(name not in name_set for name in chain)
            for chain in self.limb_chains.values()
        ):
            raise ValueError("limb_chains contain unknown or empty chains")
        return self

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(point.name for point in self.keypoints)

    @property
    def name_to_index(self) -> dict[str, int]:
        return {point.name: point.index for point in self.keypoints}

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _parse(text: str, source: str) -> AnimalSkeleton:
    try:
        document = yaml.load(text, Loader=_UniqueKeySafeLoader)
    except (yaml.YAMLError, SkeletonConfigError) as error:
        raise SkeletonConfigError(f"cannot parse skeleton {source}: {error}") from error
    if not isinstance(document, Mapping):
        raise SkeletonConfigError(f"skeleton {source} must be a YAML object")
    try:
        return AnimalSkeleton.model_validate(document)
    except ValidationError as error:
        raise SkeletonConfigError(f"invalid skeleton {source}: {error}") from error


def get_skeleton(skeleton_id: str = "dog-27") -> AnimalSkeleton:
    if skeleton_id != "dog-27":
        raise SkeletonConfigError("only dog-27 is currently built in")
    resource = resources.files("gqmr.configs.skeletons").joinpath("dog-27.yaml")
    skeleton = _parse(resource.read_text(encoding="utf-8"), "built-in:dog-27")
    if skeleton.id != skeleton_id:
        raise SkeletonConfigError("built-in skeleton ID mismatch")
    return skeleton


def load_skeleton(path: str | Path) -> AnimalSkeleton:
    skeleton_path = Path(path)
    try:
        text = skeleton_path.read_text(encoding="utf-8")
    except OSError as error:
        raise SkeletonConfigError(f"cannot read skeleton {skeleton_path}: {error}") from error
    return _parse(text, str(skeleton_path))
