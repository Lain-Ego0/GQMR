from __future__ import annotations

from pathlib import Path

import pytest

from gqmr.skeletons import SkeletonConfigError, get_skeleton, load_skeleton


def test_builtin_dog27_freezes_historical_indices_and_topology() -> None:
    skeleton = get_skeleton("dog-27")

    assert len(skeleton.keypoints) == 27
    assert skeleton.name_to_index["pelvis"] == 0
    assert skeleton.name_to_index["pelvis_duplicate"] == 1
    assert skeleton.name_to_index["neck"] == 3
    assert skeleton.name_to_index["left_shoulder"] == 6
    assert skeleton.name_to_index["left_front_toe"] == 10
    assert skeleton.name_to_index["right_shoulder"] == 11
    assert skeleton.name_to_index["right_front_toe"] == 15
    assert skeleton.name_to_index["left_hip"] == 16
    assert skeleton.name_to_index["left_hind_toe"] == 19
    assert skeleton.name_to_index["right_hip"] == 20
    assert skeleton.name_to_index["right_hind_toe"] == 23
    assert skeleton.limb_chains["FL"][0] == "left_shoulder"
    assert skeleton.limb_chains["RR"][-1] == "right_hind_toe"
    assert len(skeleton.symmetry_pairs) == 9
    assert len(skeleton.sha256) == 64
    assert skeleton.sha256 == get_skeleton("dog-27").sha256


def test_skeleton_loader_rejects_duplicate_yaml_keys(tmp_path: Path) -> None:
    source = tmp_path / "duplicate.yaml"
    source.write_text("schema_version: 1\nschema_version: 1\n", encoding="utf-8")

    with pytest.raises(SkeletonConfigError, match="duplicate YAML key"):
        load_skeleton(source)


def test_skeleton_loader_rejects_parent_cycles(tmp_path: Path) -> None:
    source = tmp_path / "cycle.yaml"
    source.write_text(
        """schema_version: 1
id: cycle
coordinate_frame: gqmr_world_x_forward_y_left_z_up
keypoints:
  - {index: 0, name: a, parent: b, side: center, role: root}
  - {index: 1, name: b, parent: a, side: center, role: root}
symmetry_pairs: []
root_landmarks: {}
limb_chains: {}
source: {}
""",
        encoding="utf-8",
    )

    with pytest.raises(SkeletonConfigError, match="parent cycle"):
        load_skeleton(source)
