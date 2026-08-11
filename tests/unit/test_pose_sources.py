from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gqmr.pose.triangulation import triangulate_keypoints
from gqmr.sources.files import (
    load_deeplabcut_csv,
    load_generic_keypoints_json,
    load_generic_keypoints_csv,
    load_sleap_csv,
    load_generic_keypoints_npz,
    save_generic_keypoints_npz,
)


def test_generic_json_and_triangulation(tmp_path: Path) -> None:
    points = np.array([[[[0.0, 0.0], [0.2, 0.1]]]], dtype=np.float32)
    document = {
        "timestamps": [0.0],
        "keypoint_names": ["a", "b"],
        "instance_ids": ["animal-0"],
        "positions": points.tolist(),
        "confidence": [[[1.0, 0.9]]],
        "valid_mask": [[[True, True]]],
        "coordinate_frame": "image_pixels_x_right_y_down",
    }
    left_path = tmp_path / "left.json"
    left_path.write_text(json.dumps(document), encoding="utf-8")
    document["positions"] = (points - np.array([[[[0.1, 0.0]]]])).tolist()
    right_path = tmp_path / "right.json"
    right_path.write_text(json.dumps(document), encoding="utf-8")
    left = load_generic_keypoints_json(left_path)
    right = load_generic_keypoints_json(right_path)
    cameras = np.array(
        [
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
            [[1, 0, 0, -1], [0, 1, 0, 0], [0, 0, 1, 0]],
        ],
        dtype=np.float64,
    )
    result = triangulate_keypoints([left, right], cameras)

    assert result.positions.shape == (1, 1, 2, 3)
    assert np.all(result.valid_mask)
    assert np.allclose(result.positions[0, 0, :, 2], 10.0, atol=1e-4)


def test_deeplabcut_csv_reader(tmp_path: Path) -> None:
    source = tmp_path / "dlc.csv"
    source.write_text(
        "scorer,net,net,net,net,net,net\n"
        "bodyparts,nose,nose,nose,tail,tail,tail\n"
        "coords,x,y,likelihood,x,y,likelihood\n"
        "0,10,20,0.9,30,40,0.8\n"
        "1,11,21,0.95,31,41,0.85\n",
        encoding="utf-8",
    )
    batch = load_deeplabcut_csv(source, fps=50.0)

    assert batch.keypoint_names == ("nose", "tail")
    assert batch.positions.shape == (2, 1, 2, 2)
    assert batch.timestamps.tolist() == [0.0, 0.02]
    assert batch.confidence[1, 0, 0] == np.float32(0.95)


def test_sleap_long_csv_reader_preserves_tracks(tmp_path: Path) -> None:
    source = tmp_path / "sleap.csv"
    source.write_text(
        "frame_idx,track,node,x,y,score\n"
        "0,dog-a,nose,1,2,0.9\n"
        "0,dog-b,nose,3,4,0.8\n"
        "1,dog-a,nose,2,3,0.95\n"
        "1,dog-b,nose,4,5,0.85\n",
        encoding="utf-8",
    )
    batch = load_sleap_csv(source, fps=25.0)

    assert batch.instance_ids == ("dog-a", "dog-b")
    assert batch.positions.shape == (2, 2, 1, 2)
    assert np.all(batch.valid_mask)

    destination = tmp_path / "sleap.npz"
    save_generic_keypoints_npz(destination, batch)
    restored = load_generic_keypoints_npz(destination)
    assert restored.instance_ids == batch.instance_ids
    assert np.array_equal(restored.positions, batch.positions, equal_nan=True)


def test_generic_long_csv_3d_reader(tmp_path: Path) -> None:
    source = tmp_path / "points.csv"
    source.write_text(
        "timestamp,instance,keypoint,x,y,z,confidence\n"
        "0.0,dog,pelvis,1,2,3,1.0\n"
        "0.1,dog,pelvis,2,3,4,0.9\n",
        encoding="utf-8",
    )
    batch = load_generic_keypoints_csv(source)

    assert batch.dimensions == 3
    assert batch.coordinate_frame == "gqmr_world_x_forward_y_left_z_up"
    assert batch.positions[:, 0, 0].tolist() == [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]
