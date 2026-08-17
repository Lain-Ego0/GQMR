from __future__ import annotations

from pathlib import Path

import av
import numpy as np

from gqmr.pose import (
    KeypointBatch,
    keypoint_batch_to_animal_motion,
    lift_ap10k_monocular_to_dog27,
)
from gqmr.pose.api import PoseBackendInfo, PoseDataError
from gqmr.pose.video_inference import infer_video_with_backend
from gqmr.skeletons import get_skeleton
from gqmr.sources.video import (
    align_keypoints_to_video,
    iter_video_frame_batches,
    read_video_frames,
)


def _write_video(destination: Path, *, frames: int = 5, fps: int = 20) -> None:
    with av.open(str(destination), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = 32
        stream.height = 24
        stream.pix_fmt = "yuv420p"
        for index in range(frames):
            image = np.full((24, 32, 3), index * 20, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


class _FixturePoseBackend:
    def describe(self) -> PoseBackendInfo:
        return PoseBackendInfo(
            api_version=1,
            name="fixture-dog-2d",
            package="tests",
            package_version="1",
            skeleton_ids=("fixture-2",),
            dimensions=(2,),
            multi_instance=False,
            batch_range=(4, 8),
            devices=("cpu",),
            output_coordinate_frame="image_pixels_x_right_y_down",
        )

    def infer(self, batch, cancel) -> KeypointBatch:
        positions = np.zeros((len(batch.frames), 1, 2, 2), dtype=np.float32)
        positions[:, 0, 0, 0] = np.arange(len(batch.frames))
        return KeypointBatch(
            timestamps=batch.timestamps,
            keypoint_names=("nose", "tail_base"),
            instance_ids=("dog-0",),
            positions=positions,
            confidence=np.ones((len(batch.frames), 1, 2), dtype=np.float32),
            valid_mask=np.ones((len(batch.frames), 1, 2), dtype=bool),
            coordinate_frame="image_pixels_x_right_y_down",
            metadata={"fixture": True},
        )


def test_keypoint_batch_to_animal_motion() -> None:
    skeleton = get_skeleton("dog-27")
    positions = np.zeros((3, 1, 27, 3), dtype=np.float32)
    batch = KeypointBatch(
        timestamps=[10.0, 10.1, 10.2],
        keypoint_names=skeleton.names,
        instance_ids=("dog",),
        positions=positions,
        confidence=np.ones((3, 1, 27)),
        valid_mask=np.ones((3, 1, 27), dtype=bool),
        coordinate_frame="gqmr_world_x_forward_y_left_z_up",
        metadata={"backend": "fixture"},
    )
    motion = keypoint_batch_to_animal_motion(batch, instance_id="dog")

    assert motion.timestamps.tolist() == [0.0, 0.09999999999999964, 0.1999999999999993]
    assert motion.keypoint_names == skeleton.names
    assert motion.metadata["source"]["instance_id"] == "dog"


def test_lift_ap10k_monocular_to_dog27() -> None:
    names = (
        "L_Eye",
        "R_Eye",
        "Nose",
        "Neck",
        "Root of tail",
        "L_Shoulder",
        "L_Elbow",
        "L_F_Paw",
        "R_Shoulder",
        "R_Elbow",
        "R_F_Paw",
        "L_Hip",
        "L_Knee",
        "L_B_Paw",
        "R_Hip",
        "R_Knee",
        "R_B_Paw",
    )
    frame = np.array(
        [
            [85, 55],
            [88, 57],
            [70, 65],
            [120, 90],
            [300, 100],
            [130, 105],
            [145, 145],
            [155, 190],
            [135, 108],
            [155, 150],
            [175, 190],
            [270, 110],
            [260, 150],
            [245, 190],
            [275, 112],
            [290, 155],
            [305, 190],
        ],
        dtype=np.float32,
    )
    positions = np.stack((frame, frame + [2, 0], frame + [4, 0]))[:, None]
    batch = KeypointBatch(
        timestamps=[3.0, 3.1, 3.2],
        keypoint_names=names,
        instance_ids=("dog",),
        positions=positions,
        confidence=np.ones((3, 1, len(names))),
        valid_mask=np.ones((3, 1, len(names)), dtype=bool),
        coordinate_frame="image_pixels_x_right_y_down",
        metadata={"fixture": True},
    )

    motion = lift_ap10k_monocular_to_dog27(batch, smoothing_window=1)
    skeleton = get_skeleton("dog-27")

    assert motion.positions.shape == (3, 27, 3)
    assert motion.keypoint_names == skeleton.names
    assert motion.timestamps.tolist() == [0.0, 0.10000000000000009, 0.20000000000000018]
    assert np.all(np.isfinite(motion.positions))
    assert np.all(motion.frame_valid)
    assert motion.metadata["source"]["format"] == (
        "experimental_monocular_ap10k_lift_v1"
    )
    assert motion.metadata["source"]["parameters"]["facing"] == "left"


def test_pyav_video_pts_and_keypoint_alignment(tmp_path: Path) -> None:
    destination = tmp_path / "video.mp4"
    _write_video(destination)
    video = read_video_frames(destination)
    batch = KeypointBatch(
        timestamps=video.timestamps,
        keypoint_names=("point",),
        instance_ids=("animal",),
        positions=np.zeros((5, 1, 1, 2)),
        confidence=np.ones((5, 1, 1)),
        valid_mask=np.ones((5, 1, 1), dtype=bool),
        coordinate_frame="image_pixels_x_right_y_down",
        metadata={},
    )
    indices, errors = align_keypoints_to_video(batch, video, tolerance_seconds=1e-9)

    assert video.frames.shape == (5, 24, 32, 3)
    assert indices.tolist() == [0, 1, 2, 3, 4]
    assert np.all(errors == 0.0)


def test_video_batches_preserve_pts_and_frame_limit(tmp_path: Path) -> None:
    destination = tmp_path / "video.mp4"
    _write_video(destination, frames=7)

    batches = list(
        iter_video_frame_batches(destination, batch_size=3, max_frames=5)
    )

    assert [len(batch.frames) for batch in batches] == [3, 2]
    assert np.concatenate([batch.pts for batch in batches]).tolist() == [
        0,
        512,
        1024,
        1536,
        2048,
    ]


def test_video_pose_inference_batches_pads_and_trims(tmp_path: Path) -> None:
    destination = tmp_path / "dog.mp4"
    _write_video(destination, frames=6)
    backend = _FixturePoseBackend()

    result = infer_video_with_backend(
        backend,
        destination,
        backend_config={"device": "cpu"},
        batch_size=4,
    )

    assert len(result.timestamps) == 6
    assert result.positions.shape == (6, 1, 2, 2)
    assert result.keypoint_names == ("nose", "tail_base")
    assert result.metadata["format"] == "gqmr_video_pose_v1"
    assert result.metadata["inference"] == {"batch_size": 4, "batches": 2}
    assert result.metadata["source_video"]["size_bytes"] == destination.stat().st_size
    assert len(result.metadata["source_video"]["sha256"]) == 64


def test_video_pose_inference_rejects_unsupported_batch_size(tmp_path: Path) -> None:
    destination = tmp_path / "dog.mp4"
    _write_video(destination)

    try:
        infer_video_with_backend(_FixturePoseBackend(), destination, batch_size=2)
    except PoseDataError as error:
        assert "backend range" in str(error)
    else:
        raise AssertionError("unsupported batch size was accepted")
