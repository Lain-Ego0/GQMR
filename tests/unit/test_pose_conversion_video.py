from __future__ import annotations

from pathlib import Path

import av
import numpy as np

from gqmr.pose import KeypointBatch, keypoint_batch_to_animal_motion
from gqmr.skeletons import get_skeleton
from gqmr.sources.video import align_keypoints_to_video, read_video_frames


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


def test_pyav_video_pts_and_keypoint_alignment(tmp_path: Path) -> None:
    destination = tmp_path / "video.mp4"
    with av.open(str(destination), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=20)
        stream.width = 32
        stream.height = 24
        stream.pix_fmt = "yuv420p"
        for index in range(5):
            image = np.full((24, 32, 3), index * 20, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
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
