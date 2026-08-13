"""Pose backend API v1 and canonical keypoint batches."""

from gqmr.pose.api import (
    KeypointBatch,
    PoseBackendInfo,
    PoseBackendV1,
    VideoFrameBatch,
    discover_pose_backends,
)
from gqmr.pose.conversion import keypoint_batch_to_animal_motion
from gqmr.pose.triangulation import triangulate_keypoints
from gqmr.pose.video_inference import infer_video_with_backend

__all__ = [
    "KeypointBatch",
    "PoseBackendInfo",
    "PoseBackendV1",
    "VideoFrameBatch",
    "discover_pose_backends",
    "keypoint_batch_to_animal_motion",
    "infer_video_with_backend",
    "triangulate_keypoints",
]
