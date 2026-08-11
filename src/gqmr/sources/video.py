"""PyAV decoding with exact presentation timestamps."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from av.error import FFmpegError

from gqmr.pose.api import KeypointBatch, PoseDataError, VideoFrameBatch


def read_video_frames(
    path: str | Path,
    *,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
    max_frames: int | None = None,
) -> VideoFrameBatch:
    if start_seconds < 0.0 or (end_seconds is not None and end_seconds <= start_seconds):
        raise PoseDataError("video time range is invalid")
    if max_frames is not None and max_frames <= 0:
        raise PoseDataError("max_frames must be positive")
    frames: list[np.ndarray] = []
    pts: list[int] = []
    time_base: Fraction | None = None
    try:
        with av.open(str(path), mode="r") as container:
            if not container.streams.video:
                raise PoseDataError("input has no video stream")
            stream = container.streams.video[0]
            for frame in container.decode(stream):
                if frame.pts is None or frame.time_base is None:
                    raise PoseDataError("decoded video frame has no PTS/time_base")
                seconds = float(frame.pts * frame.time_base)
                if seconds < start_seconds:
                    continue
                if end_seconds is not None and seconds > end_seconds:
                    break
                current_base = Fraction(frame.time_base)
                if time_base is None:
                    time_base = current_base
                if current_base != time_base:
                    raise PoseDataError("video time base changed within one stream")
                frames.append(frame.to_ndarray(format="rgb24"))
                pts.append(int(frame.pts))
                if max_frames is not None and len(frames) >= max_frames:
                    break
    except (OSError, FFmpegError) as error:
        raise PoseDataError(f"cannot decode video: {error}") from error
    if not frames or time_base is None:
        raise PoseDataError("selected video range contains no frames")
    return VideoFrameBatch(
        frames=np.stack(frames),
        pts=np.asarray(pts, dtype=np.int64),
        time_base_numerator=time_base.numerator,
        time_base_denominator=time_base.denominator,
    )


def align_keypoints_to_video(
    batch: KeypointBatch,
    video: VideoFrameBatch,
    *,
    tolerance_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(tolerance_seconds) or tolerance_seconds < 0.0:
        raise PoseDataError("alignment tolerance must be finite and non-negative")
    video_times = video.timestamps
    indices = np.searchsorted(video_times, batch.timestamps)
    indices = np.clip(indices, 0, len(video_times) - 1)
    previous = np.clip(indices - 1, 0, len(video_times) - 1)
    use_previous = np.abs(video_times[previous] - batch.timestamps) < np.abs(
        video_times[indices] - batch.timestamps
    )
    indices = np.where(use_previous, previous, indices)
    error = video_times[indices] - batch.timestamps
    if np.any(np.abs(error) > tolerance_seconds):
        raise PoseDataError("keypoint/video PTS alignment exceeds tolerance")
    return indices.astype(np.int64), error.astype(np.float64)
