"""PyAV decoding with exact presentation timestamps."""

from __future__ import annotations

from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from av.error import FFmpegError

from gqmr.pose.api import KeypointBatch, PoseDataError, VideoFrameBatch


def iter_video_frame_batches(
    path: str | Path,
    *,
    batch_size: int = 16,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
    max_frames: int | None = None,
) -> Iterator[VideoFrameBatch]:
    """Decode a video incrementally while preserving its presentation timestamps."""

    if batch_size <= 0:
        raise PoseDataError("video batch_size must be positive")
    if start_seconds < 0.0 or (end_seconds is not None and end_seconds <= start_seconds):
        raise PoseDataError("video time range is invalid")
    if max_frames is not None and max_frames <= 0:
        raise PoseDataError("max_frames must be positive")
    frames: list[np.ndarray] = []
    pts: list[int] = []
    time_base: Fraction | None = None
    decoded_frames = 0
    frame_shape: tuple[int, ...] | None = None

    def make_batch() -> VideoFrameBatch:
        assert time_base is not None
        return VideoFrameBatch(
            frames=np.stack(frames),
            pts=np.asarray(pts, dtype=np.int64),
            time_base_numerator=time_base.numerator,
            time_base_denominator=time_base.denominator,
        )

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
                image = frame.to_ndarray(format="rgb24")
                if frame_shape is None:
                    frame_shape = image.shape
                if image.shape != frame_shape:
                    raise PoseDataError("video frame dimensions changed within one stream")
                frames.append(image)
                pts.append(int(frame.pts))
                decoded_frames += 1
                if len(frames) == batch_size:
                    yield make_batch()
                    frames.clear()
                    pts.clear()
                if max_frames is not None and decoded_frames >= max_frames:
                    break
    except (OSError, FFmpegError) as error:
        raise PoseDataError(f"cannot decode video: {error}") from error
    if frames:
        yield make_batch()
    elif decoded_frames == 0:
        raise PoseDataError("selected video range contains no frames")


def read_video_frames(
    path: str | Path,
    *,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
    max_frames: int | None = None,
) -> VideoFrameBatch:
    batches = list(
        iter_video_frame_batches(
            path,
            batch_size=max_frames or 4096,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            max_frames=max_frames,
        )
    )
    return VideoFrameBatch(
        frames=np.concatenate([batch.frames for batch in batches], axis=0),
        pts=np.concatenate([batch.pts for batch in batches]),
        time_base_numerator=batches[0].time_base_numerator,
        time_base_denominator=batches[0].time_base_denominator,
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
