"""Run opt-in release performance and long-stream benchmarks."""

from __future__ import annotations

import argparse
import json
import threading
import time

import numpy as np

from gqmr.retarget import retarget_fast, retarget_high_quality
from gqmr.robots import load_robot_model
from gqmr.stream import GQMRPublisher, GQMRRecorder, build_robot_welcome
from gqmr.synthetic import generate_dog27_motion


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--stream-seconds", type=float, default=10.0)
    parser.add_argument("--stream-hz", type=float, default=200.0)
    args = parser.parse_args()
    robot = load_robot_model("unitree-go2", cache_dir=args.cache_dir)
    animal = generate_dog27_motion("trot", duration=2.0, fps=60.0)
    started = time.perf_counter()
    retarget_fast(animal, robot)
    fast_elapsed = time.perf_counter() - started
    started = time.perf_counter()
    retarget_high_quality(animal, robot)
    quality_elapsed = time.perf_counter() - started

    frame_count = int(round(args.stream_seconds * args.stream_hz))
    publisher = GQMRPublisher(
        build_robot_welcome(robot, nominal_hz=args.stream_hz),
        endpoint="tcp://127.0.0.1:*",
        ring_size=max(4096, frame_count),
    ).start()
    recorder = GQMRRecorder(publisher.bound_endpoint, credit=256)
    recorder.connect()
    recorder.validate_robot(robot)
    interval = 1.0 / args.stream_hz

    def publish() -> None:
        target = time.perf_counter()
        for _ in range(frame_count):
            publisher.publish({"qpos": robot.data.qpos, "qvel": robot.data.qvel})
            target += interval
            remaining = target - time.perf_counter()
            if remaining > 0.0:
                time.sleep(remaining)

    thread = threading.Thread(target=publish)
    started = time.perf_counter()
    thread.start()
    capture = recorder.record_frames(
        frame_count, timeout_ms=int((args.stream_seconds + 10.0) * 1000)
    )
    thread.join()
    stream_elapsed = time.perf_counter() - started
    recorder.close()
    publisher.close()
    result = {
        "fast_frames_per_second": animal.frame_count / fast_elapsed,
        "high_quality_frames_per_second": animal.frame_count / quality_elapsed,
        "stream_requested_frames": frame_count,
        "stream_received_frames": len(capture.sequence),
        "stream_gaps": len(capture.gaps),
        "stream_elapsed_seconds": stream_elapsed,
        "stream_effective_hz": frame_count / stream_elapsed,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
