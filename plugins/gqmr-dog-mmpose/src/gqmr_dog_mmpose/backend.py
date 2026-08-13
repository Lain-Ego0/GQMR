"""Optional MMPose top-down animal keypoint inference backend."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from gqmr.pose import KeypointBatch, PoseBackendInfo, VideoFrameBatch
from gqmr.pose.api import CancelToken, PoseDataError


class MMPoseDogBackend:
    """Select the highest-confidence dog instance in each frame."""

    api_version: Literal[1] = 1

    def __init__(self) -> None:
        self._pose_model: Any | None = None
        self._detector: Any | None = None
        self._keypoint_names: tuple[str, ...] = ()
        self._score_threshold = 0.15
        self._detector_confidence = 0.25
        self._detector_category_ids = [16]
        self._device = "cpu"

    def describe(self) -> PoseBackendInfo:
        return PoseBackendInfo(
            api_version=1,
            name="MMPose dog 2D",
            package="gqmr-dog-mmpose",
            package_version="0.1.0",
            skeleton_ids=("ap10k", "apt36k", "animalpose"),
            dimensions=(2,),
            multi_instance=False,
            batch_range=(1, 64),
            devices=("cpu", "cuda"),
            output_coordinate_frame="image_pixels_x_right_y_down",
        )

    def load(self, config: dict[str, Any]) -> None:
        allowed = {
            "pose_model",
            "pose_weights",
            "detector_model",
            "detector_category_ids",
            "detector_confidence",
            "device",
            "score_threshold",
        }
        unexpected = sorted(set(config) - allowed)
        if unexpected:
            raise PoseDataError(f"unknown dog-mmpose config fields: {unexpected}")
        pose_model = config.get("pose_model")
        if not isinstance(pose_model, str) or not pose_model:
            raise PoseDataError("dog-mmpose requires a non-empty pose_model")
        for key in ("pose_weights", "detector_model", "device"):
            if key in config and (not isinstance(config[key], str) or not config[key]):
                raise PoseDataError(f"dog-mmpose {key} must be a non-empty string")
        detector_model = config.get("detector_model", "yolo11n.pt")
        threshold = config.get("score_threshold", 0.15)
        if not isinstance(threshold, (int, float)) or not np.isfinite(threshold):
            raise PoseDataError("dog-mmpose score_threshold must be finite")
        if not 0.0 <= float(threshold) <= 1.0:
            raise PoseDataError("dog-mmpose score_threshold must be in [0,1]")
        detector_confidence = config.get("detector_confidence", 0.25)
        if (
            not isinstance(detector_confidence, (int, float))
            or not np.isfinite(detector_confidence)
            or not 0.0 <= float(detector_confidence) <= 1.0
        ):
            raise PoseDataError("dog-mmpose detector_confidence must be in [0,1]")
        category_ids = config.get("detector_category_ids", [16])
        if (
            not isinstance(category_ids, list)
            or not category_ids
            or any(not isinstance(value, int) or value < 0 for value in category_ids)
        ):
            raise PoseDataError(
                "dog-mmpose detector_category_ids must be a non-empty list of non-negative integers"
            )
        try:
            _apply_mmcv_compatibility_alias()
            from mmpose.apis import init_model
            from ultralytics import YOLO
        except ImportError as error:
            raise PoseDataError(
                "dog-mmpose runtime is missing; install MMPose, MMEngine, "
                "MMCV-Lite, Ultralytics and PyTorch as documented by the plugin"
            ) from error
        device = config.get("device", "cpu")
        try:
            self._pose_model = init_model(
                pose_model, config.get("pose_weights"), device=device
            )
            self._detector = YOLO(detector_model)
            self._keypoint_names = _dataset_keypoint_names(self._pose_model)
        except Exception as error:
            self._pose_model = None
            self._detector = None
            raise PoseDataError(f"cannot load dog-mmpose models: {error}") from error
        self._score_threshold = float(threshold)
        self._detector_confidence = float(detector_confidence)
        self._detector_category_ids = category_ids
        self._device = device

    def infer(self, batch: VideoFrameBatch, cancel: CancelToken) -> KeypointBatch:
        if self._pose_model is None or self._detector is None or not self._keypoint_names:
            raise PoseDataError("dog-mmpose backend is not loaded")
        if cancel.cancelled:
            raise PoseDataError("dog-mmpose inference was cancelled")
        try:
            images = [np.ascontiguousarray(frame[..., ::-1]) for frame in batch.frames]
            detections = self._detector.predict(
                source=images,
                classes=self._detector_category_ids,
                conf=self._detector_confidence,
                device=self._device,
                verbose=False,
            )
        except Exception as error:
            raise PoseDataError(f"dog-mmpose detection failed: {error}") from error
        if len(detections) != len(batch.frames):
            raise PoseDataError("dog-mmpose detector did not return one result per frame")
        keypoints = len(self._keypoint_names)
        positions = np.full((len(detections), 1, keypoints, 2), np.nan, dtype=np.float32)
        confidence = np.zeros((len(detections), 1, keypoints), dtype=np.float32)
        try:
            from mmpose.apis import inference_topdown
        except ImportError as error:
            raise PoseDataError("MMPose top-down inference API is unavailable") from error
        for frame_index, detection in enumerate(detections):
            if cancel.cancelled:
                raise PoseDataError("dog-mmpose inference was cancelled")
            boxes = getattr(detection, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            box_scores = boxes.conf.detach().cpu().numpy()
            box = boxes.xyxy[int(np.argmax(box_scores))].detach().cpu().numpy()
            try:
                samples = inference_topdown(
                    self._pose_model,
                    images[frame_index],
                    bboxes=np.asarray([box], dtype=np.float32),
                    bbox_format="xyxy",
                )
            except Exception as error:
                raise PoseDataError(f"dog-mmpose pose inference failed: {error}") from error
            if not samples:
                continue
            prediction = samples[0].pred_instances
            coordinates = np.asarray(prediction.keypoints[0], dtype=np.float32)
            scores = np.asarray(prediction.keypoint_scores[0], dtype=np.float32)
            if coordinates.shape != (keypoints, 2) or scores.shape != (keypoints,):
                raise PoseDataError("dog-mmpose returned an unexpected keypoint shape")
            positions[frame_index, 0] = coordinates
            confidence[frame_index, 0] = scores
        valid = (
            np.all(np.isfinite(positions), axis=-1)
            & np.isfinite(confidence)
            & (confidence >= self._score_threshold)
        )
        positions[~valid] = np.nan
        confidence[~np.isfinite(confidence)] = 0.0
        confidence = np.clip(confidence, 0.0, 1.0)
        return KeypointBatch(
            timestamps=batch.timestamps,
            keypoint_names=self._keypoint_names,
            instance_ids=("dog-0",),
            positions=positions,
            confidence=confidence,
            valid_mask=valid,
            coordinate_frame="image_pixels_x_right_y_down",
            metadata={
                "backend": "dog-mmpose",
                "detector": "ultralytics_yolo",
                "selection": "highest_detector_confidence",
                "detector_confidence": self._detector_confidence,
                "detector_category_ids": self._detector_category_ids,
                "score_threshold": self._score_threshold,
            },
        )

    def close(self) -> None:
        self._pose_model = None
        self._detector = None
        self._keypoint_names = ()


def _dataset_keypoint_names(model: Any) -> tuple[str, ...]:
    metadata = getattr(model, "dataset_meta", None)
    if not isinstance(metadata, dict):
        raise PoseDataError("MMPose model has no readable dataset metadata")
    mapping = metadata.get("keypoint_id2name")
    if isinstance(mapping, dict):
        try:
            names = tuple(str(mapping[index]) for index in sorted(mapping, key=int))
        except (KeyError, TypeError, ValueError) as error:
            raise PoseDataError("MMPose keypoint_id2name metadata is invalid") from error
    else:
        names = tuple(str(name) for name in metadata.get("keypoint_names", ()))
    if not names or len(set(names)) != len(names) or any(not name for name in names):
        raise PoseDataError("MMPose model keypoint names are missing or invalid")
    return names


def _apply_mmcv_compatibility_alias() -> None:
    """Allow the official Python 3.12 MMCV 2.2 wheel with MMDetection 3.2.

    MMDetection 3.2 rejects the 2.2.0 version string before importing, although
    RTMPose uses APIs and compiled operators that remain binary compatible. The
    alias is process-local and only affects OpenMMLab's import-time assertion.
    """

    try:
        import mmcv
    except ImportError:
        return
    if getattr(mmcv, "__version__", None) == "2.2.0":
        mmcv.__version__ = "2.1.0"
