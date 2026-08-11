"""Validated Stream Protocol v1 recorder and RobotMotion conversion."""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np
import zmq

from gqmr import __version__
from gqmr.assets import get_asset_spec
from gqmr.core.derivatives import angular_velocity_world, linear_velocity
from gqmr.core.motion import RobotMotion, SolverStatus
from gqmr.robots import RobotModel
from gqmr.stream.model import build_robot_welcome
from gqmr.stream.protocol import (
    PROTOCOL,
    VERSION,
    StreamProtocolError,
    decode_frame,
    decode_header,
    encode_header,
)


@dataclass(frozen=True, slots=True)
class StreamCapture:
    welcome: dict[str, Any]
    sequence: np.ndarray
    timestamp_ns: np.ndarray
    arrays: dict[str, np.ndarray]
    gaps: tuple[dict[str, Any], ...]

    def to_robot_motion(self, robot: RobotModel) -> RobotMotion:
        if self.welcome["model_sha256"] != robot.config.model_sha256:
            raise StreamProtocolError("capture model hash does not match robot")
        if "qpos" not in self.arrays:
            raise StreamProtocolError("capture does not contain qpos")
        qpos = self.arrays["qpos"]
        if len(qpos) == 0:
            raise StreamProtocolError("capture has no frames")
        root_adr = robot.root_qpos_address
        root_position = qpos[:, root_adr : root_adr + 3].copy()
        root_rotation = qpos[:, root_adr + 3 : root_adr + 7].copy()
        dof_position = qpos[:, robot.qpos_addresses].copy()
        finite = (
            np.all(np.isfinite(root_position), axis=1)
            & np.all(np.isfinite(root_rotation), axis=1)
            & np.all(np.isfinite(dof_position), axis=1)
            & (np.abs(np.linalg.norm(root_rotation, axis=1) - 1.0) < 1e-5)
            & np.all(dof_position >= robot.joint_ranges[:, 0], axis=1)
            & np.all(dof_position <= robot.joint_ranges[:, 1], axis=1)
        )
        filled_root = root_position.copy()
        filled_rotation = root_rotation.copy()
        filled_dof = dof_position.copy()
        for frame in range(len(qpos)):
            if not finite[frame]:
                if frame:
                    filled_root[frame] = filled_root[frame - 1]
                    filled_rotation[frame] = filled_rotation[frame - 1]
                    filled_dof[frame] = filled_dof[frame - 1]
                else:
                    filled_root[frame] = robot.config.default_root_position
                    filled_rotation[frame] = robot.config.default_root_rotation
                    filled_dof[frame] = robot.config.default_dof_position
        timestamps = (self.timestamp_ns - self.timestamp_ns[0]).astype(np.float64) * 1e-9
        if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0.0):
            raise StreamProtocolError("capture timestamps are not strictly increasing")

        def derivative(values: np.ndarray) -> np.ndarray:
            if len(timestamps) == 1:
                return np.zeros_like(values)
            if len(timestamps) == 2:
                slope = (values[1] - values[0]) / (timestamps[1] - timestamps[0])
                return np.stack((slope, slope))
            return linear_velocity(timestamps, values)

        if len(timestamps) >= 3:
            angular = angular_velocity_world(timestamps, filled_rotation)
        else:
            angular = np.zeros((len(timestamps), 3), dtype=np.float64)
        digest = hashlib.sha256()
        digest.update(self.sequence.astype("<i8").tobytes())
        digest.update(self.timestamp_ns.astype("<i8").tobytes())
        digest.update(qpos.astype("<f8").tobytes())
        asset = get_asset_spec(robot.config.asset_id)
        status = np.where(finite, SolverStatus.OK, SolverStatus.NUMERICAL_ERROR).astype(np.int16)
        return RobotMotion(
            timestamps=timestamps,
            dof_names=robot.config.dof_order,
            root_position=filled_root.astype(np.float32),
            root_rotation=filled_rotation.astype(np.float32),
            dof_position=filled_dof.astype(np.float32),
            root_linear_velocity=derivative(filled_root).astype(np.float32),
            root_angular_velocity=angular.astype(np.float32),
            dof_velocity=derivative(filled_dof).astype(np.float32),
            foot_contact_probability=np.full((len(qpos), 4), np.nan, dtype=np.float32),
            frame_valid=finite,
            solver_status=status,
            solver_residual=np.where(finite, 0.0, np.nan).astype(np.float32),
            metadata={
                "coordinate_frame": "gqmr_world_x_forward_y_left_z_up",
                "quaternion_order": "wxyz",
                "root_velocity_frame": "world",
                "model_id": robot.config.id,
                "model_source_commit": asset.commit,
                "model_sha256": robot.config.model_sha256,
                "robot_config_sha256": robot.config.sha256,
                "contact_order": ["FL", "FR", "RL", "RR"],
                "source_motion_sha256": digest.hexdigest(),
                "retarget_config": {
                    "mode": "mujoco_stream_v1",
                    "session_id": self.welcome["session_id"],
                    "gaps": list(self.gaps),
                },
                "created_by": {"gqmr_version": __version__},
            },
        )


class GQMRRecorder:
    def __init__(
        self,
        endpoint: str,
        *,
        requested_fields: tuple[str, ...] = ("qpos", "qvel"),
        credit: int = 256,
        context: zmq.Context | None = None,
        curve_server_key: bytes | None = None,
        curve_public_key: bytes | None = None,
        curve_secret_key: bytes | None = None,
    ) -> None:
        if (curve_public_key is None) != (curve_secret_key is None):
            raise StreamProtocolError("CurveZMQ client requires both public and secret keys")
        for key in (curve_server_key, curve_public_key, curve_secret_key):
            if key is not None and len(key) != 40:
                raise StreamProtocolError("CurveZMQ keys must be 40-byte Z85 values")
        self.endpoint = endpoint
        self.requested_fields = requested_fields
        self.credit = credit
        self._context = context or zmq.Context.instance()
        self.socket: zmq.Socket | None = None
        self.welcome: dict[str, Any] | None = None
        self._frames: list[tuple[dict[str, Any], dict[str, np.ndarray]]] = []
        self._gaps: list[dict[str, Any]] = []
        self._last_seq = -1
        self._unacked = 0
        self.curve_server_key = curve_server_key
        self.curve_public_key = curve_public_key
        self.curve_secret_key = curve_secret_key

    @property
    def gaps(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._gaps)

    def connect(self, *, timeout_ms: int = 3000) -> dict[str, Any]:
        socket = self._context.socket(zmq.DEALER)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.IDENTITY, str(uuid.uuid4()).encode("ascii"))
        if self.curve_server_key is not None:
            public = self.curve_public_key
            secret = self.curve_secret_key
            if public is None or secret is None:
                public, secret = zmq.curve_keypair()
            socket.curve_serverkey = self.curve_server_key
            socket.curve_publickey = public
            socket.curve_secretkey = secret
        socket.connect(self.endpoint)
        socket.send_multipart(
            encode_header(
                "HELLO",
                {
                    "protocol": PROTOCOL,
                    "version": VERSION,
                    "client_id": str(uuid.uuid4()),
                    "requested_fields": list(self.requested_fields),
                    "credit": self.credit,
                },
            )
        )
        if not socket.poll(timeout_ms, zmq.POLLIN):
            socket.close(0)
            raise StreamProtocolError("stream handshake timed out")
        parts = socket.recv_multipart()
        message_type, header = decode_header(parts)
        if message_type == "ERROR":
            socket.close(0)
            raise StreamProtocolError(header.get("message", "publisher rejected HELLO"))
        if message_type != "WELCOME" or header.get("protocol") != PROTOCOL or header.get("version") != VERSION:
            socket.close(0)
            raise StreamProtocolError("invalid WELCOME")
        self.socket = socket
        self.welcome = header
        return header

    def validate_robot(self, robot: RobotModel) -> None:
        if self.welcome is None:
            raise StreamProtocolError("recorder is not connected")
        if self.welcome.get("model_sha256") != robot.config.model_sha256:
            raise StreamProtocolError("WELCOME model hash does not match robot")
        expected = build_robot_welcome(robot, nominal_hz=1.0)
        for field in ("qpos_layout", "qvel_layout"):
            if self.welcome.get(field) != expected[field]:
                raise StreamProtocolError(f"WELCOME {field} does not match robot")

    def receive(self, *, timeout_ms: int = 3000) -> bool:
        if self.socket is None or self.welcome is None:
            raise StreamProtocolError("recorder is not connected")
        if not self.socket.poll(timeout_ms, zmq.POLLIN):
            raise StreamProtocolError("stream receive timed out")
        parts = self.socket.recv_multipart()
        message_type, header = decode_header(parts)
        if message_type == "FRAME":
            header, arrays = decode_frame(parts)
            if header.get("session_id") != self.welcome["session_id"]:
                raise StreamProtocolError("FRAME session mismatch")
            seq = int(header.get("seq", -1))
            if seq <= self._last_seq:
                raise StreamProtocolError("FRAME sequence is not strictly increasing")
            if self._last_seq >= 0 and seq != self._last_seq + 1:
                raise StreamProtocolError("FRAME skipped without GAP")
            missing = set(self.requested_fields) - set(arrays)
            if missing:
                raise StreamProtocolError(f"FRAME is missing requested arrays: {sorted(missing)}")
            for field, layout_name in (("qpos", "qpos_layout"), ("qvel", "qvel_layout")):
                if field not in arrays:
                    continue
                layout = self.welcome.get(layout_name)
                if not isinstance(layout, list) or not layout:
                    raise StreamProtocolError(f"WELCOME {layout_name} is invalid")
                expected_size = max(int(item["adr"]) + int(item["size"]) for item in layout)
                if arrays[field].shape != (expected_size,):
                    raise StreamProtocolError(f"FRAME {field} shape does not match WELCOME")
            self._frames.append((header, arrays))
            self._last_seq = seq
            self._unacked += 1
            if self._unacked >= 64:
                self._ack()
            return True
        if message_type == "GAP":
            self._gaps.append(header)
            self._last_seq = int(header["last_missing"])
            return False
        if message_type == "ERROR":
            raise StreamProtocolError(header.get("message", "publisher error"))
        if message_type in {"HEARTBEAT", "BYE"}:
            return False
        raise StreamProtocolError(f"unexpected stream message {message_type}")

    def record_frames(self, count: int, *, timeout_ms: int = 3000) -> StreamCapture:
        if count <= 0:
            raise StreamProtocolError("record frame count must be positive")
        deadline = time.monotonic() + timeout_ms / 1000.0
        while len(self._frames) < count:
            remaining = max(1, int((deadline - time.monotonic()) * 1000))
            if remaining <= 1 and time.monotonic() >= deadline:
                raise StreamProtocolError("recording timed out")
            self.receive(timeout_ms=remaining)
        self._ack()
        selected = self._frames[:count]
        sequence = np.asarray([item[0]["seq"] for item in selected], dtype=np.int64)
        timestamp = np.asarray([item[0]["timestamp_ns"] for item in selected], dtype=np.int64)
        names = selected[0][1].keys()
        arrays = {name: np.stack([item[1][name] for item in selected]) for name in names}
        return StreamCapture(dict(self.welcome), sequence, timestamp, arrays, tuple(self._gaps))

    def _ack(self) -> None:
        if self.socket is None or self.welcome is None or self._unacked == 0:
            return
        self.socket.send_multipart(
            encode_header(
                "ACK",
                {
                    "session_id": self.welcome["session_id"],
                    "ack_seq": self._last_seq,
                    "credit": self._unacked,
                },
            )
        )
        self._unacked = 0

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close(0)
            self.socket = None

    def __enter__(self) -> "GQMRRecorder":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
