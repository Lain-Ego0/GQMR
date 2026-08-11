"""Non-blocking threaded Stream Protocol v1 publisher."""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import zmq

from gqmr.stream.protocol import (
    PROTOCOL,
    VERSION,
    StreamProtocolError,
    decode_header,
    encode_frame,
    encode_header,
)


@dataclass(frozen=True, slots=True)
class _Frame:
    seq: int
    timestamp_ns: int
    wall_time_ns: int | None
    arrays: dict[str, np.ndarray]


@dataclass(slots=True)
class _Client:
    identity: bytes
    credit: int
    next_seq: int
    last_send_ns: int


class GQMRPublisher:
    """Copy snapshots into a bounded ring; a private thread owns the ROUTER socket."""

    def __init__(
        self,
        welcome: Mapping[str, Any],
        *,
        endpoint: str = "tcp://127.0.0.1:5570",
        ring_size: int = 4096,
        context: zmq.Context | None = None,
        curve_public_key: bytes | None = None,
        curve_secret_key: bytes | None = None,
    ) -> None:
        if ring_size <= 0:
            raise StreamProtocolError("ring_size must be positive")
        if (curve_public_key is None) != (curve_secret_key is None):
            raise StreamProtocolError("CurveZMQ publisher requires both public and secret keys")
        if curve_public_key is not None and (
            len(curve_public_key) != 40 or len(curve_secret_key or b"") != 40
        ):
            raise StreamProtocolError("CurveZMQ keys must be 40-byte Z85 values")
        if endpoint.startswith("tcp://"):
            host = endpoint[6:].rsplit(":", 1)[0]
            if host not in {"127.0.0.1", "localhost", "::1"} and (
                curve_public_key is None or curve_secret_key is None
            ):
                raise StreamProtocolError(
                    "non-loopback publishing requires CurveZMQ configuration"
                )
        self.welcome = dict(welcome)
        self.welcome.update(
            {"protocol": PROTOCOL, "version": VERSION, "session_id": str(uuid.uuid4())}
        )
        self.endpoint = endpoint
        self.ring: deque[_Frame] = deque(maxlen=ring_size)
        self._condition = threading.Condition()
        self._next_seq = 0
        self._stop = threading.Event()
        self._context = context or zmq.Context.instance()
        self._thread: threading.Thread | None = None
        self.bound_endpoint: str | None = None
        self.curve_public_key = curve_public_key
        self.curve_secret_key = curve_secret_key

    def start(self) -> "GQMRPublisher":
        if self._thread is not None:
            return self
        self._thread = threading.Thread(target=self._network_loop, name="gqmr-publisher", daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 5.0
        with self._condition:
            while self.bound_endpoint is None and time.monotonic() < deadline:
                self._condition.wait(timeout=0.05)
        if self.bound_endpoint is None:
            raise StreamProtocolError("publisher did not bind within 5 seconds")
        return self

    def publish(
        self,
        arrays: Mapping[str, np.ndarray],
        *,
        timestamp_ns: int | None = None,
        wall_time_ns: int | None = None,
    ) -> int:
        copied = {
            name: np.ascontiguousarray(value, dtype="<f8").copy()
            for name, value in arrays.items()
        }
        timestamp = time.monotonic_ns() if timestamp_ns is None else int(timestamp_ns)
        with self._condition:
            seq = self._next_seq
            self._next_seq += 1
            self.ring.append(_Frame(seq, timestamp, wall_time_ns, copied))
            self._condition.notify_all()
        return seq

    def close(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def __enter__(self) -> "GQMRPublisher":
        return self.start()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _network_loop(self) -> None:
        socket = self._context.socket(zmq.ROUTER)
        socket.setsockopt(zmq.LINGER, 0)
        if self.curve_public_key is not None and self.curve_secret_key is not None:
            socket.curve_publickey = self.curve_public_key
            socket.curve_secretkey = self.curve_secret_key
            socket.curve_server = True
        socket.bind(self.endpoint)
        endpoint = socket.getsockopt(zmq.LAST_ENDPOINT).decode("ascii")
        with self._condition:
            self.bound_endpoint = endpoint
            self._condition.notify_all()
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        clients: dict[bytes, _Client] = {}
        try:
            while not self._stop.is_set():
                events = dict(poller.poll(10))
                if socket in events:
                    message = socket.recv_multipart()
                    if len(message) < 3:
                        continue
                    identity, parts = message[0], message[1:]
                    try:
                        message_type, header = decode_header(parts)
                        if message_type == "HELLO":
                            if header.get("protocol") != PROTOCOL or header.get("version") != VERSION:
                                raise StreamProtocolError("HELLO protocol/version mismatch")
                            credit = int(header.get("credit", 0))
                            if credit <= 0 or credit > 65536:
                                raise StreamProtocolError("HELLO credit is invalid")
                            with self._condition:
                                next_seq = self.ring[0].seq if self.ring else self._next_seq
                            clients[identity] = _Client(identity, credit, next_seq, time.monotonic_ns())
                            socket.send_multipart([identity, *encode_header("WELCOME", self.welcome)])
                        elif message_type == "ACK" and identity in clients:
                            client = clients[identity]
                            if header.get("session_id") != self.welcome["session_id"]:
                                raise StreamProtocolError("ACK session mismatch")
                            credit = int(header.get("credit", 0))
                            if credit < 0 or credit > 65536:
                                raise StreamProtocolError("ACK credit is invalid")
                            client.credit += credit
                    except StreamProtocolError as error:
                        socket.send_multipart(
                            [identity, *encode_header("ERROR", {"code": "PROTOCOL", "message": str(error)})]
                        )
                with self._condition:
                    frames = tuple(self.ring)
                    next_global = self._next_seq
                oldest = frames[0].seq if frames else next_global
                frame_by_seq = {frame.seq: frame for frame in frames}
                now = time.monotonic_ns()
                for client in list(clients.values()):
                    if client.next_seq < oldest:
                        socket.send_multipart(
                            [
                                client.identity,
                                *encode_header(
                                    "GAP",
                                    {
                                        "session_id": self.welcome["session_id"],
                                        "first_missing": client.next_seq,
                                        "last_missing": oldest - 1,
                                        "reason": "ring_overflow",
                                    },
                                ),
                            ]
                        )
                        client.next_seq = oldest
                    while client.credit > 0 and client.next_seq in frame_by_seq:
                        frame = frame_by_seq[client.next_seq]
                        header = {
                            "session_id": self.welcome["session_id"],
                            "seq": frame.seq,
                            "timestamp_ns": frame.timestamp_ns,
                            "wall_time_ns": frame.wall_time_ns,
                        }
                        socket.send_multipart(
                            [client.identity, *encode_frame(header, frame.arrays)], copy=False
                        )
                        client.credit -= 1
                        client.next_seq += 1
                        client.last_send_ns = now
                    if now - client.last_send_ns >= 1_000_000_000:
                        socket.send_multipart(
                            [
                                client.identity,
                                *encode_header(
                                    "HEARTBEAT",
                                    {
                                        "session_id": self.welcome["session_id"],
                                        "last_seq": client.next_seq - 1,
                                    },
                                ),
                            ]
                        )
                        client.last_send_ns = now
        finally:
            for client in clients.values():
                try:
                    socket.send_multipart(
                        [
                            client.identity,
                            *encode_header(
                                "BYE",
                                {
                                    "session_id": self.welcome["session_id"],
                                    "last_seq": client.next_seq - 1,
                                },
                            ),
                        ],
                        flags=zmq.NOBLOCK,
                    )
                except zmq.ZMQError:
                    pass
            socket.close(0)
