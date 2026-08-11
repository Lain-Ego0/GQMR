"""Strict multipart encoding for MuJoCo Stream Protocol v1."""

from __future__ import annotations

import json
from typing import Any, Mapping

import numpy as np

from gqmr.core.errors import GQMRError

PROTOCOL = "gqmr.mujoco_stream"
VERSION = 1
MAX_HEADER_BYTES = 1024 * 1024
MAX_FRAME_BYTES = 16 * 1024 * 1024


class StreamProtocolError(GQMRError, ValueError):
    """Raised when a peer violates Stream Protocol v1."""


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StreamProtocolError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def encode_header(message_type: str, header: Mapping[str, Any]) -> list[bytes]:
    try:
        payload = json.dumps(
            header,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise StreamProtocolError(f"message header is not strict JSON: {error}") from error
    if len(payload) > MAX_HEADER_BYTES:
        raise StreamProtocolError("message header exceeds 1 MiB")
    return [message_type.encode("ascii"), payload]


def decode_header(parts: list[bytes], *, minimum_parts: int = 2) -> tuple[str, dict[str, Any]]:
    if len(parts) < minimum_parts:
        raise StreamProtocolError("multipart message has too few parts")
    if len(parts[1]) > MAX_HEADER_BYTES:
        raise StreamProtocolError("message header exceeds 1 MiB")
    try:
        message_type = parts[0].decode("ascii")
        header = json.loads(parts[1].decode("ascii"), object_pairs_hook=_reject_duplicates)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise StreamProtocolError(f"invalid stream message header: {error}") from error
    if not isinstance(header, dict):
        raise StreamProtocolError("message header must be a JSON object")
    return message_type, header


def encode_frame(header: dict[str, Any], arrays: Mapping[str, np.ndarray]) -> list[bytes]:
    descriptors: list[dict[str, Any]] = []
    payloads: list[bytes] = []
    total = 0
    for part, (name, value) in enumerate(arrays.items()):
        array = np.ascontiguousarray(value)
        if array.dtype != np.dtype("<f8"):
            array = np.ascontiguousarray(array, dtype="<f8")
        payload = array.tobytes(order="C")
        total += len(payload)
        if total > MAX_FRAME_BYTES:
            raise StreamProtocolError("FRAME payload exceeds 16 MiB")
        descriptors.append(
            {"name": name, "dtype": "<f8", "shape": list(array.shape), "part": part}
        )
        payloads.append(payload)
    document = dict(header)
    document["arrays"] = descriptors
    return [*encode_header("FRAME", document), *payloads]


def decode_frame(parts: list[bytes]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    message_type, header = decode_header(parts, minimum_parts=2)
    if message_type != "FRAME":
        raise StreamProtocolError(f"expected FRAME, got {message_type}")
    descriptors = header.get("arrays")
    if not isinstance(descriptors, list) or len(parts) != len(descriptors) + 2:
        raise StreamProtocolError("FRAME array descriptors do not match parts")
    arrays: dict[str, np.ndarray] = {}
    total = 0
    for descriptor in descriptors:
        if not isinstance(descriptor, dict) or set(descriptor) != {"name", "dtype", "shape", "part"}:
            raise StreamProtocolError("FRAME has an invalid array descriptor")
        name = descriptor["name"]
        shape = descriptor["shape"]
        part = descriptor["part"]
        if (
            not isinstance(name, str)
            or name in arrays
            or descriptor["dtype"] != "<f8"
            or not isinstance(shape, list)
            or any(not isinstance(size, int) or size < 0 for size in shape)
            or not isinstance(part, int)
            or part < 0
            or part >= len(descriptors)
        ):
            raise StreamProtocolError("FRAME has an unsafe array descriptor")
        payload = parts[part + 2]
        expected = int(np.prod(shape, dtype=np.int64)) * 8
        total += len(payload)
        if len(payload) != expected or total > MAX_FRAME_BYTES:
            raise StreamProtocolError("FRAME array byte length is invalid")
        arrays[name] = np.frombuffer(payload, dtype="<f8").reshape(shape).copy()
    return header, arrays
