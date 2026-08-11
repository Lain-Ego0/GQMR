from __future__ import annotations

import numpy as np
import pytest

from gqmr.stream.protocol import StreamProtocolError, decode_frame, encode_frame


def test_frame_multipart_roundtrip_is_little_endian_and_copy_safe() -> None:
    source = np.arange(6, dtype=np.float64).reshape(2, 3)
    parts = encode_frame(
        {"session_id": "session", "seq": 1, "timestamp_ns": 2, "wall_time_ns": None},
        {"qpos": source},
    )
    header, arrays = decode_frame(parts)

    source[:] = -1.0
    assert header["seq"] == 1
    assert arrays["qpos"].dtype == np.dtype("<f8")
    assert arrays["qpos"].tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]


def test_frame_decoder_rejects_wrong_payload_size() -> None:
    parts = encode_frame(
        {"session_id": "session", "seq": 1, "timestamp_ns": 2},
        {"qpos": np.zeros(3)},
    )
    parts[-1] = parts[-1][:-1]

    with pytest.raises(StreamProtocolError, match="byte length"):
        decode_frame(parts)
