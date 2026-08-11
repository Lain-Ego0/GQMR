"""Small spawn-safe functions used by process-job tests."""

from __future__ import annotations

import time


def add(left: int, right: int) -> int:
    return left + right


def wait_forever() -> None:
    while True:
        time.sleep(0.05)
