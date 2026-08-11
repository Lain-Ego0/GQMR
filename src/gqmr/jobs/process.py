"""Long-running internal/plugin jobs isolated with multiprocessing spawn."""

from __future__ import annotations

import importlib
import multiprocessing as mp
import queue
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

from gqmr.core.errors import GQMRError


class ProcessJobError(GQMRError, RuntimeError):
    """Raised when an isolated job fails, times out, or is cancelled."""


def _child_main(
    result_queue: mp.Queue,
    function_path: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    try:
        module_name, function_name = function_path.rsplit(":", 1)
        function = getattr(importlib.import_module(module_name), function_name)
        result_queue.put(("ok", function(*args, **kwargs)))
    except BaseException:
        result_queue.put(("error", traceback.format_exc()))


@dataclass(slots=True)
class ProcessJob:
    function_path: str
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] | None = None
    _context: Any = field(init=False, repr=False)
    _queue: Any = field(init=False, repr=False)
    _process: mp.Process | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if ":" not in self.function_path:
            raise ProcessJobError("function_path must be module:function")
        self._context = mp.get_context("spawn")
        self._queue = self._context.Queue(maxsize=1)
        self._process = None

    def start(self) -> "ProcessJob":
        if self._process is not None:
            raise ProcessJobError("job has already started")
        self._process = self._context.Process(
            target=_child_main,
            args=(self._queue, self.function_path, self.args, self.kwargs or {}),
            daemon=True,
        )
        self._process.start()
        return self

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def result(self, *, timeout: float | None = None) -> Any:
        if self._process is None:
            raise ProcessJobError("job has not started")
        try:
            status, payload = self._queue.get(timeout=timeout)
        except queue.Empty as error:
            raise ProcessJobError("job result timed out") from error
        self._process.join(timeout=0.2)
        if status == "error":
            raise ProcessJobError(payload)
        return payload

    def cancel(self, *, grace_seconds: float = 2.0) -> None:
        if self._process is None or not self._process.is_alive():
            return
        deadline = time.monotonic() + grace_seconds
        while self._process.is_alive() and time.monotonic() < deadline:
            self._process.join(timeout=min(0.05, deadline - time.monotonic()))
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.0)
        if self._process.is_alive():
            self._process.kill()
            self._process.join(timeout=1.0)

    def close(self) -> None:
        self.cancel(grace_seconds=0.0)
        self._queue.close()

    def __enter__(self) -> "ProcessJob":
        return self.start()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
