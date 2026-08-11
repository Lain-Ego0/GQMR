"""Qt thread-pool task wrapper with cooperative result cancellation."""

from __future__ import annotations

import traceback
from collections.abc import Callable
from threading import Event
from typing import Any

from PySide6.QtCore import QObject, QRunnable, Signal, Slot


class CancellationToken:
    def __init__(self) -> None:
        self._event = Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


class TaskSignals(QObject):
    succeeded = Signal(object)
    failed = Signal(str)
    finished = Signal()


class FunctionTask(QRunnable):
    def __init__(self, function: Callable[[CancellationToken], Any]) -> None:
        super().__init__()
        self.function = function
        self.token = CancellationToken()
        self.signals = TaskSignals()

    @Slot()
    def run(self) -> None:
        try:
            result = self.function(self.token)
            if not self.token.cancelled:
                self.signals.succeeded.emit(result)
        except BaseException:
            if not self.token.cancelled:
                self.signals.failed.emit(traceback.format_exc())
        finally:
            self.signals.finished.emit()
