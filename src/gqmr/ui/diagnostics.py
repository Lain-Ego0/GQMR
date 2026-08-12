"""Compact clickable problem-frame timeline."""

from __future__ import annotations

from PySide6.QtCore import QRectF, Signal, Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QWidget

from gqmr.retarget import MotionDiagnostics


class DiagnosticsTimeline(QWidget):
    frameClicked = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(34)
        self.setMaximumHeight(34)
        self.diagnostics: MotionDiagnostics | None = None
        self.current_frame = 0

    def set_diagnostics(self, diagnostics: MotionDiagnostics | None) -> None:
        self.diagnostics = diagnostics
        self.update()

    def set_frame(self, frame: int) -> None:
        self.current_frame = frame
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#ffffff"))
        rect = QRectF(8, 8, max(1, self.width() - 16), 18)
        painter.fillRect(rect, QColor("#e4dfd6"))
        diagnostics = self.diagnostics
        if diagnostics is None or len(diagnostics.invalid) == 0:
            return
        frames = len(diagnostics.invalid)
        width = rect.width() / frames
        for frame in range(frames):
            color = None
            if diagnostics.invalid[frame] or diagnostics.unreachable[frame]:
                color = QColor("#a9382f")
            elif diagnostics.self_collision[frame] or diagnostics.ground_penetration[frame] > 0.003:
                color = QColor("#d5792a")
            elif (
                diagnostics.joint_limit_proximity[frame].max() > 0.95
                or diagnostics.foot_slip_speed[frame].max() > 0.15
            ):
                color = QColor("#d6a629")
            if color is not None:
                painter.fillRect(
                    QRectF(rect.left() + frame * width, rect.top(), max(1.0, width), rect.height()),
                    color,
                )
        x = rect.left() + min(self.current_frame, frames - 1) * width
        painter.fillRect(QRectF(x, rect.top() - 3, 2, rect.height() + 6), QColor("#252422"))

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton and self.diagnostics is not None:
            frames = len(self.diagnostics.invalid)
            ratio = min(1.0, max(0.0, (event.position().x() - 8) / max(1, self.width() - 16)))
            self.frameClicked.emit(min(frames - 1, int(ratio * frames)))
            event.accept()
            return
        super().mousePressEvent(event)
