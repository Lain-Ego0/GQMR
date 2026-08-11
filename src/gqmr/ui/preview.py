"""Lightweight timeline preview for animal keypoints and robot feet."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

from gqmr.core.motion import AnimalMotion, RobotMotion
from gqmr.retarget import RetargetDiagnostics
from gqmr.skeletons import get_skeleton


class MotionPreview(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(640, 360)
        self.animal: AnimalMotion | None = None
        self.robot: RobotMotion | None = None
        self.diagnostics: RetargetDiagnostics | None = None
        self.frame = 0

    def set_motion(
        self,
        animal: AnimalMotion | None,
        robot: RobotMotion | None = None,
        diagnostics: RetargetDiagnostics | None = None,
    ) -> None:
        self.animal = animal
        self.robot = robot
        self.diagnostics = diagnostics
        self.frame = 0
        self.update()

    def set_frame(self, frame: int) -> None:
        self.frame = max(0, frame)
        self.update()

    @staticmethod
    def _project(point: np.ndarray, rect: QRectF, bounds: tuple[float, float, float, float]) -> QPointF:
        min_x, max_x, min_z, max_z = bounds
        x = rect.left() + (float(point[0]) - min_x) / max(max_x - min_x, 1e-6) * rect.width()
        y = rect.bottom() - (float(point[2]) - min_z) / max(max_z - min_z, 1e-6) * rect.height()
        return QPointF(x, y)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#111827"))
        margin = 24.0
        half = (self.width() - margin * 3) / 2
        animal_rect = QRectF(margin, 44, half, self.height() - 72)
        robot_rect = QRectF(margin * 2 + half, 44, half, self.height() - 72)
        painter.setPen(QColor("#94a3b8"))
        painter.drawText(QRectF(animal_rect.left(), 12, half, 24), Qt.AlignCenter, "AnimalMotion")
        painter.drawText(QRectF(robot_rect.left(), 12, half, 24), Qt.AlignCenter, "RobotMotion")
        painter.setPen(QPen(QColor("#334155"), 1))
        painter.drawRoundedRect(animal_rect, 10, 10)
        painter.drawRoundedRect(robot_rect, 10, 10)

        if self.animal is not None:
            frame = min(self.frame, self.animal.frame_count - 1)
            points = self.animal.positions[frame]
            finite = points[np.all(np.isfinite(points), axis=1)]
            if len(finite):
                bounds = (
                    float(np.min(finite[:, 0]) - 0.1),
                    float(np.max(finite[:, 0]) + 0.1),
                    min(0.0, float(np.min(finite[:, 2]) - 0.05)),
                    float(np.max(finite[:, 2]) + 0.1),
                )
                skeleton = get_skeleton(self.animal.metadata["skeleton_id"])
                index = {name: i for i, name in enumerate(self.animal.keypoint_names)}
                painter.setPen(QPen(QColor("#38bdf8"), 2))
                for definition in skeleton.keypoints:
                    if definition.parent is None:
                        continue
                    a = points[index[definition.parent]]
                    b = points[index[definition.name]]
                    if np.all(np.isfinite(a)) and np.all(np.isfinite(b)):
                        painter.drawLine(
                            self._project(a, animal_rect, bounds),
                            self._project(b, animal_rect, bounds),
                        )
                painter.setBrush(QColor("#f8fafc"))
                painter.setPen(Qt.NoPen)
                for point in finite:
                    center = self._project(point, animal_rect, bounds)
                    painter.drawEllipse(center, 3.2, 3.2)

        if self.robot is not None:
            frame = min(self.frame, self.robot.frame_count - 1)
            root = self.robot.root_position[frame]
            feet = (
                self.diagnostics.achieved_foot_positions[frame]
                if self.diagnostics is not None
                else np.empty((0, 3), dtype=np.float32)
            )
            points = np.vstack((root, feet))
            bounds = (
                float(np.min(points[:, 0]) - 0.12),
                float(np.max(points[:, 0]) + 0.12),
                min(0.0, float(np.min(points[:, 2]) - 0.05)),
                float(np.max(points[:, 2]) + 0.12),
            )
            root_point = self._project(root, robot_rect, bounds)
            painter.setPen(QPen(QColor("#a78bfa"), 3))
            for foot in feet:
                painter.drawLine(root_point, self._project(foot, robot_rect, bounds))
            painter.setBrush(QColor("#f59e0b"))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(root_point, 7, 7)
            painter.setBrush(QColor("#4ade80"))
            for foot in feet:
                painter.drawEllipse(self._project(foot, robot_rect, bounds), 5, 5)
