"""Qt motion preview with a real MuJoCo robot renderer."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import mujoco
import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget

from gqmr.core.motion import AnimalMotion, RobotMotion
from gqmr.retarget import RetargetDiagnostics
from gqmr.skeletons import get_skeleton

if TYPE_CHECKING:
    from gqmr.robots.model import RobotModel


_INK = QColor("#252422")
_MUTED = QColor("#716d66")
_BORDER = QColor("#d8d3ca")
_PANEL = QColor("#ffffff")
_CANVAS = QColor("#f4f1eb")
_ACCENT = QColor("#b8522f")


class MotionPreview(QWidget):
    """Show the source skeleton beside the selected MuJoCo robot model."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(720, 440)
        self.setMouseTracking(True)
        self.setToolTip("在机器人视图中拖动旋转，滚轮缩放")
        self.animal: AnimalMotion | None = None
        self.robot: RobotMotion | None = None
        self.diagnostics: RetargetDiagnostics | None = None
        self.robot_model: RobotModel | None = None
        self.frame = 0
        self._renderer: mujoco.Renderer | None = None
        self._robot_image = QImage()
        self._render_error: str | None = None
        self._render_note: str | None = None
        self._robot_rect = QRectF()
        self._drag_position: QPoint | None = None
        self._camera_azimuth = 135.0
        self._camera_elevation = -18.0
        self._camera_scale = 1.7

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
        self._render_robot_frame()

    def set_robot_model(
        self, robot_model: RobotModel | None, *, error: str | None = None
    ) -> None:
        self.close_renderer()
        self.robot_model = robot_model
        self._render_error = error
        self._render_note = None
        self._robot_image = QImage()
        if robot_model is None:
            self.update()
            return
        try:
            self._apply_neutral_scene_palette(robot_model.model)
            self._renderer = mujoco.Renderer(
                robot_model.model, height=480, width=640
            )
            self._render_error = None
            self._render_robot_frame()
        except Exception as render_error:
            self.close_renderer()
            self.robot_model = robot_model
            self._render_error = f"MuJoCo 预览不可用：{render_error}"
            self.update()

    @staticmethod
    def _apply_neutral_scene_palette(model: mujoco.MjModel) -> None:
        """Replace the upstream blue demo floor with a neutral studio palette."""

        for texture_id in range(model.ntex):
            address = int(model.tex_adr[texture_id])
            width = int(model.tex_width[texture_id])
            height = int(model.tex_height[texture_id])
            channels = int(model.tex_nchannel[texture_id])
            size = width * height * channels
            pixels = model.tex_data[address : address + size].reshape(
                height, width, channels
            )
            texture_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_TEXTURE, texture_id
            )
            if int(model.tex_type[texture_id]) == int(
                mujoco.mjtTexture.mjTEXTURE_SKYBOX
            ):
                top = np.array([236.0, 232.0, 224.0])
                bottom = np.array([250.0, 249.0, 246.0])
                gradient = np.linspace(0.0, 1.0, height)[:, None, None]
                pixels[:] = ((1.0 - gradient) * top + gradient * bottom).astype(
                    np.uint8
                )
            elif texture_name == "groundplane":
                luminance = pixels.astype(np.float64).mean(axis=2, keepdims=True)
                luminance /= 255.0
                dark = np.array([184.0, 179.0, 169.0])
                light = np.array([235.0, 232.0, 225.0])
                pixels[:] = np.clip(
                    dark + luminance * (light - dark), 0.0, 255.0
                ).astype(np.uint8)
        model.vis.rgba.haze[:] = [0.92, 0.90, 0.86, 1.0]

    def set_frame(self, frame: int) -> None:
        self.frame = max(0, frame)
        self._render_robot_frame()

    @property
    def has_robot_render(self) -> bool:
        return self.robot_model is not None and not self._robot_image.isNull()

    @property
    def rendered_robot_id(self) -> str | None:
        return self.robot_model.config.id if self.robot_model is not None else None

    def _render_robot_frame(self) -> None:
        if self._renderer is None or self.robot_model is None:
            self.update()
            return
        model = self.robot_model
        self._render_note = None
        if self.robot is not None:
            expected_hash = model.config.model_sha256
            if self.robot.metadata.get("model_sha256") != expected_hash:
                self._render_note = "当前动作属于另一机器人型号"
            else:
                frame = min(self.frame, self.robot.frame_count - 1)
                if self.robot.frame_valid[frame]:
                    try:
                        model.set_pose(
                            self.robot.root_position[frame],
                            self.robot.root_rotation[frame],
                            self.robot.dof_position[frame],
                        )
                    except Exception as error:
                        self._render_note = f"该帧无法预览：{error}"
                else:
                    self._render_note = "该帧无效，保留上一有效姿态"
        try:
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:] = model.data.xpos[model.base_body_id]
            camera.distance = max(
                float(model.model.stat.extent) * self._camera_scale, 1.1
            )
            camera.azimuth = self._camera_azimuth
            camera.elevation = self._camera_elevation
            self._renderer.update_scene(model.data, camera=camera)
            image = self._renderer.render()
            self._robot_image = QImage(
                image.data,
                image.shape[1],
                image.shape[0],
                image.strides[0],
                QImage.Format_RGB888,
            ).copy()
            self._render_error = None
        except Exception as error:
            self._render_error = f"MuJoCo 渲染失败：{error}"
        self.update()

    @staticmethod
    def _project(
        point: np.ndarray,
        rect: QRectF,
        bounds: tuple[float, float, float, float],
    ) -> QPointF:
        min_x, max_x, min_z, max_z = bounds
        x = (
            rect.left()
            + (float(point[0]) - min_x)
            / max(max_x - min_x, 1e-6)
            * rect.width()
        )
        y = (
            rect.bottom()
            - (float(point[2]) - min_z)
            / max(max_z - min_z, 1e-6)
            * rect.height()
        )
        return QPointF(x, y)

    @staticmethod
    def _draw_panel(painter: QPainter, rect: QRectF) -> None:
        painter.setPen(QPen(_BORDER, 1))
        painter.setBrush(_PANEL)
        painter.drawRoundedRect(rect, 5, 5)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), _CANVAS)
        margin = 14.0
        gap = 14.0
        available = self.width() - margin * 2 - gap
        animal_width = max(250.0, available * 0.36)
        animal_rect = QRectF(
            margin, margin, animal_width, self.height() - margin * 2
        )
        robot_rect = QRectF(
            animal_rect.right() + gap,
            margin,
            available - animal_width,
            self.height() - margin * 2,
        )
        self._robot_rect = robot_rect
        self._draw_panel(painter, animal_rect)
        self._draw_panel(painter, robot_rect)

        title_font = QFont(self.font())
        title_font.setPointSize(11)
        title_font.setWeight(QFont.DemiBold)
        painter.setFont(title_font)
        painter.setPen(_INK)
        painter.drawText(
            QRectF(animal_rect.left() + 18, animal_rect.top() + 13, 200, 26),
            Qt.AlignLeft | Qt.AlignVCenter,
            "源动作",
        )
        robot_name = (
            self.robot_model.config.id.replace("unitree-", "Unitree ").title()
            if self.robot_model is not None
            else "机器人模型"
        )
        painter.drawText(
            QRectF(robot_rect.left() + 18, robot_rect.top() + 13, 260, 26),
            Qt.AlignLeft | Qt.AlignVCenter,
            robot_name,
        )

        animal_content = animal_rect.adjusted(24, 48, -24, -24)
        if self.animal is None:
            painter.setPen(_MUTED)
            painter.drawText(animal_content, Qt.AlignCenter, "导入或生成一段动物动作")
        else:
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
                index = {
                    name: index
                    for index, name in enumerate(self.animal.keypoint_names)
                }
                painter.setPen(QPen(_ACCENT, 2.4))
                for definition in skeleton.keypoints:
                    if definition.parent is None:
                        continue
                    parent = points[index[definition.parent]]
                    child = points[index[definition.name]]
                    if np.all(np.isfinite(parent)) and np.all(np.isfinite(child)):
                        painter.drawLine(
                            self._project(parent, animal_content, bounds),
                            self._project(child, animal_content, bounds),
                        )
                painter.setBrush(_INK)
                painter.setPen(Qt.NoPen)
                for point in finite:
                    center = self._project(point, animal_content, bounds)
                    painter.drawEllipse(center, 3.2, 3.2)

        robot_content = robot_rect.adjusted(8, 44, -8, -8)
        if not self._robot_image.isNull():
            scaled = self._robot_image.scaled(
                robot_content.size().toSize(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            target = QRectF(
                robot_content.center().x() - scaled.width() / 2,
                robot_content.center().y() - scaled.height() / 2,
                scaled.width(),
                scaled.height(),
            )
            painter.drawImage(target, scaled)
        else:
            painter.setPen(_MUTED)
            message = self._render_error or "正在加载机器人模型…"
            painter.drawText(robot_content.adjusted(30, 30, -30, -30), Qt.AlignCenter, message)
        if self._render_note:
            note_rect = QRectF(
                robot_content.left() + 18,
                robot_content.bottom() - 42,
                robot_content.width() - 36,
                30,
            )
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(255, 249, 232, 235))
            painter.drawRoundedRect(note_rect, 4, 4)
            painter.setPen(QColor("#765b1c"))
            painter.drawText(note_rect, Qt.AlignCenter, self._render_note)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton and self._robot_rect.contains(
            event.position()
        ):
            self._drag_position = event.position().toPoint()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_position is not None:
            current = event.position().toPoint()
            delta = current - self._drag_position
            self._drag_position = current
            self._camera_azimuth -= delta.x() * 0.45
            self._camera_elevation = max(
                -75.0, min(5.0, self._camera_elevation + delta.y() * 0.35)
            )
            self._render_robot_frame()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton and self._drag_position is not None:
            self._drag_position = None
            self.unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:  # noqa: N802
        if self._robot_rect.contains(event.position()):
            steps = event.angleDelta().y() / 120.0
            self._camera_scale *= math.pow(0.88, steps)
            self._camera_scale = max(0.9, min(3.8, self._camera_scale))
            self._render_robot_frame()
            event.accept()
            return
        super().wheelEvent(event)

    def close_renderer(self) -> None:
        if self._renderer is not None:
            try:
                self._renderer.close()
            finally:
                self._renderer = None

    def closeEvent(self, event) -> None:  # noqa: N802
        self.close_renderer()
        super().closeEvent(event)
