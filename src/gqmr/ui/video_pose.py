"""Small 2D video-pose result preview dialog."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QDialog, QLabel, QPushButton, QVBoxLayout

from gqmr.pose import KeypointBatch
from gqmr.sources.video import read_video_frames


_AP10K_EDGES = (
    ("L_Eye", "Nose"), ("R_Eye", "Nose"), ("Nose", "Neck"),
    ("Neck", "Root of tail"), ("Neck", "L_Shoulder"),
    ("L_Shoulder", "L_Elbow"), ("L_Elbow", "L_F_Paw"),
    ("Neck", "R_Shoulder"), ("R_Shoulder", "R_Elbow"),
    ("R_Elbow", "R_F_Paw"), ("Root of tail", "L_Hip"),
    ("L_Hip", "L_Knee"), ("L_Knee", "L_B_Paw"),
    ("Root of tail", "R_Hip"), ("R_Hip", "R_Knee"),
    ("R_Knee", "R_B_Paw"),
)


def render_video_pose_frame(video_path: Path, batch: KeypointBatch) -> QImage:
    start = float(batch.metadata.get("source_video", {}).get("start_seconds", 0.0))
    video = read_video_frames(video_path, start_seconds=start, max_frames=1)
    frame = video.frames[0]
    image = QImage(
        frame.data, frame.shape[1], frame.shape[0], frame.strides[0],
        QImage.Format_RGB888,
    ).copy()
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing)
    names = {name: index for index, name in enumerate(batch.keypoint_names)}
    valid = batch.valid_mask[0, 0]
    points = batch.positions[0, 0]
    painter.setPen(QPen(QColor("#f5d04c"), max(2, image.width() // 500)))
    for first, second in _AP10K_EDGES:
        if first in names and second in names:
            first_index, second_index = names[first], names[second]
            if valid[first_index] and valid[second_index]:
                painter.drawLine(
                    QPointF(*points[first_index]), QPointF(*points[second_index])
                )
    radius = max(3.0, image.width() / 350.0)
    painter.setPen(QPen(QColor("#ffffff"), max(1, image.width() // 1000)))
    painter.setBrush(QColor("#b8522f"))
    for index, point in enumerate(points):
        if valid[index] and np.all(np.isfinite(point)):
            painter.drawEllipse(QPointF(*point), radius, radius)
    painter.end()
    return image


class VideoPosePreviewDialog(QDialog):
    def __init__(self, video_path: Path, output_path: Path, batch: KeypointBatch, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("狗视频 2D 关键点提取结果")
        self.resize(920, 700)
        layout = QVBoxLayout(self)
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setMinimumSize(640, 420)
        image_label.setPixmap(
            QPixmap.fromImage(render_video_pose_frame(video_path, batch)).scaled(
                880, 580, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )
        summary = QLabel(
            f"{batch.timestamps.size} 帧 · {len(batch.keypoint_names)} 个关键点 · "
            f"有效率 {float(np.mean(batch.valid_mask)):.1%}\n已保存：{output_path}\n"
            "当前为图像坐标系 2D 结果，还需时序 3D 提升才能生成 DOG27 动画。"
        )
        summary.setWordWrap(True)
        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.accept)
        layout.addWidget(image_label, 1)
        layout.addWidget(summary)
        layout.addWidget(close_button, 0, Qt.AlignRight)
