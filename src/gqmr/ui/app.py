"""GQMR PySide6 GUI MVP."""

from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from PySide6.QtCore import QThreadPool, QTimer, Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QInputDialog,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QStatusBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from gqmr import __version__
from gqmr.core.io import load_motion, save_motion
from gqmr.core.motion import AnimalMotion, RobotMotion
from gqmr.editing import EditStack, filter_robot_motion, make_robot_loop
from gqmr.exporters import export_deepmimic_json, export_isaaclab_amp_v232
from gqmr.project import (
    add_resource,
    load_project,
    materialize_resource,
    new_project,
    pack_project,
    save_project,
)
from gqmr.project.model import EditCommand
from gqmr.retarget import replay_quality_report, retarget_fast, retarget_high_quality
from gqmr.robots import load_robot_model
from gqmr.sources.files import load_legacy_dog27
from gqmr.synthetic import generate_dog27_motion
from gqmr.ui.preview import MotionPreview
from gqmr.ui.worker import FunctionTask


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"GQMR {__version__} — Quadruped Motion Studio")
        self.resize(1280, 780)
        self.thread_pool = QThreadPool.globalInstance()
        self.active_task: FunctionTask | None = None
        self.animal_motion: AnimalMotion | None = None
        self.robot_motion: RobotMotion | None = None
        self.diagnostics = None
        self.animal_path: Path | None = None
        self.robot_path: Path | None = None
        self.project = new_project()
        self.project_path: Path | None = None
        self.play_timer = QTimer(self)
        self.edit_stack: EditStack | None = None
        self.edit_target = "animal"
        self.play_timer.timeout.connect(self._advance_frame)
        self._build_ui()
        self._build_menu()
        self._apply_style()

    def _build_ui(self) -> None:
        central = QWidget()
        root_layout = QVBoxLayout(central)
        title = QLabel("General Quadruped Motion Retargeting")
        title.setObjectName("title")
        subtitle = QLabel("导入动物骨架 → MuJoCo 重定向 → 质量检查 → AMP 导出")
        subtitle.setObjectName("subtitle")
        root_layout.addWidget(title)
        root_layout.addWidget(subtitle)

        splitter = QSplitter(Qt.Horizontal)
        controls = QWidget()
        controls.setMinimumWidth(280)
        controls.setMaximumWidth(360)
        controls_layout = QVBoxLayout(controls)

        source_group = QGroupBox("1. 动物运动")
        source_layout = QVBoxLayout(source_group)
        source_buttons = QHBoxLayout()
        self.demo_button = QPushButton("生成 MIT 演示")
        self.import_button = QPushButton("导入文件")
        source_buttons.addWidget(self.demo_button)
        source_buttons.addWidget(self.import_button)
        self.source_label = QLabel("尚未加载")
        self.source_label.setWordWrap(True)
        source_layout.addLayout(source_buttons)
        source_layout.addWidget(self.source_label)
        controls_layout.addWidget(source_group)

        retarget_group = QGroupBox("2. 目标机器人")
        retarget_layout = QFormLayout(retarget_group)
        self.robot_combo = QComboBox()
        self.robot_combo.addItems(["unitree-go2", "unitree-b2"])
        self.retarget_mode = QComboBox()
        self.retarget_mode.addItems(["快速 DLS", "高质量接触锁定"])
        self.cache_edit = QLineEdit()
        self.cache_edit.setPlaceholderText("留空使用 GQMR 默认资产缓存")
        self.retarget_button = QPushButton("运行快速 DLS 重定向")
        self.cancel_button = QPushButton("取消任务")
        self.cancel_button.setEnabled(False)
        retarget_layout.addRow("机器人", self.robot_combo)
        retarget_layout.addRow("求解模式", self.retarget_mode)
        retarget_layout.addRow("资产缓存", self.cache_edit)
        retarget_layout.addRow(self.retarget_button)
        retarget_layout.addRow(self.cancel_button)
        controls_layout.addWidget(retarget_group)

        export_group = QGroupBox("3. 验证与导出")
        export_layout = QFormLayout(export_group)
        self.format_combo = QComboBox()
        self.format_combo.addItems(["isaaclab_amp_v232", "canonical", "deepmimic"])
        self.quality_button = QPushButton("运行 MuJoCo 质量检查")
        self.export_button = QPushButton("导出结果")
        export_layout.addRow("格式", self.format_combo)
        export_layout.addRow(self.quality_button)
        export_layout.addRow(self.export_button)
        controls_layout.addWidget(export_group)
        controls_layout.addStretch(1)

        center = QWidget()
        center_layout = QVBoxLayout(center)
        self.preview = MotionPreview()
        playback = QHBoxLayout()
        self.play_button = QPushButton("播放")
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_label = QLabel("0 / 0")
        playback.addWidget(self.play_button)
        playback.addWidget(self.frame_slider, 1)
        playback.addWidget(self.frame_label)
        center_layout.addWidget(self.preview, 1)
        center_layout.addLayout(playback)

        report_panel = QWidget()
        report_panel.setMinimumWidth(300)
        report_layout = QVBoxLayout(report_panel)
        report_layout.addWidget(QLabel("质量报告 / 任务日志"))
        self.report = QTextEdit()
        self.report.setReadOnly(True)
        report_layout.addWidget(self.report)

        splitter.addWidget(controls)
        splitter.addWidget(center)
        splitter.addWidget(report_panel)
        splitter.setStretchFactor(1, 1)
        root_layout.addWidget(splitter, 1)
        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())

        self.demo_button.clicked.connect(self.generate_demo)
        self.import_button.clicked.connect(self.import_motion)
        self.retarget_button.clicked.connect(self.start_retarget)
        self.cancel_button.clicked.connect(self.cancel_task)
        self.quality_button.clicked.connect(self.run_quality)
        self.export_button.clicked.connect(self.export_motion)
        self.play_button.clicked.connect(self.toggle_playback)
        self.frame_slider.valueChanged.connect(self.set_frame)
        self._update_enabled()

    def _build_menu(self) -> None:
        menu = self.menuBar().addMenu("工程")
        actions = [
            ("新建", self.new_project),
            ("打开…", self.open_project),
            ("保存", self.save_current_project),
            ("打包工程…", self.pack_current_project),
        ]
        for label, callback in actions:
            action = QAction(label, self)
            action.triggered.connect(callback)
            menu.addAction(action)
        edit_menu = self.menuBar().addMenu("编辑")
        edit_actions = [
            ("撤销", self.undo_edit),
            ("重做", self.redo_edit),
            ("裁剪时间范围…", self.trim_dialog),
            ("变速…", self.time_scale_dialog),
            ("重采样…", self.resample_dialog),
            ("机器人平滑滤波…", self.filter_dialog),
            ("机器人循环闭合", self.loop_current_robot),
        ]
        for label, callback in edit_actions:
            action = QAction(label, self)
            action.triggered.connect(callback)
            edit_menu.addAction(action)
        help_menu = self.menuBar().addMenu("帮助")
        about = QAction("关于 GQMR", self)
        about.triggered.connect(self.show_about)
        help_menu.addAction(about)

    def show_about(self) -> None:
        QMessageBox.about(
            self,
            "关于 GQMR",
            f"GQMR {__version__}\n\n"
            "General Quadruped Motion Retargeting\n"
            "GQMR: MIT License\n"
            "PySide6 / Qt / shiboken6: LGPL-3.0 (dynamic libraries)\n"
            "MuJoCo: Apache-2.0; Unitree assets: BSD-3-Clause\n\n"
            "完整第三方说明见 docs/THIRD_PARTY_LICENSES.md",
        )

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #0f172a; color: #e2e8f0; font-size: 13px; }
            QLabel#title { font-size: 25px; font-weight: 700; color: #f8fafc; }
            QLabel#subtitle { color: #94a3b8; margin-bottom: 8px; }
            QGroupBox { border: 1px solid #334155; border-radius: 9px; margin-top: 12px; padding-top: 12px; font-weight: 600; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QPushButton { background: #1e293b; border: 1px solid #475569; border-radius: 6px; padding: 8px; }
            QPushButton:hover { background: #334155; }
            QPushButton:disabled { color: #64748b; background: #172033; }
            QLineEdit, QComboBox, QTextEdit { background: #111827; border: 1px solid #334155; border-radius: 5px; padding: 6px; }
            QSlider::groove:horizontal { height: 5px; background: #334155; }
            QSlider::handle:horizontal { width: 14px; margin: -5px 0; border-radius: 7px; background: #38bdf8; }
            """
        )

    def _cache_dir(self) -> Path | None:
        text = self.cache_edit.text().strip()
        return Path(text) if text else None

    def _log(self, value) -> None:
        if isinstance(value, str):
            self.report.setPlainText(value)
        else:
            self.report.setPlainText(json.dumps(value, ensure_ascii=False, indent=2))

    def _update_enabled(self) -> None:
        busy = self.active_task is not None
        self.retarget_button.setEnabled(self.animal_motion is not None and not busy)
        self.quality_button.setEnabled(self.robot_motion is not None and not busy)
        self.export_button.setEnabled(self.robot_motion is not None and not busy)
        self.cancel_button.setEnabled(busy)

    def set_animal_motion(self, motion: AnimalMotion, path: Path | None = None) -> None:
        self.animal_motion = motion
        self.animal_path = path
        self.robot_motion = None
        self.robot_path = None
        self.diagnostics = None
        self.edit_stack = EditStack(motion)
        self.edit_target = "animal"
        self.source_label.setText(
            f"{path or motion.metadata['source'].get('gait', 'memory')}\n{motion.frame_count} 帧 / {motion.duration:.3f} s"
        )
        self.preview.set_motion(motion)
        self.frame_slider.setRange(0, motion.frame_count - 1)
        self.set_frame(0)
        self._update_enabled()

    def generate_demo(self) -> None:
        motion = generate_dog27_motion("trot", duration=2.0, fps=60.0)
        self.set_animal_motion(motion)
        self._log({"generated": "MIT dog-27 trot", "frames": motion.frame_count})

    def import_motion(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self, "导入动物运动", "", "Motion (*.npz *.txt);;All files (*)"
        )
        if not filename:
            return
        path = Path(filename)
        try:
            motion = load_motion(path) if path.suffix.lower() == ".npz" else load_legacy_dog27(path)
            if not isinstance(motion, AnimalMotion):
                raise ValueError("选择的 NPZ 不是 AnimalMotion")
            self.set_animal_motion(motion, path)
        except Exception as error:
            QMessageBox.critical(self, "导入失败", str(error))

    def _run_task(self, function, success) -> None:
        if self.active_task is not None:
            return
        task = FunctionTask(function)
        self.active_task = task
        task.signals.succeeded.connect(success)
        task.signals.failed.connect(lambda text: self._log(text))
        task.signals.finished.connect(self._task_finished)
        self.thread_pool.start(task)
        self.statusBar().showMessage("任务运行中…")
        self._update_enabled()

    def _task_finished(self) -> None:
        self.active_task = None
        self.statusBar().showMessage("就绪", 3000)
        self._update_enabled()

    def cancel_task(self) -> None:
        if self.active_task is not None:
            self.active_task.token.cancel()
            self.statusBar().showMessage("已请求取消，等待安全检查点…")

    def start_retarget(self) -> None:
        if self.animal_motion is None:
            return
        animal = self.animal_motion
        robot_id = self.robot_combo.currentText()
        cache = self._cache_dir()
        high_quality = self.retarget_mode.currentIndex() == 1

        def work(token):
            robot = load_robot_model(robot_id, cache_dir=cache)
            result = (
                retarget_high_quality(animal, robot)
                if high_quality
                else retarget_fast(animal, robot)
            )
            return robot, *result

        def complete(result) -> None:
            robot, motion, diagnostics = result
            self.robot_motion = motion
            self.diagnostics = diagnostics
            self.edit_stack = EditStack(motion)
            self.edit_target = "robot"
            self.preview.set_motion(self.animal_motion, motion, diagnostics)
            self.frame_slider.setRange(0, motion.frame_count - 1)
            self._log(replay_quality_report(motion, robot))
            self._update_enabled()

        self._run_task(work, complete)

    def run_quality(self) -> None:
        if self.robot_motion is None:
            return
        motion = self.robot_motion
        robot_id = self.robot_combo.currentText()
        cache = self._cache_dir()
        self._run_task(
            lambda token: replay_quality_report(
                motion, load_robot_model(robot_id, cache_dir=cache)
            ),
            self._log,
        )

    def export_motion(self) -> None:
        if self.robot_motion is None:
            return
        format_name = self.format_combo.currentText()
        suffix = ".json" if format_name == "deepmimic" else ".npz"
        filename, _ = QFileDialog.getSaveFileName(self, "导出", f"motion{suffix}")
        if not filename:
            return
        destination = Path(filename)
        motion = self.robot_motion
        robot_id = self.robot_combo.currentText()
        cache = self._cache_dir()

        def work(token):
            robot = load_robot_model(robot_id, cache_dir=cache)
            if format_name == "canonical":
                save_motion(destination, motion)
            elif format_name == "deepmimic":
                export_deepmimic_json(motion, destination)
            else:
                export_isaaclab_amp_v232(motion, robot, destination)
            return {"exported": str(destination), "format": format_name}

        self._run_task(work, self._log)

    def _edit_command(self, kind: str, parameters: dict) -> EditCommand:
        return EditCommand(
            command_id=str(uuid.uuid4()),
            kind=kind,
            resource_id=str(uuid.uuid4()),
            parameters=parameters,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def _show_edited_motion(self, motion) -> None:
        if isinstance(motion, AnimalMotion):
            self.animal_motion = motion
            self.robot_motion = None
            self.diagnostics = None
            self.edit_target = "animal"
            self.preview.set_motion(motion)
        else:
            self.robot_motion = motion
            self.edit_target = "robot"
            self.diagnostics = None
            self.preview.set_motion(self.animal_motion, motion, self.diagnostics)
        self.frame_slider.setRange(0, motion.frame_count - 1)
        self.set_frame(min(self.frame_slider.value(), motion.frame_count - 1))
        self._log(
            {
                "edited": self.edit_target,
                "frames": motion.frame_count,
                "duration_seconds": motion.duration,
                "history": motion.metadata.get("edit_history", []),
            }
        )
        self._update_enabled()

    def undo_edit(self) -> None:
        if self.edit_stack is not None:
            self._show_edited_motion(self.edit_stack.undo())

    def redo_edit(self) -> None:
        if self.edit_stack is not None:
            self._show_edited_motion(self.edit_stack.redo())

    def trim_dialog(self) -> None:
        if self.edit_stack is None:
            return
        current = self.edit_stack.current()
        start, ok = QInputDialog.getDouble(
            self, "裁剪", "开始时间 (s)", 0.0, 0.0, current.duration, 4
        )
        if not ok:
            return
        end, ok = QInputDialog.getDouble(
            self, "裁剪", "结束时间 (s)", current.duration, start, current.duration, 4
        )
        if ok:
            self._show_edited_motion(
                self.edit_stack.push(
                    self._edit_command("trim", {"start": start, "end": end})
                )
            )

    def time_scale_dialog(self) -> None:
        if self.edit_stack is None:
            return
        speed, ok = QInputDialog.getDouble(
            self, "变速", "速度倍率", 1.0, 0.01, 20.0, 3
        )
        if ok:
            self._show_edited_motion(
                self.edit_stack.push(
                    self._edit_command("time_scale", {"speed": speed})
                )
            )

    def resample_dialog(self) -> None:
        if self.edit_stack is None:
            return
        fps, ok = QInputDialog.getInt(self, "重采样", "FPS", 60, 1, 1000)
        if ok:
            self._show_edited_motion(
                self.edit_stack.push(self._edit_command("resample", {"fps": fps}))
            )

    def filter_dialog(self) -> None:
        if self.robot_motion is None:
            return
        window, ok = QInputDialog.getInt(
            self, "平滑滤波", "奇数窗口帧数", 9, 3, self.robot_motion.frame_count, 2
        )
        if not ok:
            return
        try:
            filtered = filter_robot_motion(self.robot_motion, window_frames=window)
            self.edit_stack = EditStack(filtered)
            self._show_edited_motion(filtered)
        except Exception as error:
            QMessageBox.critical(self, "滤波失败", str(error))

    def loop_current_robot(self) -> None:
        if self.robot_motion is None:
            return
        try:
            loop = make_robot_loop(self.robot_motion)
            self.edit_stack = EditStack(loop)
            self._show_edited_motion(loop)
        except Exception as error:
            QMessageBox.critical(self, "循环闭合失败", str(error))

    def toggle_playback(self) -> None:
        if self.play_timer.isActive():
            self.play_timer.stop()
            self.play_button.setText("播放")
        else:
            self.play_timer.start(16)
            self.play_button.setText("暂停")

    def _advance_frame(self) -> None:
        maximum = self.frame_slider.maximum()
        value = self.frame_slider.value() + 1
        self.frame_slider.setValue(0 if value > maximum else value)

    def set_frame(self, frame: int) -> None:
        self.preview.set_frame(frame)
        self.frame_label.setText(f"{frame + 1} / {self.frame_slider.maximum() + 1}")

    def new_project(self) -> None:
        self.project = new_project()
        self.project_path = None
        self.statusBar().showMessage("已新建工程", 3000)

    def open_project(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "打开 GQMR 工程", "", "GQMR (*.gqmr)")
        if not filename:
            return
        try:
            self.project = load_project(filename)
            self.project_path = Path(filename)
            active = self.project.active_animal_motion
            if active is not None:
                resource_path = materialize_resource(
                    filename, self.project, active
                )
                motion = load_motion(resource_path)
                if isinstance(motion, AnimalMotion):
                    self.set_animal_motion(motion, resource_path)
            active_robot = self.project.retarget.get("active_robot_motion")
            if active_robot is not None:
                resource_path = materialize_resource(
                    filename, self.project, active_robot
                )
                motion = load_motion(resource_path)
                if isinstance(motion, RobotMotion):
                    self.robot_motion = motion
                    self.robot_path = resource_path
                    self.edit_stack = EditStack(motion)
                    self.edit_target = "robot"
                    self.preview.set_motion(self.animal_motion, motion)
                    self.frame_slider.setRange(0, motion.frame_count - 1)
                    self._update_enabled()
            self._log({"opened_project": filename, "resources": len(self.project.resources)})
        except Exception as error:
            QMessageBox.critical(self, "打开失败", str(error))

    def save_current_project(self) -> None:
        if self.project_path is None:
            filename, _ = QFileDialog.getSaveFileName(self, "保存 GQMR 工程", "project.gqmr", "GQMR (*.gqmr)")
            if not filename:
                return
            self.project_path = Path(filename)
        project = self.project
        try:
            if self.animal_motion is not None and self.animal_path is None:
                self.animal_path = self.project_path.with_name(
                    f"{self.project_path.stem}.animal.npz"
                )
                save_motion(self.animal_path, self.animal_motion)
            if self.robot_motion is not None and self.robot_path is None:
                self.robot_path = self.project_path.with_name(
                    f"{self.project_path.stem}.robot.npz"
                )
                save_motion(self.robot_path, self.robot_motion)
            if self.animal_path is not None and not any(
                resource.uri == str(self.animal_path.resolve())
                for resource in project.resources.values()
            ):
                project = add_resource(project, self.animal_path, make_active="animal")
            if self.robot_path is not None and not any(
                resource.uri == str(self.robot_path.resolve())
                for resource in project.resources.values()
            ):
                project = add_resource(project, self.robot_path, make_active="robot")
            save_project(self.project_path, project, source_path=self.project_path)
            self.project = project
            self.statusBar().showMessage(f"已保存 {self.project_path}", 4000)
        except Exception as error:
            QMessageBox.critical(self, "保存失败", str(error))

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.active_task is not None:
            self.active_task.token.cancel()
            self.thread_pool.waitForDone(2000)
        event.accept()

    def pack_current_project(self) -> None:
        filename, _ = QFileDialog.getSaveFileName(self, "打包 GQMR 工程", "portable.gqmr", "GQMR (*.gqmr)")
        if not filename:
            return
        try:
            pack_project(filename, self.project, source_path=self.project_path)
            self.statusBar().showMessage(f"已打包 {filename}", 4000)
        except Exception as error:
            QMessageBox.critical(self, "打包失败", str(error))


def main() -> int:
    application = QApplication.instance() or QApplication(sys.argv)
    application.setApplicationName("GQMR")
    application.setOrganizationName("GQMR")
    window = MainWindow()
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
