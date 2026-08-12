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
    QFrame,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QInputDialog,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QStatusBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from gqmr import __version__
from gqmr.assets import default_asset_root, get_asset_spec
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
from gqmr.pose import keypoint_batch_to_animal_motion
from gqmr.retarget import (
    diagnose_motion,
    evaluate_motion_suite,
    replay_quality_report,
    retarget_fast,
    retarget_high_quality,
)
from gqmr.ui.diagnostics import DiagnosticsTimeline
from gqmr.robots import available_robot_configs, load_robot_model
from gqmr.sources.files import (
    load_generic_keypoints_csv,
    load_generic_keypoints_json,
    load_generic_keypoints_npz,
    load_legacy_dog27,
)
from gqmr.synthetic import available_motion_presets, generate_dog27_preset
from gqmr.ui.preview import MotionPreview
from gqmr.ui.worker import FunctionTask


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"GQMR {__version__} — 四足动作重定向")
        self.resize(1440, 900)
        self.setMinimumSize(1120, 720)
        self.thread_pool = QThreadPool.globalInstance()
        self.active_task: FunctionTask | None = None
        self.animal_motion: AnimalMotion | None = None
        self.robot_motion: RobotMotion | None = None
        self.diagnostics = None
        self.motion_diagnostics = None
        self.repair_start_frame = 0
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
        QTimer.singleShot(0, self.refresh_robot_preview)

    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("appRoot")
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        header = QFrame()
        header.setObjectName("topBar")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(24, 14, 24, 14)
        header_layout.setSpacing(14)
        brand = QLabel("GQMR")
        brand.setObjectName("brand")
        product = QLabel("四足动作重定向工作台")
        product.setObjectName("productName")
        self.project_label = QLabel("未保存工程")
        self.project_label.setObjectName("projectState")
        header_layout.addWidget(brand)
        header_layout.addWidget(product)
        header_layout.addStretch(1)
        header_layout.addWidget(self.project_label)
        root_layout.addWidget(header)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setObjectName("workspaceSplitter")
        splitter.setHandleWidth(1)
        controls = QFrame()
        controls.setObjectName("sidebar")
        controls.setMinimumWidth(300)
        controls.setMaximumWidth(340)
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(18, 18, 18, 18)
        controls_layout.setSpacing(14)

        source_group = QGroupBox("输入动作")
        source_layout = QVBoxLayout(source_group)
        source_layout.setSpacing(10)
        self.motion_preset_combo = QComboBox()
        for preset in available_motion_presets():
            self.motion_preset_combo.addItem(preset.label, preset.id)
        self.motion_preset_combo.setCurrentIndex(
            self.motion_preset_combo.findData("trot_standard")
        )
        self.motion_preset_combo.setToolTip("固定参数、可重复生成，适合跨机器人对比")
        source_buttons = QHBoxLayout()
        self.demo_button = QPushButton("生成所选动作")
        self.import_button = QPushButton("导入动作…")
        source_buttons.addWidget(self.demo_button)
        source_buttons.addWidget(self.import_button)
        self.source_label = QLabel("尚未载入动作")
        self.source_label.setObjectName("sourceStatus")
        self.source_label.setWordWrap(True)
        self.source_label.setMinimumHeight(58)
        source_layout.addWidget(self.motion_preset_combo)
        source_layout.addLayout(source_buttons)
        source_layout.addWidget(self.source_label)
        controls_layout.addWidget(source_group)

        retarget_group = QGroupBox("机器人与求解")
        retarget_layout = QFormLayout(retarget_group)
        retarget_layout.setContentsMargins(14, 18, 14, 14)
        retarget_layout.setHorizontalSpacing(10)
        retarget_layout.setVerticalSpacing(10)
        self.robot_combo = QComboBox()
        for robot_id in available_robot_configs():
            self.robot_combo.addItem(get_asset_spec(robot_id).display_name, robot_id)
        self.retarget_mode = QComboBox()
        self.retarget_mode.addItem("快速（适合预览）", "fast")
        self.retarget_mode.addItem("高质量（接触优化）", "high-quality")
        self.cache_edit = QLineEdit()
        self.cache_edit.setText(str(default_asset_root()))
        self.cache_edit.setToolTip("完整机器人模型位于该目录的 assets/ 子目录")
        self.retarget_button = QPushButton("开始重定向")
        self.retarget_button.setObjectName("primaryButton")
        self.cancel_button = QPushButton("停止任务")
        self.cancel_button.setEnabled(False)
        self.batch_scope_combo = QComboBox()
        self.batch_scope_combo.addItem("当前机器人 × 全部动作", "current_robot")
        self.batch_scope_combo.addItem("全部机器人 × 全部动作", "all_robots")
        self.batch_button = QPushButton("批量评估泛化性能")
        self.batch_button.setToolTip("逐项运行固定动作测试集，并输出统一质量指标")
        self.model_status = QLabel("模型尚未加载")
        self.model_status.setObjectName("modelStatus")
        self.model_status.setWordWrap(True)
        self.model_status.setMinimumHeight(58)
        retarget_layout.addRow("目标型号", self.robot_combo)
        retarget_layout.addRow("处理方式", self.retarget_mode)
        retarget_layout.addRow("资产根目录", self.cache_edit)
        retarget_layout.addRow(self.model_status)
        retarget_layout.addRow(self.retarget_button)
        retarget_layout.addRow("评估范围", self.batch_scope_combo)
        retarget_layout.addRow(self.batch_button)
        retarget_layout.addRow(self.cancel_button)
        controls_layout.addWidget(retarget_group)

        export_group = QGroupBox("检查与输出")
        export_layout = QFormLayout(export_group)
        export_layout.setContentsMargins(14, 18, 14, 14)
        export_layout.setHorizontalSpacing(10)
        export_layout.setVerticalSpacing(10)
        self.format_combo = QComboBox()
        self.format_combo.addItem("Isaac Lab AMP", "isaaclab_amp_v232")
        self.format_combo.addItem("GQMR 标准 NPZ", "canonical")
        self.format_combo.addItem("DeepMimic JSON", "deepmimic")
        self.quality_button = QPushButton("生成质量报告")
        self.diagnose_button = QPushButton("分析问题帧")
        self.mark_repair_start_button = QPushButton("设为修复起点")
        self.repair_button = QPushButton("重算选中区间")
        self.export_button = QPushButton("导出…")
        export_layout.addRow("输出格式", self.format_combo)
        export_layout.addRow(self.quality_button)
        export_layout.addRow(self.diagnose_button)
        export_layout.addRow(self.mark_repair_start_button)
        export_layout.addRow(self.repair_button)
        export_layout.addRow(self.export_button)
        controls_layout.addWidget(export_group)
        controls_layout.addStretch(1)

        center = QWidget()
        center.setObjectName("workspace")
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(20, 18, 20, 18)
        center_layout.setSpacing(12)
        preview_header = QHBoxLayout()
        preview_title = QLabel("动作预览")
        preview_title.setObjectName("sectionTitle")
        self.preview_status = QLabel("拖动模型可旋转，滚轮可缩放")
        self.preview_status.setObjectName("previewHint")
        preview_header.addWidget(preview_title)
        preview_header.addStretch(1)
        preview_header.addWidget(self.preview_status)
        self.preview = MotionPreview()
        playback_panel = QFrame()
        playback_panel.setObjectName("playbackBar")
        playback = QHBoxLayout(playback_panel)
        playback.setContentsMargins(10, 8, 12, 8)
        self.play_button = QPushButton("播放")
        self.play_button.setObjectName("playButton")
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_label = QLabel("0 / 0")
        self.frame_label.setObjectName("frameCounter")
        playback.addWidget(self.play_button)
        playback.addWidget(self.frame_slider, 1)
        playback.addWidget(self.frame_label)
        self.diagnostics_timeline = DiagnosticsTimeline()
        self.diagnostics_label = QLabel("尚未运行问题帧分析")
        self.diagnostics_label.setWordWrap(True)
        report_header = QHBoxLayout()
        report_title = QLabel("运行记录")
        report_title.setObjectName("sectionTitle")
        report_header.addWidget(report_title)
        report_header.addStretch(1)
        self.report = QTextEdit()
        self.report.setObjectName("reportView")
        self.report.setReadOnly(True)
        self.report.setMaximumHeight(190)
        self.report.setPlaceholderText("质量报告、导出结果和错误信息会显示在这里。")
        center_layout.addLayout(preview_header)
        center_layout.addWidget(self.preview, 1)
        center_layout.addWidget(playback_panel)
        center_layout.addWidget(self.diagnostics_timeline)
        center_layout.addWidget(self.diagnostics_label)
        center_layout.addLayout(report_header)
        center_layout.addWidget(self.report)

        controls_scroll = QScrollArea()
        controls_scroll.setObjectName("sidebarScroll")
        controls_scroll.setFrameShape(QFrame.NoFrame)
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setMinimumWidth(300)
        controls_scroll.setMaximumWidth(340)
        controls_scroll.setWidget(controls)
        splitter.addWidget(controls_scroll)
        splitter.addWidget(center)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 1120])
        root_layout.addWidget(splitter, 1)
        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())

        self.demo_button.clicked.connect(self.generate_demo)
        self.import_button.clicked.connect(self.import_motion)
        self.retarget_button.clicked.connect(self.start_retarget)
        self.batch_button.clicked.connect(self.start_batch_evaluation)
        self.cancel_button.clicked.connect(self.cancel_task)
        self.quality_button.clicked.connect(self.run_quality)
        self.diagnose_button.clicked.connect(self.run_diagnostics)
        self.mark_repair_start_button.clicked.connect(self.mark_repair_start)
        self.repair_button.clicked.connect(self.repair_selected_range)
        self.export_button.clicked.connect(self.export_motion)
        self.play_button.clicked.connect(self.toggle_playback)
        self.frame_slider.valueChanged.connect(self.set_frame)
        self.diagnostics_timeline.frameClicked.connect(self.frame_slider.setValue)
        self.robot_combo.currentIndexChanged.connect(
            lambda _: self.refresh_robot_preview()
        )
        self.cache_edit.editingFinished.connect(self.refresh_robot_preview)
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
            "MuJoCo: Apache-2.0; Unitree / ANYmal / Deep Robotics assets: BSD-3-Clause\n\n"
            "完整第三方说明见 docs/THIRD_PARTY_LICENSES.md",
        )

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            * {
                font-family: "Noto Sans CJK SC", "Microsoft YaHei", sans-serif;
                font-size: 14px;
                color: #292724;
            }
            QMainWindow, QWidget#appRoot, QWidget#workspace {
                background: #f4f1eb;
            }
            QFrame#topBar {
                background: #ffffff;
                border-bottom: 1px solid #d8d3ca;
            }
            QLabel#brand {
                font-size: 22px;
                font-weight: 750;
                letter-spacing: 1px;
                color: #252422;
            }
            QLabel#productName {
                font-size: 15px;
                color: #716d66;
            }
            QLabel#projectState {
                padding: 5px 10px;
                border: 1px solid #d8d3ca;
                border-radius: 3px;
                background: #faf9f6;
                color: #625e57;
            }
            QFrame#sidebar {
                background: #ebe7df;
                border-right: 1px solid #d8d3ca;
            }
            QScrollArea#sidebarScroll {
                background: #ebe7df;
                border: none;
            }
            QScrollArea#sidebarScroll > QWidget > QWidget {
                background: #ebe7df;
            }
            QGroupBox {
                background: #ffffff;
                border: 1px solid #d3cec5;
                border-radius: 4px;
                margin-top: 13px;
                padding-top: 9px;
                font-size: 15px;
                font-weight: 650;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #34312d;
            }
            QLabel#sourceStatus, QLabel#modelStatus {
                background: #f7f5f0;
                border-left: 3px solid #b8522f;
                padding: 9px 10px;
                color: #4e4a44;
                line-height: 1.35;
            }
            QLabel#sectionTitle {
                font-size: 17px;
                font-weight: 700;
                color: #252422;
            }
            QLabel#previewHint {
                color: #716d66;
            }
            QPushButton {
                min-height: 34px;
                padding: 4px 12px;
                background: #ffffff;
                border: 1px solid #bdb7ad;
                border-radius: 4px;
                color: #302d29;
                font-weight: 550;
            }
            QPushButton:hover {
                background: #f4f1eb;
                border-color: #908980;
            }
            QPushButton:pressed {
                background: #e5e0d7;
            }
            QPushButton:disabled {
                color: #aaa49b;
                background: #f1eee8;
                border-color: #d9d4cb;
            }
            QPushButton#primaryButton {
                background: #b8522f;
                border-color: #9e4325;
                color: #ffffff;
                font-weight: 700;
            }
            QPushButton#primaryButton:hover {
                background: #a64829;
            }
            QPushButton#playButton {
                min-width: 72px;
                background: #34312d;
                border-color: #34312d;
                color: #ffffff;
            }
            QLineEdit, QComboBox {
                min-height: 32px;
                padding: 2px 8px;
                background: #ffffff;
                border: 1px solid #bdb7ad;
                border-radius: 3px;
                selection-background-color: #d9a58f;
            }
            QLineEdit:focus, QComboBox:focus {
                border: 1px solid #b8522f;
            }
            QComboBox::drop-down {
                width: 24px;
                border: none;
            }
            QFrame#playbackBar {
                background: #ffffff;
                border: 1px solid #d8d3ca;
                border-radius: 4px;
            }
            QLabel#frameCounter {
                min-width: 72px;
                color: #4e4a44;
            }
            QTextEdit#reportView {
                background: #ffffff;
                border: 1px solid #d8d3ca;
                border-radius: 4px;
                padding: 9px;
                color: #34312d;
                font-family: "Noto Sans Mono CJK SC", "DejaVu Sans Mono", monospace;
                font-size: 13px;
                selection-background-color: #d9a58f;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #d4cfc6;
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal {
                background: #b8522f;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                width: 16px;
                margin: -6px 0;
                border-radius: 8px;
                background: #ffffff;
                border: 2px solid #b8522f;
            }
            QMenuBar {
                background: #ffffff;
                border-bottom: 1px solid #d8d3ca;
                padding: 2px 8px;
            }
            QMenuBar::item:selected, QMenu::item:selected {
                background: #eee9e1;
            }
            QMenu {
                background: #ffffff;
                border: 1px solid #cfc9bf;
            }
            QStatusBar {
                background: #ffffff;
                border-top: 1px solid #d8d3ca;
                color: #625e57;
            }
            """
        )

    def _cache_dir(self) -> Path | None:
        text = self.cache_edit.text().strip()
        return Path(text) if text else None

    def _selected_robot_id(self) -> str:
        return str(self.robot_combo.currentData())

    def refresh_robot_preview(self) -> None:
        robot_id = self._selected_robot_id()
        display_name = self.robot_combo.currentText()
        self.model_status.setText(f"正在加载 {display_name}…")
        try:
            robot = load_robot_model(robot_id, cache_dir=self._cache_dir())
            self.preview.set_robot_model(robot)
            self.model_status.setText(
                f"{display_name} 已加载\n模型、网格与哈希校验通过"
            )
            self.preview_status.setText("拖动模型可旋转，滚轮可缩放")
        except Exception as error:
            self.preview.set_robot_model(None, error=str(error))
            self.model_status.setText(f"{display_name} 不可用\n{error}")
            self.preview_status.setText("请先安装或修复机器人资产")

    def _log(self, value) -> None:
        if isinstance(value, str):
            self.report.setPlainText(value)
        else:
            self.report.setPlainText(json.dumps(value, ensure_ascii=False, indent=2))

    def _update_enabled(self) -> None:
        busy = self.active_task is not None
        self.retarget_button.setEnabled(self.animal_motion is not None and not busy)
        self.quality_button.setEnabled(self.robot_motion is not None and not busy)
        self.diagnose_button.setEnabled(self.robot_motion is not None and not busy)
        self.mark_repair_start_button.setEnabled(
            self.robot_motion is not None and not busy
        )
        self.repair_button.setEnabled(
            self.robot_motion is not None
            and self.animal_motion is not None
            and not busy
        )
        self.export_button.setEnabled(self.robot_motion is not None and not busy)
        self.cancel_button.setEnabled(busy)
        self.batch_button.setEnabled(not busy)

    def set_animal_motion(self, motion: AnimalMotion, path: Path | None = None) -> None:
        self.animal_motion = motion
        self.animal_path = path
        self.robot_motion = None
        self.robot_path = None
        self.diagnostics = None
        self.motion_diagnostics = None
        self.diagnostics_timeline.set_diagnostics(None)
        self.edit_stack = EditStack(motion)
        self.edit_target = "animal"
        source = motion.metadata.get("source", {})
        source_name = path.name if path is not None else source.get(
            "preset_label", source.get("gait", "内存动作")
        )
        self.source_label.setText(
            f"{source_name}\n{motion.frame_count} 帧 · {motion.duration:.3f} 秒"
        )
        self.preview.set_motion(motion)
        self.frame_slider.setRange(0, motion.frame_count - 1)
        self.set_frame(0)
        self._update_enabled()

    def generate_demo(self) -> None:
        preset_id = str(self.motion_preset_combo.currentData())
        motion = generate_dog27_preset(preset_id)
        self.set_animal_motion(motion)
        source = motion.metadata["source"]
        self._log(
            {
                "generated": source["preset_label"],
                "preset_id": source["preset_id"],
                "gait": source["gait"],
                "speed_mps": source["speed_mps"],
                "cycle_hz": source["cycle_hz"],
                "turn_rate_rad_s": source["turn_rate_rad_s"],
                "frames": motion.frame_count,
                "license": source["license"],
            }
        )

    def import_motion(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "导入动物运动",
            "",
            "AnimalMotion / 3D keypoints (*.npz *.json *.csv *.txt);;All files (*)",
        )
        if not filename:
            return
        path = Path(filename)
        try:
            suffix = path.suffix.lower()
            if suffix == ".txt":
                motion = load_legacy_dog27(path)
            elif suffix == ".json":
                motion = keypoint_batch_to_animal_motion(
                    load_generic_keypoints_json(path)
                )
            elif suffix == ".csv":
                motion = keypoint_batch_to_animal_motion(
                    load_generic_keypoints_csv(path)
                )
            elif suffix == ".npz":
                try:
                    motion = load_motion(path)
                except Exception:
                    motion = keypoint_batch_to_animal_motion(
                        load_generic_keypoints_npz(path)
                    )
            else:
                raise ValueError("支持 .npz、.json、.csv 和旧版 dog-27 .txt")
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
        robot_id = self._selected_robot_id()
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
            self.motion_diagnostics = None
            self.diagnostics_timeline.set_diagnostics(None)
            self.edit_stack = EditStack(motion)
            self.edit_target = "robot"
            self.preview.set_robot_model(robot)
            self.preview.set_motion(self.animal_motion, motion, diagnostics)
            self.frame_slider.setRange(0, motion.frame_count - 1)
            self._log(replay_quality_report(motion, robot))
            self._update_enabled()

        self._run_task(work, complete)

    def start_batch_evaluation(self) -> None:
        cache = self._cache_dir()
        high_quality = self.retarget_mode.currentIndex() == 1
        if self.batch_scope_combo.currentData() == "all_robots":
            robot_ids = list(available_robot_configs())
        else:
            robot_ids = [self._selected_robot_id()]
        self._run_task(
            lambda token: evaluate_motion_suite(
                robot_ids,
                cache_dir=cache,
                high_quality=high_quality,
                cancelled=lambda: token.cancelled,
            ),
            self._log,
        )

    def run_quality(self) -> None:
        if self.robot_motion is None:
            return
        motion = self.robot_motion
        robot_id = self._selected_robot_id()
        cache = self._cache_dir()
        self._run_task(
            lambda token: replay_quality_report(
                motion, load_robot_model(robot_id, cache_dir=cache)
            ),
            self._log,
        )

    def run_diagnostics(self) -> None:
        if self.robot_motion is None:
            return
        motion = self.robot_motion
        retarget_diagnostics = self.diagnostics
        robot_id = self._selected_robot_id()
        cache = self._cache_dir()

        def complete(diagnostics) -> None:
            self.motion_diagnostics = diagnostics
            self.diagnostics_timeline.set_diagnostics(diagnostics)
            problems = diagnostics.problem_frames
            self._log(
                {
                    "problem_frames": problems.tolist(),
                    "problem_frame_count": int(len(problems)),
                    "invalid_frames": int(diagnostics.invalid.sum()),
                    "unreachable_frames": int(diagnostics.unreachable.sum()),
                    "self_collision_frames": int(diagnostics.self_collision.sum()),
                }
            )
            self.set_frame(self.frame_slider.value())

        self._run_task(
            lambda token: diagnose_motion(
                motion,
                load_robot_model(robot_id, cache_dir=cache),
                retarget_diagnostics,
            ),
            complete,
        )

    def mark_repair_start(self) -> None:
        self.repair_start_frame = self.frame_slider.value()
        self.statusBar().showMessage(
            f"局部修复起点：第 {self.repair_start_frame + 1} 帧", 3000
        )

    def repair_selected_range(self) -> None:
        if self.animal_motion is None or self.robot_motion is None:
            return
        start = min(self.repair_start_frame, self.frame_slider.value())
        stop = max(self.repair_start_frame, self.frame_slider.value())
        if stop <= start:
            self.statusBar().showMessage(
                "请先设置修复起点，再移动到区间终点", 4000
            )
            return
        animal = self.animal_motion
        initial = self.robot_motion
        robot_id = self._selected_robot_id()
        cache = self._cache_dir()

        def work(token):
            robot = load_robot_model(robot_id, cache_dir=cache)
            motion, diagnostics = retarget_high_quality(
                animal,
                robot,
                frame_range=(start, stop),
                initial_motion=initial,
            )
            return robot, motion, diagnostics

        def complete(result) -> None:
            robot, motion, diagnostics = result
            self.robot_motion = motion
            self.diagnostics = diagnostics
            self.motion_diagnostics = None
            self.diagnostics_timeline.set_diagnostics(None)
            self.edit_stack = EditStack(motion)
            self.preview.set_robot_model(robot)
            self.preview.set_motion(self.animal_motion, motion, diagnostics)
            self.frame_slider.setValue(stop)
            self._log(
                {
                    "local_repair": [start, stop],
                    "solver_residual_rmse_m": float(
                        (motion.solver_residual[start : stop + 1] ** 2).mean()
                        ** 0.5
                    ),
                }
            )
            self._update_enabled()

        self._run_task(work, complete)

    def export_motion(self) -> None:
        if self.robot_motion is None:
            return
        format_name = str(self.format_combo.currentData())
        suffix = ".json" if format_name == "deepmimic" else ".npz"
        filename, _ = QFileDialog.getSaveFileName(self, "导出", f"motion{suffix}")
        if not filename:
            return
        destination = Path(filename)
        motion = self.robot_motion
        robot_id = self._selected_robot_id()
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
        self.motion_diagnostics = None
        self.diagnostics_timeline.set_diagnostics(None)
        self.diagnostics_label.setText("尚未运行问题帧分析")
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
        self.diagnostics_timeline.set_frame(frame)
        self.frame_label.setText(f"{frame + 1} / {self.frame_slider.maximum() + 1}")
        if self.motion_diagnostics is not None:
            messages = self.motion_diagnostics.frame_messages(frame)
            position = self.motion_diagnostics.root_position_correction[frame]
            rotation = self.motion_diagnostics.root_rotation_correction[frame]
            summary = "、".join(messages) if messages else "未发现明显问题"
            self.diagnostics_label.setText(
                f"第 {frame + 1} 帧：{summary} | "
                f"根位置修正 {float((position ** 2).sum() ** 0.5) * 1000.0:.1f} mm | "
                f"根旋转修正 {float((rotation ** 2).sum() ** 0.5) * 57.2958:.1f}°"
            )

    def new_project(self) -> None:
        self.project = new_project()
        self.project_path = None
        self.project_label.setText("未保存工程")
        self.statusBar().showMessage("已新建工程", 3000)

    def open_project(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "打开 GQMR 工程", "", "GQMR (*.gqmr)")
        if not filename:
            return
        try:
            self.project = load_project(filename)
            self.project_path = Path(filename)
            self.project_label.setText(self.project_path.name)
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
                    robot_index = self.robot_combo.findData(motion.metadata["model_id"])
                    if robot_index >= 0:
                        self.robot_combo.setCurrentIndex(robot_index)
                    self.robot_motion = motion
                    self.robot_path = resource_path
                    self.diagnostics = None
                    self.motion_diagnostics = None
                    self.diagnostics_timeline.set_diagnostics(None)
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
            self.project_label.setText(self.project_path.name)
            self.statusBar().showMessage(f"已保存 {self.project_path}", 4000)
        except Exception as error:
            QMessageBox.critical(self, "保存失败", str(error))

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.active_task is not None:
            self.active_task.token.cancel()
            self.thread_pool.waitForDone(2000)
        self.preview.close_renderer()
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
