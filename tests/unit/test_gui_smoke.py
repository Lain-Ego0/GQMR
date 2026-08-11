from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MUJOCO_GL", "egl")

from PySide6.QtWidgets import QApplication

from gqmr.assets import default_asset_root
from gqmr.core.io import load_motion
from gqmr.synthetic import generate_dog27_motion
from gqmr.ui.app import MainWindow


def test_gui_window_and_preview_smoke() -> None:
    application = QApplication.instance() or QApplication([])
    window = MainWindow()
    motion = generate_dog27_motion("trot", duration=0.2, fps=20.0)

    window.set_animal_motion(motion)
    window.show()
    application.processEvents()

    assert window.windowTitle().startswith("GQMR")
    assert window.frame_slider.maximum() == motion.frame_count - 1
    assert window.retarget_button.isEnabled()
    window.close()


def test_gui_renders_go2_and_b2_models() -> None:
    asset_root = os.environ.get("GQMR_TEST_ASSET_ROOT") or os.environ.get(
        "GQMR_TEST_ASSET_CACHE"
    )
    application = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.cache_edit.setText(str(asset_root or default_asset_root()))

    go2_index = window.robot_combo.findData("unitree-go2")
    window.robot_combo.setCurrentIndex(go2_index)
    window.refresh_robot_preview()
    assert window.preview.rendered_robot_id == "unitree-go2"
    assert window.preview.has_robot_render

    repository = Path(__file__).resolve().parents[2]
    motion = load_motion(repository / "examples/demo/trot.go2.npz")
    window.preview.set_motion(None, motion)
    window.preview.set_frame(5)
    assert window.preview.has_robot_render

    b2_index = window.robot_combo.findData("unitree-b2")
    window.robot_combo.setCurrentIndex(b2_index)
    window.refresh_robot_preview()
    assert window.preview.rendered_robot_id == "unitree-b2"
    assert window.preview.has_robot_render
    window.close()
