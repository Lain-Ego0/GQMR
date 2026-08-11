from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

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
