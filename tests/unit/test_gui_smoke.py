from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from PySide6.QtWidgets import QApplication, QFileDialog

from gqmr.assets import default_asset_root
from gqmr.core.io import load_motion
from gqmr.pose.api import KeypointBatch, PoseBackendInfo
from gqmr.synthetic import available_motion_presets, generate_dog27_motion
from gqmr.ui import app as app_module
from gqmr.ui.app import MainWindow


def _make_ap10k_batch() -> KeypointBatch:
    names = (
        "L_Eye",
        "R_Eye",
        "Nose",
        "Neck",
        "Root of tail",
        "L_Shoulder",
        "L_Elbow",
        "L_F_Paw",
        "R_Shoulder",
        "R_Elbow",
        "R_F_Paw",
        "L_Hip",
        "L_Knee",
        "L_B_Paw",
        "R_Hip",
        "R_Knee",
        "R_B_Paw",
    )
    frame = np.array(
        [
            [85, 55],
            [88, 57],
            [70, 65],
            [120, 90],
            [300, 100],
            [130, 105],
            [145, 145],
            [155, 190],
            [135, 108],
            [155, 150],
            [175, 190],
            [270, 110],
            [260, 150],
            [245, 190],
            [275, 112],
            [290, 155],
            [305, 190],
        ],
        dtype=np.float32,
    )
    positions = np.stack((frame, frame + [2, 0], frame + [4, 0]))[:, None]
    return KeypointBatch(
        timestamps=[0.0, 0.1, 0.2],
        keypoint_names=names,
        instance_ids=("dog-0",),
        positions=positions,
        confidence=np.ones((3, 1, len(names))),
        valid_mask=np.ones((3, 1, len(names)), dtype=bool),
        coordinate_frame="image_pixels_x_right_y_down",
        metadata={"fixture": True},
    )


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
    assert not window.diagnose_button.isEnabled()
    assert not window.repair_button.isEnabled()
    assert window.environment_button.isEnabled()
    window.close()


def test_gui_exposes_repeatable_generalization_motion_suite() -> None:
    application = QApplication.instance() or QApplication([])
    window = MainWindow()

    expected = [preset.id for preset in available_motion_presets()]
    actual = [
        window.motion_preset_combo.itemData(index)
        for index in range(window.motion_preset_combo.count())
    ]
    assert actual == expected
    right_index = window.motion_preset_combo.findData("turn_right")
    window.motion_preset_combo.setCurrentIndex(right_index)
    window.generate_demo()

    assert window.animal_motion is not None
    assert window.animal_motion.metadata["source"]["preset_id"] == "turn_right"
    assert "右转" in window.source_label.text()
    assert window.batch_scope_combo.findData("all_robots") >= 0
    window.close()


def test_gui_exposes_installed_dog_video_pose_backend(monkeypatch) -> None:
    class FixtureBackend:
        def describe(self) -> PoseBackendInfo:
            return PoseBackendInfo(
                api_version=1,
                name="MMPose dog 2D",
                package="fixture",
                package_version="1",
                skeleton_ids=("ap10k",),
                dimensions=(2,),
                multi_instance=False,
                batch_range=(1, 64),
                devices=("cuda",),
                output_coordinate_frame="image_pixels_x_right_y_down",
            )

    monkeypatch.setattr(
        app_module, "discover_pose_backends", lambda: {"dog-mmpose": FixtureBackend}
    )
    application = QApplication.instance() or QApplication([])
    window = MainWindow()

    backend_index = window.pose_backend_combo.findData("dog-mmpose")
    assert backend_index >= 0
    assert window.pose_config_edit.text().endswith("dog-mmpose.cuda.json")
    assert not window.video_extract_button.isEnabled()

    window.video_pose_path = Path("fixture-dog.mp4")
    window._update_enabled()
    assert window.video_extract_button.isEnabled()
    assert "2D" in window.video_extract_button.text()
    window.close()


def test_gui_monocular_lift_generates_and_loads_animal_motion(
    tmp_path: Path, monkeypatch
) -> None:
    application = QApplication.instance() or QApplication([])
    window = MainWindow()
    batch = _make_ap10k_batch()
    source = tmp_path / "dog.2d.npz"
    output = tmp_path / "dog.experimental.animal.npz"
    window._set_video_pose_batch(batch, source)
    window.monocular_preprocess.setChecked(False)

    assert window.monocular_lift_button.isEnabled()
    assert window.monocular_instance_combo.currentData() == "dog-0"
    assert window.monocular_facing_combo.currentData() == "auto"
    assert window.monocular_torso_length.value() == 0.48
    assert window.monocular_body_width.value() == 0.28

    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(output), "NPZ (*.npz)"),
    )

    def run_immediately(function, success, failure=None) -> None:
        try:
            success(function(SimpleNamespace(cancelled=False)))
        except Exception as error:
            if failure is not None:
                failure(str(error))
            raise

    window._run_task = run_immediately
    window.start_monocular_lift()

    assert output.is_file()
    assert window.animal_motion is not None
    assert window.animal_motion.metadata["source"]["format"] == (
        "experimental_monocular_ap10k_lift_v1"
    )
    assert window.animal_path == output
    assert window.retarget_button.isEnabled()
    assert "dog-27" in window.monocular_status.text()
    window.close()


def test_gui_renders_go2_and_b2_models() -> None:
    asset_root = os.environ.get("GQMR_TEST_ASSET_ROOT") or os.environ.get(
        "GQMR_TEST_ASSET_CACHE"
    )
    application = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.cache_edit.setText(str(asset_root or default_asset_root()))
    assert {
        window.robot_combo.itemData(index)
        for index in range(window.robot_combo.count())
    } == {
        "unitree-go2", "unitree-go1", "unitree-a1", "unitree-a2",
        "unitree-b2", "anybotics-anymal-c",
        "deeprobotics-lite3",
    }

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
