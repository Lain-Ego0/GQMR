from __future__ import annotations

from types import SimpleNamespace

import gqmr.retarget.benchmark as benchmark


def test_motion_suite_reports_every_robot_action_and_preserves_failures(monkeypatch) -> None:
    def fake_load(robot_id, cache_dir=None):
        if robot_id == "broken":
            raise RuntimeError("missing asset")
        return SimpleNamespace(config=SimpleNamespace(id=robot_id))

    def fake_retarget(animal, robot):
        preset_id = animal.metadata["source"]["preset_id"]
        if preset_id == "pace_standard":
            raise RuntimeError("solver failed")
        return SimpleNamespace(preset_id=preset_id), None

    def fake_report(motion, robot):
        return {
            "valid_frame_ratio": 1.0,
            "solver_residual_rmse_m": 0.01,
            "solver_residual_p95_m": 0.02,
            "joint_limit_violation_frames": 0,
            "self_collision_frames": 0,
            "maximum_ground_penetration_m": 0.0,
            "mean_contact_foot_speed_mps": 0.03,
        }

    monkeypatch.setattr(benchmark, "load_robot_model", fake_load)
    monkeypatch.setattr(benchmark, "retarget_fast", fake_retarget)
    monkeypatch.setattr(benchmark, "replay_quality_report", fake_report)

    report = benchmark.evaluate_motion_suite(
        ["working", "broken"], duration=0.1, fps=20.0
    )

    assert report["summary"]["requested_evaluations"] == 16
    assert report["summary"]["completed_evaluations"] == 7
    assert report["summary"]["failed_evaluations"] == 2
    assert {row["preset_id"] for row in report["results"]} == {
        "walk_slow",
        "walk_standard",
        "trot_slow",
        "trot_standard",
        "trot_fast",
        "turn_left",
        "turn_right",
    }
    assert {error["robot_id"] for error in report["errors"]} == {
        "working",
        "broken",
    }


def test_motion_suite_cancellation_returns_partial_report(monkeypatch) -> None:
    monkeypatch.setattr(
        benchmark,
        "load_robot_model",
        lambda robot_id, cache_dir=None: SimpleNamespace(
            config=SimpleNamespace(id=robot_id)
        ),
    )
    report = benchmark.evaluate_motion_suite(
        ["robot"], cancelled=lambda: True
    )

    assert report["cancelled"] is True
    assert report["summary"]["completed_evaluations"] == 0
    assert report["summary"]["requested_evaluations"] == 8
