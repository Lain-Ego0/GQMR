"""Motion retargeting algorithms."""

from gqmr.retarget.animal_preprocess import (
    BodyScaleEstimate,
    RootEstimate,
    estimate_body_scale,
    estimate_contact_probability,
    estimate_root_motion,
)
from gqmr.retarget.benchmark import evaluate_motion_suite
from gqmr.retarget.diagnostics import MotionDiagnostics, diagnose_motion
from gqmr.retarget.dynamics import PDReplayConfig, simulate_pd_tracking
from gqmr.retarget.fast import (
    FastRetargetConfig,
    FastRetargetError,
    RetargetDiagnostics,
    retarget_fast,
)
from gqmr.retarget.high_quality import (
    HighQualityRetargetConfig,
    retarget_high_quality,
)
from gqmr.retarget.quality import replay_quality_report
from gqmr.retarget.preprocess import (
    AnimalPreprocessConfig,
    AnimalPreprocessReport,
    GroundEstimate,
    estimate_ground_plane,
    preprocess_animal_motion,
    reestimate_contact_and_ground,
)

__all__ = [
    "BodyScaleEstimate",
    "RootEstimate",
    "estimate_body_scale",
    "estimate_contact_probability",
    "estimate_root_motion",
    "FastRetargetConfig",
    "FastRetargetError",
    "RetargetDiagnostics",
    "retarget_fast",
    "replay_quality_report",
    "MotionDiagnostics",
    "diagnose_motion",
    "evaluate_motion_suite",
    "HighQualityRetargetConfig",
    "retarget_high_quality",
    "PDReplayConfig",
    "simulate_pd_tracking",
    "AnimalPreprocessConfig",
    "AnimalPreprocessReport",
    "GroundEstimate",
    "estimate_ground_plane",
    "preprocess_animal_motion",
    "reestimate_contact_and_ground",
]
