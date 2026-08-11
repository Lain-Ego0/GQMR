"""Read-only candidate recognition for v1-compatible MJCF robots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco

from gqmr.robots.model import RobotModelError


def inspect_mjcf_candidates(path: str | Path) -> dict[str, Any]:
    try:
        model = mujoco.MjModel.from_xml_path(str(path))
    except (OSError, ValueError) as error:
        raise RobotModelError(f"cannot load candidate MJCF {path}: {error}") from error
    free = [
        joint_id
        for joint_id in range(model.njnt)
        if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE)
    ]
    unsupported: list[str] = []
    scalar: list[dict[str, Any]] = []
    for joint_id in range(model.njnt):
        if joint_id in free:
            continue
        joint_type = int(model.jnt_type[joint_id])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if joint_type not in {
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        }:
            unsupported.append(name or f"joint-{joint_id}")
            continue
        scalar.append(
            {
                "name": name,
                "type": "hinge"
                if joint_type == int(mujoco.mjtJoint.mjJNT_HINGE)
                else "slide",
                "limited": bool(model.jnt_limited[joint_id]),
                "range": model.jnt_range[joint_id].tolist(),
                "qpos_address": int(model.jnt_qposadr[joint_id]),
                "dof_address": int(model.jnt_dofadr[joint_id]),
            }
        )
    names_by_type: dict[str, list[str]] = {}
    for label, object_type, count in (
        ("bodies", mujoco.mjtObj.mjOBJ_BODY, model.nbody),
        ("sites", mujoco.mjtObj.mjOBJ_SITE, model.nsite),
        ("geoms", mujoco.mjtObj.mjOBJ_GEOM, model.ngeom),
    ):
        names_by_type[label] = [
            name
            for object_id in range(count)
            if (name := mujoco.mj_id2name(model, object_type, object_id))
        ]
    leg_order = ("FL", "FR", "RL", "RR")
    suggested_dofs: list[str] = []
    tokens = ("hip", "thigh", "calf", "knee", "ankle")
    for leg in leg_order:
        candidates = [
            item["name"]
            for item in scalar
            if item["name"] and item["name"].upper().startswith(leg)
        ]
        candidates.sort(
            key=lambda name: next(
                (index for index, token in enumerate(tokens) if token in name.lower()),
                99,
            )
        )
        suggested_dofs.extend(candidates)
    foot_candidates: dict[str, list[dict[str, str]]] = {}
    for leg in leg_order:
        matches: list[dict[str, str]] = []
        for object_type in ("sites", "bodies", "geoms"):
            for name in names_by_type[object_type]:
                lower = name.lower()
                if name.upper().startswith(leg) and any(
                    token in lower for token in ("foot", "toe", "calf", "ankle")
                ):
                    matches.append({"type": object_type[:-1], "name": name})
        foot_candidates[leg] = matches
    reasons: list[str] = []
    if len(free) != 1:
        reasons.append("v1 requires exactly one free root joint")
    if unsupported:
        reasons.append(f"unsupported non-scalar joints: {unsupported}")
    if len(scalar) != 12:
        reasons.append(f"v1 requires 12 scalar joints, found {len(scalar)}")
    if any(not item["name"] or not item["limited"] for item in scalar):
        reasons.append("all scalar joints must be named and hard-limited")
    if len(suggested_dofs) != 12:
        reasons.append("could not infer 3 prefixed joints for every FL/FR/RL/RR leg")
    if any(not values for values in foot_candidates.values()):
        reasons.append("one or more legs have no semantic foot candidate")
    return {
        "path": str(path),
        "nq": model.nq,
        "nv": model.nv,
        "nu": model.nu,
        "free_joint_count": len(free),
        "scalar_joints": scalar,
        "unsupported_joints": unsupported,
        "named_objects": names_by_type,
        "suggested_dof_order": suggested_dofs,
        "foot_candidates": foot_candidates,
        "v1_candidate": not reasons,
        "reasons": reasons,
        "notice": "candidate only; user confirmation is required before saving a config",
    }
