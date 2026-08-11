"""Build Stream Protocol layout declarations from a validated RobotModel."""

from __future__ import annotations

import mujoco

from gqmr.robots import RobotModel


def build_robot_welcome(robot: RobotModel, *, nominal_hz: float) -> dict:
    def joint_type(joint_id: int) -> str:
        return (
            "hinge"
            if int(robot.model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_HINGE)
            else "slide"
        )

    return {
        "model_id": robot.config.id,
        "model_sha256": robot.config.model_sha256,
        "coordinate_frame": "mujoco_model",
        "quaternion_order": "wxyz",
        "qpos_layout": [
            {
                "joint": "$root",
                "type": "free",
                "adr": robot.root_qpos_address,
                "size": 7,
            },
            *[
                {
                    "joint": name,
                    "type": joint_type(int(joint_id)),
                    "adr": int(address),
                    "size": 1,
                }
                for name, joint_id, address in zip(
                    robot.config.dof_order, robot.joint_ids, robot.qpos_addresses
                )
            ],
        ],
        "qvel_layout": [
            {
                "joint": "$root",
                "type": "free",
                "adr": robot.root_dof_address,
                "size": 6,
            },
            *[
                {
                    "joint": name,
                    "type": joint_type(int(joint_id)),
                    "adr": int(address),
                    "size": 1,
                }
                for name, joint_id, address in zip(
                    robot.config.dof_order, robot.joint_ids, robot.dof_addresses
                )
            ],
        ],
        "site_names": [],
        "nominal_hz": float(nominal_hz),
        "clock": "monotonic_ns",
    }
