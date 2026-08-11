from __future__ import annotations

from pathlib import Path

import numpy as np

from gqmr.robots.config import RobotConfig
from gqmr.robots.model import RobotModel


def _synthetic_xml() -> str:
    legs = []
    # Deliberately declare FR before FL so business order cannot rely on MJCF order.
    for leg, x, y in (
        ("FR", 0.2, -0.1),
        ("FL", 0.2, 0.1),
        ("RR", -0.2, -0.1),
        ("RL", -0.2, 0.1),
    ):
        legs.append(
            f"""
            <body name="{leg}_hip" pos="{x} {y} 0">
              <joint name="{leg}_hip_joint" axis="1 0 0" range="-1 1"/>
              <geom type="sphere" size="0.01" contype="0" conaffinity="0"/>
              <body name="{leg}_thigh" pos="0 {'0.1' if y > 0 else '-0.1'} 0">
                <joint name="{leg}_thigh_joint" axis="0 1 0" range="-2 2"/>
                <geom type="sphere" size="0.01" contype="0" conaffinity="0"/>
                <body name="{leg}_calf" pos="0 0 -0.2">
                  <joint name="{leg}_calf_joint" axis="0 1 0" range="-2 0"/>
                  <body name="{leg}_foot" pos="0 0 -0.2"/>
                  <geom name="{leg}" type="sphere" size="0.02" pos="0 0 -0.2"/>
                </body>
              </body>
            </body>
            """
        )
    return f"""
    <mujoco model="synthetic-quadruped">
      <compiler angle="radian"/>
      <worldbody>
        <body name="base_link" pos="0 0 0.3">
          <freejoint name="floating_base"/>
          <geom type="box" size="0.1 0.05 0.03" contype="0" conaffinity="0"/>
          {''.join(legs)}
        </body>
      </worldbody>
    </mujoco>
    """


def _synthetic_config() -> RobotConfig:
    dofs = tuple(
        f"{leg}_{joint}_joint"
        for leg in ("FL", "FR", "RL", "RR")
        for joint in ("hip", "thigh", "calf")
    )
    return RobotConfig.model_validate(
        {
            "schema_version": 1,
            "id": "synthetic",
            "asset_id": "synthetic",
            "model": "scene.xml",
            "model_sha256": "a" * 64,
            "root_joint": "$root",
            "base_body": "base_link",
            "dof_order": dofs,
            "feet": {
                leg: {
                    "body": f"{leg}_foot",
                    "local_position": [0, 0, 0],
                    "contact_geoms": [leg],
                }
                for leg in ("FL", "FR", "RL", "RR")
            },
            "default_root_position": [0, 0, 0.3],
            "default_root_rotation": [1, 0, 0, 0],
            "default_dof_position": [0, 0.5, -1] * 4,
        }
    )


def test_business_dof_order_and_foot_jacobian(tmp_path: Path) -> None:
    model_path = tmp_path / "synthetic.xml"
    model_path.write_text(_synthetic_xml(), encoding="utf-8")
    robot = RobotModel.from_xml_path(model_path, _synthetic_config())
    assert tuple(robot.config.dof_order[:3]) == (
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
    )
    assert robot.qpos_addresses.tolist()[:3] != [7, 8, 9]
    assert robot.foot_positions().shape == (4, 3)

    pose = np.array([0.1, 0.4, -0.9] * 4)
    robot.set_pose([0, 0, 0.3], [1, 0, 0, 0], pose)
    analytic = robot.foot_jacobian("FL")
    finite = np.zeros_like(analytic)
    step = 1e-6
    for index in range(12):
        plus = pose.copy()
        minus = pose.copy()
        plus[index] += step
        minus[index] -= step
        robot.set_pose([0, 0, 0.3], [1, 0, 0, 0], plus)
        plus_position = robot.foot_position("FL")
        robot.set_pose([0, 0, 0.3], [1, 0, 0, 0], minus)
        minus_position = robot.foot_position("FL")
        finite[:, index] = (plus_position - minus_position) / (2 * step)
    relative_error = np.linalg.norm(analytic - finite) / np.linalg.norm(finite)
    assert relative_error < 1e-4
