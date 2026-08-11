"""Validated MuJoCo model binding and kinematics in business DOF order."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray

from gqmr.assets import status_asset
from gqmr.core.coordinates import normalize_quaternions
from gqmr.core.errors import GQMRError
from gqmr.robots.config import LEG_ORDER, RobotConfig, get_robot_config


class RobotModelError(GQMRError, ValueError):
    """Raised when a MuJoCo model violates the v1 robot contract."""


@dataclass(frozen=True, slots=True)
class FootBinding:
    body_id: int
    local_position: NDArray[np.float64]
    contact_geom_ids: tuple[int, ...]


class RobotModel:
    """One MuJoCo model plus one private MjData and validated name bindings."""

    def __init__(self, model: mujoco.MjModel, config: RobotConfig) -> None:
        self.model = model
        self.config = config
        self.data = mujoco.MjData(model)
        self.root_joint_id = self._resolve_root_joint()
        self.root_qpos_address = int(model.jnt_qposadr[self.root_joint_id])
        self.root_dof_address = int(model.jnt_dofadr[self.root_joint_id])
        self.base_body_id = self._name_id(mujoco.mjtObj.mjOBJ_BODY, config.base_body)
        self.joint_ids, self.qpos_addresses, self.dof_addresses = self._resolve_dofs()
        self.actuator_ids = self._resolve_actuators()
        self.joint_ranges = np.ascontiguousarray(model.jnt_range[self.joint_ids], dtype=np.float64)
        self.feet = self._resolve_feet()
        self.set_pose(
            config.default_root_position,
            config.default_root_rotation,
            config.default_dof_position,
        )

    @classmethod
    def from_xml_path(cls, path: str | os.PathLike[str], config: RobotConfig) -> "RobotModel":
        try:
            model = mujoco.MjModel.from_xml_path(str(path))
        except (ValueError, OSError) as error:
            raise RobotModelError(f"cannot load MuJoCo model {path}: {error}") from error
        return cls(model, config)

    def _name_id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise RobotModelError(f"MuJoCo model is missing {object_type.name} {name!r}")
        return int(object_id)

    def _resolve_root_joint(self) -> int:
        free_type = int(mujoco.mjtJoint.mjJNT_FREE)
        free_joint_ids = np.flatnonzero(self.model.jnt_type == free_type)
        if len(free_joint_ids) != 1:
            raise RobotModelError("v1 requires exactly one free root joint")
        joint_id = int(free_joint_ids[0])
        actual_name = mujoco.mj_id2name(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
        )
        if self.config.root_joint != "$root" and actual_name != self.config.root_joint:
            raise RobotModelError(
                f"configured root joint {self.config.root_joint!r} does not match {actual_name!r}"
            )
        return joint_id

    def _resolve_dofs(self) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.int32]]:
        allowed_types = {
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        }
        scalar_names: list[str] = []
        for joint_id in range(self.model.njnt):
            if joint_id == self.root_joint_id:
                continue
            if int(self.model.jnt_type[joint_id]) not in allowed_types:
                raise RobotModelError("v1 only supports scalar hinge/slide joints")
            name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
            )
            if not name:
                raise RobotModelError("all scalar joints must have names")
            scalar_names.append(name)
        if set(scalar_names) != set(self.config.dof_order):
            missing = sorted(set(self.config.dof_order) - set(scalar_names))
            extra = sorted(set(scalar_names) - set(self.config.dof_order))
            raise RobotModelError(
                f"configured DOF set does not match model; missing={missing}, extra={extra}"
            )
        joint_ids = np.array(
            [self._name_id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.config.dof_order],
            dtype=np.int32,
        )
        if not np.all(self.model.jnt_limited[joint_ids]):
            raise RobotModelError("all v1 scalar DOFs must have hard joint limits")
        qpos = np.asarray(self.model.jnt_qposadr[joint_ids], dtype=np.int32)
        dof = np.asarray(self.model.jnt_dofadr[joint_ids], dtype=np.int32)
        return joint_ids, qpos, dof

    def _resolve_feet(self) -> dict[str, FootBinding]:
        result: dict[str, FootBinding] = {}
        for leg in LEG_ORDER:
            foot = self.config.feet[leg]
            body_id = self._name_id(mujoco.mjtObj.mjOBJ_BODY, foot.body)
            if foot.contact_geoms:
                geom_ids = tuple(
                    self._name_id(mujoco.mjtObj.mjOBJ_GEOM, name)
                    for name in foot.contact_geoms
                )
            else:
                geom_ids = tuple(
                    geom_id
                    for geom_id in range(self.model.ngeom)
                    if int(self.model.geom_bodyid[geom_id]) == body_id
                    and (
                        int(self.model.geom_contype[geom_id]) != 0
                        or int(self.model.geom_conaffinity[geom_id]) != 0
                    )
                )
            if not geom_ids:
                raise RobotModelError(f"foot {leg} has no collision geoms")
            result[leg] = FootBinding(
                body_id=body_id,
                local_position=np.asarray(foot.local_position, dtype=np.float64),
                contact_geom_ids=geom_ids,
            )
        return result

    def _resolve_actuators(self) -> NDArray[np.int32]:
        actuator_ids: list[int] = []
        for joint_id, name in zip(self.joint_ids, self.config.dof_order):
            matches = [
                actuator_id
                for actuator_id in range(self.model.nu)
                if int(self.model.actuator_trnid[actuator_id, 0]) == int(joint_id)
            ]
            if len(matches) > 1:
                raise RobotModelError(
                    f"DOF {name!r} has multiple directly bound actuators"
                )
            actuator_ids.append(matches[0] if matches else -1)
        return np.asarray(actuator_ids, dtype=np.int32)

    def set_pose(
        self,
        root_position: ArrayLike,
        root_rotation_wxyz: ArrayLike,
        dof_position: ArrayLike,
    ) -> None:
        root_position_array = np.asarray(root_position, dtype=np.float64)
        dof_position_array = np.asarray(dof_position, dtype=np.float64)
        root_rotation = normalize_quaternions(root_rotation_wxyz)
        if root_position_array.shape != (3,) or not np.all(np.isfinite(root_position_array)):
            raise RobotModelError("root_position must be finite shape (3,)")
        if root_rotation.shape != (4,):
            raise RobotModelError("root_rotation must have shape (4,)")
        if dof_position_array.shape != (len(self.config.dof_order),) or not np.all(
            np.isfinite(dof_position_array)
        ):
            raise RobotModelError("dof_position has invalid shape or non-finite values")
        lower, upper = self.joint_ranges[:, 0], self.joint_ranges[:, 1]
        tolerance = 1e-6
        if np.any(dof_position_array < lower - tolerance) or np.any(
            dof_position_array > upper + tolerance
        ):
            raise RobotModelError("dof_position violates a hard joint limit")
        dof_position_array = np.clip(dof_position_array, lower, upper)
        self.data.qpos[:] = self.model.qpos0
        start = self.root_qpos_address
        self.data.qpos[start : start + 3] = root_position_array
        self.data.qpos[start + 3 : start + 7] = root_rotation
        self.data.qpos[self.qpos_addresses] = dof_position_array
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def body_position(self, body_name: str) -> NDArray[np.float64]:
        body_id = self._name_id(mujoco.mjtObj.mjOBJ_BODY, body_name)
        return np.asarray(self.data.xpos[body_id], dtype=np.float64).copy()

    def named_bodies(self) -> tuple[str, ...]:
        """Return all named non-world bodies in stable MuJoCo body-ID order."""

        names: list[str] = []
        for body_id in range(1, self.model.nbody):
            name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, body_id
            )
            if name:
                names.append(name)
        return tuple(names)

    def body_pose(self, body_name: str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return world position and wxyz orientation for a named body."""

        body_id = self._name_id(mujoco.mjtObj.mjOBJ_BODY, body_name)
        return (
            np.asarray(self.data.xpos[body_id], dtype=np.float64).copy(),
            np.asarray(self.data.xquat[body_id], dtype=np.float64).copy(),
        )

    def foot_position(self, leg: str) -> NDArray[np.float64]:
        try:
            binding = self.feet[leg]
        except KeyError as error:
            raise RobotModelError(f"unknown leg {leg!r}") from error
        rotation = self.data.xmat[binding.body_id].reshape(3, 3)
        return np.asarray(
            self.data.xpos[binding.body_id] + rotation @ binding.local_position,
            dtype=np.float64,
        ).copy()

    def foot_positions(self) -> NDArray[np.float64]:
        return np.stack([self.foot_position(leg) for leg in LEG_ORDER])

    def foot_jacobian(self, leg: str) -> NDArray[np.float64]:
        try:
            binding = self.feet[leg]
        except KeyError as error:
            raise RobotModelError(f"unknown leg {leg!r}") from error
        point = self.foot_position(leg)
        jacobian_position = np.zeros((3, self.model.nv), dtype=np.float64)
        jacobian_rotation = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jac(
            self.model,
            self.data,
            jacobian_position,
            jacobian_rotation,
            point,
            binding.body_id,
        )
        return np.ascontiguousarray(jacobian_position[:, self.dof_addresses])

    def foot_jacobians(self) -> NDArray[np.float64]:
        return np.stack([self.foot_jacobian(leg) for leg in LEG_ORDER])

    def collision_metrics(self) -> tuple[int, float]:
        """Return self-contact count and maximum ground-plane penetration depth."""

        self_contacts = 0
        maximum_ground_penetration = 0.0
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            body_a = int(self.model.geom_bodyid[contact.geom1])
            body_b = int(self.model.geom_bodyid[contact.geom2])
            if body_a != 0 and body_b != 0:
                self_contacts += 1
            elif contact.dist < 0.0:
                world_geom = contact.geom1 if body_a == 0 else contact.geom2
                if int(self.model.geom_type[world_geom]) != int(
                    mujoco.mjtGeom.mjGEOM_PLANE
                ):
                    continue
                maximum_ground_penetration = max(
                    maximum_ground_penetration, float(-contact.dist)
                )
        return self_contacts, maximum_ground_penetration


def load_robot_model(
    robot_id: str, *, cache_dir: str | os.PathLike[str] | None = None
) -> RobotModel:
    config = get_robot_config(robot_id)
    status = status_asset(config.asset_id, cache_dir=cache_dir)
    if not status.valid:
        raise RobotModelError(
            f"asset {config.asset_id} is not verified ({status.state}); run gqmr assets install"
        )
    if status.model_sha256 != config.model_sha256:
        raise RobotModelError("robot config model_sha256 does not match the asset manifest")
    expected_path = Path(status.install_path) / config.model
    if expected_path != Path(status.model_path):
        raise RobotModelError("robot config model path does not match the asset manifest")
    return RobotModel.from_xml_path(expected_path, config)
