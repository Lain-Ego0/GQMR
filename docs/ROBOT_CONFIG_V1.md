# GQMR Robot Config v1

> 状态：实施协议 v1  
> 后端：MuJoCo only

## 1. 适用模型

v1 机器人必须满足：

- 恰好一个 free root joint；
- 恰好 12 个具名标量 hinge/slide DOF；
- 每个标量 DOF 都有硬限位；
- 四条腿语义固定为 `FL、FR、RL、RR`；
- 足端使用具名 body 加局部点表达；
- 模型资产已经通过 GQMR 内置 manifest 验证。

ball joint、闭链、轮腿、耦合关节、无硬限位 DOF 和多个自由根在加载阶段拒绝。

## 2. YAML 结构

```yaml
schema_version: 1
id: unitree-go2
asset_id: unitree-go2
model: unitree_robots/go2/scene.xml
model_sha256: 48baeb791c25c3fdaca0163c614145ade0e29d710ee9fcce9d8a5f551e3ca2e1
root_joint: $root
base_body: base_link
dof_order: [FL_hip_joint, FL_thigh_joint, FL_calf_joint, ...]
feet:
  FL:
    body: FL_foot
    local_position: [0.0, 0.0, 0.0]
    contact_geoms: [FL]
default_root_position: [0.0, 0.0, 0.27]
default_root_rotation: [1.0, 0.0, 0.0, 0.0]
default_dof_position: [0.0, 0.9, -1.8, ...]
```

配置使用 PyYAML safe loader 和 Pydantic v2，拒绝重复 key 和未知字段。`model` 必须是安全 POSIX 相对路径；所有 hash 使用 64 位小写十六进制。

## 3. 根关节

`root_joint: $root` 是业务保留别名，表示模型中唯一的 free joint：

- Go2 上游 free joint 无名称；
- B2 上游名称为 `floating_base_joint`；
- 两者在 GQMR 业务层都解析为 `$root`。

如果配置使用具体名称，则必须与上游名称完全一致。无 free joint 或多个 free joint 都拒绝加载。

根 qpos 通过 MuJoCo `jnt_qposadr` 获取，布局为世界位置 3 + `wxyz` 四元数 4。根 qvel 地址通过 `jnt_dofadr` 获取，业务层不会硬编码数组偏移。

## 4. DOF 顺序

`dof_order` 是所有核心数据、IK 和导出的唯一业务顺序。加载器通过名称解析每个 joint 的：

- MuJoCo joint ID；
- qpos address；
- qvel/DOF address；
- joint type；
- hard range。

模型声明顺序、qpos 顺序、qvel 顺序和 actuator 顺序都不能替代 `dof_order`。模型中的具名标量 DOF 集合必须与配置完全一致。

当前 Go2/B2 的业务顺序均为：

```text
FL hip/thigh/calf,
FR hip/thigh/calf,
RL hip/thigh/calf,
RR hip/thigh/calf
```

## 5. 足端与接触

足端世界位置为：

```text
body_xpos + body_xmat @ local_position
```

Go2 有 `FL_foot/FR_foot/RL_foot/RR_foot` body，局部点为零，并有具名 `FL/FR/RL/RR` contact geom。

B2 没有 foot body/site。配置绑定对应 calf body，局部点为 `[0, 0, -0.35] m`。`contact_geoms` 为空时，加载器收集该 calf body 下所有 `contype != 0` 或 `conaffinity != 0` 的 collision geom，用于后续接触聚合。

## 6. 配置身份

`robot_config_sha256` 是 Pydantic 验证后配置的规范 JSON SHA-256：

- UTF-8；
- JSON key 排序；
- 紧凑分隔符；
- 禁止 NaN/Infinity。

当前值：

| 配置 | `robot_config_sha256` |
|---|---|
| Unitree Go2 | `68e508dbbcb210b8a6b155535ca4f7232b8af9f13f973c39098d6db34c00d1a0` |
| Unitree B2 | `3188c8673e3a4ce8fead8401e087f15f218c68c7669808923d5f9e58146009d6` |

B2 值包含可用于重定向的站立默认姿态：根高 `0.52 m`，每腿 `[0.0, 0.8, -1.6] rad`。上游折叠 `home` 不作为产品默认姿态。

## 7. 数值 API

每个 `RobotModel` 拥有自己的 `MjData`，提供：

- 设置世界系根位置、`wxyz` 根旋转和业务顺序 12 DOF；
- 查询具名 body 位置；
- 查询 FL/FR/RL/RR 足端世界位置；
- 查询单足 `3×12` 或四足 `4×3×12` 平移 Jacobian；
- 查询接触 geom IDs、qpos/qvel 地址和关节限位。

Jacobian 使用 `mj_jac` 计算完整 `3×nv` 矩阵后，按配置解析出的业务 DOF 地址重排，不假设 MuJoCo 内部顺序。
