# GQMR Motion Schema v1

> Schema 状态：冻结  
> 四元数：`[w, x, y, z]`  
> 坐标：右手，X 前、Y 左、Z 上  
> NPZ 加载：必须使用 `allow_pickle=False`

## 1. 通用规则

- 所有数组使用 C contiguous 布局写出。
- 浮点运动数组使用 little-endian `float32`，时间戳使用 little-endian `float64`。
- 字符串数组使用 NumPy Unicode dtype，不使用 object dtype。
- `metadata_json` 是 UTF-8 JSON 编码后的 `uint8[M]`。
- JSON 中不允许 `NaN`、`Infinity` 或任意可执行对象。
- `timestamps[0]` 规范化为 `0.0`，并严格递增。
- 坐标单位固定为米、弧度、秒、米每秒和弧度每秒。
- 空间数据出现 NaN 的帧必须同时将对应有效掩码置为 false；导出器默认拒绝 NaN。

## 2. AnimalMotion v1

文件扩展名建议：`.animal.npz`。

| Key | dtype/shape | 必填 | 说明 |
|---|---|---:|---|
| `schema_id` | Unicode scalar | 是 | 固定 `gqmr.animal_motion` |
| `schema_version` | Unicode scalar | 是 | 固定 `1.0` |
| `timestamps` | `float64[T]` | 是 | 严格递增，从 0 开始 |
| `keypoint_names` | Unicode `[K]` | 是 | 唯一、非空 |
| `positions` | `float32[T,K,3]` | 是 | 世界坐标 |
| `confidence` | `float32[T,K]` | 是 | `[0,1]` |
| `valid_mask` | `bool[T,K]` | 是 | 点是否可用于求解 |
| `contact_probability` | `float32[T,4]` | 是 | 顺序 FL、FR、RL、RR；未知填 NaN |
| `frame_valid` | `bool[T]` | 是 | 整帧是否满足最小骨架要求 |
| `metadata_json` | `uint8[M]` | 是 | 见下方 |

`metadata_json` 必须包含：

```json
{
  "coordinate_frame": "gqmr_world_x_forward_y_left_z_up",
  "length_unit": "m",
  "time_unit": "s",
  "skeleton_id": "dog-27",
  "skeleton_sha256": "...",
  "contact_order": ["FL", "FR", "RL", "RR"],
  "contact_source": "unknown|heuristic|mujoco|manual|mixed",
  "source": {},
  "created_by": {"gqmr_version": "..."}
}
```

## 3. RobotMotion v1

文件扩展名建议：`.robot.npz`。

v1 只表达自由根节点加标量 DOF。多自由度关节必须先转换为等价的具名标量 DOF，否则拒绝加载。

| Key | dtype/shape | 必填 | 说明 |
|---|---|---:|---|
| `schema_id` | Unicode scalar | 是 | 固定 `gqmr.robot_motion` |
| `schema_version` | Unicode scalar | 是 | 固定 `1.0` |
| `timestamps` | `float64[T]` | 是 | 严格递增，从 0 开始 |
| `dof_names` | Unicode `[N]` | 是 | 模型标量 DOF 名称和顺序 |
| `root_position` | `float32[T,3]` | 是 | 世界坐标 |
| `root_rotation` | `float32[T,4]` | 是 | `wxyz`，把根局部向量旋转到世界坐标 |
| `dof_position` | `float32[T,N]` | 是 | 弧度或米，类型由模型定义 |
| `root_linear_velocity` | `float32[T,3]` | 是 | 世界坐标 |
| `root_angular_velocity` | `float32[T,3]` | 是 | 世界坐标角速度向量 |
| `dof_velocity` | `float32[T,N]` | 是 | rad/s 或 m/s |
| `foot_contact_probability` | `float32[T,4]` | 是 | FL、FR、RL、RR |
| `frame_valid` | `bool[T]` | 是 | 是否允许默认导出 |
| `solver_status` | `int16[T]` | 是 | 状态码 |
| `solver_residual` | `float32[T]` | 是 | 加权残差 |
| `metadata_json` | `uint8[M]` | 是 | 模型、坐标和生成信息 |

状态码：

| 值 | 名称 | 含义 |
|---:|---|---|
| 0 | `OK` | 正常收敛 |
| 1 | `DEGRADED_ROOT` | 根姿态使用历史方向或降级估计 |
| 2 | `MAX_ITER` | 达到迭代上限但结果有限 |
| 3 | `UNREACHABLE` | 足端目标不可达 |
| 4 | `MISSING_INPUT` | 输入关键点不足 |
| 5 | `NUMERICAL_ERROR` | 非有限数或分解失败 |
| 6 | `INTERPOLATED` | 用户确认后由短区间修复生成 |

`metadata_json` 必须包含：

```json
{
  "coordinate_frame": "gqmr_world_x_forward_y_left_z_up",
  "quaternion_order": "wxyz",
  "root_velocity_frame": "world",
  "model_id": "unitree-go2",
  "model_source_commit": "ae6a8403e272733e9996ef59990880330496177f",
  "model_sha256": "...",
  "robot_config_sha256": "...",
  "contact_order": ["FL", "FR", "RL", "RR"],
  "source_motion_sha256": "...",
  "retarget_config": {},
  "created_by": {"gqmr_version": "..."}
}
```

## 4. 验证规则

- 时间戳差值必须全部 `> 0`。
- 所有名称唯一；模型中的 DOF 集合必须与 `dof_names` 完全一致，顺序可以由名称重排。
- `dof_names` 的规范顺序来自机器人配置 `dof_order`，不能从 MJCF 声明、`qpos` 地址或 actuator 顺序隐式推断。
- 四元数范数与 1 的差必须 `< 1e-5`，相邻帧点积为负时在写出前翻转后一个四元数。
- 置信度范围为 `[0,1]`；接触概率允许 NaN 表示未知，否则范围为 `[0,1]`。
- `frame_valid=true` 时，根、DOF 和速度字段必须全部有限。
- `solver_status>=3` 的帧默认必须是 `frame_valid=false`，除非人工修复为状态 6。
- 模型 hash 不匹配时禁止回放或导出，除非用户执行明确的模型迁移操作。

## 5. 速度和重采样

- 非等间隔时间戳先在原时间轴进行导数估计，再重采样。
- 位置和标量 DOF 使用分段三次 Hermite；数据不足时退化为线性插值。
- 根旋转使用最短弧 SLERP。
- 根线速度和根角速度始终从规范化根轨迹重新计算，不直接假设外部 MuJoCo `qvel` 已经使用目标坐标表达。
- 根角速度由相邻旋转的 quaternion log map 除以时间差计算，不对欧拉角差分。
- 中间帧使用中心差分，边界使用二阶单边差分；不足 3 帧时拒绝生成训练速度。
- 滤波先处理位置/DOF，再重新计算速度。旋转滤波在 SO(3) 切空间完成。

## 6. Isaac Lab v2.3.2 导出

导出器固定名称 `isaaclab_amp_v232`，默认重采样为 60 Hz。输出字段严格匹配 Isaac Lab v2.3.2 `MotionLoader`：

```text
fps                         int scalar
dof_names                   Unicode[N]
body_names                  Unicode[B]
dof_positions               float32[T,N]
dof_velocities              float32[T,N]
body_positions              float32[T,B,3]
body_rotations              float32[T,B,4] wxyz
body_linear_velocities      float32[T,B,3] world
body_angular_velocities     float32[T,B,3] world
```

所有 body 状态由固定 hash 的 MuJoCo 模型对 `RobotMotion` 做 FK 生成，不能信任来源文件里可能过期的 body cache。默认 Go2 body 包括全部具名 body；AMP 环境至少使用 `base_link` 和四个足端 body。

导出验收：原版 `MotionLoader` 成功加载；随机采样 1000 个时间点；恢复到 MuJoCo 后 body 位置误差 `<1e-5 m`、旋转误差 `<1e-5 rad`。
