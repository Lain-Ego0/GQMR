# GQMR MuJoCo Stream Protocol v1

> 状态：冻结  
> 传输：ZeroMQ ROUTER/DEALER  
> 编码：UTF-8 JSON header + little-endian raw ndarray frames

## 1. 范围

协议只负责从外部 MuJoCo 进程可靠、非阻塞地采集状态。v1 不提供远程控制、执行任意命令、模型上传或 Python 对象反序列化。

外部进程中的 `GQMRPublisher.publish()` 只把快照复制进本地有界环形缓冲，不进行网络阻塞。专用发送线程拥有 ZeroMQ socket。

## 2. 连接模型

- Publisher 使用 ROUTER，默认绑定 `tcp://127.0.0.1:5570`。
- GQMR recorder 使用 DEALER，主动连接 publisher。
- 默认只允许 loopback。绑定非 loopback 时必须配置 CurveZMQ server/client key。
- 一个 publisher 可以服务多个只读 recorder；每个客户端有独立信用窗口和丢帧统计。

## 3. 消息

所有消息第一段是 ASCII 类型，第二段是 JSON header；`FRAME` 后跟零个或多个原始数组段。

### HELLO

客户端发送：

```json
{
  "protocol": "gqmr.mujoco_stream",
  "version": 1,
  "client_id": "uuid",
  "requested_fields": ["qpos", "qvel", "site_xpos"],
  "credit": 256
}
```

### WELCOME

Publisher 返回：

```json
{
  "protocol": "gqmr.mujoco_stream",
  "version": 1,
  "session_id": "uuid",
  "model_id": "unitree-go2",
  "model_sha256": "...",
  "coordinate_frame": "mujoco_model",
  "quaternion_order": "wxyz",
  "qpos_layout": [
    {"joint":"$root","type":"free","adr":0,"size":7},
    {"joint":"FL_hip_joint","type":"hinge","adr":7,"size":1}
  ],
  "qvel_layout": [
    {"joint":"$root","type":"free","adr":0,"size":6},
    {"joint":"FL_hip_joint","type":"hinge","adr":6,"size":1}
  ],
  "site_names": [],
  "nominal_hz": 500.0,
  "clock": "monotonic_ns"
}
```

无名 free joint 使用保留名称 `$root`。客户端必须在开始录制前校验协议版本、模型 hash、完整 layout、字段形状和名称。任何不匹配均发送 `ERROR` 并停止录制。

### FRAME

```json
{
  "session_id": "uuid",
  "seq": 123,
  "timestamp_ns": 1234567890,
  "wall_time_ns": null,
  "arrays": [
    {"name": "qpos", "dtype": "<f8", "shape": [19], "part": 0},
    {"name": "qvel", "dtype": "<f8", "shape": [18], "part": 1}
  ]
}
```

- `seq` 从 0 开始严格递增。
- `timestamp_ns` 来自源进程 monotonic clock，不得使用可回拨的系统墙钟排序。
- 每个数组段长度必须与 dtype、shape 精确一致，最大单帧负载默认 16 MiB。
- 非有限值不会自动丢弃，但 recorder 将该帧标为无效并报告。

### ACK

客户端每消费最多 64 帧或每 100 ms 返回累计 ACK：

```json
{"session_id":"uuid","ack_seq":123,"credit":64}
```

### GAP

Publisher 本地缓冲溢出、客户端跟不上或源主动跳帧时发送：

```json
{"session_id":"uuid","first_missing":124,"last_missing":140,"reason":"ring_overflow"}
```

Recorder 必须在时间轴保存 GAP，不得插值后伪装为原始连续采集。后处理可以生成派生修复轨迹。

### HEARTBEAT / ERROR / BYE

- 空闲 1 秒发送 HEARTBEAT。
- 3 秒没有数据或 heartbeat，客户端进入 disconnected 状态。
- ERROR 包含稳定错误代码和可读消息。
- 正常关闭发送 BYE，并包含最后序列号。

## 4. 缓冲和背压

- Publisher 环形缓冲默认 4096 帧，写入不阻塞仿真线程。
- 缓冲满时丢最旧帧，并为每个受影响客户端发送 GAP。
- 客户端初始信用 256；只在收到 ACK 后补充发送额度。
- Recorder 落盘使用独立线程和分块临时文件，完成后转换为 canonical motion。
- v1 不做压缩；若负载超过 1 MiB/帧，再通过协议能力协商增加压缩，不修改已有消息语义。

## 5. qpos 语义转换

- 原始 `qpos/qvel` 作为审计数据保存时必须绑定模型 hash。
- Recorder 使用握手 layout 和 MuJoCo 模型交叉解析自由根和标量 DOF，不直接硬编码索引。
- MuJoCo free joint 四元数按 `wxyz` 读取。
- canonical DOF 顺序来自机器人配置 `dof_order`，不是 qpos 或 actuator 顺序。
- canonical 世界系根速度由根轨迹重新计算；原始 qvel 保留用于审计和一致性检查。
- 外部源若发送 site/body 状态，它们只是交叉检查或源语义映射输入；与 qpos FK 不一致超过 `1e-5 m` 时报告错误。
- 从源机器人生成 `AnimalMotion` 必须经过显式语义映射配置，例如 base、hip、knee 和 foot；不能把目标机器人特定 qpos 直接称为统一动物骨架。

## 6. 验收

- 在参考机上以 200 Hz 连续录制 30 分钟。
- 协议自身丢帧率 `<0.1%`，全部丢帧都有 GAP。
- 启用 publisher 后源仿真实时因子下降 `<5%`。
- 正常网络往返的端到端 p95 延迟 `<50 ms`。
- 断线重连产生新 session，不把两个 monotonic clock 区间静默拼接。
- 错误模型 hash、错误 dtype、超长消息和乱序序列都有自动测试。
