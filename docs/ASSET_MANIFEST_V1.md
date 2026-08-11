# GQMR Asset Manifest v1

> 状态：实施协议 v1  
> 适用资产：`unitree-go2`、`unitree-b2`

## 1. 可信根

内置 manifest 固定以下上游来源：

```text
repository:     https://github.com/unitreerobotics/unitree_mujoco
commit:         ae6a8403e272733e9996ef59990880330496177f
archive_sha256: 824a51b228c317348866180b1214ed736621d2163006d682156d54b6a55da711
license:        BSD-3-Clause
```

运行时不接受远端 manifest，不跟随 branch/tag，不使用下载后自动发现的文件替换内置清单。升级资产必须修改仓库中的 manifest 并经过人工审查。

## 2. Manifest 结构

```json
{
  "schema_id": "gqmr.asset_manifest",
  "schema_version": 1,
  "asset_id": "unitree-go2",
  "display_name": "Unitree Go2",
  "source": {
    "repository": "...",
    "commit": "40 lowercase hex",
    "archive_url": "...fixed commit...tar.gz",
    "archive_sha256": "64 lowercase hex",
    "archive_prefix": "single archive root/"
  },
  "license": {"spdx": "BSD-3-Clause", "path": "LICENSE"},
  "model_path": "unitree_robots/go2/scene.xml",
  "model_sha256": "64 lowercase hex",
  "files": [
    {"path": "LICENSE", "sha256": "...", "size": 1559}
  ]
}
```

所有 `path` 使用 POSIX 相对路径，禁止绝对路径、`.`、`..` 和重复路径。文件必须同时匹配精确字节数与 SHA-256。

## 3. 模型身份 hash

`model_sha256` 标识会影响 MuJoCo 模型语义的 XML/mesh 集合。它不等于 `scene.xml` 单文件 hash，也不包含许可证文件。

按 manifest 中 `files` 的声明顺序，对许可证以外的每个文件依次输入：

```text
UTF8(path) + NUL
ASCII(file_sha256) + NUL
ASCII(decimal_size) + LF
```

最终取上述完整字节流的 SHA-256。文件内容、路径、大小或 manifest 声明顺序变化都会产生新的模型身份。

当前值：

| 资产 | `model_sha256` |
|---|---|
| Unitree Go2 | `48baeb791c25c3fdaca0163c614145ade0e29d710ee9fcce9d8a5f551e3ca2e1` |
| Unitree B2 | `2ebeb90cb3cee67b4ae37e719244454b854719db126d9394ed89d3f0c9ec76e5` |

`RobotMotion.metadata_json.model_sha256`、缓存键、回放和导出模型绑定使用此值。

## 4. 安装规则

默认缓存根通过 `platformdirs.user_cache_path("gqmr")` 获取；测试和离线部署可以显式传入缓存根。安装目录为：

```text
<cache>/assets/<asset_id>/<source_commit>/
```

安装器流程：

1. 下载或读取用户指定的固定提交 tar.gz。
2. 在解压前校验整个归档 SHA-256。
3. 检查所有归档成员路径、重复名、类型、数量和总解压大小。
4. 只提取 manifest 声明的文件到同目录临时目录。
5. 每个文件边写边校验大小和 SHA-256，并执行 `fsync`。
6. 全部通过后原子激活目录；已有损坏目录只有显式 `--repair` 才替换。

状态校验不信任安装记录，会重新读取所有声明文件计算 hash；符号链接、意外文件、缺失文件或安装记录不一致都会使状态变为 `corrupt`。

## 5. 离线包

离线包扩展名建议为 `.gqmr-assets`，格式为 ZIP64：

```text
asset.gqmr-assets
├── pack.json
├── manifest.json
└── files/
    ├── LICENSE
    └── unitree_robots/...
```

`pack.json` 固定包含 `asset_id`、内置 manifest 的规范 JSON SHA-256 和 `model_sha256`。解包时必须：

- 用 `asset_id` 选择随软件发布的内置 manifest；
- 要求包内 manifest 与内置 manifest JSON 语义完全一致；
- 要求 ZIP 成员集合与 manifest 完全一致；
- 检查重复名、路径穿越、非普通文件、异常压缩比和解压总量；
- 对每个文件重新校验大小与 SHA-256 后再原子激活。

离线包不允许引入新资产 ID 或覆盖内置可信 manifest。
