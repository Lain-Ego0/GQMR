#!/usr/bin/env bash
set -euo pipefail

repository_root="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
external_root="${GQMR_EXTERNAL_DATA_ROOT:-$repository_root/external_data}"
source_root="$repository_root/motion_imitation/retarget_motion/data"
dog_root="$external_root/ai4animation-dog27"

mkdir -p "$dog_root/motions" "$external_root/pferd" "$external_root/rgbd-dog" "$external_root/acinoset"

cp "$source_root/LICENSE.txt" "$dog_root/LICENSE.txt"
for source_path in "$source_root"/*_joint_pos.txt; do
    cp "$source_path" "$dog_root/motions/$(basename -- "$source_path")"
done

GQMR_EXTERNAL_ROOT="$external_root" GQMR_REPOSITORY_ROOT="$repository_root" \
    python3 - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

repository_root = Path(os.environ["GQMR_REPOSITORY_ROOT"])
external_root = Path(os.environ["GQMR_EXTERNAL_ROOT"])
motion_root = external_root / "ai4animation-dog27" / "motions"

files = []
for path in sorted(motion_root.glob("*_joint_pos.txt")):
    files.append(
        {
            "path": str(path.relative_to(external_root)),
            "bytes": path.stat().st_size,
            "frames": sum(1 for line in path.open(encoding="utf-8") if line.strip()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    )

manifest = {
    "schema": "gqmr.local_external_datasets.v1",
    "git_tracked": False,
    "datasets": [
        {
            "id": "ai4animation-dog27",
            "installed": True,
            "source": "https://github.com/sebastianstarke/AI4Animation",
            "license": "CC-BY-NC-4.0",
            "redistributable_with_gqmr": False,
            "files": files,
            "total_frames": sum(item["frames"] for item in files),
        },
        {
            "id": "pferd",
            "installed": any((external_root / "pferd").iterdir()),
            "source": "https://doi.org/10.7910/DVN/2EXONE",
            "license": "custom-dataverse-terms-review-required",
            "redistributable_with_gqmr": False,
        },
        {
            "id": "rgbd-dog",
            "installed": any((external_root / "rgbd-dog").iterdir()),
            "source": "https://github.com/CAMERA-Bath/RGBD-Dog",
            "license": "academic-only-signed-agreement",
            "redistributable_with_gqmr": False,
        },
        {
            "id": "acinoset",
            "installed": any((external_root / "acinoset").iterdir()),
            "source": "https://github.com/African-Robotics-Unit/AcinoSet",
            "license": "not-clearly-specified-review-required",
            "redistributable_with_gqmr": False,
        },
    ],
}
(external_root / "LOCAL_DATASETS.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(
    f"Installed {len(files)} AI4Animation dog-27 clips "
    f"({manifest['datasets'][0]['total_frames']} frames) into {external_root}"
)
PY

printf '%s\n' "PFERD:    https://doi.org/10.7910/DVN/2EXONE -> $external_root/pferd/"
printf '%s\n' "RGBD-Dog: https://github.com/CAMERA-Bath/RGBD-Dog -> $external_root/rgbd-dog/"
printf '%s\n' "AcinoSet: https://github.com/African-Robotics-Unit/AcinoSet -> $external_root/acinoset/"
