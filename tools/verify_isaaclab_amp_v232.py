"""Execute the frozen Isaac Lab v2.3.2 MotionLoader against a GQMR AMP file."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("motion_loader", type=Path)
    parser.add_argument("amp", type=Path)
    parser.add_argument("--samples", type=int, default=1000)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "isaaclab_v232_motion_loader", args.motion_loader
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import MotionLoader source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loader = module.MotionLoader(str(args.amp), torch.device("cpu"))
    times = loader.sample_times(args.samples)
    outputs = loader.sample(args.samples, times=times)
    if not all(torch.isfinite(value).all() for value in outputs):
        raise RuntimeError("MotionLoader sampling produced non-finite values")
    reversed_dofs = list(reversed(loader.dof_names))
    if loader.get_dof_index(reversed_dofs) != list(reversed(range(loader.num_dofs))):
        raise RuntimeError("MotionLoader DOF name reordering failed")
    result = {
        "frames": loader.num_frames,
        "dofs": loader.num_dofs,
        "bodies": loader.num_bodies,
        "samples": args.samples,
        "finite": True,
        "name_reordering": True,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
