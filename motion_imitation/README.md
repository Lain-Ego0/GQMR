# Quadruped Motion Retargeting

This trimmed project keeps only the motion-retargeting part of the original
`motion_imitation` repository. It converts dog motion-capture joint positions
to quadruped robot poses with PyBullet inverse kinematics.

## Contents

- `retarget_motion/retarget_motion.py`: retargeting and visualization
- `retarget_motion/retarget_config_*.py`: A1, Laikago and Vision60 mappings
- `retarget_motion/data/*_joint_pos.txt`: source motion-capture joint positions

## Install

```bash
python3 -m pip install -r requirements.txt
```

## Run

The default target is Laikago:

```bash
python3 retarget_motion/retarget_motion.py
```

To target A1 or Vision60, change the active config import near the top of
`retarget_motion.py`. Generated motion files are written into the
`retarget_motion/` directory.

The original license is retained in `LICENSE.txt`, and the source motion data
license is retained in `retarget_motion/data/LICENSE.txt`.
