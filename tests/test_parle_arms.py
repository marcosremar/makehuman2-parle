"""Regression for the length of the arms (2026-11-10): run with `Blender -b --python-exit-code 1 -P tests/test_parle_arms.py`.

The women's upper arm was 13.3% of the height against 15.4% in two whole-body scans; the men's were close to their reference and only even out. Each bone stretches evenly to its share, the shoulder stays and the hand is carried whole.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import parle_polish  # noqa: E402

HEIGHT = 1.63
JOINT, ELBOW, WRIST = np.array([0.17, 0.0, 1.30]), np.array([0.27, 0.0, 1.108]), np.array([0.36, 0.01, 0.93])
upper_m, fore_m = np.linalg.norm(ELBOW - JOINT), np.linalg.norm(WRIST - ELBOW)
upper, fore = parle_polish.arm_scales(HEIGHT, upper_m, fore_m, "female")
assert abs(upper * upper_m / HEIGHT - parle_polish.ARM_UPPER_SHARE["female"]) < 1e-9 and abs(fore * fore_m / HEIGHT - parle_polish.ARM_FORE_SHARE["female"]) < 1e-9, (upper, fore)
assert upper > 1.05 and parle_polish.arm_scales(HEIGHT, upper_m, fore_m, "male")[0] > upper
# A body far from the measure moves only as far as the sleeve allows.
assert parle_polish.arm_scales(HEIGHT, 0.1, 0.5, "female") == (parle_polish.ARM_SCALE_LIMITS[1], parle_polish.ARM_SCALE_LIMITS[0])
shift = parle_polish.arm_shift(JOINT, ELBOW, WRIST, upper, fore)
moved = lambda point: shift(np.array([point], float))[0]
assert np.abs(moved(JOINT)).max() < 1e-12 and np.abs(moved(JOINT + (ELBOW - JOINT) * 0.08)).max() < 1e-12
new_elbow, new_wrist = ELBOW + moved(ELBOW), WRIST + moved(WRIST)
assert abs(np.linalg.norm(new_elbow - JOINT) - upper * upper_m) < 1e-9
assert abs(np.linalg.norm(new_wrist - new_elbow) - fore * fore_m) < 1e-9
# The bones keep pointing the same way, so the clips, which only turn them, play as before.
assert np.linalg.norm(np.cross(new_elbow - JOINT, ELBOW - JOINT)) < 1e-12 and np.linalg.norm(np.cross(new_wrist - new_elbow, WRIST - ELBOW)) < 1e-12
# The hand goes whole: a fingertip moves exactly as the wrist does.
tip = WRIST + (WRIST - ELBOW) * 0.9 + [0.02, 0.03, 0.0]
assert np.abs(moved(tip) - moved(WRIST)).max() < 1e-12
# Even stretch, no jump: a millimetre along the arm never opens more than the stretch of the bone.
steps = np.linspace(0, 1, 2001)
for start, end, scale in ((JOINT, ELBOW, upper), (ELBOW, WRIST, fore)):
    line = start + np.outer(steps, end - start)
    gaps = np.linalg.norm(np.diff(line + shift(line), axis=0), axis=1) / np.linalg.norm(np.diff(line, axis=0), axis=1)
    assert gaps.max() < 1 + (scale - 1) / (1 - parle_polish.ARM_STILL_SHARE) + 1e-6 and gaps.min() > 1 - 1e-6, (gaps.min(), gaps.max())
print("ok")
