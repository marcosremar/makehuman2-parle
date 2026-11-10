"""Regression for the shape of the shoulders (2026-11-09): run with `Blender -b --python-exit-code 1 -P tests/test_parle_shoulders.py`.

Lowered from the bind pose the top of the arm stood out as a ball and the slope from the neck was flat. The cap goes in, the slope rises, and the rest of the body stays.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import parle_polish  # noqa: E402

NECK = np.array([0.0, 0.0, 1.45])
LEFT, RIGHT = np.array([0.19, 0.0, 1.36]), np.array([-0.19, 0.0, 1.36])
ELBOWS = [np.array([0.38, 0.0, 1.15]), np.array([-0.38, 0.0, 1.15])]
CAP, RISE = parle_polish.SHOULDER_CAP_M["female"], parle_polish.SHOULDER_RISE_M["female"]
field = parle_polish.shoulder_field(NECK, [LEFT, RIGHT], ELBOWS, CAP, RISE)
arm = (ELBOWS[0] - LEFT) / np.linalg.norm(ELBOWS[0] - LEFT)
top = np.array([0.0, 0.0, 1.0]) - arm[2] * arm
top /= np.linalg.norm(top)


def moved(point):
    return field(np.array([point], float))[0] - point


cap = LEFT + top * 0.06 + arm * 0.03
step = moved(cap)
assert step @ top < -0.7 * CAP, step
mirror = cap * np.array([-1, 1, 1])
assert abs(moved(mirror) - step * np.array([-1, 1, 1])).max() < 1e-9
slope = np.array([0.08, 0.0, 1.42])
assert moved(slope)[2] > 0.7 * RISE, moved(slope)
for still in ([0.0, -0.12, 1.2], [0.0, 0.0, 1.62], [0.45, 0.0, 1.05], [0.0, 0.0, 1.0], [0.05, -0.1, 1.28]):
    assert np.abs(moved(still)).max() < 1e-6, (still, moved(still))
# A garment is only as whole as the field is smooth: no jump between points a millimetre apart around the shoulder.
grid = np.mgrid[0.02:0.36:0.001, 1.25:1.6:0.002].reshape(2, -1).T
points = np.stack([grid[:, 0], np.full(len(grid), 0.02), grid[:, 1]], axis=1)
jump = np.linalg.norm((field(points + [0.001, 0, 0]) - (points + [0.001, 0, 0])) - (field(points) - points), axis=1)
assert jump.max() < 0.0015, jump.max()
# Men (2026-11-10): the end of the shoulder comes down and in, the neck, the chest and the forearm stay, and a woman's shoulder is the same as before.
MALE = [parle_polish.SHOULDER_CAP_M["male"], parle_polish.SHOULDER_RISE_M["male"], parle_polish.SHOULDER_DROP_M["male"], parle_polish.SHOULDER_IN_M["male"]]
man = parle_polish.shoulder_field(NECK, [LEFT, RIGHT], ELBOWS, *MALE)
flat = parle_polish.shoulder_field(NECK, [LEFT, RIGHT], ELBOWS, MALE[0], MALE[1])
extra = lambda point: (man(np.array([point], float)) - flat(np.array([point], float)))[0]
tip = extra(LEFT + [0.02, 0.0, 0.04])
assert tip[2] < -0.9 * MALE[2] and tip[0] < -0.9 * MALE[3], tip
assert abs(extra(RIGHT + [-0.02, 0.0, 0.04]) - tip * np.array([-1, 1, 1])).max() < 1e-9
for still in ([0.0, -0.12, 1.2], [0.0, 0.0, 1.62], [0.45, 0.0, 1.05], [0.0, 0.0, 1.0], [0.05, -0.1, 1.28], [0.1, 0.0, 1.5]):
    assert np.abs(extra(still)).max() < 1e-6, (still, extra(still))
assert parle_polish.SHOULDER_DROP_M["female"] == 0 and parle_polish.SHOULDER_IN_M["female"] == 0
jump = np.linalg.norm((man(points + [0.001, 0, 0]) - (points + [0.001, 0, 0])) - (man(points) - points), axis=1)
assert jump.max() < 0.0015, jump.max()
print("ok")
