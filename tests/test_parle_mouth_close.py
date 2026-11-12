"""Regression for the thread of tooth in the middle of /m/ (2026-11-11): run with `.local/venv/bin/python tests/test_parle_mouth_close.py`.

The cast had no lip-closing shape, so the game sealed /p b m/ by closing the jaw, and a thread of tooth showed between the lips.
`mouthClose` at the weight of `jawOpen` must bring the lower lip back to its rest while the chin stays open.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
from parle_mouth_close import closing_weight, lower_lip, mouth_close  # noqa: E402

# A profile strip every 2 mm: upper lip (z > 0), lower lip (0 to -6 mm), the skin under it and the chin (down to -40 mm).
heights = np.arange(0.01, -0.04, -0.002)
rest = np.array([[x, 0.0, z] for x in (-0.01, 0.0, 0.01) for z in heights])
lower_face = rest[:, 2] < 0
# The jaw carries the lower face 8 mm down and 2 mm back; the upper lip stays.
jaw = np.where(lower_face[:, None], np.array([0.0, 0.002, -0.008]), 0.0)
# The lip visemes move the lips (both), most at the meeting line.
lip_move = np.where(np.abs(rest[:, 2]) < 0.007, 0.004, 0.0004)
lip = lower_lip(rest, jaw, [lip_move])
assert len(lip) and np.all(rest[lip, 2] < 0) and np.all(rest[lip, 2] > -0.007), "the lower lip is the moved part the jaw carries"

close = mouth_close(rest, jaw, [lip_move])
opened = rest + jaw
sealed = rest + jaw + close
assert np.allclose(sealed[lip], rest[lip]), "jawOpen plus mouthClose at the same weight: the lower lip back at rest, lips meeting"
upper = rest[:, 2] > 0
assert np.allclose(close[upper], 0), "the upper lip does not move"
chin = rest[:, 2] < -0.025
assert np.allclose(sealed[chin], opened[chin]), "the chin stays with the jaw"
weight = closing_weight(rest, jaw, lip)
middle = rest[:, 0] == 0.0
column = weight[middle & lower_face][np.argsort(-rest[middle & lower_face, 2])]
assert np.all(np.diff(column) <= 1e-12), "the weight only falls going down from the lip"
assert np.any((column > 0) & (column < 1)), "it fades under the lip, without a step"
assert not mouth_close(rest, jaw, []).any(), "without lip visemes, nothing"
print("ok")
