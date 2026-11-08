"""Regression for the tongue that stayed up while the jaw opened (2026-11-08): run with `.local/venv/bin/python tests/test_parle_tongue.py`."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
from parle_tongue import ROOT_SHARE, TIP_SHARE, carried, is_mouth_key, lower_teeth, tongue_share  # noqa: E402

tongue = np.array([[0, -0.05, 0], [0, -0.03, 0], [0, 0.0, 0]], dtype=float)
share = tongue_share(tongue)
assert abs(share[0] - TIP_SHARE) < 1e-9 and abs(share[2] - ROOT_SHARE) < 1e-9, share
assert share[0] > share[1] > share[2], "the tongue must take less of the jaw towards the throat"

jaw = np.array([[0, 0, -0.017], [0, 0, -0.016], [0, 0, -0.001], [0, 0, 0]], dtype=float)
lower = lower_teeth(jaw)
assert lower.tolist() == [True, True, False, False], lower

move = carried(tongue, jaw, lower)
assert move is not None and move[0, 2] < -0.012, "the tip stayed up while the jaw opened"
assert abs(move[2, 2]) < abs(move[0, 2]) / 2, "the root moved as much as the tip"
assert carried(tongue, np.zeros_like(jaw), lower) is None, "a shape that does not move the jaw must not move the tongue"
assert is_mouth_key("viseme_aa") and is_mouth_key("jawOpen") and not is_mouth_key("faceChin")
print("ok")
