"""Regression for the tongue that stayed up while the jaw opened (2026-11-08): run with `.local/venv/bin/python tests/test_parle_tongue.py`."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
from parle_tongue import LOWER_TEETH_TUCK, ROOT_SHARE, ROUND_KEYS, SMILE_LOWER_TUCK, SMILE_TEETH_KEYS, smile_tuck, TIP_SHARE, carried, hidden_in_round, is_mouth_key, lower_teeth, tongue_share, tucked  # noqa: E402

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
# The lower row tucks behind the lower lip, the upper row stays (2026-11-16: both rows showed in every open "a", "dente cerrado").
tuck = tucked(jaw)
assert np.allclose(tuck[:2], LOWER_TEETH_TUCK) and np.allclose(tuck[2:], 0.0), f"the upper teeth moved, or the lower did not: {tuck}"
assert LOWER_TEETH_TUCK[2] < 0 and LOWER_TEETH_TUCK[1] > 0, "the lower teeth go down and back"
print("ok")
# Round 70: no teeth in the rounded shapes ("o", "u": "um bico em anel com os dentes à mostra", both judges): there the upper row
# goes up behind the upper lip and the lower one down, and no other shape is touched.
hide = hidden_in_round(jaw)
assert np.all(hide[:2, 2] < 0) and np.all(hide[2:, 2] > 0) and np.allclose(hide[:, :2], 0), f"the rows do not part behind the lips: {hide}"
assert set(ROUND_KEYS) == {"viseme_O", "viseme_U"}, "the teeth hide in a shape that is not rounded"

# Round 88 (four judges of four, all eight: "sem a fileira de baixo"): in the smile the lower row goes down and back, half on each
# side, and the upper row stays; the open shapes tuck the lower row at least 5 mm down.
smile = smile_tuck(jaw, 0.5)
assert np.allclose(smile[:2], np.asarray(SMILE_LOWER_TUCK) * 0.5) and np.allclose(smile[2:], 0.0), f"the smile moves the upper row, or not the lower: {smile}"
assert set(SMILE_TEETH_KEYS) == {"mouthSmileLeft", "mouthSmileRight"} and SMILE_LOWER_TUCK[2] < -0.002, "the smile does not take the lower row down"
assert LOWER_TEETH_TUCK[2] <= -0.005, f"the lower row is tucked under 5 mm: {LOWER_TEETH_TUCK}"
print("ok sorriso")
