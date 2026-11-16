"""Regression for the style by kind of person (2026-11-15): run with `python tests/test_parle_style.py` (no Blender).

The owner asked for the faces of one animated series, by kind of person. The nose of each kind grows about a middle behind its tip,
across more than to the front, and the upper lip and the cheeks stay; every kind has its numbers; a sex with no style is refused."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import parle_style  # noqa: E402

assert set(parle_style.STYLE) == {"male", "male-old", "female", "female-old"}, "a kind of person has no style"
assert parle_style.kind_of("male", {"age": "old"}) == "male-old" and parle_style.kind_of("female", {}) == "female"
try:
    parle_style.kind_of("dog", {})
    raise AssertionError("a body that is neither a woman nor a man gets a style")
except ValueError:
    pass
men, old = parle_style.STYLE["male"], parle_style.STYLE["male-old"]
assert men["nose"] > parle_style.STYLE["female"]["nose"] and old["nose"] > men["nose"], "the series' men have the bigger nose, the old the biggest"
assert old["brow"] > men["brow"] > parle_style.STYLE["female"]["brow"], "the eyebrows of the kinds are not in the series' order"

tip = np.array([0.0, -0.1, 1.6])
field = parle_style.nose_field(tip, 1.6, 0.9)
side = np.array([[0.01, -0.09, 1.6]])
assert field(side)[0, 0] > side[0, 0] * 1.2, "the nose does not grow across"
assert abs(field(tip[None])[0, 1] - tip[1]) < abs(field(side)[0, 0] - side[0, 0]), "the nose comes out more than it widens"
lip, cheek = np.array([[0.0, -0.095, 1.6 - 0.02]]), np.array([[0.045, -0.07, 1.6]])
assert np.allclose(field(lip), lip, atol=2e-4) and np.allclose(field(cheek), cheek), "the upper lip or the cheek moves with the nose"
grey = np.array([[0.6, 0.5, 0.45]])
assert np.ptp(parle_style.warmer(grey, 1.3)) > np.ptp(grey), "the skin is not more saturated"
# The men's nose was pressed flat (owner, 2026-11-16, "o nariz dele está amassado"): it grows to the front more than across, and the
# ball of the sculpture only fills out, leaving the bridge above it where it was and bringing the tip forward.
for kind in ("male", "male-old"):
    assert parle_style.STYLE[kind]["forward"] > 1.0, f"the nose of {kind} is pushed back into the face"
    across, back, _ = parle_style.SCULPT[kind]["nose_ball"]
    assert back >= across, f"the nose ball of {kind} is wider than it is deep: a flat nose"
eyes = [(np.array([0.032, -0.06, 1.66]), 0.012), (np.array([-0.032, -0.06, 1.66]), 0.012)]
sculpt = parle_style.sculpt_field(tip, eyes, "male")
bridge = np.array([[0.0, -0.085, 1.628]])
assert sculpt(tip[None])[0, 1] < tip[1] - 0.001, "the tip of the nose does not come out into its ball"
assert np.allclose(sculpt(bridge), bridge), "the ball of the nose pulls the bridge in"
print("ok")
