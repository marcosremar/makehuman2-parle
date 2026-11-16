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
# Simple volumes (round 27): a bump on a smoothed patch goes, the volume stays, and the rims of the eyes and the lips are left.
grid = np.stack(np.meshgrid(np.arange(9.0), np.arange(9.0)), -1).reshape(-1, 2)
flat = np.c_[grid, np.zeros(len(grid))]
flat[40, 2] = 1.0
links = [(i, i + 1) for i in range(81) if i % 9 < 8] + [(i, i + 9) for i in range(72)]
done = parle_style.smoothed(flat, np.array(links), np.ones(81), rounds=10)
assert done[40, 2] < 0.4 and abs(done[:, 2].sum() - 1.0) < 0.5, "the bump is not smoothed out, or the volume went"
rim = eyes[0][0] + np.array([0.0, -0.012, 0.0])
lips = tip + np.array([0.0, 0.006, -0.04])
neck = tip + np.array([0.0, 0.06, -0.12])
weight = parle_style.simple_weight(np.array([rim, lips, neck]), tip, eyes)
assert weight[0] < 0.01 and weight[1] < 0.01 and weight[2] > 0.9, f"the smoothing takes the eye or the lips, or leaves the neck: {weight}"
# The lower face shorter (round 31): the chin comes up most, the mouth part of the way, and nothing above the nose tip nor far down
# the neck moves.
short = parle_style.lower_face_field(tip, eyes, "male")
probe = np.array([[0.0, -0.09, tip[2] - 0.083], [0.0, -0.09, tip[2] - 0.04], [0.0, -0.1, tip[2] + 0.01], [0.0, -0.05, 1.4]])
up = (short(probe) - probe)[:, 2]
assert up[0] > up[1] > 0.002 and abs(up[2]) < 1e-9 and abs(up[3]) < 1e-9, f"the lower face is not shortened as it should be: {up}"
# Worn meshes follow the skin (rounds 36 to 43: hairline frayed, earrings and glasses off the face, a torn collar): a point on the
# skin's moved patch goes with it, one far from it stays.
skin_before = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.5, 0.5, 0.5], [0.51, 0.5, 0.5]])
skin_after = skin_before + np.array([[0.0, -0.005, 0.0]] * 3 + [[0.0, 0.0, 0.0]] * 2)
follow = parle_style.follow_field(skin_before, skin_after)(np.array([[0.005, 0.005, 0.01], [0.3, 0.3, 0.3]]))
assert abs(follow[0, 1] - 0.0) < 1e-4, "the hair over the moved skin stays where it was"
assert np.allclose(follow[1], [0.3, 0.3, 0.3]), "a point far from the moved skin moves"
# The lid lifted by the blink run backwards no longer folds at the edge of the blink (rounds 24 to 46: a pale arc over each
# eyebrow): along a chain, the abrupt edge of a move becomes a slope, the pinned rim kept.
chain = np.zeros((12, 3))
chain[:4, 2] = 0.004
edges_chain = np.array([[i, i + 1] for i in range(11)])
pin = np.zeros(12, bool)
pin[0] = True
evened = parle_style.spread(chain, edges_chain, pin, np.arange(12) < 10, 200)
assert evened[0, 2] == 0.004, "the lid's rim moved"
assert np.abs(np.diff(evened[:, 2])).max() < 0.5 * np.abs(np.diff(chain[:, 2])).max(), "the edge of the lift stays a fold"
assert evened[11, 2] == 0.0, "a point past the reach moved"
# The skin that stayed counts too (round 47: an eyebrow went down with the lid under it): a point next to skin that did not move
# stays, even with moved skin a centimetre off.
lid_before = np.array([[0.0, 0.0, 0.0], [0.002, 0.0, 0.0], [0.0, 0.0, 0.012], [0.002, 0.0, 0.012]])
lid_after = lid_before + np.array([[0.0, 0.0, -0.003]] * 2 + [[0.0, 0.0, 0.0]] * 2)
brow = parle_style.follow_field(lid_before, lid_after)(np.array([[0.001, -0.002, 0.013]]))
assert abs(brow[0, 2] - 0.013) < 0.0005, "the eyebrow goes down with the lid under the skin it sits on"
# The old open their eyes less than the young (round 47: white above the iris of the old woman and the old man).
assert parle_style.EYE_TALL_LEAST_BY_KIND["female-old"] < parle_style.EYE_TALL_LEAST_BY_KIND["female"], "the old open as wide as the young"
assert parle_style.EYE_TALL_LEAST_BY_KIND["male-old"] < parle_style.EYE_TALL_LEAST_BY_KIND["male"], "the old open as wide as the young"
print("ok")
