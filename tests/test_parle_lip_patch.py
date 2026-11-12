"""Regression for the lips of the women as a patch of the map (2026-11-12). Run with `.local/venv/bin/python tests/test_parle_lip_patch.py`.

Painted on the skin the lips had some 12 by 45 texels: "an edgeless puffed pad under smeared colour", "red paint bleeds past ragged
corners" (the judge by sight, rounds 15 to 18). The rule of where the lip is (`lip_reach`, `lip_place`) now serves the mesh too
(`parle_lip_cut`), and a woman's lips are drawn on a patch of their own (`with_lip_patch`), the skin of her face left bare."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import parle_lips  # noqa: E402

HALF = 0.025
# A mouth 50 mm wide whose lips meet on a level line at the front of the face.
LINE = {"middle": 0.0, "half": HALF, "centres": np.array((-0.02, 0.0, 0.02)), "medians": np.zeros((3, 3)), "curve": np.zeros(parle_lips.SLIT_DEGREE + 1)}
over, under = parle_lips.lip_reach(np.array((0.0, parle_lips.BOW_AT_M, -parle_lips.BOW_AT_M, HALF, 2 * HALF)), HALF)
assert over[1] == over[2] > over[0] > 0 and over[3] == over[4] == 0 and under[0] > 0 == under[3], f"the reach of the lips across the mouth: {over}, {under}"

points = np.array([
    (0.0, 0.0, over[0] / 2),            # the middle of the upper lip
    (0.0, 0.0, over[0] + 0.002),        # the skin over it
    (0.0, 0.0, -under[0] / 2),          # the middle of the lower lip
    (0.0, 0.0, -under[0] - 0.002),      # the chin under it
    (0.0, 0.0, over[0]),                # on the upper border
    (0.0, parle_lips.FRONT_M + 0.002, 0.001),   # inside the mouth
    (HALF * 1.1, 0.0, 0.0005),          # past a corner
    (-HALF / 2, 0.0, 0.0),              # on the line where the lips meet, a quarter of the way across
])
inside, across, up = parle_lips.lip_place(LINE, points)
assert inside[0] > 0 and inside[2] > 0 and (inside[[1, 3, 5, 6]] < 0).all() and abs(inside[4]) < 1e-9, f"which points are lip: {inside}"
assert abs(inside[1] + 0.002) < 1e-9 and abs(inside[5] + 0.002) < 1e-9, f"how far outside the lip, in metres: {inside}"
assert abs(up[0] - 0.75) < 1e-9 and abs(up[2] - 0.25) < 1e-9 and up[4] == 1 and up[1] == 1 and up[3] == 0 and up[7] == 0.5, f"how high on the lips: {up}"
assert across[0] == 0.5 and abs(across[7] - 0.25) < 1e-9 and across[6] == 1, f"how far across the lips: {across}"

# On the map the places of the lips stay inside the patch, clear of its rim, with the upper border at the top.
left, top, wide, tall = parle_lips.LIP_PATCH_PX
edge = parle_lips.LIP_PATCH_EDGE_PX
corners = parle_lips.patch_uv(np.array((0.0, 1.0, 0.5)), np.array((0.0, 1.0, 0.5))) * parle_lips.SIZE
assert np.allclose(corners[0], (left + edge, parle_lips.SIZE - (top + tall - edge))) and np.allclose(corners[1], (left + wide - edge, parle_lips.SIZE - (top + edge))), f"the patch on the map: {corners}"

# The picture: a plain skin with the lip of the old map marked on the face. The face keeps its skin; the patch has the lips.
skin = np.full((parle_lips.SIZE, parle_lips.SIZE, 3), (150, 170, 215), np.uint8)
share = np.zeros((parle_lips.SIZE, parle_lips.SIZE), np.float32)
share[500:520, 480:560] = 1
out = parle_lips.with_lip_patch(skin, share)
rest = np.ones(skin.shape[:2], bool)
rest[top - edge:top + tall + edge, left - edge:left + wide + edge] = False
assert (out[rest] == skin[rest]).all(), "the skin outside the patch changed: the lip is no longer painted on the face"
lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.float32)
own = cv2.cvtColor(skin, cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
at = lambda across, up: lab[int(round(top + tall - edge - up * (tall - 2 * edge))), int(round(left + edge + across * (wide - 2 * edge)))]
lower, upper, meet, corner = at(0.5, 0.25), at(0.5, 0.85), at(0.5, 0.48), at(0.0, 0.5)
assert lower[0] < own[0] - 15 and lower[1] > own[1] + 8, f"the lower lip is not darker and rosier than the skin: {lower} against {own}"
assert 0.74 < lower[0] / own[0] < 0.9, f"the lip is {lower[0] / own[0]:.2f} of the skin's lightness: lipstick, or no lip"
assert upper[0] < lower[0] - 3 and meet[0] < upper[0] - 5, f"the upper lip is not darker than the lower, or the line where they meet is not the darkest: {upper[0]}, {lower[0]}, {meet[0]}"
assert np.abs(corner - own).max() < 6, f"at the corner of the mouth the colour is not the skin's: {corner} against {own}"
print("ok")
