"""Regression for the paint of a drawn face (2026-11-16): run with `python tests/test_parle_paint.py` (no Blender).

Round 20 of the loop judged by sight against the owner's chosen series: smudged grey eyebrows, grained skin, no blush. The eyebrow
becomes a clean dark band (light grey on the old), the skin map loses its grain, and the cheeks and the nose tip take a warm blush
that the forehead and the back of the head do not."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import parle_paint  # noqa: E402

brow = np.zeros((4, 4, 4), np.float32)
brow[:, :, :3] = 0.5
brow[:, :, 3] = np.linspace(0.0, 1.0, 16).reshape(4, 4)
young, old = parle_paint.drawn_brow(brow, False), parle_paint.drawn_brow(brow, True)
assert young[..., :3].max() < 0.3, "the young's eyebrow is not dark"
assert old[..., :3].min() > 0.55, "the old's eyebrow is not light grey"
alpha = young[..., 3].ravel()
assert ((alpha < 0.02) | (alpha > 0.98)).mean() >= 0.6, "the eyebrow's edge is still a soft smudge"

rng = np.random.default_rng(1)
skin = 0.6 + 0.1 * rng.standard_normal((64, 64, 3)).astype(np.float32)
smoothed = parle_paint.box_blur(skin, 3)
assert smoothed.shape == skin.shape and smoothed.std() < skin.std() / 3, "the skin keeps its grain"
assert abs(smoothed.mean() - skin.mean()) < 0.01, "the blur changes the skin's colour"

eyes = [(np.array([0.032, -0.06, 1.66]), 0.012), (np.array([-0.032, -0.06, 1.66]), 0.012)]
tip = np.array([0.0, -0.1, 1.62])
cheek = eyes[0][0] + np.array([0.12, -0.25, -0.72]) * 0.064
forehead, back = np.array([0.0, -0.08, 1.75]), np.array([0.0, 0.08, 1.66])
share = parle_paint.blush_share(np.array([cheek, tip, forehead, back]), tip, eyes, "male")
assert share[0] > 0.8 * parle_paint.BLUSH_SHARE["male"] and share[1] > 0.6 * parle_paint.BLUSH_SHARE["male"], "no blush on the cheek or the nose"
assert share[2] < 0.02 and share[3] < 0.001, "the forehead or the back of the head blush"
assert parle_paint.BLUSH_SHARE["male-old"] > parle_paint.BLUSH_SHARE["male"], "the old blush less than the young"
red = parle_paint.blushed(np.full((1, 1, 3), 0.6, np.float32), np.ones((1, 1)))
assert red[0, 0, 0] > red[0, 0, 2], "the blush is not warm"
print("ok")
