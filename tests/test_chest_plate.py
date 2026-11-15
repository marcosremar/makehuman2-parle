"""Regression for the bare chest of the muscled man (2026-11-15): run with `.local/venv/bin/python tests/test_chest_plate.py`.

His pectorals were the base mesh's two round mounds under the sculpture's relief: 4.3 cm out over a hard ledge, or, smoothed, two soft
mounds with a cone at each nipple, and from the side both read as breasts (owner: "o peitoral do homem negro está errado"). Now the
trunk's own curve is fitted from under the mounds at each height and a plate goes on it (core/chest_plate.py).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from core import chest_plate  # noqa: E402

# The plate: nothing under its lower edge, its lower edge higher toward the armpit, thickest at the bottom and thinner up to the collarbone.
low_middle = chest_plate.EDGE_LOW + chest_plate.EDGE_RISE * 0.3
assert chest_plate.plate_depth(np.array([low_middle - chest_plate.EDGE_FALL - 0.01]), np.array([0.3]))[0] == 0.0
just_above = chest_plate.plate_depth(np.array([low_middle + 0.01]), np.array([0.3]))[0]
assert just_above > 0.8 * chest_plate.THICKEST_M * 0.98, just_above
# At the same height, close to the armpit the edge is higher: the point is still under it.
outer = chest_plate.plate_depth(np.array([low_middle + 0.01]), np.array([0.7]))[0]
assert outer < just_above, (outer, just_above)
top = chest_plate.plate_depth(np.array([0.98]), np.array([0.3]))[0]
assert 0.5 * just_above < top < just_above, (top, just_above)
# Nothing on the breastbone or past the armpit.
assert chest_plate.plate_depth(np.array([0.8]), np.array([0.0]))[0] == 0.0
assert chest_plate.plate_depth(np.array([0.8]), np.array([0.95]))[0] == 0.0

# The trunk's curve without the mounds: a trunk 30 cm across whose front falls back 6 cm to the flanks, with a mound 15 mm tall and
# 8 cm wide on each side of the breastbone. Fitted from under, the curve at the top of a mound keeps less than a third of it (the
# three terms across cannot follow the trunk exactly: 4.3 mm of the 15 stay).
xs = np.linspace(-0.15, 0.15, 61)
levels = np.linspace(0.4, 1.05, 40)
x, level = (grid.ravel() for grid in np.meshgrid(xs, levels))
trunk = 0.12 - 0.06 * (x / 0.15) ** 2
mounds = 0.015 * (np.exp(-((x - 0.07) ** 2 + ((level - 0.72) * 0.4) ** 2) / (2 * 0.02 ** 2))
                  + np.exp(-((x + 0.07) ** 2 + ((level - 0.72) * 0.4) ** 2) / (2 * 0.02 ** 2)))
points = np.stack([x, -(trunk + mounds), level], axis=1)
curve = chest_plate.trunk_curve(points, 0.0, level, np.ones(len(x), bool))
peak = np.argmax(mounds)
assert abs(curve[peak] - trunk[peak]) < mounds[peak] / 3, (curve[peak], trunk[peak], mounds[peak])
# At heights away from the mounds, the curve keeps to the trunk: its roundness across is not flattened away.
plain = np.abs(level - 0.72) > 0.2
assert np.abs(curve[plain] - trunk[plain]).max() < 0.003, np.abs(curve[plain] - trunk[plain]).max()
print("test_chest_plate: ok")
