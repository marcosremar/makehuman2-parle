"""Regression for the lash strip that tore in the blink (2026-11-09): run with Blender's Python, `blender -b --python-exit-code 1 -P tests/test_parle_lash.py`."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
from parle_polish import elevation, lid_curve, lid_turn, turned_about_eye  # noqa: E402

# The lid moves only over x from -1 to 1 cm; the strip is wider. Past the lid it keeps the last value, it does not extrapolate.
x = np.linspace(-0.01, 0.01, 9)
moved = np.stack([x * 0, -0.01 * (1 - (x / 0.01) ** 2), x * 0], axis=1)
across = np.array([-0.02, 0.0, 0.02])
curve = lid_curve(moved, x, across)
assert abs(curve[0, 1]) < 1e-4, f"the tip flew: {curve[0, 1]}"
assert abs(curve[2, 1]) < 1e-4, f"the tip flew: {curve[2, 1]}"
assert abs(curve[1, 1] + 0.01) < 1e-4, f"the middle of the lid is lost: {curve[1, 1]}"

# Turning keeps every point on its sphere around the eye, and lands where the lid landed.
centre = np.array([0.01, -0.02, 1.5])
up = np.radians(35.0)
start = centre + np.array([[0.0, -0.012 * np.cos(up), 0.012 * np.sin(up)]])
down = np.radians(-30.0)
closed = centre + np.array([[0.0, -0.012 * np.cos(down), 0.012 * np.sin(down)]])
turn = elevation(closed, centre) - elevation(start, centre)
landed = turned_about_eye(start, centre, turn)
assert np.allclose(landed, closed, atol=1e-7), landed
assert abs(np.linalg.norm(landed - centre) - 0.012) < 1e-7

# The turn along the strip is read inside the lid's own span.
span = np.linspace(-0.01, 0.01, 7)
rim_start = centre + np.stack([span, np.full(7, -0.012 * np.cos(up)), np.full(7, 0.012 * np.sin(up))], axis=1)
rim_closed = centre + np.stack([span, np.full(7, -0.012 * np.cos(down)), np.full(7, 0.012 * np.sin(down))], axis=1)
wide = lid_turn(rim_start, rim_closed, centre, rim_start[:, 0], np.array([centre[0] - 0.05, centre[0], centre[0] + 0.05]))
assert np.allclose(wide, down - up, atol=1e-6), wide
print("ok")
