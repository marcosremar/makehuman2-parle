"""Regression for the mouth that looked lit from inside (2026-11-08): run with `.local/venv/bin/python tests/test_parle_mouth.py`."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import parle_mouth as mouth  # noqa: E402

# Upper arch: front incisors at y=-0.1, molars 50 mm behind; lower arch likewise; the gum 20 mm above the tip.
teeth = np.array([[0, -0.1, 0], [0, -0.1, 0.02], [0, -0.05, 0], [0, -0.1, -0.05], [0, -0.05, -0.05]], float)
lower = np.array([False, False, False, True, True])
shade = mouth.teeth_shade(teeth, lower)
assert shade[0] == 1.0, f"the incisor must stay lit: {shade[0]}"
assert shade[2] <= mouth.TEETH_DEEP + 1e-6, f"the molars must be in the dark: {shade[2]}"
assert shade[1] <= mouth.GUM_SHADE + 1e-6, f"the gum must be darker than the tooth: {shade[1]}"

tongue = np.array([[0, -0.08, 0], [0, -0.04, 0], [0, -0.03, 0]], float)
tongue_light = mouth.tongue_shade(tongue)
assert tongue_light[0] == mouth.TONGUE_FRONT < 0.7, "the tip of the tongue is already behind the teeth"
assert tongue_light[0] > tongue_light[1] >= tongue_light[2] and abs(tongue_light[2] - mouth.TONGUE_DEEP) < 1e-6, tongue_light

uv = np.array([[[0, 0], [0.5, 0], [0, 1]]], float)
canvas = mouth.shade_map(16, 16, uv, np.array([[0.2, 0.2, 0.2]], float))
assert np.nanmax(canvas) == np.nanmin(canvas) and abs(np.nanmin(canvas) - 0.2) < 1e-6
assert np.isnan(canvas).any(), "the map must say where no triangle falls"
filled = mouth.grow(canvas, 40)
assert not np.isnan(filled).any() and abs(filled.max() - 0.2) < 1e-6, "islands must bleed their own shade past their edge"

pixels = np.full((16, 16, 4), 0.8, np.float32)
dark = mouth.darken(pixels, np.full((16, 16), 0.5, np.float32))
assert abs(dark[0, 0, 0] - 0.4) < 1e-6 and dark[0, 0, 3] == pixels[0, 0, 3], "alpha must not change"

# The tip sits at the lip's edge on the widest "a": it must come down; 3 mm below already shows.
assert mouth.incisor_drop(0.0) == mouth.SHOW_TARGET_M
assert mouth.incisor_drop(0.003) == 0.0
assert mouth.incisor_drop(None) == 0.0

upper_tip = np.array([[0.0, -0.1, -0.003], [0.0, -0.1, 0.01], [0.0, -0.05, 0.0], [0.0, -0.1, -0.03], [0.0, -0.05, -0.03]])
arch_lower = np.array([False, False, False, True, True])
no_move = np.zeros_like(upper_tip)
jaw_teeth = np.zeros_like(upper_tip)
jaw_teeth[3:, 2] = -0.017
lip = np.array([[0.0, -0.11, 0.0], [0.0, -0.11, -0.02]], float)  # the upper edge, 3 mm above the tip; then the lower lip the jaw takes
lip_jaw = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -0.02]])
show = mouth.incisor_show(upper_tip, no_move, jaw_teeth, arch_lower, lip, np.zeros_like(lip), lip_jaw)
assert show is not None and abs(show - 0.003) < 1e-6, show
print("ok")
