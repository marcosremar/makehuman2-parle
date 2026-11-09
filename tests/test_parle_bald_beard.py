"""Regressions for the balding line and the beard border (2026-11-08): run with `.local/venv/bin/python tests/test_parle_bald_beard.py`.

The beard mask came from the bake's noise: its border was ragged, with specks around it and the old dark halo left on the skin outside.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
from parle_bald import bare_crown  # noqa: E402
from skin_style import tidy_beard  # noqa: E402

TOP = 1.7
FRONT = -0.1
BACK = 0.1
head = np.array([[0, FRONT, TOP - 0.1], [0, BACK, TOP - 0.1], [0, 0, TOP]], dtype=float)
spots = np.array([
    [0, -0.03, TOP],          # top of the head
    [0, -0.09, TOP - 0.05],   # high forehead
    [0.07, 0.0, TOP - 0.1],   # side, above the ear
    [0, 0.1, TOP - 0.1],      # back of the head
    [0, -0.09, TOP - 0.13],   # brow
], dtype=float)
assert bare_crown(spots, head).tolist() == [True, True, False, False, False]

size = 400
skin = np.full((size, size, 3), 200.0, np.float32)
beard = np.zeros((size, size), np.float32)
beard[150:250, 100:300] = 1
# A ragged border, a loose speck and the dark halo the bake leaves around the beard.
beard[150:156, ::9] = 0
beard[100:104, 200:204] = 1
skin[140:150, 100:300] = 110
skin[100:104, 200:204] = 60
lips = np.zeros((size, size), np.float32)
lips[120:135, 180:220] = 1
skin[120:135, 180:220] = 90

clean, soft = tidy_beard(skin, beard, lips)
assert soft[102, 202] < 0.05, "a loose speck is not beard"
assert soft[200, 200] > 0.95, "the middle of the beard stays"
assert abs(float(soft[153, 130:270].max()) - float(soft[153, 130:270].min())) < 0.45, "the border runs smooth"
assert clean[145, 200, 0] > 170, f"the halo outside the beard heals to skin: {clean[145, 200, 0]}"
assert clean[127, 200, 0] < 120, "the lip is not healed away"
print("ok")
