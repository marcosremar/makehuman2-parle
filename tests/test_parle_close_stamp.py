"""Regression for garments longer than the game's (2026-11-07): run with `.local/venv/bin/python tests/test_parle_close_stamp.py`."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
from parle_stamp import close_stamp, main_colour, shoe_colour  # noqa: E402

SIZE = 400
GROUND = np.array([200, 0, 200])
BLUE = np.array([180, 90, 40], dtype=np.uint8)
HEM = np.array([250, 250, 250], dtype=np.uint8)
CLOTH = np.array([[[0, 0], [SIZE - 1, 0], [SIZE - 1, SIZE - 1]], [[0, 0], [SIZE - 1, SIZE - 1], [0, SIZE - 1]]], dtype=np.int32)


def leg():
    """Print down to row 200, a white fray just above that, and nothing below: the game's shorts are shorter than the new ones."""
    image = np.full((SIZE, SIZE, 3), GROUND, np.uint8)
    image[:200] = BLUE
    image[185:200] = HEM
    return image


# Lúcia and m15: the white fray of the old hem ended up in the middle of the thigh, and Seu Manoel's dark cuff above the ankle.
closed, _, unreached = close_stamp(leg(), GROUND, CLOTH)
assert (np.abs(closed[190, 200].astype(int) - BLUE) <= 2).all(), closed[190, 200]
assert (np.abs(closed[350, 200].astype(int) - BLUE) <= 2).all(), closed[350, 200]
assert unreached[350, 200] and unreached[190, 200] and not unreached[50, 200]

# Rays that miss here and there are not a hem: the first fix took André's whole plaid away.
plaid = np.full((SIZE, SIZE, 3), BLUE, np.uint8)
plaid[::40] = HEM
plaid[100, 100] = GROUND
closed, _, unreached = close_stamp(plaid, GROUND, CLOTH)
assert not unreached.any()
assert (closed[40, 200] == HEM).all()
assert (closed[100, 100] != GROUND).any()

# Seu Nelson: the median of green, wine and white stripes is a brown that is not on the shirt.
stripes = np.array([[60, 120, 40]] * 30 + [[60, 30, 130]] * 30 + [[240, 240, 240]] * 40, dtype=np.uint8)
assert main_colour(stripes)[1] < 0.6
# Fábio: a shaded red and one exact white trim. The red is the garment, though no single shade of it is as common as the white.
shaded = np.array([[20, 20, 150 + (index % 50)] for index in range(800)] + [[250, 250, 250]] * 200, dtype=np.uint8)
colour, plain = main_colour(shaded)
assert colour[2] > 100 and colour[0] < 60 and plain >= 0.6

# Fábio: only a white trim was reached, and the shorts are the red measured on the game body.
trim = np.full((SIZE, SIZE, 3), GROUND, np.uint8)
trim[:40] = HEM
closed, _, _ = close_stamp(trim, GROUND, CLOTH)
assert (np.abs(closed[350, 200].astype(int) - GROUND) <= 2).all(), closed[350, 200]

# Cláudia: a plain shirt loses the shadow of the old hem at the edge of the cloth...
shirt = np.full((SIZE, SIZE, 3), BLUE, np.uint8)
shirt[-20:] = BLUE // 3
closed, _, _ = close_stamp(shirt, GROUND, CLOTH)
assert (np.abs(closed[-5, 200].astype(int) - BLUE) <= 2).all(), closed[-5, 200]
# ...and a striped one keeps its dark stripes there.
dark = np.full((SIZE, SIZE, 3), BLUE // 4, np.uint8)
dark[np.arange(SIZE) % 80 < 40] = BLUE
closed, _, _ = close_stamp(dark, GROUND, CLOTH)
assert (closed == dark).all()

# Helena: a shoe of nearly the skin's colour read as a bare foot; on darker skin the same shoe is a shoe.
assert shoe_colour("#dbbd94", (225, 190, 150)) != "#dbbd94"
assert shoe_colour("#dbbd94", (110, 70, 45)) == "#dbbd94"
print("ok")
