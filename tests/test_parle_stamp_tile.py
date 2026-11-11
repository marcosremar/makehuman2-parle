"""Regression for the print that repeats joined to the baked picture (2026-11-11): Carlos's plaid came straight but his buttons and the
undershirt at his neck must stay, and Dona Rosa's plain bodice too. Run with `.local/venv/bin/python tests/test_parle_stamp_tile.py`."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
from stamp_tile import joined  # noqa: E402

SIZE = 400
GREEN, DARK, WHITE, PALE, BLACK = (70, 190, 60), (40, 60, 70), (245, 245, 245), (215, 235, 165), (20, 20, 20)
grid = np.indices((SIZE, SIZE))
# The print laid straight on the cloth: a plaid of 32 pixels.
laid = np.full((SIZE, SIZE, 3), GREEN, np.uint8)
laid[(grid[0] % 32 < 6) | (grid[1] % 32 < 6)] = DARK
# The picture baked from the game: the same plaid in waves down to row 250, with a button and the undershirt on it, a plain black part
# below (a bodice), and a column the bake did not reach (a sleeve), in the ground colour.
wave = (8 * np.sin(grid[0] / 25.0)).astype(int)
baked = np.full((SIZE, SIZE, 3), GREEN, np.uint8)
baked[((grid[0] + wave.T) % 32 < 6) | ((grid[1] + wave) % 32 < 6)] = DARK
baked[100:106, 150:156] = WHITE
baked[30:70, 180:220] = PALE
baked[250:] = BLACK
baked[:, 330:] = GREEN
unreached = np.zeros((SIZE, SIZE), np.uint8)
unreached[:, 330:] = 255
cloth = np.array([[[0, 0], [SIZE - 41, 0], [SIZE - 41, SIZE - 1]], [[0, 0], [SIZE - 41, SIZE - 1], [0, SIZE - 1]]], dtype=np.int32)
out, share = joined(baked, laid, cloth, unreached)
same = lambda where, colour: (np.abs(out[where].astype(int) - colour) <= 12).all()
# The waves give way to the straight plaid: a point that is a dark line on the laid print and green on the baked one comes out dark.
rows, columns = np.nonzero((laid[:200, :140] == DARK).all(axis=2) & (baked[:200, :140] == GREEN).all(axis=2))
assert len(rows) > 50 and all(same((row, column), DARK) for row, column in zip(rows[::7], columns[::7])), "the waves of the game's plaid are still there"
# What sits on the print without being of it stays: the button, and the undershirt at the neck.
assert same((103, 153), WHITE), out[103, 153]
assert same((50, 200), PALE), out[50, 200]
# The plain part stays plain, and what the bake did not reach takes the print.
assert same((330, 100), BLACK), out[330, 100]
assert same((100, 345), laid[100, 345]) and same((300, 345), laid[300, 345]), "the sleeve the bake did not reach went without print"
# Past the cloth, each point takes the colour of the cloth next to it: a picture read between two points never shows another colour.
assert same((100, SIZE - 20), out[100, SIZE - 42]), (out[100, SIZE - 20], out[100, SIZE - 42])
assert 0.5 < share < 0.9, share
print("ok")
