"""Regression for the loose bits of skin under the clothes (2026-11-15): run with `.local/venv/bin/python tests/test_skin_islands.py`.

A bit of skin the cut under the clothes left on its own came out through the cloth in gestures (the back of the pharmacist's T-shirt,
the gym man's shorts). `skin_islands.loose` finds the small pieces; a piece split along a texture seam (the same place twice) is one.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import skin_islands  # noqa: E402


def strip(count, x0, step=0.01):
    """A row of `count` points joined one to the next."""
    points = np.stack([x0 + np.arange(count) * step, np.zeros(count), np.zeros(count)], axis=1)
    edges = np.stack([np.arange(count - 1), np.arange(1, count)], axis=1)
    return points, edges


big, big_edges = strip(1200, 0.0)
small, small_edges = strip(16, 100.0)
# The big piece is split along a seam: its second half has its own copy of the point where the halves meet.
seam = big[600:601].copy()
points = np.concatenate([big, small, seam])
edges = np.concatenate([np.delete(big_edges, 599, axis=0), small_edges + len(big), [[len(big) + len(small), 599]]])
piece, size = skin_islands.pieces(points, edges)
assert list(size) == [1200, 16], size
assert piece[0] == piece[1199] == 0 and piece[len(big)] == 1
found = skin_islands.loose(points, edges)
assert set(found) == set(range(len(big), len(big) + len(small))), found
# Nothing is loose on a body in one piece.
assert len(skin_islands.loose(big, big_edges)) == 0
print("test_skin_islands: ok")
