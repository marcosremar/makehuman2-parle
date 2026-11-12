"""Regression for the measures of how alike the two halves of a face are (2026-11-12). Run with
`.local/venv/bin/python tests/test_parle_symmetry.py`.

A face made of the Face Mesh points the measures read, its two halves mirror images: every measure is nothing. Then one thing at a
time is put off, by a known amount, and each measure answers by what the sums say."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import parle_symmetry  # noqa: E402

MID, WIDE, PUPILS_APART = 500.0, 400.0, 180.0
# One side of the face, as (how far from the middle, height on the picture); the twin of each point is its mirror image.
SIDE = {33: (130, 300), 133: (50, 302), 159: (90, 295), 145: (90, 305), 107: (40, 268), 105: (90, 265), 70: (140, 272), 48: (45, 420), 61: (70, 500), 234: (200, 380), 172: (160, 560)}
TWIN = {33: 263, 133: 362, 159: 386, 145: 374, 107: 336, 105: 334, 70: 300, 48: 278, 61: 291, 234: 454, 172: 397}


def face():
    points = np.zeros((478, 2))
    for index, (aside, height) in SIDE.items():
        points[index] = (MID - aside, height)
        points[TWIN[index]] = (MID + aside, height)
    for place, index in enumerate(parle_symmetry.MIDDLE):
        points[index] = (MID, 150 + 25 * place)
    points[parle_symmetry.PUPILS[0]], points[parle_symmetry.PUPILS[1]] = (MID - PUPILS_APART / 2, 300), (MID + PUPILS_APART / 2, 300)
    return points


even = parle_symmetry.measures(face())
assert set(even) == set(parle_symmetry.NAMES), f"measures without a name, or names without a measure: {set(even) ^ set(parle_symmetry.NAMES)}"
assert all(abs(value) < 1e-9 for value in even.values()), f"a face whose halves mirror each other is not even: {even}"

# The mouth 10 pixels to one side: its right corner stands 10 further from the middle and its left one 10 nearer, 20 between them,
# as a share of the distance between the pupils; the five other pairs stay.
points = face()
points[[61, 291], 0] += 10
found = parle_symmetry.measures(points)
assert abs(found["sym_sides"] - 2 * 10 / PUPILS_APART) < 1e-9, f"the mouth to one side: {found}"
assert found["sym_level"] < 1e-9, f"a mouth moved sideways is not higher on one side: {found['sym_level']}"

# One eye 8 pixels higher: four of the eleven pairs of twins are 8 apart in height.
points = face()
points[[263, 362, 386, 374], 1] -= 8
found = parle_symmetry.measures(points)
assert abs(found["sym_level"] - 4 * 8 / 11 / WIDE) < 1e-9 and found["sym_sides"] < 1e-9, f"one eye higher: {found}"

# The tip of the nose 6 pixels to one side: three of the twenty points down the middle are off the line.
points = face()
points[[4, 1, 2], 0] += 6
found = parle_symmetry.measures(points)
assert abs(found["sym_midline"] - 3 * 6 / 20 / WIDE) < 1e-9 and found["sym_sides"] < 1e-9, f"a crooked nose: {found}"

# One eye more open, its upper lid 2 pixels higher: 12 tall against 10, and 28 under its eyebrow against 30; the other four sizes stay.
points = face()
points[386, 1] -= 2
found = parle_symmetry.measures(points)
assert abs(found["sym_size"] - (2 / 11 + 2 / 29) / 6) < 1e-9, f"one eye more open: {found['sym_size']}"

# The size of the picture does not count.
assert all(abs(parle_symmetry.measures(points * 3)[name] - found[name]) < 1e-9 for name in found), "a bigger picture of the same face reads different"
print("ok")
