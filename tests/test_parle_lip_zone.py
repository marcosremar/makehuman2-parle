"""Regression for the lip map (2026-11-08): run with `.local/venv/bin/python tests/test_parle_lip_zone.py`.

The old map was a soft patch centred 46 mm below the nose tip, 19 mm under the mouth: the women's lower lip took colour down to the
chin and looked too big. The map now follows the mouth's slit.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
from parle_lips import SIZE, lip_zone  # noqa: E402

STEP = 0.002
xs = np.arange(-0.04, 0.0401, STEP)
SLIT_Z = 1.55
TIP_Z = SLIT_Z + 0.027
points, triangles = [], []


def sheet(z_from, z_to):
    """A flat sheet of skin facing -y; the two sheets share no edge, so the gap between them is the slit."""
    zs = np.arange(z_from, z_to + 1e-6, STEP)
    first = len(points)
    for z in zs:
        for x in xs:
            points.append((x, 0.0, z))
    width = len(xs)
    for row in range(len(zs) - 1):
        for col in range(width - 1):
            a = first + row * width + col
            triangles.extend([(a, a + 1, a + width), (a + 1, a + width + 1, a + width)])


sheet(SLIT_Z - 0.03, SLIT_Z - 0.0005)
sheet(SLIT_Z + 0.0005, SLIT_Z + 0.03)
points.append((0.0, -0.02, TIP_Z))  # the nose tip, in front
points = np.array(points)
triangles = np.array(triangles)
uv = np.stack([(points[triangles][:, :, 0] + 0.05) / 0.1, (points[triangles][:, :, 2] - SLIT_Z + 0.05) / 0.1], axis=2)
zone = lip_zone({"tri": triangles, "uv": uv}, points)


def at(x, z):
    return zone[int(round((1 - (z - SLIT_Z + 0.05) / 0.1) * SIZE)), int(round((x + 0.05) / 0.1 * SIZE))]


# The middle of each lip is lip.
assert at(0, SLIT_Z + 0.003) > 0.9 and at(0, SLIT_Z - 0.004) > 0.9, (at(0, SLIT_Z + 0.003), at(0, SLIT_Z - 0.004))
# 15 mm below the slit is chin, not lip: the old patch painted it.
assert at(0, SLIT_Z - 0.015) < 0.05, at(0, SLIT_Z - 0.015)
# Above the upper lip is skin.
assert at(0, SLIT_Z + 0.012) < 0.05
# The lip narrows to nothing past the mouth's corner.
assert at(0.039, SLIT_Z - 0.004) < 0.05
print("ok")

# The shaved chin uses the same measured lip (Seu Nelson, 2026-11-08): the old fixed band 41-53 mm under the nose sat on the lower
# lip and the chin below it, and its warm tone read as a dark stain under the mouth.
from parle_lips import lip_points  # noqa: E402

lip = lip_points({"tri": triangles, "uv": uv}, points)
below_mouth = (np.abs(points[:, 0]) < 0.01) & (points[:, 2] < SLIT_Z - 0.012) & (points[:, 2] > SLIT_Z - 0.026)
assert not lip[below_mouth].any()
print("ok raspado")
