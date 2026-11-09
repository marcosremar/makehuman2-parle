"""Regressions for the woman's lower lip and chin (2026-11-09): run with `blender -b --python-exit-code 1 -P tests/test_parle_lower_lip.py`.

The lower lip read as a crumpled ridge over a hollow, three lips with the upper one, and the chin stuck out once the mouth had gone back.
"""
import sys
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
from parle_face import chin_back, held_in, relaxed_under_lip  # noqa: E402

# A strip of face seen in profile, every 3 mm from the lip down: a ridge at the lip's lower half and a hollow under it.
heights = np.arange(0.0, -0.06, -0.003)
depths = np.zeros_like(heights)
depths[3] = -0.003
depths[5] = 0.003
rest = np.array([[x, depth, height] for x in (-0.02, -0.01, 0.0, 0.01, 0.02) for height, depth in zip(heights, depths)])
columns = len(heights)
lip = np.where(rest[:, 2] > -0.009)[0]
mesh = bpy.data.meshes.new("strip")
faces = [(row * columns + i, row * columns + i + 1, (row + 1) * columns + i + 1, (row + 1) * columns + i) for row in range(4) for i in range(columns - 1)]
mesh.from_pydata(rest.tolist(), [], faces)
obj = bpy.data.objects.new("Body", mesh)
middle = 2 * columns
moved = relaxed_under_lip(obj, rest, np.zeros_like(rest), lip, 6)
after = rest + moved
assert np.ptp(after[middle + 3 : middle + 6, 1]) < 0.6 * np.ptp(rest[middle + 3 : middle + 6, 1]), "the ridge and the hollow melt into one form"
assert abs(moved[middle, 1]) < 1e-9, "the line where the lips meet stays"
back = chin_back(rest, lip, 0.003)
assert back[middle + 10, 1] > 0.0029, "the chin goes back as far as the mouth did"
assert abs(back[middle, 1]) < 1e-9, "the lip does not go twice"
# The lining goes down with the jaw, but never forward or out through the cheek.
points = np.array([[0.03, 0.05, 0.0], [-0.03, 0.05, 0.0]])
pushed = held_in(points, np.array([[0.01, -0.01, -0.02], [-0.01, 0.002, -0.02]]), np.array([0, 1]), 0.0)
assert pushed[0].tolist() == [0.0, 0.0, -0.02], "forward and outward moves go, the drop stays"
assert pushed[1].tolist() == [0.0, 0.002, -0.02], "a move backwards stays"
print("ok")
