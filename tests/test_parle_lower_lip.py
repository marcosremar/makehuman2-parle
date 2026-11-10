"""Regressions for the woman's lower lip and chin (2026-11-09): run with `blender -b --python-exit-code 1 -P tests/test_parle_lower_lip.py`.

The lower lip read as a crumpled ridge over a hollow, three lips with the upper one, and the chin stuck out once the mouth had gone back.
"""
import sys
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
from parle_face import LIPS_FULL, chin_back, full_lips, held_in, relaxed_corners, relaxed_under_lip  # noqa: E402

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

# Corners (owner, 2026-11-09): the meeting line fell towards the corners and the face read as sulking. They go up; the middle stays, and both lips move alike.


class Points(list):
    def foreach_get(self, _name, target):
        target[:] = np.asarray(self).ravel()


class Key:
    def __init__(self, points):
        self.data = Points(points.tolist())


columns = np.arange(-0.03, 0.0301, 0.003)
face = np.array([[x, 0.0, z] for x in columns for z in (0.004, 0.001, -0.001, -0.004, 0.03)])
lipped = face[:, 2] < 0.02
shut = face.copy()
shut[lipped, 1] -= 0.01
open_jaw = face.copy()
open_jaw[face[:, 2] < 0, 2] -= 0.01
lifted = full_lips({"viseme_U": Key(shut), "viseme_PP": Key(shut), "jawOpen": Key(open_jaw)}, face, LIPS_FULL["female"])
at = lambda x, z: int(np.where((np.abs(face[:, 0] - x) < 1e-6) & (np.abs(face[:, 2] - z) < 1e-6))[0][0])
corner_x = columns[np.abs(columns - 0.027).argmin()]
assert lifted[at(corner_x, 0.001), 2] > 0.003, "the corner goes up"
assert abs(lifted[at(0.0, 0.001), 2]) < 0.0005, "the middle stays"
assert abs(lifted[at(corner_x, 0.001), 2] - lifted[at(corner_x, -0.001), 2]) < 0.0005, "both lips go up together, the mouth stays shut"

# The corner ridge (owner, 2026-11-09): a little knob at each corner faced the ceiling and lit up as a pale patch in the player. It melts; the middle of the lips stays.
grid_x, grid_z = np.arange(-0.045, 0.0451, 0.003), np.arange(-0.021, 0.0211, 0.003)
sheet = np.array([[x, 0.0, z] for x in grid_x for z in grid_z])
rows = len(grid_z)
ridge = np.where((np.abs(np.abs(sheet[:, 0]) - 0.027) < 1e-6) & (np.abs(sheet[:, 2]) < 1e-6))[0]
sheet[ridge, 1] -= 0.004
lip_points = np.where((np.abs(sheet[:, 2]) < 0.0035) & (np.abs(sheet[:, 0]) < 0.0285))[0]
quads = [(c * rows + r, c * rows + r + 1, (c + 1) * rows + r + 1, (c + 1) * rows + r) for c in range(len(grid_x) - 1) for r in range(rows - 1)]
grid_mesh = bpy.data.meshes.new("corners")
grid_mesh.from_pydata(sheet.tolist(), [], quads)
grid_obj = bpy.data.objects.new("Corners", grid_mesh)
relaxed_move = relaxed_corners(grid_obj, sheet, np.zeros_like(sheet), lip_points)
assert (sheet[ridge, 1] + relaxed_move[ridge, 1] > -0.002).all(), "the knob at each corner melts to less than half"
centre = np.where((np.abs(sheet[:, 0]) < 1e-6) & (np.abs(sheet[:, 2]) < 1e-6))[0]
assert np.abs(relaxed_move[centre]).max() < 1e-9, "the middle of the lips stays"
assert abs(sheet[0, 1] + relaxed_move[0, 1]) < 1e-9, "the far cheek stays"
print("ok corners")
