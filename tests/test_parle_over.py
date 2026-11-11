"""Regression for the two last touches on a finished body (2026-11-11). The top hangs outside the bottom: after the build of the kit the
hem of Carlos's shirt stood up to 11 mm inside his trousers, and the waistband showed through the plaid in two slits. And the points
of the skin at the same place share a normal: the seam between his head and his neck showed as a line across the nape. Run with
`Blender -b --factory-startup --python-exit-code 1 -P tests/test_parle_over.py`."""
import math
import sys
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import parle_polish  # noqa: E402

bpy.ops.wm.read_factory_settings(use_empty=True)


def tube(name, radius, low, high, rings=11, around=32):
    """A tube about the z axis, open at both ends, its faces looking out."""
    points = [(radius * math.cos(2 * math.pi * turn / around), radius * math.sin(2 * math.pi * turn / around), low + (high - low) * ring / (rings - 1))
              for ring in range(rings) for turn in range(around)]
    faces = [(ring * around + turn, ring * around + (turn + 1) % around, (ring + 1) * around + (turn + 1) % around, (ring + 1) * around + turn)
             for ring in range(rings - 1) for turn in range(around)]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(points, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


# Trousers of 15 cm of radius up to the waist at 1.0 m, and a shirt of 16 cm that hangs over them down to 0.9 m.
trousers, shirt = tube("Outfit_Bottom", 0.15, 0.5, 1.0), tube("Outfit_Top", 0.16, 0.9, 1.4)
before = parle_polish.rest_points(shirt).copy()
assert parle_polish.top_over_bottom([shirt], [trousers]) == 0, "a shirt that already hangs outside was moved"
# The front of the hem goes 11 mm inside the trousers, as after the build of the slim kit.
dented = before.copy()
hem = (before[:, 2] < 0.95) & (before[:, 1] < -0.12)
dented[hem, :2] *= 0.139 / 0.16
shirt.data.vertices.foreach_set("co", dented.ravel())
shirt.data.update()
moved = parle_polish.top_over_bottom([shirt], [trousers])
after = parle_polish.rest_points(shirt)
radius = np.linalg.norm(after[:, :2], axis=1)
over = after[:, 2] <= 1.0
# The flat faces of a tube of 32 sides stand up to 0.7 mm inside its radius: the gap is counted from them.
assert moved >= hem.sum() and radius[over].min() >= 0.15 * math.cos(math.pi / 32) + parle_polish.OVER_GAP_M - 1e-4, (moved, radius[over].min())
# Only what was short of the gap moved, and the cloth above the waist stays where it was.
far = before[:, 2] > 1.1
assert np.abs(after[far] - dented[far]).max() < 1e-6, "cloth far above the trousers moved"
assert np.abs(after[:, 2] - dented[:, 2]).max() < 0.004, "the hem went up or down"
# A dress has no bottom under it, and nothing happens.
assert parle_polish.top_over_bottom([shirt], []) == 0
print("ok")

# The points of the skin that stand at the same place get the same normal (2026-11-11): the glTF splits a point wherever the picture has
# a seam, each half lit by its own faces alone, and where the head meets the neck the light drew the seam as a line across the nape.
bpy.ops.wm.read_factory_settings(use_empty=True)
# Two faces that meet at a ridge along y, each with its own two points on the ridge, and a third face far away.
points = [(-1, 0, 0), (0, 0, 1), (0, 1, 1), (-1, 1, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0), (0, 1, 1), (5, 0, 0), (6, 0, 0), (6, 1, 0), (5, 1, 0)]
mesh = bpy.data.meshes.new("skin")
mesh.from_pydata(points, [], [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11)])
mesh.update()
skin = bpy.data.objects.new("Body", mesh)
bpy.context.scene.collection.objects.link(skin)
before = np.array([corner.vector[:] for corner in mesh.corner_normals])
assert abs(before[1] @ before[4]) < 0.1, "the two sides of the ridge already shared a normal"
assert parle_polish.seamless_normals(skin) == 2
after = np.array([corner.vector[:] for corner in mesh.corner_normals])
# Corners 1 and 2 of the first face and 0 and 3 of the second are the ridge: they now look straight up, the mean of the two sides.
for corner in (1, 2, 4, 7):
    assert np.allclose(after[corner], (0, 0, 1), atol=1e-4), (corner, after[corner])
# The points with no twin keep the normal of their face.
for corner in (0, 3, 5, 6, 8, 9, 10, 11):
    assert np.allclose(after[corner], before[corner], atol=1e-4), (corner, after[corner], before[corner])
print("ok")

# A piece with bone weights but no points rides the skin without breaking (2026-11-11): the short hair of the office worker has
# vertex groups and no vertex, and its empty list of nearest points came out as floats, which numpy refuses as indices.
bpy.ops.wm.read_factory_settings(use_empty=True)
empty = bpy.data.objects.new("Acc_Hair_short", bpy.data.meshes.new("Acc_Hair_short"))
bpy.context.scene.collection.objects.link(empty)
empty.vertex_groups.new(name="upperarm01.L")
tree = parle_polish.KDTree(1)
tree.insert((0, 0, 0), 0)
tree.balance()
shares = parle_polish.over_skin(empty, {"L": np.zeros(0)}, tree, {"L": np.ones(1)})
assert shares["L"].shape == (0,), shares
print("ok")
