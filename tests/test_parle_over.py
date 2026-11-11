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

# The legs of a pair of shorts end on a level line (2026-11-11): the bottoms of a kit have few faces, and the hem of the pharmacy
# attendant's shorts went round each thigh in teeth of 2 cm. Shorts are sewn of a front and a back panel, so each opening is the hems
# of both. The waist and the seams between the panels stay as they are.
def panel(name, front):
    """Half of a pair of shorts: the front or the back of two legs 20 cm apart, each half a tube of 8 cm, from 0.60 to 0.90 m, its hem in teeth."""
    points, faces = [], []
    for leg in (-0.10, 0.10):
        start = len(points)
        for ring in range(7):
            for turn in range(9):
                angle = math.pi * turn / 8 + (math.pi if front else 0.0)
                high = 0.60 + 0.05 * ring + (0.02 if ring == 0 and turn % 2 else 0.0)
                points.append((leg + 0.08 * math.cos(angle), 0.08 * math.sin(angle), high))
        faces += [(start + ring * 9 + turn, start + ring * 9 + turn + 1, start + (ring + 1) * 9 + turn + 1, start + (ring + 1) * 9 + turn) for ring in range(6) for turn in range(8)]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(points, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


halves = [panel("Outfit_Bottom", True), panel("Outfit_Bottom.001", False)]
ragged = [parle_polish.rest_points(half).copy() for half in halves]
assert parle_polish.level_hems(halves) == 2, "the two openings of a pair of shorts sewn of two panels were not found"
for half, was in zip(halves, ragged):
    now = parle_polish.rest_points(half)
    hem = was[:, 2] < 0.63
    assert np.ptp(now[hem, 2]) < 1e-4, f"the hem still goes up and down: {np.ptp(now[hem, 2]) * 1000:.1f} mm"
    assert np.allclose(now[~hem], was[~hem], atol=1e-7), "waist or seams moved with the hem"
    assert np.allclose(now[hem][:, :2], was[hem][:, :2], atol=1e-4), "the hem came nearer the leg, or further from it"

# A skirt is one opening, right across the middle.
skirt = tube("Outfit_Bottom", 0.20, 0.5, 0.9, rings=7, around=24)
wavy = parle_polish.rest_points(skirt).copy()
wavy[wavy[:, 2] < 0.501, 2] += np.where(np.arange((wavy[:, 2] < 0.501).sum()) % 2 == 0, 0.0, 0.02)
skirt.data.vertices.foreach_set("co", wavy.ravel())
skirt.data.update()
assert parle_polish.level_hems([skirt]) == 1 and np.ptp(parle_polish.rest_points(skirt)[wavy[:, 2] < 0.53, 2]) < 1e-4, "the hem of a skirt was not levelled as one opening"
assert parle_polish.level_hems([tube("Outfit_Top", 0.16, 0.9, 1.4)]) == 0, "the hem of a shirt was taken for the leg of a bottom"
print("ok")

# A neck a quarter shorter (2026-11-11, the women): by their weights alone, the points of the skin step from the neck's share of the
# loss to the head's in a ring or two, and all the loss closed up there in a dark fold. The skin shares it out evenly from the chest,
# which keeps none, to the head, which keeps all. A neck of 21 rings: 6 of the chest, 9 of the neck, 6 of the head.
throat = tube("Body", 0.06, 1.30, 1.50, rings=21, around=16)
rows = np.repeat(np.arange(21), 16)
rigid, loose = (rows >= 15).astype(float), ((rows >= 6) & (rows < 15)).astype(float)
stepped = np.where(rows >= 15, 1.0, np.where(rows >= 13, 0.9, np.where(rows >= 6, 0.2, 0.0)))
even = parle_polish.even_follow(throat, stepped, rigid, loose).reshape(21, 16)
assert np.ptp(even, axis=1).max() < 1e-6, "the points of one ring of the neck come down by different shares"
by_ring = even.mean(axis=1)
assert np.abs(by_ring[:6]).max() == 0 and np.abs(by_ring[15:] - 1).max() == 0, f"the chest came down, or the head did not come down whole: {by_ring}"
steps = np.diff(by_ring[5:16])
assert steps.min() > 0.05 and steps.max() < 0.15, f"the loss of the neck is not shared out evenly ring by ring (a tenth each): {steps.round(3)}"
assert np.diff(stepped.reshape(21, 16).mean(axis=1)).max() > 0.6, "the neck of this test never stepped"
print("ok")
