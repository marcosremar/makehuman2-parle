"""Regression for the edges of the game's hair on a cast head (2026-11-11). The owner asked how to make the avatar more beautiful, face
and body in motion; in the photographs the first thing after the face was hair that looked torn: the long hair of the pharmacy's
attendant ended in saw teeth where it is cut at the neck and stood in a knot of spikes round each ear, and the cap of hair on Carlos's
skull went round his nape in square steps. Run with `Blender -b --factory-startup --python-exit-code 1 -P tests/test_parle_hair.py`."""
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import parle_hair  # noqa: E402

bpy.ops.wm.read_factory_settings(use_empty=True)


def made(name, points, faces):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(float(value) for value in point) for point in points], [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def sheet(name, radius, low, high, around=24, rings=9, skew=0.0, start=0.0, sweep=2 * math.pi):
    """A sheet round the z axis from `low` to `high`; `skew` leans its rows, so that a level cut crosses its faces."""
    closed = sweep >= 2 * math.pi - 1e-9
    columns = around if closed else around + 1
    points = [(radius * math.cos(start + sweep * turn / around), radius * math.sin(start + sweep * turn / around), low + (high - low) * ring / (rings - 1) + skew * math.sin(3 * sweep * turn / around))
              for ring in range(rings) for turn in range(columns)]
    faces = [(ring * columns + turn, ring * columns + (turn + 1) % columns, (ring + 1) * columns + (turn + 1) % columns, (ring + 1) * columns + turn)
             for ring in range(rings - 1) for turn in range(around)]
    return made(name, points, faces)


def border(obj):
    """The points on the open edges of a mesh, in the world."""
    count = {}
    for polygon in obj.data.polygons:
        corners = list(polygon.vertices)
        for first, second in zip(corners, corners[1:] + corners[:1]):
            key = (min(first, second), max(first, second))
            count[key] = count.get(key, 0) + 1
    on = sorted({index for key, times in count.items() if times == 1 for index in key})
    return np.array([obj.matrix_world @ obj.data.vertices[index].co for index in on])


# Long hair cut at the neck: a sheet round the head whose rows lean, so the level of the cut crosses its faces. The hem ends on the level.
neck = Vector((0.0, 0.0, 1.40))
level = neck.z + parle_hair.CHEST_ABOVE_NECK_M
hair = sheet("Hair_Long", 0.11, 1.30, 1.62, skew=0.012)
parle_hair.trim_chest_hair(hair, neck)
edge = border(hair)
front = edge[(edge[:, 1] < neck.y + parle_hair.CHEST_FRONT_M - 0.005) & (edge[:, 2] < level + 0.03)]
assert len(front) >= 6, f"the hair in front of the neck was not cut at all: {len(front)} points on its hem"
assert np.abs(front[:, 2] - level).max() < 1e-4, f"the hem of cut hair goes up and down in teeth: {np.ptp(front[:, 2]) * 1000:.1f} mm"

# Hair pushed out of the skin: a head with an ear on it, and a sheet of hair that the ear stands through. Out of it, the hair has no spikes.
head = sheet("Body", 0.085, 1.30, 1.75, around=48, rings=24)
bump = np.array([vertex.co for vertex in head.data.vertices])
ear = np.exp(-(((bump[:, 2] - 1.55) / 0.02) ** 2) - ((np.arctan2(bump[:, 1], bump[:, 0])) / 0.25) ** 2)
bump[:, :2] *= (1 + 0.6 * ear * (1 + 0.3 * np.sin(bump[:, 2] * 900)))[:, None]
head.data.vertices.foreach_set("co", bump.ravel())
head.data.update()
curtain = sheet("Hair_Side", 0.11, 1.44, 1.66, around=40, rings=23, start=-1.2, sweep=2.4)
before = np.array([vertex.co for vertex in curtain.data.vertices])
parle_hair.clear_of_body(curtain, [head], 1.42)
after = np.array([vertex.co for vertex in curtain.data.vertices])
went = np.linalg.norm(after - before, axis=1)
assert went.max() > 0.01, f"the ear did not push the hair at all: {went.max() * 1000:.1f} mm"
rows = after.reshape(23, 41, 3)
out = np.linalg.norm(rows[:, :, :2], axis=2)
jump = max(np.abs(np.diff(out, axis=0)).max(), np.abs(np.diff(out, axis=1)).max())
assert jump < 0.4 * went.max(), f"hair pushed off the ear stands in spikes: {jump * 1000:.1f} mm between two neighbours, of {went.max() * 1000:.1f} mm of push"
assert went.max() < 0.05, f"hair pushed off the ear went further than any ear stands out: {went.max() * 1000:.0f} mm"
assert np.abs(after[:, 2] - before[:, 2]).max() < 1e-6, "hair above the mouth was pushed up or down, not out from the head"

# A cap of hair on the skull whose edge goes in square steps: it is a cap, and its edge is smoothed; hair that stands off the skull is no cap.
def dome(name, radius, rim_deg, around=48, rings=12, teeth=0.0):
    """Part of a ball about the middle of the head, from near its top down to `rim_deg` from it; `teeth` lifts every other pair of points of the rim."""
    centre = np.array((0.0, 0.0, 1.70))
    points = []
    for ring in range(rings):
        for turn in range(around):
            polar = math.radians(8 + (rim_deg - 8) * ring / (rings - 1))
            if ring == rings - 1 and (turn // 2) % 2:
                polar -= teeth
            angle = 2 * math.pi * turn / around
            points.append(centre + radius * np.array((math.sin(polar) * math.cos(angle), math.sin(polar) * math.sin(angle), math.cos(polar))))
    faces = [((ring + 1) * around + turn, (ring + 1) * around + (turn + 1) % around, ring * around + (turn + 1) % around, ring * around + turn)
             for ring in range(rings - 1) for turn in range(around)]
    return made(name, points, faces)


skull = dome("Body", 0.090, 170, rings=24)
cap = dome("Hair_curly_top", 0.094, 100, teeth=math.radians(9))
assert parle_hair.hugs_scalp(cap, skull), "a cap of hair lying on the skull is not seen as one"
assert not parle_hair.hugs_scalp(dome("Hair_Curls", 0.13, 100), skull), "hair that stands 4 cm off the skull is taken for a cap"
rim = lambda: border(cap)[border(cap)[:, 2] < 1.74][:, 2]
rough = np.ptp(rim())
parle_hair.smooth_hairline(cap)
assert rough > 0.012 and np.ptp(rim()) < 0.5 * rough, f"the square steps of a cap's edge are as tall as before: {np.ptp(rim()) * 1000:.1f} mm of {rough * 1000:.1f}"

# A cap cut along a seam of its picture down the nape: the two sides of the seam are one line, and stay one line.
def seamed(name, radius, rim_deg, around=24, rings=8, teeth=0.0):
    """The same part of a ball, its points doubled along the seam at the back: a column of points at the start and one at the end, at the same places."""
    centre, points = np.array((0.0, 0.0, 1.70)), []
    for ring in range(rings):
        for turn in range(around + 1):
            polar = math.radians(8 + (rim_deg - 8) * ring / (rings - 1))
            if ring == rings - 1 and (turn // 2) % 2 and 0 < turn < around:
                polar -= teeth
            angle = math.pi / 2 + 2 * math.pi * turn / around
            points.append(centre + radius * np.array((math.sin(polar) * math.cos(angle), math.sin(polar) * math.sin(angle), math.cos(polar))))
    columns = around + 1
    faces = [((ring + 1) * columns + turn, (ring + 1) * columns + turn + 1, ring * columns + turn + 1, ring * columns + turn) for ring in range(rings - 1) for turn in range(around)]
    return made(name, points, faces)


split = seamed("Hair_Scalp", 0.094, 100, teeth=math.radians(9))
parle_hair.smooth_hairline(split)
after = np.array([vertex.co for vertex in split.data.vertices]).reshape(8, 25, 3)
assert np.abs(after[:, 0] - after[:, -1]).max() < 1e-6, f"the seam of a cap opened into a slit: {np.abs(after[:, 0] - after[:, -1]).max() * 1000:.2f} mm apart"

# A cap stops at the ear: it is not pushed out over it as hair that hangs is. Behind an ear, a point of the cap has the skull right
# under it and the flap of the ear further out on the same level: it clears the skull by a little, and does not jump out past the flap.
centre = np.array((0.0, 0.0, 1.70))
ball = lambda radius, polar, angle: centre + radius * np.array((math.sin(polar) * math.cos(angle), math.sin(polar) * math.sin(angle), math.cos(polar)))
flap_points = [ball(0.13, math.radians(84 + 2 * row), -0.2 + 0.05 * column) for row in range(9) for column in range(9)]
flap_faces = [((row + 1) * 9 + column, (row + 1) * 9 + column + 1, row * 9 + column + 1, row * 9 + column) for row in range(8) for column in range(8)]
bare = [tuple(vertex.co) for vertex in skull.data.vertices]
eared = made("Body", bare + flap_points, [tuple(polygon.vertices) for polygon in skull.data.polygons] + [tuple(len(bare) + index for index in face) for face in flap_faces])
low_cap = dome("Hair_Scalp", 0.094, 115, around=48, rings=14)
was = np.array([vertex.co for vertex in low_cap.data.vertices])
assert parle_hair.hugs_scalp(low_cap, eared)
parle_hair.clear_of_body(low_cap, [eared], 1.5)
went_cap = np.linalg.norm(np.array([vertex.co for vertex in low_cap.data.vertices]) - was, axis=1)
assert went_cap.max() < 0.015, f"a cap on the skull was pushed out past the ear like hair that hangs (that is 42 mm here): {went_cap.max() * 1000:.0f} mm"

# And pushed out of the skin, the two sides of a seam go together: a cap sunk 3 mm into the skull comes out whole.
sunk = seamed("Hair_Scalp", 0.087, 100)
parle_hair.clear_of_body(sunk, [skull], 1.5)
out = np.array([vertex.co for vertex in sunk.data.vertices]).reshape(8, 25, 3)
assert np.linalg.norm(out[3] - centre, axis=1).min() > 0.09, "a cap sunk into the skull was left inside it"
assert np.abs(out[:, 0] - out[:, -1]).max() < 1e-6, f"pushed out of the skin, the seam of a cap opened: {np.abs(out[:, 0] - out[:, -1]).max() * 1000:.2f} mm apart"
print("ok")
