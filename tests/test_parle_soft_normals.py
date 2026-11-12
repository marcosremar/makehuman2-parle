"""Regression for the folds of the face lit as smooth skin (2026-11-12). Run with
`Blender -b --factory-startup --python-exit-code 1 -P tests/test_parle_soft_normals.py`.

The judge by sight, every round: under smooth, drawn features the skin of the cast is still that of "a tired realistic adult",
the fold beside the nose and the hollow under each eye catching the light. `parle_soft_normals.soften` gives the points of a
zone the normal of the skin around them, so a fold in the zone is lit as the skin it folds: the points stay where they are,
the skin outside the zone keeps its normals to the last digit, an edge that is left out is not counted as skin around the
others, and two points at the same place (a seam of the picture) still share one normal."""
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Euler
from mathutils.bvhtree import BVHTree

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import parle_polish  # noqa: E402
import parle_soft_normals  # noqa: E402

bpy.ops.wm.read_factory_settings(use_empty=True)
# A sheet of skin that faces +z, a point every 2 mm, with a groove along y in the middle of it: 4 mm to each side and 2 mm
# deep, its walls leaning 27 degrees. A seam of the picture runs along one rim of the groove, where a wall meets the flat skin:
# the points of that line are there twice, one for the faces on each side, lit 27 degrees apart until they are joined.
STEP, HALF_WIDE, HALF_TALL = 0.002, 0.06, 0.04
GROOVE_HALF, GROOVE_DEEP, SEAM_AT = 0.004, 0.002, 0.004
ACROSS = np.round(np.arange(-HALF_WIDE, HALF_WIDE + 1e-9, STEP), 6)
ALONG = np.round(np.arange(-HALF_TALL, HALF_TALL + 1e-9, STEP), 6)
SEAM_COLUMN = int(np.flatnonzero(np.isclose(ACROSS, SEAM_AT))[0])
# The zone: whole as far as 12 mm from the line y = 0 and nothing from 24 mm on, so the groove runs through it and out of it.
# The skin around a point is taken four times as wide as the groove is to each side: a fold as wide as the skin it is averaged
# over keeps half of its lean (a groove 6 mm to each side under 12 mm kept 12 of its 29 degrees, 2026-11-12).
ZONE_WHOLE, ZONE_NONE = 0.012, 0.024
RADIUS = 0.016


def depth(x):
    return np.where(np.abs(x) < GROOVE_HALF, -GROOVE_DEEP / 2 * (1 + np.cos(np.pi * x / GROOVE_HALF)), 0.0)


def grooved_sheet(step_at=None):
    """The sheet as an object, where its points are, and which point of the seam each of its twins doubles. With `step_at`,
    the skin beyond that x drops 10 mm over one row of faces: an edge like the rim of a lid."""
    wide = len(ACROSS)
    points = np.array([(x, y, float(depth(x))) for y in ALONG for x in ACROSS])
    if step_at is not None:
        points[points[:, 0] < step_at, 2] -= 0.01
    twins = np.array([row * wide + SEAM_COLUMN for row in range(len(ALONG))])
    index = lambda row, col, right: len(points) + row if right and col == SEAM_COLUMN else row * wide + col
    faces = []
    for row in range(len(ALONG) - 1):
        for col in range(wide - 1):
            right = col >= SEAM_COLUMN
            faces.append((index(row, col, right), index(row, col + 1, right), index(row + 1, col + 1, right), index(row + 1, col, right)))
    every = np.concatenate([points, points[twins]])
    mesh = bpy.data.meshes.new("Body")
    mesh.from_pydata([tuple(point) for point in every], [], faces)
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    obj = bpy.data.objects.new("Body", mesh)
    bpy.context.scene.collection.objects.link(obj)
    # The skin of a character stands in the world turned and smaller than it was modelled (the rig of the cast is scaled).
    obj.rotation_euler = Euler((0.3, -0.2, 0.9))
    obj.scale = (0.9, 0.9, 0.9)
    bpy.context.view_layer.update()
    return obj, every, twins


def corner_normals(obj):
    """The normal of each corner of each face, and the point it belongs to."""
    mesh = obj.data
    found = np.empty(len(mesh.loops) * 3)
    mesh.corner_normals.foreach_get("vector", found)
    point_of = np.empty(len(mesh.loops), dtype=np.int64)
    mesh.loops.foreach_get("vertex_index", point_of)
    return found.reshape(-1, 3), point_of


off_flat = lambda normals: np.degrees(np.arccos(np.clip(normals[:, 2], -1, 1)))
apart_by = lambda first, second: np.degrees(np.arccos(np.clip((first * second).sum(axis=1), -1, 1)))
# Blender keeps a custom normal as two 16-bit angles from the normal of its corner: written back as it was read, a normal comes
# out up to 0.014 degrees away (2026-11-12, on this sheet). Less than this is the same normal.
SAME_DEG = 0.05


def point_normals(obj, count):
    """One normal for each point: its corners all carry the same one on a smooth mesh."""
    corners, point_of = corner_normals(obj)
    out = np.zeros((count, 3))
    out[point_of] = corners
    assert apart_by(corners, out[point_of]).max() < SAME_DEG, f"the corners of a point stand {apart_by(corners, out[point_of]).max():.3f} degrees apart"
    return out


sheet, POINTS, TWINS = grooved_sheet()
zone = parle_soft_normals.fade(np.abs(POINTS[:, 1]), ZONE_WHOLE, ZONE_NONE)
wall = (np.abs(POINTS[:, 0]) > 0.001) & (np.abs(POINTS[:, 0]) < GROOVE_HALF)
assert parle_polish.seamless_normals(sheet) == len(ALONG), "the seam of the test is not a line of doubled points"
before = point_normals(sheet, len(POINTS))
assert off_flat(before[wall]).min() > 20, f"the walls of the groove of the test lean {off_flat(before[wall]).min():.0f} to {off_flat(before[wall]).max():.0f} degrees"
assert apart_by(before[TWINS], before[len(POINTS) - len(TWINS):]).max() < SAME_DEG, "the two sides of the seam do not start with one normal"
where = parle_polish.read(sheet.data.vertices).copy()

changed = parle_soft_normals.soften(sheet, zone, RADIUS * 0.9)
after = point_normals(sheet, len(POINTS))
# Only the normals change: the points stay, so the outline of the face does.
assert np.array_equal(parle_polish.read(sheet.data.vertices), where), "a point moved"
places = len(POINTS) - len(TWINS)
assert changed == int((zone[:places] >= parle_soft_normals.CHANGED_FROM).sum()), f"{changed} places changed, of the {int((zone[:places] > 0).sum())} in the zone"
# Where the zone is whole, the groove is lit as the flat skin it folds: within a few degrees of the normal of the sheet.
whole = zone >= 1
assert (whole & wall).sum() > 20, f"the test has {(whole & wall).sum()} points of the walls of the groove in the whole of the zone"
assert off_flat(after[whole]).max() < 4, f"in the whole of the zone a normal still leans {off_flat(after[whole]).max():.1f} degrees ({off_flat(before[whole]).max():.1f} before)"
# Outside the zone nothing changes: the corners are written back as they were read.
outside = zone <= 0
assert (outside & wall).sum() > 30 and apart_by(after[outside], before[outside]).max() < SAME_DEG, f"outside the zone a normal turned {apart_by(after[outside], before[outside]).max():.3f} degrees"
# Between the two the change fades: a wall leans less than it did and more than in the whole of the zone, and less the
# further into the zone it is.
fading = (zone > 0.05) & (zone < 0.95) & wall
assert fading.sum() > 15 and (off_flat(after[fading]) < off_flat(before[fading])).all(), "in the fading edge of the zone a wall leans as much as before"
assert off_flat(after[fading]).max() > 4, "the zone has no fading edge: the change is whole right up to it"
row = wall & np.isclose(POINTS[:, 0], -0.002)
lean = off_flat(after[row])[np.argsort(-np.abs(POINTS[row, 1]))]
assert (np.diff(lean[: len(lean) // 2]) < 1e-3).all(), f"along one wall, from outside the zone to its middle, the lean does not go down: {np.round(lean[: len(lean) // 2], 1).tolist()}"
# The two sides of the seam still share one normal, all along it: in the zone, in its edge and outside.
assert apart_by(after[TWINS], after[places:]).max() < SAME_DEG, f"the two sides of the seam stand {apart_by(after[TWINS], after[places:]).max():.3f} degrees apart after the change"
assert off_flat(before[TWINS][zone[TWINS] >= 1]).min() > 10, "the seam of the test is not on the rim of the groove"
bpy.data.objects.remove(sheet)

# Without the step that joins the seams first, the two sides of a seam in the whole of the zone come out with one normal all
# the same: points at the same place weigh and turn as one.
apart, _, _ = grooved_sheet()
split = point_normals(apart, len(POINTS))
assert apart_by(split[TWINS], split[places:]).min() > 20, f"the two sides of the seam of the test start {apart_by(split[TWINS], split[places:]).min():.1f} degrees apart"
parle_soft_normals.soften(apart, zone, RADIUS * 0.9)
joined = point_normals(apart, len(POINTS))
inside = zone[TWINS] >= 1
assert inside.sum() > 10 and apart_by(joined[TWINS][inside], joined[places:][inside]).max() < SAME_DEG, f"in the whole of the zone the two sides of a seam stand {apart_by(joined[TWINS][inside], joined[places:][inside]).max():.3f} degrees apart"
bpy.data.objects.remove(apart)

# An edge that stays is not counted as skin around the others. 34 mm from the groove, further than the skin around a point
# goes, the skin drops 10 mm over one row of faces, like the rim of a lid into the socket: counted, the wall of that drop leans
# the normals of the skin beside it; left out, with the skin below it, they are those of the flat skin.
STEP_AT = -0.034
lean_beside = {}
for name, leave in (("counted", False), ("left out", True)):
    stepped, _, _ = grooved_sheet(step_at=STEP_AT)
    parle_polish.seamless_normals(stepped)
    start = point_normals(stepped, len(POINTS))
    rim = POINTS[:, 0] < STEP_AT + STEP * 1.5
    took = np.where(rim, 0.0, 1.0)
    parle_soft_normals.soften(stepped, took, RADIUS * 0.9, left_out=rim if leave else None)
    now = point_normals(stepped, len(POINTS))
    beside = (POINTS[:, 0] > STEP_AT + STEP * 1.5) & (POINTS[:, 0] < STEP_AT + 0.006) & (np.abs(POINTS[:, 1]) < HALF_TALL - RADIUS)
    lean_beside[name] = off_flat(now[beside]).max()
    assert apart_by(now[rim], start[rim]).max() < SAME_DEG, f"the edge itself turned {apart_by(now[rim], start[rim]).max():.3f} degrees"
    bpy.data.objects.remove(stepped)
assert lean_beside["left out"] < 1 and lean_beside["counted"] > 5, f"beside an edge the skin leans {lean_beside['left out']:.1f} degrees with the edge left out and {lean_beside['counted']:.1f} with it counted"

# What shows and what does not: a pocket behind the skin, like the lining of the mouth behind the lips, does not show, and
# the skin in front of it does. The face looks at -y.
bpy.ops.wm.read_factory_settings(use_empty=True)
# The skin in front is 40 cm wide and the pocket 1 cm behind its middle: from no way a face is looked at is the pocket seen past it.
front = [(-0.2, 0.0, -0.2), (0.2, 0.0, -0.2), (0.2, 0.0, 0.2), (-0.2, 0.0, 0.2)]
behind = [(-0.01, 0.01, -0.01), (0.01, 0.01, -0.01), (0.01, 0.01, 0.01), (-0.01, 0.01, 0.01)]
spot = np.array(front + behind)
facing = np.tile((0.0, -1.0, 0.0), (8, 1))
cover = BVHTree.FromPolygons(spot.tolist(), [(0, 1, 2, 3), (4, 5, 6, 7)])
shown = parle_soft_normals.shown_places(spot, facing, np.ones(8, bool), cover)
assert shown[:4].all() and not shown[4:].any(), f"of the skin in front {shown[:4].tolist()} shows and of the pocket behind it {shown[4:].tolist()}"

# A face with no eye globes, or no skin, is left as it is.
assert parle_soft_normals.soft_normals([]) == 0, "something changed on a character with no pieces"
print("ok")
