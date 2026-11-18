"""Regressions for the pieces over the clothes of who serves at a counter (2026-11-18): the apron came out lumpy, like modelling clay,
with a saw-toothed border where the shirt crossed it, and the badge was a crumpled blot. Run with
`Blender -b --factory-startup --python-exit-code 1 -P tests/test_parle_workwear.py`."""
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils.bvhtree import BVHTree

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import parle_workwear  # noqa: E402

bpy.ops.wm.read_factory_settings(use_empty=True)


def sheet(name, size=0.4, cells=24, y=0.0, bump=0.0, seed=1):
    """A square sheet in the x-z plane at depth `y`, facing -y (the front of a body), with random bumps of `bump` metres."""
    rng = np.random.default_rng(seed)
    points = []
    for row in range(cells + 1):
        for column in range(cells + 1):
            points.append((-size / 2 + size * column / cells, y + bump * float(rng.standard_normal()), -size / 2 + size * row / cells))
    faces = [(row * (cells + 1) + column, row * (cells + 1) + column + 1, (row + 1) * (cells + 1) + column + 1, (row + 1) * (cells + 1) + column)
             for row in range(cells) for column in range(cells)]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(points, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def depths(obj):
    return np.array([vertex.co.y for vertex in obj.data.vertices])


# The shirt under the apron is wrinkled (2 mm bumps); the apron sits 1 cm in front of it with its own 4 mm lumps.
shirt = sheet("shirt", bump=0.002, seed=3)
apron = sheet("apron", size=0.3, cells=20, y=-0.01, bump=0.004, seed=5)
before = depths(apron)
lift = parle_workwear.lift_over(apron, [shirt], 0.012)
after = depths(apron)
shirt_front = depths(shirt).min()
surface = BVHTree.FromPolygons([tuple(vertex.co) for vertex in shirt.data.vertices], [tuple(polygon.vertices) for polygon in shirt.data.polygons])
gaps = np.array([surface.find_nearest(vertex.co, 0.1)[3] for vertex in apron.data.vertices])
assert gaps.min() > 0.012 - 0.0015, f"every point of the apron is the gap off the shirt's wrinkles ({gaps.min():.4f})"
assert np.std(np.diff(after.reshape(21, 21), axis=1)) < 0.5 * np.std(np.diff(before.reshape(21, 21), axis=1)), "the lumps are smoothed away"
xs = np.array([vertex.co.x for vertex in apron.data.vertices])
assert np.ptp(xs) > 0.29, "smoothing does not shrink the apron"
assert lift > 0

# The lift a lump needs spreads over its neighbours and never falls under what a point itself needs.
edges = np.array([(0, 1), (1, 2), (2, 3), (3, 4)])
need = np.array([0.0, 0.0, 0.01, 0.0, 0.0])
field = parle_workwear.lift_field(need, edges, 5, rings=1, rounds=2)
assert field[2] >= 0.01 and field[1] > 0.004 and field[3] > 0.004, "the lift of one point is spread to its neighbours"

# The badge is a flat plate: a crumpled patch comes out on one plane, in front of the shirt.
badge = sheet("badge", size=0.06, cells=6, y=-0.02, bump=0.004, seed=7)
bend = parle_workwear.rigid_badge(badge, [shirt], [])
plate = np.array([tuple(vertex.co) for vertex in badge.data.vertices])
assert bend > 0.003, "the patch was crumpled"
assert np.linalg.svd(plate - plate.mean(axis=0), compute_uv=False)[2] < 1e-5, "the badge lies on one plane"
assert min(surface.find_nearest(vertex.co, 0.1)[3] for vertex in badge.data.vertices) > parle_workwear.BADGE_GAP_M - 1e-4, "off the shirt"

# One pleat down the middle of the skirt: the middle comes forward (to -y), the sides of it go back, the bib above stays.
skirt = sheet("skirt", size=0.3, cells=30, y=-0.05)
parle_workwear.pleat(skirt, waist_z=0.05, middle_x=0.0)
points = np.array([tuple(vertex.co) for vertex in skirt.data.vertices])
low = points[:, 2] < -0.05
middle = low & (np.abs(points[:, 0]) < 0.005)
side = low & (np.abs(np.abs(points[:, 0]) - 0.056) < 0.006)
above = points[:, 2] > 0.06
assert points[middle, 1].mean() < -0.05 - 0.005, "the ridge of the pleat comes forward"
assert points[side, 1].mean() > -0.05 + 0.002, "beside it the cloth folds back"
assert np.abs(points[above, 1] + 0.05).max() < 1e-6, "above the waist nothing moves"
print("ok", round(lift, 4), round(bend, 4), math.isfinite(lift))
