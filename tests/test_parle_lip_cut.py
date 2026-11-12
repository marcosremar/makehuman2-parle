"""Regression for the border of the lips as an edge of the mesh (2026-11-12). Run with
`Blender -b --factory-startup --python-exit-code 1 -P tests/test_parle_lip_cut.py`.

Painted on the skin, a woman's lips were "an edgeless puffed pad under smeared colour" with "ragged corners" (the judge by sight,
rounds 15 to 18): a dozen texels of the map tall. `parle_lip_cut.lips_with_an_edge` cuts the skin along the border of the lips and
maps the faces inside it to the patch of the lips on the map. The first cut took the crossing of each edge on a straight line
between its ends: on the bow of the upper lip it fell outside the lip, the pieces were not lip, and the border came out in steps."""
import sys
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import parle_lip_cut  # noqa: E402
import parle_lips  # noqa: E402
import parle_polish  # noqa: E402

bpy.ops.wm.read_factory_settings(use_empty=True)
# A face that looks at -y: a sheet with 4 mm between its points, the tip of a nose 32 mm above the mouth, and the mouth a slit
# 40 mm wide where the lower lip has a row of points of its own.
STEP = 0.004
XS = np.round(np.arange(-0.04, 0.0401, STEP), 6)
LOW_ZS, UP_ZS = np.round(np.arange(-0.04, 0.0001, STEP), 6), np.round(np.arange(0.0, 0.0601, STEP), 6)
SLIT_HALF = 0.02
points, number = [], {}
for sheet, zs in (("low", LOW_ZS), ("up", UP_ZS)):
    for z in zs:
        for x in XS:
            shared = z == 0 and abs(x) > SLIT_HALF + 1e-9
            if sheet == "up" and shared:
                number[(sheet, x, z)] = number[("low", x, z)]
                continue
            number[(sheet, x, z)] = len(points)
            points.append((x, -0.02 if (x == 0 and abs(z - 0.032) < 1e-9) else 0.0, z))
faces = [(number[(sheet, XS[i], zs[j])], number[(sheet, XS[i + 1], zs[j])], number[(sheet, XS[i + 1], zs[j + 1])], number[(sheet, XS[i], zs[j + 1])])
         for sheet, zs in (("low", LOW_ZS), ("up", UP_ZS)) for j in range(len(zs) - 1) for i in range(len(XS) - 1)]
mesh = bpy.data.meshes.new("Body")
mesh.from_pydata(points, [], faces)
body = bpy.data.objects.new("Body", mesh)
bpy.context.scene.collection.objects.link(body)
uv = mesh.uv_layers.new(name="UVMap")
for loop in mesh.loops:
    x, _, z = points[loop.vertex_index]
    uv.data[loop.index].uv = (0.3 + (x + 0.04) / 0.08 * 0.6, 0.2 + (z + 0.04) / 0.1 * 0.6)
body.shape_key_add(name="Basis")
jaw = body.shape_key_add(name="jawOpen")
before = np.array(points)
for index in np.flatnonzero((before[:, 2] < 0) | (np.arange(len(points)) < len(LOW_ZS) * len(XS))):
    jaw.data[index].co.z -= 0.003
count = len(points)

lips, added = parle_lip_cut.lips_with_an_edge(body, low=parle_lips.LIP_LOW_M)
assert lips > 20 and added > 20, f"the cut: {lips} faces of lip, {added} points added"
now = parle_polish.read(mesh.shape_keys.key_blocks[0])
assert np.allclose(now[:count], before, atol=1e-7), "a point of the kit moved, or lost its number"
mesh.calc_loop_triangles()
triangles = np.array([triangle.vertices[:] for triangle in mesh.loop_triangles])
line = parle_lips.mouth_line({"tri": triangles}, now)
inside = parle_lips.lip_place(line, now)[0]
# The points the cut made are on the border of the lips: by halving, not on a straight line between the ends of the edge.
assert np.abs(inside[count:]).max() < 0.0001, f"a point of the cut stands {1000 * np.abs(inside[count:]).max():.3f} mm from the border"
# Which faces are lip: the ones whose places are on the patch. They are inside the border, the rest outside, with no face astride.
left, top, wide, tall = parle_lips.LIP_PATCH_PX
on_patch = lambda place: left / parle_lips.SIZE <= place[0] <= (left + wide) / parle_lips.SIZE and 1 - (top + tall) / parle_lips.SIZE <= place[1] <= 1 - top / parle_lips.SIZE
area = 0.0
uv = mesh.uv_layers.active
for polygon in mesh.polygons:
    places = [tuple(uv.data[index].uv) for index in polygon.loop_indices]
    middle = parle_lips.lip_place(line, np.mean(now[list(polygon.vertices)], axis=0)[None, :])[0][0]
    if all(on_patch(place) for place in places):
        assert middle > -1e-4, f"a face on the patch lies {-middle * 1000:.2f} mm outside the lips"
        area += polygon.area
    else:
        assert not any(on_patch(place) for place in places) and middle < 1e-4, f"a face astride the border, or a face of lip left off the patch ({middle * 1000:.2f} mm inside)"
across = np.linspace(-line["half"], line["half"], 4001)
over, under = parle_lips.lip_reach(across, line["half"])
whole = float(((over + under)[:-1] * np.diff(across)).sum())
assert abs(area / whole - 1) < 0.03, f"the faces of lip cover {area * 1e6:.1f} mm2 where the lips have {whole * 1e6:.1f}: the border is not the border of the lips"
# A shape goes with the cut: a point made on an edge takes of the shape what the two ends of the edge take.
moved = parle_polish.read(mesh.shape_keys.key_blocks["jawOpen"]) - now
lower = now[count:, 2] < -1e-4
upper = (now[count:, 2] > 1e-4) & (np.abs(now[count:, 0]) < SLIT_HALF)
assert lower.sum() > 5 and upper.sum() > 5, f"the test has {lower.sum()} points of the cut on the lower lip and {upper.sum()} on the upper"
assert np.allclose(moved[count:][lower, 2], -0.003, atol=1e-6) and np.allclose(moved[count:][upper], 0, atol=1e-6), "the points of the cut did not go with the open jaw"
print("ok")
