"""
    License information: data/licenses/makehuman_license.txt

    The border of a woman's lips as an edge of the mesh (2026-11-12). Run in Blender, by `parle_polish.natural_body`.

    Painted on the skin, the lips had some 12 by 45 texels of the map: "an edgeless puffed pad under smeared colour", "red paint
    bleeds past ragged corners along a wavy lip line, like smeared lipstick" (the judge by sight, rounds 15 to 18, always the
    first fault of the face at rest). Here the mesh is cut along the border of the lips (`parle_lips.lip_place`, the same rule
    the painting had), and the faces inside it take their places on the patch of the lips on the map (`parle_lips.patch_uv`),
    where `parle_lips.with_lip_patch` draws them. The colour then ends on an edge, as sharp as the screen, whatever the map.
"""
import bmesh
import numpy as np
from mathutils import Vector

import parle_lips
from parle_polish import read, to_world

# Edges are looked at where both their points are no further outside the lip than the first of these. A point of the kit no
# further from the border than the second is on it and its edges are left whole. Where an edge crosses the border is found by
# halving, this many times: the border bends (the bow of the upper lip, the corners), and taken on a straight line between the
# two ends the crossing fell up to a few tenths of a millimetre outside, the pieces were then not lip, and the upper border came
# out in steps, face by face (2026-11-12, the first cut). An edge is cut no nearer one of its ends than the share, so no sliver
# of a face is made.
LIP_CUT_NEAR_M = 0.03
LIP_CUT_ON_M = 0.00002
LIP_CUT_HALVINGS = 14
LIP_CUT_END = 0.02
# Two points of the cut no further apart than this are one point. At a corner of the mouth the border of the upper lip and the
# border of the lower one meet, and an edge of each can cross there: three pairs on the pharmacy's attendant (2026-11-12). Left
# apart they make slivers of faces, and each of the two has the moves of its own lip: the game reads two points in the same place
# as copies of one along a seam of the map (babylon-cinema `test/npcs/cast-mouth-arkit.spec.ts`).
LIP_CUT_SAME_M = 0.00005


def lips_with_an_edge(body, low=parle_lips.WOMAN_LIP_LOW_M):
    """Cuts the skin along the border of the lips and maps the faces of the lips to their patch: how many faces are lip, and how
    many points the cut added. Shape keys, bone weights and the rest of the map go with the faces; the points of the kit keep
    their numbers."""
    mesh = body.data
    blocks = mesh.shape_keys.key_blocks if mesh.shape_keys else None
    world = to_world(body, read(blocks[0]) if blocks else read(mesh.vertices))
    mesh.calc_loop_triangles()
    triangles = np.zeros(len(mesh.loop_triangles) * 3, dtype=np.int64)
    mesh.loop_triangles.foreach_get("vertices", triangles)
    line = parle_lips.mouth_line({"tri": triangles.reshape(-1, 3)}, world)
    inside = parle_lips.lip_place(line, world, low)[0]
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    crossed = []
    for edge in bm.edges:
        first, second = (inside[vert.index] for vert in edge.verts)
        if min(first, second) < -LIP_CUT_NEAR_M or abs(first) <= LIP_CUT_ON_M or abs(second) <= LIP_CUT_ON_M or (first > 0) == (second > 0):
            continue
        crossed.append(edge)
    count = len(bm.verts)
    if crossed:
        start = world[[edge.verts[0].index for edge in crossed]]
        step = world[[edge.verts[1].index for edge in crossed]] - start
        starts_in = inside[[edge.verts[0].index for edge in crossed]] > 0
        near, far = np.zeros(len(crossed)), np.ones(len(crossed))
        for _ in range(LIP_CUT_HALVINGS):
            half = (near + far) / 2
            same = (parle_lips.lip_place(line, start + step * half[:, None], low)[0] > 0) == starts_in
            near, far = np.where(same, half, near), np.where(same, far, half)
        shares = np.clip((near + far) / 2, LIP_CUT_END, 1 - LIP_CUT_END)
        made = [bmesh.utils.edge_split(edge, edge.verts[0], float(share))[1] for edge, share in zip(crossed, shares)]
        bmesh.ops.remove_doubles(bm, verts=made, dist=LIP_CUT_SAME_M)
        made = [vert for vert in made if vert.is_valid]
        bmesh.ops.connect_verts(bm, verts=made)
        bmesh.ops.triangulate(bm, faces=[face for face in bm.faces if len(face.verts) > 4])
    bm.verts.index_update()
    bm.verts.ensure_lookup_table()
    inside, across, up = parle_lips.lip_place(line, to_world(body, np.array([vert.co[:] for vert in bm.verts])), low)
    # The points the cut made are the border, whatever their place reads back as.
    inside[count:] = 0.0
    place = parle_lips.patch_uv(across, up)
    layer = bm.loops.layers.uv.active
    lips = 0
    for face in bm.faces:
        if layer is None or any(inside[vert.index] < -LIP_CUT_ON_M for vert in face.verts) or not any(inside[vert.index] > LIP_CUT_ON_M for vert in face.verts):
            continue
        for loop in face.loops:
            loop[layer].uv = Vector(place[loop.vert.index])
        lips += 1
    added = len(bm.verts) - count
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return lips, added
