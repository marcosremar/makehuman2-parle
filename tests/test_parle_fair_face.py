"""Regression for the skin of the face (2026-11-12). Run with
`Blender -b --factory-startup --python-exit-code 1 -P tests/test_parle_fair_face.py`.

Owner, 2026-11-11: "o rosto da Lúcia está muito feio". In clay, next to Sofia's face, the cast's had the folds of a real one from the
nose to the chin, and the ruler of `parle_beauty`, which reads points, saw none of them. `parle_polish.fair_face` evens the skin where
the soft traits of the face act and keeps the lids, the nose and where the lips meet. And a flake of the lining of the mouth showed
on each cheek beside the nose: `parle_polish.lining_behind_skin` puts the inside of the mouth back behind the skin."""
import sys
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import parle_mouth  # noqa: E402
import parle_polish  # noqa: E402

bpy.ops.wm.read_factory_settings(use_empty=True)


def sheet(xs, zs, depth):
    """A sheet of skin facing -y: its points, row by row, and its faces. `depth` gives y at each point."""
    points = [(x, depth(x, z), z) for z in zs for x in xs]
    wide = len(xs)
    faces = [(row * wide + col, row * wide + col + 1, (row + 1) * wide + col + 1, (row + 1) * wide + col) for row in range(len(zs) - 1) for col in range(wide - 1)]
    return points, faces


def rig_with_head():
    data = bpy.data.armatures.new("rig")
    rig = bpy.data.objects.new("rig", data)
    bpy.context.scene.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    for name, low, high in (("Neck", 0.0, 0.1), ("Head", 0.1, 0.3)):
        bone = data.edit_bones.new(name)
        bone.head, bone.tail = (0, 0, low), (0, 0, high)
    data.edit_bones["Head"].parent = data.edit_bones["Neck"]
    bpy.ops.object.mode_set(mode="OBJECT")
    return rig


# A patch of the face 8 cm across, a point every 4 mm: a dome 1 cm high, which is the shape of the head, with a fold 2 mm deep every
# 16 mm across it, which is what a real face has from the nose to the mouth. Two rows 2 mm apart in the middle are where the lips meet.
xs = np.arange(-0.04, 0.0401, 0.004)
zs = np.concatenate([np.arange(-0.04, -0.0039, 0.004), (-0.001, 0.001), np.arange(0.004, 0.0401, 0.004)])
dome = lambda x, z: -0.01 * np.cos(np.pi * x / 0.16) * np.cos(np.pi * z / 0.16)
fold = lambda x, z: 0.002 * np.sin(2 * np.pi * x / 0.016)
points, faces = sheet(xs, zs, lambda x, z: dome(x, z) + fold(x, z))
# Behind it, a piece that is inside the head: it has skin ahead of it.
inner, inner_faces = sheet(xs[8:13], zs[4:9], lambda x, z: 0.01 + fold(x, z))
first_inner = len(points)
mesh = bpy.data.meshes.new("Body")
mesh.from_pydata(points + inner, [], faces + [tuple(index + first_inner for index in face) for face in inner_faces])
body = bpy.data.objects.new("Body", mesh)
bpy.context.scene.collection.objects.link(body)
rest = np.array(points + inner)
x, z = rest[:, 0], rest[:, 2]
outer = np.arange(len(rest)) < first_inner
lids = outer & (z > 0.03)
lips = outer & (np.abs(z) < 0.006) & (np.abs(x) < 0.022)
meet = lips & (np.abs(z) < 0.002)
off_head = outer & (x < -0.03)
body.shape_key_add(name="Basis")
for name, where, move in (("faceCheek", np.ones(len(rest), bool), (0, 0, 0.001)), ("eyeBlinkLeft", lids, (0, 0, -0.004)), ("faceLipVolume", lips, (0, -0.002, 0)),
                          ("jawOpen", outer & (z < 0), (0, 0, -0.02)), ("viseme_aa", outer & (z < 0), (0, 0, -0.01))):
    key = body.shape_key_add(name=name)
    for index in np.flatnonzero(where):
        key.data[index].co = [a + b for a, b in zip(key.data[index].co, move)]
group = body.vertex_groups.new(name="Head")
group.add([int(index) for index in np.flatnonzero(~off_head)], 1.0, "REPLACE")
body.vertex_groups.new(name="Neck").add([int(index) for index in np.flatnonzero(off_head)], 1.0, "REPLACE")
rig = rig_with_head()
keys = body.data.shape_keys.key_blocks
before = {key.name: parle_polish.read(key) - rest for key in keys}

moved, far = parle_polish.fair_face(body, rig)
now = parle_polish.read(keys[0])
went = np.linalg.norm(now - rest, axis=1)
soft = outer & ~lids & ~lips & ~off_head & (np.abs(x) < 0.024) & (np.abs(z) > 0.01) & (np.abs(z) < 0.024)
left = np.abs(now[soft, 1] - dome(x[soft], z[soft])).max()
assert moved > 100 and 0.001 < far < 0.004, f"the skin of the face did not move, or went far: {moved} points, {far * 1000:.1f} mm"
assert left < 0.0006, f"a fold 2 mm deep is still {left * 1000:.2f} mm deep after the skin is evened"
# The shape of the head stays: the skin lies on the dome, not inside it, where steps toward the neighbours alone would have shrunk it.
inside = float((now[soft, 1] - dome(x[soft], z[soft])).mean())
assert abs(inside) < 0.0002, f"the evened skin lies {inside * 1000:.2f} mm off the dome of the face"
# What keeps its points: the lids, where the lips meet, what is not skin of the head, and what is inside the head.
for name, where in (("the lids", lids), ("the meeting of the lips", meet), ("skin that is not of the head", off_head), ("what is inside the head", ~outer)):
    assert went[where].max() < 1e-7, f"{name} moved {went[where].max() * 1000:.2f} mm"
# The lips take part with a small weight: the rest of them moves, and less than the skin beside them with the same fold.
rest_of_lips = lips & ~meet & (np.abs(x) < 0.014)
beside = outer & ~lips & (np.abs(z) < 0.006) & (np.abs(x) > 0.024) & (np.abs(x) < 0.03) | soft
assert 1e-5 < went[rest_of_lips].max() < 0.6 * went[beside].max(), f"the lips: {went[rest_of_lips].max() * 1000:.2f} mm against {went[beside].max() * 1000:.2f} beside them"
# Every key still makes the move it made, and the mesh is its rest key.
for key in keys[1:]:
    assert np.allclose(parle_polish.read(key) - now, before[key.name], atol=1e-7), f"{key.name} no longer makes the move it made"
assert np.allclose(parle_polish.read(body.data.vertices), now, atol=1e-7), "the mesh and its rest key part ways"
# A body with none of the soft traits of the face is left alone.
bare = bpy.data.objects.new("Bare", bpy.data.meshes.new("Bare"))
bare.data.from_pydata(points, [], faces)
assert parle_polish.fair_face(bare, rig) == (0, 0.0), "a body without the traits of a face was smoothed"

# The eyebrows lie on the skin. A strip that starts 1 mm inside the dome of the brow and ends 3 mm above it, as the kit's did after
# the face had moved under it (from the side it stood out like a visor), ends BROW_OFF_SKIN_M off the skin all along, and its shape
# keys still make the move they made.
bpy.ops.wm.read_factory_settings(use_empty=True)
brow_dome = lambda x, z: -0.01 * np.cos(np.pi * x / 0.16) * np.cos(np.pi * z / 0.16)
skin_points, skin_faces = sheet(np.arange(-0.04, 0.0401, 0.004), np.arange(-0.04, 0.0401, 0.004), brow_dome)
mesh = bpy.data.meshes.new("Body")
mesh.from_pydata(skin_points, [], skin_faces)
body = bpy.data.objects.new("Body", mesh)
bpy.context.scene.collection.objects.link(body)
strip_xs = np.arange(-0.03, 0.0301, 0.005)
strip = [(x, brow_dome(x, z) + 0.001 - 0.004 * (x + 0.03) / 0.06, z) for z in (0.0, 0.004) for x in strip_xs]
wide = len(strip_xs)
mesh = bpy.data.meshes.new("eyebrow")
mesh.from_pydata(strip, [], [(col, col + 1, wide + col + 1, wide + col) for col in range(wide - 1)])
brow = bpy.data.objects.new("Human.eyebrow005", mesh)
bpy.context.scene.collection.objects.link(brow)
brow.shape_key_add(name="Basis")
raised = brow.shape_key_add(name="browInnerUp")
for point in raised.data:
    point.co.z += 0.003
far = parle_polish.brows_on_skin([body, brow])
now = parle_polish.read(brow.data.shape_keys.key_blocks[0])
off = np.array([dome_y - point[1] for point, dome_y in zip(now, brow_dome(now[:, 0], now[:, 2]))])
assert 0.0025 < far < 0.0035, f"the eyebrow of the test stood {far * 1000:.1f} mm off the skin at most, not 3"
assert np.allclose(off, parle_polish.BROW_OFF_SKIN_M, atol=0.0003), f"the eyebrow does not lie on the skin: {off * 1000} mm off it"
assert np.allclose(parle_polish.read(brow.data.shape_keys.key_blocks["browInnerUp"]) - now, (0, 0, 0.003), atol=1e-7), "the raised eyebrow no longer rises as it did"
assert parle_polish.brows_on_skin([body]) == 0.0, "a body with no eyebrow has one laid on it"

# The inside of the mouth. Its points are the piece of the map the inside of the mouth sits on (`parle_mouth.mouth_piece`): a sheet
# behind the cheek, in the box of the teeth, whose map coordinates are on that island. One of its faces stands 1 mm in front of the cheek
# (its corners behind, its middle through: the cheek dips there) and another half a millimetre behind it.
bpy.ops.wm.read_factory_settings(use_empty=True)
cheek_xs, cheek_zs = np.arange(-0.03, 0.0301, 0.003), np.arange(-0.03, 0.0301, 0.003)
dip = lambda x, z: 0.004 * np.exp(-((x - 0.012) ** 2 + (z - 0.012) ** 2) / 0.004 ** 2)
cheek, cheek_faces = sheet(cheek_xs, cheek_zs, dip)
lining_xs, lining_zs = np.array((0.006, 0.018)), np.array((0.006, 0.018))
lining, lining_faces = sheet(lining_xs, lining_zs, lambda x, z: 0.0025)
deep, deep_faces = sheet(np.array((-0.02, 0.02)), np.array((-0.02, 0.0)), lambda x, z: 0.02)
count = (len(cheek), len(lining), len(deep))
first_lining, first_deep = count[0], count[0] + count[1]
mesh = bpy.data.meshes.new("Body")
# The lining and the deep of the mouth are one piece: a face joins them.
join = [(first_lining, first_lining + 1, first_deep + 1, first_deep)]
mesh.from_pydata(cheek + lining + deep, [], cheek_faces + [tuple(index + first_lining for index in face) for face in lining_faces] + [tuple(index + first_deep for index in face) for face in deep_faces] + join)
body = bpy.data.objects.new("Body", mesh)
bpy.context.scene.collection.objects.link(body)
layer = mesh.uv_layers.new(name="UVMap")
inside_uv = (parle_mouth.BODY_ISLAND_U + 0.05, 1 - (parle_mouth.BODY_ISLAND_V + 0.05))
for loop in mesh.loops:
    layer.data[loop.index].uv = inside_uv if loop.vertex_index >= first_lining else (0.2, 0.9)
body.shape_key_add(name="Basis")
lip = body.shape_key_add(name="faceLipVolume")
lip.data[0].co = [a + b for a, b in zip(lip.data[0].co, (0, -0.002, 0))]
teeth_mesh = bpy.data.meshes.new("teeth")
teeth_mesh.from_pydata([(-0.025, 0.004, -0.025), (0.025, 0.004, -0.025), (0.025, 0.03, 0.025), (-0.025, 0.03, 0.025)], [], [(0, 1, 2, 3)])
teeth = bpy.data.objects.new("Human.teeth_base", teeth_mesh)
bpy.context.scene.collection.objects.link(teeth)
start = parle_polish.read(body.data.shape_keys.key_blocks[0])
piece = np.arange(len(start)) >= first_lining
assert dip(0.012, 0.012) > 0.0025 + 0.001, "the lining of the test does not come through the cheek"
went = parle_polish.lining_behind_skin([body, teeth])
after = parle_polish.read(body.data.shape_keys.key_blocks[0])
assert went == count[1], f"{went} points of the lining went back, of the {count[1]} of the face that came through"
assert np.allclose(after[~piece], start[~piece], atol=1e-9), "the skin of the face moved for the lining"
assert np.allclose(after[first_deep:], start[first_deep:], atol=1e-9), "the deep of the mouth, which was behind the skin, moved"
assert np.allclose(after[piece][:, [0, 2]], start[piece][:, [0, 2]], atol=1e-9), "the lining went another way than back"
behind = after[first_lining:first_deep, 1] - dip(0.012, 0.012)
assert (behind >= parle_polish.LINING_BEHIND_M - 1e-6).all() and (behind < 0.006).all(), f"the lining is {behind * 1000} mm behind the dip of the cheek"
assert parle_polish.lining_behind_skin([body, teeth]) == 0, "a lining already behind the skin went back again"
print("ok")
