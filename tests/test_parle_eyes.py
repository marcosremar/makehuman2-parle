"""Regression for the size of the eyes (2026-11-11). The owner found the faces of the cast "muito feio", and bigger eyes are what most
goes with a woman's face being found attractive (Cunningham 1986), so the women's grow more than the men's. Not without end: grown 1.4
times, the skin round each eye stood as a bag, the points between the lids and the end of the field pressed into a fifth of the room
they had. Run with `Blender -b --factory-startup --python-exit-code 1 -P tests/test_parle_eyes.py`."""
import sys
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import parle_polish  # noqa: E402

bpy.ops.wm.read_factory_settings(use_empty=True)
# Two globes of 12 mm of radius, 64 mm apart, at the height of 1.5 m.
RADIUS, HALF, HIGH = 0.012, 0.032, 1.5
BALL = np.array([(RADIUS * np.cos(turn) * np.sin(tilt), RADIUS * np.sin(turn) * np.sin(tilt), RADIUS * np.cos(tilt))
                 for turn in np.linspace(0, 2 * np.pi, 16, endpoint=False) for tilt in np.linspace(0, np.pi, 9)])
# The skin of one cheek: a point every millimetre from 5 to 45 mm below the middle of the eye.
FAR = np.arange(0.005, 0.0451, 0.001)


def cloud(name, points):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(float(value) for value in point) for point in points], [], [])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def grown(scale):
    """How far below the middle of the eye each point of the cheek ends, and the radius of the globes, with the eyes grown `scale` times."""
    eye = cloud("eye", np.concatenate([BALL + (-HALF, 0, HIGH), BALL + (HALF, 0, HIGH)]))
    skin = cloud("Body", [(HALF, 0.0, HIGH - far) for far in FAR])
    parle_polish.bigger_eyes([skin, eye], eye, scale)
    out = HIGH - parle_polish.rest_points(skin)[:, 2], parle_polish.eye_radius(eye)
    for obj in (eye, skin):
        bpy.data.objects.remove(obj)
    return out


women, men = parle_polish.eye_scale("female-slim"), parle_polish.eye_scale("male-muscled")
assert men == parle_polish.EYE_SCALE < women, f"the women's eyes grow no more than the men's: {women} against {men}"
assert parle_polish.eye_scale("female-normal") == parle_polish.eye_scale("female-muscled") == women, "the women's eyes differ from kit to kit"
# On photographs, eyes are judged too large from 116.3% on, with a spread of 9.1 (Seyama & Nagayama 2009), and on a drawn face bigger
# eyes hardly matter (the same authors, 2007). The head of the cast is drawn: a quarter bigger than a real one on the women, its skin
# evened out (2026-11-12). So past what a photograph bears, and no further than the skin round the eye has room for.
assert 1.163 < women <= 1.3, f"the women's eyes: {women}"

after, radius = grown(women)
reach = parle_polish.EYE_REACH_WIDE_M if women > parle_polish.EYE_SCALE else parle_polish.EYE_REACH_M
core, beyond = FAR <= parle_polish.EYE_CORE_M, FAR >= reach
assert abs(radius - RADIUS * women) < 1e-6, f"the globe did not grow with the eye: {radius}"
assert np.allclose(after[core], FAR[core] * women, atol=1e-6), "the lids did not grow with the globe"
assert np.allclose(after[beyond], FAR[beyond], atol=1e-7), "the skin past the reach of the eye moved"
room = np.diff(after) / np.diff(FAR)
assert room.min() > 0.45, f"the skin round the eye is pressed to less than half its room, the start of a fold: {room.min():.2f}"
# The men's, which grow less, settle in the nearer reach.
assert np.allclose(grown(men)[0][FAR >= parle_polish.EYE_REACH_M], FAR[FAR >= parle_polish.EYE_REACH_M], atol=1e-7), "the men's eyes reach as far as the women's"
# Why not bigger: at 1.4, in the reach the eyes had then, the same skin had a fifth of its room and stood as a bag under each eye
# (2026-11-11); and in the wider reach it still has less than at the women's size.
wide = parle_polish.EYE_REACH_WIDE_M
parle_polish.EYE_REACH_WIDE_M = parle_polish.EYE_REACH_M
pressed = np.diff(grown(1.4)[0]) / np.diff(FAR)
parle_polish.EYE_REACH_WIDE_M = wide
assert pressed.min() < 0.3, f"at 1.4 the skin is not pressed as it was when the bags showed: {pressed.min():.2f}"
assert room.min() > pressed.min() + 0.2, f"the wider reach gives the skin no more room than it had when the bags showed: {room.min():.2f}"
assert (np.diff(grown(1.7)[0]) / np.diff(FAR)).min() < room.min() - 0.1, "eyes far bigger still leave the skin as much room"

# A point of the globe that on this face stands below the lower lid, in front of the cheek: Dona Marta had one under each eye, a
# white speck (2026-11-11). It goes behind the skin; the ones seen through the opening stay.
import parle_face  # noqa: E402

SKIN_Y, STEP = -0.0125, 0.001
across, down = np.arange(-0.06, 0.0601, STEP), np.arange(-0.025, 0.0251, STEP)
corners = [(x, SKIN_Y, HIGH + z) for z in down for x in across]
wide = len(across)
in_opening = lambda x, z: abs(abs(x) - HALF) < 0.009 and abs(z) < 0.004
faces = [(row * wide + col, row * wide + col + 1, (row + 1) * wide + col + 1, (row + 1) * wide + col)
         for row in range(len(down) - 1) for col in range(wide - 1) if not in_opening(across[col] + STEP / 2, down[row] + STEP / 2)]
skin_mesh = bpy.data.meshes.new("Body")
skin_mesh.from_pydata(corners, [], faces)
skin_mesh.update()
skin = bpy.data.objects.new("Body", skin_mesh)
bpy.context.scene.collection.objects.link(skin)
globes = np.concatenate([BALL + (-HALF, 0, HIGH), BALL + (HALF, 0, HIGH)])
below = int(np.argmin(np.linalg.norm(globes - (HALF + 0.003, -0.0095, HIGH - 0.0068), axis=1)))
pole = int(np.argmin(np.linalg.norm(globes - (HALF, -RADIUS, HIGH), axis=1)))
assert not in_opening(globes[below][0], globes[below][2] - HIGH) and in_opening(globes[pole][0], globes[pole][2] - HIGH), "the two points are not where the test wants them"
globes[below][1] = SKIN_Y - 0.001
eye = cloud("Human.low-poly", globes)
parle_face.tuck_eyes(skin, [eye])
now = parle_polish.rest_points(eye)
assert now[below][1] > SKIN_Y + 0.0005, f"a point of the globe under the lid still stands in front of the cheek: {now[below][1] - SKIN_Y:+.4f} m from the skin"
assert np.allclose(now[pole], globes[pole], atol=1e-7), "a point seen through the lids was moved"
assert np.allclose(np.delete(now, below, axis=0), np.delete(globes, below, axis=0), atol=1e-7), "points behind the skin were moved"

# What shows is read on the face, not on a list of the kit's (2026-11-12): with the women's lids opened past the kit's, points the
# kit hides stood in the opening, and each went back behind the wall of the socket, a bite out of the iris. Here the socket has a
# wall 8 mm behind the opening, and the points of the globe seen through the opening, beside its middle, stay where they are.
wall = [(HALF - 0.02, -0.004, HIGH - 0.02), (HALF + 0.02, -0.004, HIGH - 0.02), (HALF + 0.02, -0.004, HIGH + 0.02), (HALF - 0.02, -0.004, HIGH + 0.02)]
deep_mesh = bpy.data.meshes.new("Body")
deep_mesh.from_pydata(corners + wall, [], faces + [tuple(range(len(corners), len(corners) + 4))])
deep_mesh.update()
deep = bpy.data.objects.new("Body", deep_mesh)
bpy.context.scene.collection.objects.link(deep)
open_eye = cloud("Human.low-poly", globes)
beside = [index for index, point in enumerate(globes) if in_opening(point[0], point[2] - HIGH) and point[1] < -0.009 and index != pole]
assert len(beside) >= 2, f"the test has no point of the globe in the opening beside its middle: {beside}"
parle_face.tuck_eyes(deep, [open_eye])
opened = parle_polish.rest_points(open_eye)
assert np.allclose(opened[beside + [pole]], globes[beside + [pole]], atol=1e-7), "points of the globe seen through the lids went back into the socket"
assert opened[below][1] > SKIN_Y + 0.0005, "with a socket behind it, the point in front of the cheek stayed there"
for obj in (deep, open_eye):
    bpy.data.objects.remove(obj)

# And at the very end, whatever moved skin and globes apart after that step (the head of a kit grows by its bones, its skin point by
# point): the same point out in front of the cheek again goes back behind it, with no list to read.
bpy.data.objects.remove(eye)
again = cloud("Human.low-poly", globes)
assert parle_polish.globes_behind_skin([skin, again]) == 1, "the point of the globe in front of the cheek was not the one moved, or not the only one"
last = parle_polish.rest_points(again)
assert last[below][1] > SKIN_Y + 0.0005 and np.allclose(np.delete(last, below, axis=0), np.delete(globes, below, axis=0), atol=1e-7), "the globes after the last touch"
assert parle_polish.globes_behind_skin([skin, again]) == 0, "a globe already behind the skin was moved again"

# The globe has few faces: the middle of one may stand in front of the skin with its corners all behind it. Here a face of three
# corners, 1 mm behind the cheek under the opening, and a dent in the cheek 2 mm deep over its middle.
bpy.data.objects.remove(again)
dent = np.array(corners)
dented = np.linalg.norm(dent[:, (0, 2)] - (HALF, HIGH - 0.012), axis=1) < 0.0025
dent[dented, 1] += 0.002
skin.data.vertices.foreach_set("co", dent.ravel())
skin.data.update()
patch = [(HALF - 0.006, SKIN_Y + 0.001, HIGH - 0.016), (HALF + 0.006, SKIN_Y + 0.001, HIGH - 0.016), (HALF, SKIN_Y + 0.001, HIGH - 0.0065)]
seen_through = [(-HALF, -RADIUS, HIGH), (HALF - 0.007, SKIN_Y + 0.0015, HIGH - 0.002), (HALF + 0.007, SKIN_Y + 0.0015, HIGH + 0.002), (HALF, -RADIUS, HIGH)]
flat = bpy.data.meshes.new("Human.low-poly")
flat.from_pydata(patch + seen_through, [], [(0, 1, 2)])
flat.update()
low = bpy.data.objects.new("Human.low-poly", flat)
bpy.context.scene.collection.objects.link(low)
middle_of = lambda obj: parle_polish.rest_points(obj)[:3].mean(axis=0)
assert middle_of(low)[1] < SKIN_Y + 0.002, "the middle of the face is not in front of the dented cheek to begin with"
moved = parle_polish.globes_behind_skin([skin, low])
after = parle_polish.rest_points(low)
assert moved >= 1 and middle_of(low)[1] > SKIN_Y + 0.002 + 0.0005, f"the middle of a face of the globe still stands in front of the skin: {moved} corners moved, {middle_of(low)[1] - SKIN_Y - 0.002:+.4f} m behind it"
assert np.allclose(after[3:], np.array(seen_through), atol=1e-7), "corners seen through the lids were moved"

# Past the outer corner of the opening nothing of the globe is meant to show (2026-11-12): there the white came out through the skin
# in a flag that ran on from the opening, with no strip of lid between it and the middle of the eye, so the rule above let it be.
# A face of the globe from inside the opening out past its corner, its far corner a millimetre in front of the cheek and more
# than 40 degrees round the ball from the front: that corner goes back behind the skin, the two in the opening stay.
bpy.data.objects.remove(low)
flag = [(HALF + 0.018, SKIN_Y - 0.001, HIGH), (HALF + 0.004, SKIN_Y - 0.001, HIGH - 0.002), (HALF + 0.004, SKIN_Y - 0.001, HIGH + 0.002)]
frame = [(-HALF, -0.02, HIGH), (HALF, -0.02, HIGH), (HALF - 0.018, SKIN_Y + 0.002, HIGH)]
skin.data.vertices.foreach_set("co", np.array(corners).ravel())
skin.data.update()


def flagged():
    made = bpy.data.meshes.new("Human.low-poly")
    made.from_pydata(flag + frame, [], [(0, 1, 2)])
    made.update()
    obj = bpy.data.objects.new("Human.low-poly", made)
    bpy.context.scene.collection.objects.link(obj)
    return obj


rule = parle_polish.EYE_SIDE_FRONT
parle_polish.EYE_SIDE_FRONT = -1.0
before_rule = flagged()
assert parle_polish.globes_behind_skin([skin, before_rule]) == 0, "the flag of the test goes back without the rule for the side of the ball"
bpy.data.objects.remove(before_rule)
parle_polish.EYE_SIDE_FRONT = rule
low = flagged()
went_back = parle_polish.globes_behind_skin([skin, low])
tips = parle_polish.rest_points(low)
assert went_back == 1 and tips[0][1] > SKIN_Y + 0.0005, f"the globe past the outer corner still stands in front of the skin: {went_back} points went, {tips[0][1] - SKIN_Y:+.4f} m"
assert np.allclose(tips[1:], np.array(flag + frame)[1:], atol=1e-7), "points of the globe in the opening went with it"

# The globe of the kit is a ball of eight sides: the middle of a face stood more than a millimetre inside the ball, and the rim of a
# lid floated over it (the judge by sight, 2026-11-12: "ragged lid edges, a torn line under the lower lid"). Rounded, every face
# is four, twice over, the new points out on the ball; the kit's own points, the shape keys and the UVs follow.
import bmesh  # noqa: E402
from mathutils import Vector  # noqa: E402

for obj in (skin, low):
    bpy.data.objects.remove(obj)
EYES = (Vector((-HALF, 0, HIGH)), Vector((HALF, 0, HIGH)))
build = bmesh.new()
for middle in EYES:
    for vertex in bmesh.ops.create_uvsphere(build, u_segments=8, v_segments=6, radius=RADIUS)["verts"]:
        vertex.co += middle
bmesh.ops.triangulate(build, faces=build.faces[:])
kit_mesh = bpy.data.meshes.new("Human.low-poly")
build.to_mesh(kit_mesh)
build.free()
kit_mesh.uv_layers.new(name="UVMap")
kit = bpy.data.objects.new("Human.low-poly", kit_mesh)
bpy.context.scene.collection.objects.link(kit)
kit.shape_key_add(name="Basis")
look_up = kit.shape_key_add(name="eyeLookUpLeft")
for point in look_up.data:
    point.co.z += 0.005
before = parle_polish.rest_points(kit)
faces_before = len(kit_mesh.polygons)


def inside_ball(obj):
    """How far inside the ball of its eye the middle of the face furthest in stands."""
    points = parle_polish.rest_points(obj)
    middles = np.array([points[list(polygon.vertices)].mean(axis=0) for polygon in obj.data.polygons])
    return float((RADIUS - np.min([np.linalg.norm(middles - np.array(eye_middle), axis=1) for eye_middle in EYES], axis=0)).max())


flat_by = inside_ball(kit)
parents = parle_polish.round_globes(kit)
after = parle_polish.rest_points(kit)
assert len(kit_mesh.polygons) == faces_before * 4 ** parle_polish.EYE_ROUND_LEVELS and {len(polygon.vertices) for polygon in kit_mesh.polygons} == {3}, f"the faces of the rounded globe: {len(kit_mesh.polygons)} from {faces_before}"
assert np.allclose(after[:len(before)], before, atol=1e-7), "the kit's own points of the globe moved"
assert len(parents) == len(after) - len(before) and all(max(pair) < len(before) + index for index, pair in enumerate(parents)), "the parents of the new points"
far = np.min([np.linalg.norm(after - np.array(eye_middle), axis=1) for eye_middle in EYES], axis=0)
assert np.allclose(far, RADIUS, atol=2e-6), f"new points of the globe off its ball by up to {np.abs(far - RADIUS).max() * 1000:.3f} mm"
assert flat_by > 0.0008 and inside_ball(kit) < 0.0001, f"the faces of the globe stand {inside_ball(kit) * 1000:.2f} mm inside the ball (they stood {flat_by * 1000:.2f})"
keys = kit_mesh.shape_keys.key_blocks
assert [key.name for key in keys] == ["Basis", "eyeLookUpLeft"] and np.allclose(parle_polish.read(keys[1]) - parle_polish.read(keys[0]), (0, 0, 0.005), atol=1e-6), "a shape key of the globe did not follow its new points"
assert len(kit_mesh.uv_layers) == 1 and len(kit_mesh.uv_layers[0].data) == len(kit_mesh.loops), "the UVs of the globe did not follow"
# And its faces, a sixteenth of what they were, are looked at in as many places for their size as the kit's own were.
big, small = np.array([(0, 0, 0), (0.013, 0, 0), (0, 0.013, 0)]), np.array([(0, 0, 0), (0.003, 0, 0), (0, 0.003, 0)])
assert parle_polish.face_steps(big) == parle_polish.EYE_FACE_STEPS and parle_polish.face_steps(small) == 3 and parle_polish.face_steps(small / 4) == 1, "in how many steps a face of the globe is looked at"
print("ok")
