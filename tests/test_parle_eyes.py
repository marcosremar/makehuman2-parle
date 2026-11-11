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
# On photographs, eyes are judged too large from 116.3% on, with a spread of 9.1 (Seyama & Nagayama 2009): no further than one spread past it.
assert women <= 1.163 + 0.091, f"the women's eyes are past where a real face's read as too large: {women}"

after, radius = grown(women)
core, beyond = FAR <= parle_polish.EYE_CORE_M, FAR >= parle_polish.EYE_REACH_M
assert abs(radius - RADIUS * women) < 1e-6, f"the globe did not grow with the eye: {radius}"
assert np.allclose(after[core], FAR[core] * women, atol=1e-6), "the lids did not grow with the globe"
assert np.allclose(after[beyond], FAR[beyond], atol=1e-7), "the skin past the reach of the eye moved"
room = np.diff(after) / np.diff(FAR)
assert room.min() > 0.5, f"the skin round the eye is pressed to less than half its room, the start of a fold: {room.min():.2f}"
# Why not bigger: at 1.4 the same skin has a fifth of its room, and stood as a bag under each eye (2026-11-11).
pressed = np.diff(grown(1.4)[0]) / np.diff(FAR)
assert pressed.min() < 0.3, f"at 1.4 the skin is not pressed as it was when the bags showed: {pressed.min():.2f}"

# A point of the globe that the kit lists as seen through the lids, but that on this face stands below the lower lid, in front of the
# cheek: Dona Marta had one under each eye, a white speck (2026-11-11). It goes behind the skin; the ones seen through the opening stay.
import json  # noqa: E402
import tempfile  # noqa: E402

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
with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as listed:
    json.dump([below, pole], listed)
parle_face.tuck_eyes(skin, [eye], listed.name)
now = parle_polish.rest_points(eye)
assert now[below][1] > SKIN_Y + 0.0005, f"a point of the globe under the lid still stands in front of the cheek: {now[below][1] - SKIN_Y:+.4f} m from the skin"
assert np.allclose(now[pole], globes[pole], atol=1e-7), "a point seen through the lids was moved"
assert np.allclose(np.delete(now, below, axis=0), np.delete(globes, below, axis=0), atol=1e-7), "points behind the skin were moved"

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
print("ok")
