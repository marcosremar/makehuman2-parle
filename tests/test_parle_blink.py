"""Regression for the blink (2026-11-12). The look of the cast is moved into the rest shape, and what is moved into the rest shape goes
into every shape key, the blink too: with the women's lids opened at rest (`parle_face.FAIR_LOOK`) the blink would stop short of
closing by as much, a slit of eye left showing. Run with `Blender -b --factory-startup --python-exit-code 1 -P tests/test_parle_blink.py`."""
import sys
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import parle_face  # noqa: E402

bpy.ops.wm.read_factory_settings(use_empty=True)
# Two upper lids, a lower lid and a cheek: one point each. The left eye is at -x.
POINTS = np.array([(-0.03, 0.0, 0.006), (0.03, 0.0, 0.006), (-0.03, 0.0, -0.006), (-0.03, 0.0, -0.03)])
UPPER_LEFT, UPPER_RIGHT, LOWER_LEFT, CHEEK = range(4)
mesh = bpy.data.meshes.new("Body")
mesh.from_pydata([tuple(point) for point in POINTS], [], [])
body = bpy.data.objects.new("Body", mesh)
bpy.context.scene.collection.objects.link(body)
body.shape_key_add(name="Basis")
MOVES = {
    "faceEyeOpen": {UPPER_LEFT: (0, 0, 0.002), UPPER_RIGHT: (0, 0, 0.002), LOWER_LEFT: (0, 0, -0.001), CHEEK: (0, 0, -0.0003)},
    "eyeWideLeft": {UPPER_LEFT: (0, 0, 0.0015)},
    "eyeWideRight": {UPPER_RIGHT: (0, 0, 0.0015)},
    "eyeBlinkLeft": {UPPER_LEFT: (0, 0, -0.011), LOWER_LEFT: (0, 0, 0.0002)},
    "eyeBlinkRight": {UPPER_RIGHT: (0, 0, -0.011)},
}
for name, moves in MOVES.items():
    key = body.shape_key_add(name=name)
    for index, move in moves.items():
        key.data[index].co = [key.data[index].co[axis] + move[axis] for axis in range(3)]
keys = mesh.shape_keys.key_blocks
shut = {name: parle_face.key_points(keys[name]).copy() for name in parle_face.BLINKS}
rest = parle_face.key_points(keys[0]).copy()

look = {"faceEyeOpen": 0.5, "eyeWideLeft": 1.0, "eyeWideRight": 1.0, "faceJaw": -0.7}
lids = sum(((parle_face.key_points(keys[name]) - rest) * value for name, value in look.items() if name in keys), np.zeros_like(rest))
assert lids[UPPER_LEFT][2] > 0.002, "the look of the test does not open the lids"
parle_face.move_rest(body, lids)
opened = parle_face.key_points(keys["eyeBlinkLeft"]).copy()
assert opened[UPPER_LEFT][2] - shut["eyeBlinkLeft"][UPPER_LEFT][2] > 0.002, "the test does not start from a blink that stops short"
taken = parle_face.blink_still_closes(keys, look)
now = {name: parle_face.key_points(keys[name]) for name in parle_face.BLINKS}
assert taken >= 2, f"how many points of the blinks were seen to: {taken}"
# The lid a blink shuts closes where the kit closes it; the other eye, which that blink does not move, keeps the look.
assert np.allclose(now["eyeBlinkLeft"][UPPER_LEFT], shut["eyeBlinkLeft"][UPPER_LEFT], atol=1e-7), f"the left lid shuts {1000 * (now['eyeBlinkLeft'][UPPER_LEFT][2] - shut['eyeBlinkLeft'][UPPER_LEFT][2]):+.2f} mm from where the kit shuts it"
assert np.allclose(now["eyeBlinkRight"][UPPER_RIGHT], shut["eyeBlinkRight"][UPPER_RIGHT], atol=1e-7), "the right lid does not shut where the kit shuts it"
assert np.allclose(now["eyeBlinkLeft"][UPPER_RIGHT], rest[UPPER_RIGHT] + lids[UPPER_RIGHT], atol=1e-7), "a blink of the left eye moved the right one"
assert np.allclose(now["eyeBlinkLeft"][CHEEK], rest[CHEEK] + lids[CHEEK], atol=1e-7), "a point the blink does not move lost the look"
# A point the blink barely moves gives back only a little: no step where the lid ends.
back = (shut["eyeBlinkLeft"] + lids - now["eyeBlinkLeft"])[LOWER_LEFT][2] / lids[LOWER_LEFT][2]
assert 0 < back < 0.5, f"the lower lid, which the blink moves 0.2 mm, gave back {back:.2f} of the look"
# The rest shape itself keeps the look, and so does every other shape.
assert np.allclose(parle_face.key_points(keys[0]), rest + lids, atol=1e-7) and np.allclose(parle_face.key_points(keys["eyeWideLeft"]) - parle_face.key_points(keys[0]), [MOVES["eyeWideLeft"].get(index, (0, 0, 0)) for index in range(4)], atol=1e-7), "the rest shape or another shape changed"
# A look that leaves the lids alone leaves the blinks alone.
kept = {name: parle_face.key_points(keys[name]).copy() for name in parle_face.BLINKS}
assert parle_face.blink_still_closes(keys, {"faceJaw": -0.7}) == 0 and all(np.allclose(parle_face.key_points(keys[name]), kept[name]) for name in kept), "a look without lid traits changed the blink"
print("ok")
