"""Regression for the look the faces of the cast aim at (2026-11-11). The owner found them "muito feio" and asked for a measure of beauty
as the criterion (`parle_beauty`); what it pointed at goes into the rest shape through the kit's own morphs, a bias on top of each
person's traits (`parle_face.FAIR_LOOK`). Run with `Blender -b --factory-startup --python-exit-code 1 -P tests/test_parle_fair_look.py`."""
import sys
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import parle_face  # noqa: E402

bpy.ops.wm.read_factory_settings(use_empty=True)
women, men = parle_face.FAIR_LOOK["female"], parle_face.FAIR_LOOK["male"]
# The directions the literature gives for a woman's face (Cunningham 1986): smaller nose, smaller jaw, shorter chin, higher brows, eyes further apart.
assert all(women[name] < 0 for name in ("faceNoseWidth", "faceJaw", "faceChin")), f"nose, jaw or chin go the wrong way: {women}"
assert all(women[name] > 0 for name in ("faceBrowHeight", "faceEyeSpacing")), f"brows or eyes go the wrong way: {women}"
# A bias, not another face: no more than a person's own traits take (0.55), and twice these pinched the face below the cheeks.
traits = {name: value for name, value in women.items() if name.startswith("face")}
assert all(abs(value) <= 0.4 for value in traits.values()), f"a bias as strong as a person's own traits: {traits}"
assert not any(name.startswith("face") for name in men), f"the men's faces, which score next to Rafa's, are bent too: {men}"
# The corners of the mouth rest a little up on everybody, the same on both sides, and far from a smile.
for look in (women, men):
    assert 0 < look["mouthSmileLeft"] == look["mouthSmileRight"] <= 0.2, f"the mouth at rest: {look}"

# The jaw rests closed just as far as the lips need to meet. A quarter of `jawOpen` carried the lower lip 7 mm up over the upper one and
# 5 mm out in front of it, a mouth with one lip (2026-11-11, `parle_profile`); with none the lips stand apart and the teeth show (2026-11-07).
closed = -parle_face.CALM_LOOK["jawOpen"]
assert 0.08 <= closed <= 0.15, f"the jaw at rest is closed {closed} of the way: past the lips' meeting, or short of it"

# Baked into the rest shape: the rest moves by each morph times its weight, and every key still makes the same move from it.
mesh = bpy.data.meshes.new("Body")
mesh.from_pydata([(0, 0, 0), (0.02, 0, 0), (-0.02, 0, 0), (0, 0, 0.05)], [], [])
body = bpy.data.objects.new("Body", mesh)
bpy.context.scene.collection.objects.link(body)
body.shape_key_add(name="Basis")
moves = {"faceChin": (0, (0, 0, -0.01)), "mouthSmileLeft": (1, (0, 0, 0.01)), "mouthSmileRight": (2, (0, 0, 0.01)), "viseme_aa": (0, (0, 0, -0.03))}
for name, (index, move) in moves.items():
    key = body.shape_key_add(name=name)
    key.data[index].co = [a + b for a, b in zip(key.data[index].co, move)]
keys = body.data.shape_keys.key_blocks
rest = parle_face.key_points(keys[0]).copy()
before = {name: parle_face.key_points(keys[name]) - rest for name in moves}
parle_face.calm_look([body], fair={"faceChin": -0.5, "mouthSmileLeft": 0.1, "mouthSmileRight": 0.1})
now = parle_face.key_points(keys[0])
assert np.allclose(now[0] - rest[0], (0, 0, 0.005), atol=1e-7), f"the chin did not rise by half its morph: {now[0] - rest[0]}"
assert np.allclose(now[1] - rest[1], (0, 0, 0.001), atol=1e-7) and np.allclose(now[2] - rest[2], (0, 0, 0.001), atol=1e-7), "the corners of the mouth did not rise"
assert np.allclose(now[3], rest[3], atol=1e-7), "a point no morph moves was moved"
assert np.allclose(np.array([vertex.co for vertex in body.data.vertices]), now, atol=1e-7), "the mesh and its rest key part ways"
for name in moves:
    assert np.allclose(parle_face.key_points(keys[name]) - now, before[name], atol=1e-7), f"{name} no longer makes the move it made"
print("ok")
