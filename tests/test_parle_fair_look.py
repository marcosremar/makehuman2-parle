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
# The directions the literature gives for a woman's face (Cunningham 1986): smaller nose, smaller jaw, shorter chin, bigger eyes
# further apart. Its higher eyebrows are not kept: on the cast they read as a startled arch, and Sofia's sit close over the eye (below).
assert all(women[name] < 0 for name in ("faceNoseWidth", "faceJaw", "faceChin")), f"nose, jaw or chin go the wrong way: {women}"
assert women["faceEyeSpacing"] > 0 and "faceEyeSize" not in women, f"the eyes go the wrong way, or grow by a trait the globe does not have: {women}"
# How far: at first "a bias, not another face", no trait past 0.4. On 2026-11-12 the measure gave the women 9.0 to 9.3 and a judge
# by sight 3.5 in 10, the owner agreeing: "a tired, realistic adult" with a square lower face. The jaw now narrows as far as the kit
# takes it without folding once the skin is evened out, and nothing else goes past half of a trait.
traits = {name: value for name, value in women.items() if name.startswith("face")}
assert -0.8 <= traits["faceJaw"] < -0.4 and all(abs(value) <= 0.5 for name, value in traits.items() if name != "faceJaw"), f"how far the traits of the women go: {traits}"
assert not any(name.startswith("face") for name in men), f"the men's faces, which score next to Rafa's, are bent too: {men}"
# The eyebrows keep the height and the slope they had: let down they pressed on the lid, and with the inner end let down too they
# climbed to a far peak, a scowl (the judge by sight, 2026-11-12). What changes is their shape (`BROW_ARCH`, below). And the nose is
# shorter with a round tip.
assert 0 <= women["faceBrowHeight"] <= 0.3 and "faceBrowAngle" not in women and "browInnerUp" not in women, f"the women's eyebrows: {women}"
assert women["faceNoseLength"] < 0 < women["noseShape-ponta-arredondada"], f"the women's nose: {women}"
# The corners of the mouth rest a little up on everybody, the same on both sides, and far from a smile. The women's need less of it:
# with the jaw where the lips meet their corners already sit above the middle of the lips, and with the men's 0.12 they sat 7% of the
# width of the mouth above it, past Sofia's and those of the women of the game (2026-11-12).
for look in (women, men):
    assert 0 < look["mouthSmileLeft"] == look["mouthSmileRight"] <= 0.2, f"the mouth at rest: {look}"
assert women["mouthSmileLeft"] <= 0.06 < men["mouthSmileLeft"], f"the women's mouth smiles at rest as much as the men's: {women['mouthSmileLeft']}"
# The nose is narrowed little: with the kit's narrow straight nose on top of -0.7 the cheek drew in beside the wing and the lining of
# the mouth showed through it, a flake on each cheek (2026-11-12).
assert women["faceNoseWidth"] >= -0.4 and "noseShape-reto-estreito" not in women, f"the women's nose is narrowed as it was when the lining came through: {women}"
# The lids: more open than the calm look leaves them, and still down to the top of the iris, which is drawn bigger. At -0.25 with
# the iris of a real eye the whole iris showed and the judge by sight read a stare; at -0.4 and -0.3 a flat slit.
assert parle_face.CALM_LOOK["faceEyeOpen"] < women["faceEyeOpen"] <= -0.2, f"the women's lids: {women['faceEyeOpen']}"

# The eyebrows keep more of the kit's thickness on the women: at three quarters they read as a pencil line on a smooth face (2026-11-12).
assert parle_face.BROW_THIN["male"] < parle_face.BROW_THIN["female"] <= 1, f"how thick the eyebrows are left: {parle_face.BROW_THIN}"
brow = np.array([(sign * x, 0.0, 0.001 * side) for sign in (-1, 1) for x in (0.01, 0.02, 0.03, 0.04, 0.05) for side in (-1, 1)])
thinner = brow + parle_face.brow_line_move(brow, 0.5)
assert np.allclose(thinner[:, 2], brow[:, 2] * 0.5, atol=1e-9) and np.allclose(thinner[:, :2], brow[:, :2]), "an eyebrow is not thinned about its middle line"
assert np.allclose(parle_face.brow_line_move(brow, 1.0), 0), "an eyebrow left whole moves"
# And it thins to its outer end: whole by the nose, half at the temple, evenly between, on both sides. The women's was a boomerang,
# thickest at mid-length (2026-11-12).
tapered = brow + parle_face.brow_line_move(brow, 1.0, 0.5)
left = np.abs(tapered[:, 2]) / 0.001
assert np.allclose(left[np.abs(brow[:, 0]) == 0.01], 1, atol=1e-9) and np.allclose(left[np.abs(brow[:, 0]) == 0.05], 0.5, atol=1e-9) and np.allclose(left[np.abs(brow[:, 0]) == 0.03], 0.75, atol=1e-9), f"an eyebrow does not thin evenly to its outer end: {left}"
assert parle_face.BROW_OUTER["female"] < parle_face.BROW_OUTER["male"] == 1, f"which eyebrows thin to the temple: {parle_face.BROW_OUTER}"
# And it stops short of the temple: its inner end stays, its outer end comes in by the share that is cut, the rest evenly.
shorter = brow + parle_face.brow_line_move(brow, 1.0, 1.0, 0.8)
assert np.allclose(np.abs(shorter[np.abs(brow[:, 0]) == 0.01, 0]), 0.01, atol=1e-9) and np.allclose(np.abs(shorter[np.abs(brow[:, 0]) == 0.05, 0]), 0.042, atol=1e-9), "an eyebrow is not shortened from its inner end"
assert np.allclose(shorter[:, 1:], brow[:, 1:], atol=1e-9) and parle_face.BROW_LONG["female"] < parle_face.BROW_LONG["male"] == 1, "a shorter eyebrow changes its thickness, or the men's is cut"
# And its arch comes down: an eyebrow that rises 4 mm at mid-length over the line between its ends rises 2 with half the arch,
# its ends where they were and its thickness the same.
arched = np.array([(sign * x, 0.0, 0.004 * (1 - ((x - 0.03) / 0.02) ** 2) + 0.001 * side) for sign in (-1, 1) for x in (0.01, 0.02, 0.03, 0.04, 0.05) for side in (-1, 1)])
lower = arched + parle_face.brow_line_move(arched, 1.0, 1.0, 1.0, 0.5)
peak = lambda brows: float(brows[np.abs(brows[:, 0]) == 0.03, 2].mean())
ends = np.abs(arched[:, 0]) != 0.03
assert abs(peak(lower) - 0.002) < 2e-4 and np.allclose(lower[np.isin(np.abs(arched[:, 0]), (0.01, 0.05))], arched[np.isin(np.abs(arched[:, 0]), (0.01, 0.05))], atol=2e-4), f"half the arch of an eyebrow is {peak(lower) * 1000:.2f} mm high, not 2"
assert np.allclose(lower[0::2, 2] - lower[1::2, 2], arched[0::2, 2] - arched[1::2, 2], atol=1e-9) and parle_face.BROW_ARCH["female"] < parle_face.BROW_ARCH["male"] == 1, "a flatter eyebrow changes its thickness, or the men's is flattened"

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
