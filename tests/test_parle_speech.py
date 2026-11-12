"""Regression for how far the mouth of the cast opens in speech (2026-11-12). Run with
`Blender -b --factory-startup --python-exit-code 1 -P tests/test_parle_speech.py`.

The owner asked for the faces to be looked at in every movement of the mouth, and found them "ainda bem feio". With the weights the
player gives every face, the cast's mouth opened two to three times as far as Rafa's and Sofia's, into a dark hole with the lower
teeth in a row: the kit's open shapes move the chin 41 to 45 mm where theirs move it 7 to 26. `parle_face.calm_speech` keeps a share
of those shapes (`SPEECH_OPEN`), on the body, the teeth and the tongue alike."""
import sys
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import parle_face  # noqa: E402

bpy.ops.wm.read_factory_settings(use_empty=True)


def piece(name, moves):
    """Three points, and a shape key for each move: the first point goes down by that much."""
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([(0, 0, 0), (0.01, 0, 0), (0, 0, 0.02)], [], [])
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.shape_key_add(name="Basis")
    for key_name, drop in moves.items():
        key = obj.shape_key_add(name=key_name)
        key.data[0].co = (0, 0, -drop)
    return obj


moves = {"jawOpen": 0.045, "viseme_aa": 0.041, "viseme_PP": 0.008}
body, teeth, hat = piece("Body", moves), piece("Human.teeth_base", {"jawOpen": 0.045, "viseme_aa": 0.041}), piece("Hat", {})
shares = parle_face.SPEECH_OPEN["female"]
# The body comes twice in the list, as it did in the cast (it is among the meshes of the character and was named ahead of them
# too): it took every share twice over and its mouth opened less than the teeth, which then dropped below the lip (2026-11-12).
parle_face.calm_speech([body, body, teeth, hat], shares)
for obj in (body, teeth):
    keys = obj.data.shape_keys.key_blocks
    rest = parle_face.key_points(keys[0])
    assert np.allclose(rest, [(0, 0, 0), (0.01, 0, 0), (0, 0, 0.02)], atol=1e-9), f"the rest shape of {obj.name} moved"
    for name in ("jawOpen", "viseme_aa"):
        drop = -(parle_face.key_points(keys[name]) - rest)[0, 2]
        assert abs(drop - moves[name] * shares[name]) < 1e-7, f"{name} of {obj.name} opens {drop * 1000:.1f} mm, not {shares[name]} of {moves[name] * 1000:.0f}"
        assert np.allclose((parle_face.key_points(keys[name]) - rest)[1:], 0, atol=1e-9), f"{name} of {obj.name} moves points it did not move"
# The body and the teeth open by the same share: the jaw carries both, and cut on one alone the teeth would come through the lip.
drops = [-(parle_face.key_points(obj.data.shape_keys.key_blocks["jawOpen"]) - parle_face.key_points(obj.data.shape_keys.key_blocks[0]))[0, 2] for obj in (body, teeth)]
assert abs(drops[0] - drops[1]) < 1e-9, f"the body and the teeth part: {drops}"
# A shape that is not of the open mouth stays whole.
still = -(parle_face.key_points(body.data.shape_keys.key_blocks["viseme_PP"]) - parle_face.key_points(body.data.shape_keys.key_blocks[0]))[0, 2]
assert abs(still - 0.008) < 1e-7, f"the closed lips were cut too: {still}"

# The shares: every one keeps less than the whole shape and more than a third of it, the wide "a" least; the peak of an "a" in the
# player (0.55 of viseme_aa and 0.2 of jawOpen) lands between Sofia's 9 mm and Rafa's 14, on the woman measured (41 and 45 mm) and
# on the man (31 and 35), where the kit's was 32: at 7 mm the judge by sight found every open shape the same flat slit.
for sex, own in (("female", (0.041, 0.045)), ("male", (0.031, 0.035))):
    kept = parle_face.SPEECH_OPEN[sex]
    assert all(0.33 < share < 1 for share in kept.values()) and kept["viseme_aa"] == min(kept.values()), f"the shares of the {sex} mouth: {kept}"
    peak = 0.55 * own[0] * kept["viseme_aa"] + 0.2 * own[1] * kept["jawOpen"]
    assert 0.009 < peak < 0.014, f"the peak of an \"a\" on the {sex} mouth is {peak * 1000:.1f} mm, far from the leads' or back at a slit"

# The rounded shapes (2026-11-12): at the player's weights the kit's "o" was "a wide letterbox gape where Sofia makes a small round
# pucker" and its "ch" the same half-open slit as four other shapes (the judge by sight, round 14). `mixed_speech` makes each again
# from the shapes a mesh has, read before any is written.
bpy.ops.wm.read_factory_settings(use_empty=True)


def shaped(name, moves):
    """One point, and a shape key for each move."""
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([(0, 0, 0)], [], [])
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.shape_key_add(name="Basis")
    for key_name, move in moves.items():
        obj.shape_key_add(name=key_name).data[0].co = move
    return obj


OWN = {"viseme_O": (0.004, 0, -0.012), "viseme_U": (-0.006, -0.003, -0.001), "viseme_CH": (0, -0.004, -0.004), "viseme_aa": (0.002, 0, -0.02),
       "jawOpen": (0, 0, -0.03), "mouthSmileLeft": (0.005, 0, 0.003), "mouthSmileRight": (0.005, 0, 0.003), "viseme_PP": (0, 0, 0.001)}
face = shaped("Body", OWN)
jaw_only = {name: OWN[name] for name in ("viseme_O", "viseme_aa", "jawOpen")}
teeth = shaped("Human.teeth_base", jaw_only)
mix = parle_face.SPEECH_MIX
parle_face.mixed_speech([face, face, teeth, shaped("Hat", {})], mix)
for obj, own in ((face, OWN), (teeth, jaw_only)):
    keys = obj.data.shape_keys.key_blocks
    assert np.allclose(parle_face.key_points(keys[0]), 0, atol=1e-9), f"the rest shape of {obj.name} moved"
    for name, move in own.items():
        want = sum((np.array(own[part]) * share for part, share in mix[name].items() if part in own), np.zeros(3)) if name in mix else np.array(move)
        assert np.allclose(parle_face.key_points(keys[name])[0], want, atol=1e-7), f"{name} of {obj.name} is {parle_face.key_points(keys[name])[0]}, not {want}"
# What the mix says: the "o" takes more of the pucker than of its own opening, the "u" and the "ch" more pucker than they had, the
# "u" closing the jaw so the lips meet, and the "a" lifts the corners of the mouth, the same on both sides, and far from a smile.
assert mix["viseme_O"]["viseme_U"] > 2 * mix["viseme_O"]["viseme_O"] > 0.5, f"the \"o\": {mix['viseme_O']}"
assert 1 < mix["viseme_U"]["viseme_U"] <= 1.3 and mix["viseme_U"]["jawOpen"] < 0 and mix["viseme_CH"]["viseme_CH"] == 1 and mix["viseme_CH"]["viseme_U"] >= 0.5, f"the pucker: {mix['viseme_U']}, {mix['viseme_CH']}"
assert mix["viseme_aa"]["viseme_aa"] == 1 and 0 < mix["viseme_aa"]["mouthSmileLeft"] == mix["viseme_aa"]["mouthSmileRight"] <= 0.5, f"the \"a\": {mix['viseme_aa']}"
# At the player's peak of an "o" (0.51 of the shape, the jaw added by the player), the mouth is narrower than it is tall is not
# asked here: only that the pucker in it is past half of the kit's whole "u", where it was a third.
assert 0.51 * mix["viseme_O"]["viseme_U"] > 0.5, "the pucker in the player's \"o\""
print("ok")
