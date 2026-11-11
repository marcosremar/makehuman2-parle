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
parle_face.calm_speech([body, teeth, hat], shares)
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
# player (0.55 of viseme_aa and 0.2 of jawOpen) lands between Rafa's 14 mm and two thirds of the 32 it was, on the woman measured (41
# and 45 mm) and on the man (31 and 35): at 15 mm the judge by sight found every open shape the same flat slit.
for sex, own in (("female", (0.041, 0.045)), ("male", (0.031, 0.035))):
    kept = parle_face.SPEECH_OPEN[sex]
    assert all(0.33 < share < 1 for share in kept.values()) and kept["viseme_aa"] == min(kept.values()), f"the shares of the {sex} mouth: {kept}"
    peak = 0.55 * own[0] * kept["viseme_aa"] + 0.2 * own[1] * kept["jawOpen"]
    assert 0.014 < peak < 0.021, f"the peak of an \"a\" on the {sex} mouth is {peak * 1000:.1f} mm, far from the leads' or back at a scream"
print("ok")
