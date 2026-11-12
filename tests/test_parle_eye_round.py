"""Regression for the eye that opened round (2026-11-12). Run with
`Blender -b --factory-startup --python-exit-code 1 -P tests/test_parle_eye_round.py`.

The owner found the eyes of one of the women of the cast "esbugalhado" and approved another's: from the front the opening of the
first was 0.49 as tall as it was wide, the approved one's 0.42. `parle_polish.eyes_no_rounder` brings the upper lid of an eye
rounder than EYE_TALL_MOST down by a share of its own blink: the rest shape and every other shape take it, and the blink still
ends where it ended. An eye that is not rounder stays."""
import sys
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import parle_polish  # noqa: E402

RADIUS = 0.025
BALLS = (np.array((-0.03, 0.0, 0.0)), np.array((0.03, 0.0, 0.0)))
BALL = BALLS[0]
SKIN = RADIUS + 0.0005
# The skin covers the whole front of the ball: a lower lid up to 1 mm under the middle of the eye and an upper lid whose edge
# stands `opened` over it in the middle and comes down to it 14 mm either side, the corners of an opening 28 mm wide.
ACROSS = np.round(np.arange(-0.027, 0.0271, 0.001), 6)
LOWER = np.round(np.arange(-0.027, -0.0009, 0.001), 6)
ROWS, TOP, CORNER_AT = 20, 0.027, 0.014


def globe_points():
    points, faces = [], []
    rings, sides = 12, 24
    for centre in BALLS:
        first = len(points)
        for ring in range(rings + 1):
            tilt = np.pi * ring / rings
            for side in range(sides):
                turn = 2 * np.pi * side / sides
                points.append(tuple(centre + RADIUS * np.array((np.sin(tilt) * np.cos(turn), np.sin(tilt) * np.sin(turn), np.cos(tilt)))))
        for ring in range(rings):
            for side in range(sides):
                faces.append((first + ring * sides + side, first + ring * sides + (side + 1) % sides, first + (ring + 1) * sides + (side + 1) % sides, first + (ring + 1) * sides + side))
    return points, faces


def on_skin(dx, dz):
    return np.array((BALL[0] + dx, BALL[1] - np.sqrt(max(SKIN * SKIN - dx * dx - dz * dz, 0.0)), BALL[2] + dz))


def sheet(rows):
    return [(row * len(ACROSS) + col, row * len(ACROSS) + col + 1, (row + 1) * len(ACROSS) + col + 1, (row + 1) * len(ACROSS) + col) for row in range(rows - 1) for col in range(len(ACROSS) - 1)]


def lid_height(dx, row, apart):
    edge = LOWER[-1] + apart * max(0.0, 1 - (dx / CORNER_AT) ** 2)
    return edge + (TOP - edge) * row / (ROWS - 1)


def face(opened):
    """A lower lid, and an upper lid `opened` above it that a blink brings down past it."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    points, faces = globe_points()
    mesh = bpy.data.meshes.new("Human.low-poly")
    mesh.from_pydata(points, [], faces)
    eye = bpy.data.objects.new("Human.low-poly", mesh)
    bpy.context.scene.collection.objects.link(eye)
    lower = [on_skin(dx, dz) for dz in LOWER for dx in ACROSS]
    upper = [on_skin(dx, lid_height(dx, row, opened)) for row in range(ROWS) for dx in ACROSS]
    mesh = bpy.data.meshes.new("Body")
    mesh.from_pydata([tuple(point) for point in lower + upper], [], sheet(len(LOWER)) + [tuple(index + len(lower) for index in corner) for corner in sheet(ROWS)])
    body = bpy.data.objects.new("Body", mesh)
    bpy.context.scene.collection.objects.link(body)
    body.shape_key_add(name="Basis")
    blink = body.shape_key_add(name="eyeBlinkRight")
    for index, point in enumerate(on_skin(dx, lid_height(dx, row, -0.0005)) for row in range(ROWS) for dx in ACROSS):
        blink.data[len(lower) + index].co = tuple(point)
    other = body.shape_key_add(name="faceEyeSpacing")
    for point in other.data:
        point.co.x += 0.002
    return body, eye, len(lower)


MOST, STEP = parle_polish.EYE_TALL_MOST, parle_polish.EYE_TALL_STEP_M

# An eye 16 mm tall and 28 mm wide, rounder than the cast's may be.
body, eye, first_upper = face(0.016)
keys = body.data.shape_keys.key_blocks
rest, shut, apart = (parle_polish.read(keys[name]).copy() for name in ("Basis", "eyeBlinkRight", "faceEyeSpacing"))
found = parle_polish.eyes_no_rounder([body, eye])
assert list(found) == ["eyeBlinkRight"], f"what was looked at: {list(found)}"
wide, was, tall, share = found["eyeBlinkRight"]
assert 0.0265 < wide < 0.0295 and 0.0150 < was < 0.0170, f"the test does not start from a round eye of 28 by 16 mm: {wide * 1000:.1f} by {was * 1000:.1f} mm"
assert was > MOST * wide, "the test's eye is not rounder than the cast's may be"
assert MOST * wide - 3 * STEP <= tall <= MOST * wide, f"the opening came to {tall * 1000:.1f} mm, the most for its width is {MOST * wide * 1000:.1f}"
assert 0.1 < share <= parle_polish.EYE_TALL_BLINK_MOST, f"the share of the blink taken: {share}"
now = parle_polish.read(keys["Basis"])
edge = slice(first_upper, first_upper + len(ACROSS))
over_the_eye = np.abs(ACROSS) < 0.008
down, asked = (rest - now)[edge, 2], share * (rest - shut)[edge, 2]
assert (np.abs(down - asked)[over_the_eye] < 0.0006).all() and (down[over_the_eye] > 0.002).all(), f"the edge of the upper lid came down {down[over_the_eye] * 1000} mm, that share of its blink is {asked[over_the_eye] * 1000}"
assert (np.linalg.norm(now[first_upper:] - BALL, axis=1) >= RADIUS).all(), "the lowered lid cuts into the ball: a lid goes straight across the chord of its globe"
assert np.allclose(parle_polish.read(keys["eyeBlinkRight"]), shut, atol=1e-7), "the blink no longer ends where it ended"
assert np.allclose(parle_polish.read(keys["faceEyeSpacing"]) - now, apart - rest, atol=1e-7), "another shape lost its move"
assert np.allclose(parle_polish.read(body.data.vertices), now, atol=1e-7), "the points of the mesh are not the rest shape"
assert np.allclose(now[:first_upper], rest[:first_upper], atol=1e-7), "the lower lid, which this blink does not move, moved"
again = parle_polish.eyes_no_rounder([body, eye])["eyeBlinkRight"]
assert again[3] == 0 and np.allclose(parle_polish.read(keys["Basis"]), now, atol=1e-7), f"a second pass moved the lid again: {again}"

# An almond of 28 by 10 mm stays as it is.
body, eye, first_upper = face(0.010)
keys = body.data.shape_keys.key_blocks
rest = parle_polish.read(keys["Basis"]).copy()
wide, was, tall, share = parle_polish.eyes_no_rounder([body, eye])["eyeBlinkRight"]
assert share == 0 and was == tall and np.allclose(parle_polish.read(keys["Basis"]), rest, atol=1e-7), f"an eye that is not round was moved: {share} of the blink"

# Where the most of a blink the rest may take is not enough, it takes that much and no more: past it the eye is sleepy.
body, eye, first_upper = face(0.016)
wide, was, tall, share = parle_polish.eyes_no_rounder([body, eye], most=0.2)["eyeBlinkRight"]
assert share == parle_polish.EYE_TALL_BLINK_MOST and 0.2 * wide < tall < was - 0.004, f"asked for a slit, the eye took {share} of the blink and stands {tall * 1000:.1f} mm tall"

# No globe, or no blink: nothing to do.
bpy.data.objects.remove(eye)
assert parle_polish.eyes_no_rounder([body]) == {}, "a body with no globe was looked at"
print("ok")
