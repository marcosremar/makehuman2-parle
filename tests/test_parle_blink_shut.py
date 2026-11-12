"""Regression for the lid that shuts (2026-11-12). Run with
`Blender -b --factory-startup --python-exit-code 1 -P tests/test_parle_blink_shut.py`.

The blink of a kit stops the upper lid a millimetre above the lower one. On the head of the kit the lashes cover that slit; the
cast has none (owner: "sem cílios"), and with the eyes shut each one showed a dark slit with a thread of white in it. With a lid
shut, `parle_polish.blinks_shut` lays the upper lid on the globe and brings it down until nothing of the globe shows from the
front. Only the shape that shuts the lid changes."""
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import parle_polish  # noqa: E402

RADIUS = 0.025
BALLS = (np.array((-0.03, 0.0, 0.0)), np.array((0.03, 0.0, 0.0)))
BALL = BALLS[0]
SKIN = RADIUS + 0.0005
# The skin covers the front of the ball and past it on every side: a lower lid up to 1 mm under the middle of the eye, and an upper
# lid whose edge is 5 mm over the middle of the eye in the middle and comes down to the lower lid 14 mm either side, the corners
# of the opening. The upper lid has this many rows from its edge up to 18 mm over the middle of the eye.
ACROSS = np.round(np.arange(-0.022, 0.0221, 0.001), 6)
LOWER = np.round(np.arange(-0.018, -0.0009, 0.001), 6)
ROWS, TOP, CORNER_AT, OPEN = 14, 0.018, 0.014, 0.006


def globe_points():
    """Two balls of faces, one for each eye."""
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


def on_skin(dx, dz, far=SKIN):
    return np.array((BALL[0] + dx, BALL[1] - np.sqrt(max(far * far - dx * dx - dz * dz, 0.0)), BALL[2] + dz))


def sheet(rows):
    return [(row * len(ACROSS) + col, row * len(ACROSS) + col + 1, (row + 1) * len(ACROSS) + col + 1, (row + 1) * len(ACROSS) + col) for row in range(rows - 1) for col in range(len(ACROSS) - 1)]


def lid_height(dx, row, apart):
    """How high a point of the upper lid stands: its edge (row 0) `apart` above the lower lid over the middle of the eye and on it
    from the corners out, its last row at the top."""
    edge = LOWER[-1] + apart * max(0.0, 1 - (dx / CORNER_AT) ** 2)
    return edge + (TOP - edge) * row / (ROWS - 1)


def face(leaves, sunk=None):
    """A lower lid, and an upper lid that a blink brings down to `leaves` above it. `sunk`, a point of the upper lid, shuts 1 mm inside the ball."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    points, faces = globe_points()
    mesh = bpy.data.meshes.new("Human.low-poly")
    mesh.from_pydata(points, [], faces)
    eye = bpy.data.objects.new("Human.low-poly", mesh)
    bpy.context.scene.collection.objects.link(eye)
    lower = [on_skin(dx, dz) for dz in LOWER for dx in ACROSS]
    upper = [on_skin(dx, lid_height(dx, row, OPEN)) for row in range(ROWS) for dx in ACROSS]
    mesh = bpy.data.meshes.new("Body")
    mesh.from_pydata([tuple(point) for point in lower + upper], [], sheet(len(LOWER)) + [tuple(index + len(lower) for index in corner) for corner in sheet(ROWS)])
    body = bpy.data.objects.new("Body", mesh)
    bpy.context.scene.collection.objects.link(body)
    body.shape_key_add(name="Basis")
    blink = body.shape_key_add(name="eyeBlinkRight")
    closed = [on_skin(dx, lid_height(dx, row, leaves)) for row in range(ROWS) for dx in ACROSS]
    if sunk is not None:
        row, col = sunk
        closed[row * len(ACROSS) + col] = on_skin(ACROSS[col], lid_height(ACROSS[col], row, leaves), RADIUS - 0.001)
    for index, point in enumerate(closed):
        blink.data[len(lower) + index].co = tuple(point)
    other = body.shape_key_add(name="faceEyeSpacing")
    for point in other.data:
        point.co.x += 0.002
    return body, eye, len(lower)


def shows(body, eye, name):
    """How tall, at most, the strip of globe that shows from the front with that shape whole."""
    points = parle_polish.read(body.data.shape_keys.key_blocks[name])
    skin = BVHTree.FromPolygons([tuple(point) for point in points], [tuple(polygon.vertices) for polygon in body.data.polygons])
    globe = BVHTree.FromPolygons([tuple(vertex.co) for vertex in eye.data.vertices], [tuple(polygon.vertices) for polygon in eye.data.polygons])
    step, most = 0.0002, 0.0
    for dx in np.arange(-0.02, 0.0201, 0.001):
        tall = 0.0
        for dz in np.arange(-0.015, 0.0151, step):
            start = Vector((BALL[0] + dx, BALL[1] - RADIUS - 0.05, BALL[2] + dz))
            ball, cover = globe.ray_cast(start, Vector((0, 1, 0)), 0.2)[3], skin.ray_cast(start, Vector((0, 1, 0)), 0.2)[3]
            tall += step if ball is not None and (cover is None or cover > ball + 1e-5) else 0.0
        most = max(most, tall)
    return most


body, eye, first_upper = face(0.0012)
keys = body.data.shape_keys.key_blocks
rest, shut, apart = (parle_polish.read(keys[name]).copy() for name in ("Basis", "eyeBlinkRight", "faceEyeSpacing"))
assert 0.0009 < shows(body, eye, "eyeBlinkRight") < 0.0016, f"the test does not start from a slit: {shows(body, eye, 'eyeBlinkRight') * 1000:.2f} mm of globe showing"
found = parle_polish.blinks_shut([body, eye])
was, left = found["eyeBlinkRight"]
assert list(found) == ["eyeBlinkRight"] and 0.0008 <= was <= 0.0017 and left == 0, f"what was found: {found}"
assert shows(body, eye, "eyeBlinkRight") == 0, f"with the lid shut {shows(body, eye, 'eyeBlinkRight') * 1000:.2f} mm of globe still shows"
now = parle_polish.read(keys["eyeBlinkRight"])
down = (shut - now)[:, 2]
lowest, highest = slice(first_upper, first_upper + len(ACROSS)), slice(len(rest) - len(ACROSS), len(rest))
over_the_eye, past_the_corners = np.abs(ACROSS) < 0.008, np.abs(ACROSS) >= CORNER_AT
assert (down[lowest][over_the_eye] > 0.001).all() and (down[lowest] < 0.003).all(), f"the edge of the lid came down {down[lowest] * 1000} mm"
assert np.allclose(now[lowest][past_the_corners], shut[lowest][past_the_corners], atol=1e-7), "the lid moved past the corners of the opening, where it does not shut"
assert np.allclose(now[highest], shut[highest], atol=1e-7) and np.allclose(now[:first_upper], shut[:first_upper], atol=1e-7), "the top of the lid, which the blink does not move, or the lower lid moved"
assert (np.diff(down[first_upper:].reshape(ROWS, len(ACROSS)), axis=0) <= 1e-7).all(), "a point higher on the lid came down more than the one under it"
assert (np.linalg.norm(now[first_upper:] - BALL, axis=1) > RADIUS).all(), "the shut lid went into the ball"
assert np.allclose(parle_polish.read(keys[0]), rest, atol=1e-7) and np.allclose(parle_polish.read(keys["faceEyeSpacing"]), apart, atol=1e-7), "the rest shape or another shape changed"

# A lid that already shuts on its globe is left as it is.
body, eye, first_upper = face(-0.0005)
shut = parle_polish.read(body.data.shape_keys.key_blocks["eyeBlinkRight"]).copy()
assert parle_polish.blinks_shut([body, eye]) == {"eyeBlinkRight": (0.0, 0.0)} and np.allclose(parle_polish.read(body.data.shape_keys.key_blocks["eyeBlinkRight"]), shut, atol=1e-7), "a lid that shuts was moved"

# A point of the lid that shuts inside the ball comes out on it: the globe is rounder than the one the kit shut its lids on.
body, eye, first_upper = face(-0.0005, sunk=(2, len(ACROSS) // 2))
index = first_upper + 2 * len(ACROSS) + len(ACROSS) // 2
inside = RADIUS - np.linalg.norm(parle_polish.read(body.data.shape_keys.key_blocks["eyeBlinkRight"])[index] - BALL)
assert inside > 0.0009, f"the test does not start from a point of the lid inside the ball: {inside * 1000:.2f} mm"
parle_polish.blinks_shut([body, eye])
off = np.linalg.norm(parle_polish.read(body.data.shape_keys.key_blocks["eyeBlinkRight"])[index] - BALL) - RADIUS
assert abs(off - parle_polish.LID_OFF_BALL_M) < 1e-5 and shows(body, eye, "eyeBlinkRight") == 0, f"the point of the lid that shut inside the ball stands {off * 1000:.2f} mm off it"

# No globe, or no blink: nothing to do.
bpy.data.objects.remove(eye)
assert parle_polish.blinks_shut([body]) == {}, "a body with no globe was looked at"
print("ok")
