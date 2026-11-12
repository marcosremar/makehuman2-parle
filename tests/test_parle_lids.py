"""Regression for the fold of the upper lids (2026-11-12). Run with
`Blender -b --factory-startup --python-exit-code 1 -P tests/test_parle_lids.py`.

The judge by sight, round 16, on the eyes of the cast: "lidless symmetric cut-outs with thin sharp rims". Cut through the middle of
an eye, the skin of Lúcia went from the rim of the opening to the brow in one flat wall, where the upper lids of Rafa and of Sofia
stand off the globe and the skin above them goes in again. `parle_polish.lids_with_a_fold` gives the upper lid that fold: the rim
of the opening, the lower lid, the socket and the corners stay, and a blink still shuts the lid where the kit shuts it."""
import sys
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import parle_polish  # noqa: E402

bpy.ops.wm.read_factory_settings(use_empty=True)
# The right eye of a face that looks at -y: its ball, and a sheet of skin over the front of it. The skin lies 1 mm off the ball
# as far as 7.5 mm above and below the middle of the eye and leaves it from there (the brow, the cheek); inside the opening of the
# lids it is the socket, 6 mm inside the ball.
BALL, RADIUS = np.array((-0.03, 0.0, 0.0)), 0.025
HALF_WIDE, HALF_TALL = 0.012, 0.005
STEPS = np.round(np.arange(-0.018, 0.0181, 0.001), 6)


def in_opening(dx, dz):
    return (dx / HALF_WIDE) ** 2 + (dz / HALF_TALL) ** 2 < 1


def skin_point(dx, dz):
    far = RADIUS - 0.006 if in_opening(dx, dz) else RADIUS + 0.001 + 0.9 * max(0.0, abs(dz) - 0.0075)
    return (BALL[0] + dx, BALL[1] - np.sqrt(far * far - dx * dx - dz * dz), BALL[2] + dz)


GRID = [(dx, dz) for dz in STEPS for dx in STEPS]
POINTS = np.array([skin_point(dx, dz) for dx, dz in GRID])
WIDE = len(STEPS)
FACES = [(row * WIDE + col, row * WIDE + col + 1, (row + 1) * WIDE + col + 1, (row + 1) * WIDE + col) for row in range(WIDE - 1) for col in range(WIDE - 1)]
DX, DZ = np.array(GRID).T
SOCKET = np.array([in_opening(dx, dz) for dx, dz in GRID])
off_ball = lambda found: np.linalg.norm(found - BALL, axis=1) - RADIUS


def face_with_lids(points=POINTS):
    mesh = bpy.data.meshes.new("Body")
    mesh.from_pydata([tuple(point) for point in points], [], FACES)
    obj = bpy.data.objects.new("Body", mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.shape_key_add(name="Basis")
    # A blink brings the upper lid down 8 mm, as far up as 9 mm above the middle of the eye; another trait moves the whole face.
    blink = obj.shape_key_add(name="eyeBlinkLeft")
    for index in np.flatnonzero(~SOCKET & (DZ > 0) & (DZ < 0.0095)):
        blink.data[index].co.z -= 0.008
    apart = obj.shape_key_add(name="faceEyeSpacing")
    for point in apart.data:
        point.co.x += 0.002
    return obj


def world_of(obj):
    edges = np.zeros(len(obj.data.edges) * 2, dtype=np.int64)
    obj.data.edges.foreach_get("vertices", edges)
    normals = np.array([vertex.normal[:] for vertex in obj.data.vertices])
    return normals, edges.reshape(-1, 2)


body = face_with_lids()
assert (np.array([vertex.normal[:] for vertex in body.data.vertices])[~SOCKET, 1] < -0.3).all(), "the skin of the test does not face the front"
keys = body.data.shape_keys.key_blocks
shut_before, apart_before = parle_polish.read(keys["eyeBlinkLeft"]).copy(), parle_polish.read(keys["faceEyeSpacing"]).copy()
normals, edges = world_of(body)
way, rim = parle_polish.lid_ways(POINTS, normals, np.arange(len(POINTS)), edges, BALL, RADIUS)
assert 40 < rim.sum() < 80 and not rim[SOCKET].any(), f"the rim of the test has {rim.sum()} points"
assert np.isinf(way[SOCKET]).all() and (way[rim] == 0).all(), "the socket has a way to the rim, or the rim is not at none from itself"
upper, middle_of_lid = ~SOCKET & ~rim & (DZ > 0), np.abs(DX) < 0.0015

moved, furthest = parle_polish.lids_with_a_fold(body, [(BALL, RADIUS)], fold=0.002)
now = parle_polish.read(keys[0])
step = off_ball(now) - off_ball(POINTS)
assert moved > 60 and 0.001 < furthest < 0.0021, f"the fold moved {moved} points, the furthest {furthest * 1000:.2f} mm"
assert np.allclose(np.cross(now - POINTS, POINTS - BALL), 0, atol=1e-9), "a point left the line from the middle of its ball"
# What stays: the rim of the opening, the socket, the lower lid, the corners and beyond, and the skin far from the rim.
for name, still in (("the rim", rim), ("the socket", SOCKET), ("the lower lid", DZ <= 0), ("the corners", np.abs(DX) >= HALF_WIDE), ("skin far from the rim", ~np.isfinite(way))):
    assert np.allclose(now[still], POINTS[still], atol=1e-9), f"{name} moved {1000 * np.abs(now - POINTS)[still].max():.2f} mm"
# Over the middle of the eye the lid stands off the ball as far as it is asked to, a little way from the rim...
lip = upper & middle_of_lid & (way > parle_polish.LID_LIP_AT_M) & (way < parle_polish.LID_PLATE_M)
assert lip.sum() >= 3, f"the test has {lip.sum()} points where the lid stands"
stands = off_ball(now)[lip]
assert (stands > parle_polish.LID_LIP_M * parle_polish.LID_PLATE_SHARE - 1e-4).all() and (stands < parle_polish.LID_LIP_M + 1e-4).all() and (step[lip] > 0.0005).any(), f"the lid stands {stands * 1000} mm off the ball"
# ... and the skin above it goes in, most where the fold is, by what was asked there.
fold = upper & middle_of_lid & (np.abs(way - parle_polish.LID_FOLD_AT_M) < 0.0012)
assert fold.sum() >= 3 and 0.0016 < -step[fold].min() < 0.002 + 1e-6, f"the fold goes in {-1000 * step[fold].min():.2f} mm over the middle of the eye"
past = upper & (way > parle_polish.LID_FOLD_AT_M + parle_polish.LID_FOLD_REACH_M[1] + 0.0005)
assert past.sum() > 20 and np.allclose(step[past], 0, atol=1e-7), "skin past the reach of the fold moved"
# The fold runs out sooner on the outer side of the eye (the one further from the middle of the face, -x here) than on the inner.
outer, inner = upper & (np.abs(DX + 0.007) < 0.0005), upper & (np.abs(DX - 0.007) < 0.0005)
assert np.abs(step[inner]).max() > 1.2 * np.abs(step[outer]).max() > 0, f"at the same way from its corner, the outer side moved {1000 * np.abs(step[outer]).max():.2f} mm and the inner {1000 * np.abs(step[inner]).max():.2f}"
# A blink shuts the lid where the kit shuts it; a point the blink does not move carries the fold into it; other shapes keep their move.
shut_now = parle_polish.read(keys["eyeBlinkLeft"])
blinked = np.linalg.norm(shut_before - POINTS, axis=1) > parle_polish.BLINK_WHOLE_M
assert (blinked & (np.abs(step) > 0.0005)).sum() > 20 and np.allclose(shut_now[blinked], shut_before[blinked], atol=1e-7), "the lid does not shut where the kit shuts it"
above = ~blinked & (np.abs(step) > 0.0002)
assert above.sum() >= 5 and np.allclose(shut_now[above] - shut_before[above], now[above] - POINTS[above], atol=1e-7), "skin the blink does not move lost the fold while the eye is shut"
assert np.allclose(parle_polish.read(keys["faceEyeSpacing"]) - now, apart_before - POINTS, atol=1e-7), "another shape lost its move"
assert np.allclose(parle_polish.read(body.data.vertices), now, atol=1e-7), "the mesh and its rest shape differ"

# With the whole fold the skin goes in no nearer the ball than the lids lie on it.
bpy.data.objects.remove(body)
whole = face_with_lids()
parle_polish.lids_with_a_fold(whole, [(BALL, RADIUS)])
least = np.minimum(off_ball(POINTS), parle_polish.LID_OFF_BALL_M)
deep = off_ball(parle_polish.read(whole.data.shape_keys.key_blocks[0]))
assert (deep >= least - 1e-7).all() and (deep[upper] < off_ball(POINTS)[upper] - 0.002).any(), f"the fold took skin {1000 * (least - deep).max():.2f} mm nearer the ball than it may, or made no fold"
bpy.data.objects.remove(whole)

# A face with no opening over the ball is left as it is.
closed = face_with_lids(np.array([skin_point(dx, 0.02 + abs(dz)) for dz in STEPS for dx in STEPS]))
assert parle_polish.lids_with_a_fold(closed, [(BALL, RADIUS)]) == (0, 0.0), "a face with no opening of the lids was moved"

# The way to the rim is the straight one to its line. The skin of a kit has few points there: two points of the rim 12 mm apart
# over the middle of the eye, and the lip of the lid a point 1 mm above the line between them, tied to both and to nothing else
# of the rim; counted along the edges it was 6 mm from the rim, and got the fold where the lid should stand.
on_ball = lambda dx, dz, far: np.array((BALL[0] + dx, BALL[1] - np.sqrt(far * far - dx * dx - dz * dz), BALL[2] + dz))
few = np.array([on_ball(-0.006, 0, RADIUS + 0.001), on_ball(0.006, 0, RADIUS + 0.001), on_ball(0, 0.001, RADIUS + 0.001), on_ball(0, -0.002, RADIUS - 0.006)])
turned = (few - BALL) / np.linalg.norm(few - BALL, axis=1, keepdims=True)
way, rim = parle_polish.lid_ways(few, turned, np.arange(4), np.array([(0, 1), (0, 2), (1, 2), (0, 3), (1, 3)]), BALL, RADIUS)
assert rim.tolist() == [True, True, False, False] and np.isinf(way[3]), f"the rim of the few points: {rim.tolist()}"
assert 0.0008 < way[2] < 0.002, f"the lip 1 mm from the line of the rim is {way[2] * 1000:.1f} mm from it"
print("ok")
