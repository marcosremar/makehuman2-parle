"""Regression for the nostrils (2026-11-12). Run with
`Blender -b --factory-startup --python-exit-code 1 -P tests/test_parle_nose.py`.

The judge by sight, rounds 13 to 18, on the nose of the cast: "torn, asymmetric dark notches with hanging spikes; the right one
is a ripped hole". Under each wing of the nose the skin of the kit goes up into the nose in a notch 1.7 to 2.9 mm deep, with a
few points on walls the light does not reach. `parle_nose.shallow_nostrils` closes each notch with the fillet a ball leaves when
rolled over it, and presses a shallow dent into the fillet, the same on both sides; the tip of the nose, its middle line and the
skin away from the notches stay, and every shape key takes the same step."""
import sys
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import parle_nose  # noqa: E402

# The base of a nose that looks at -y: a plate that looks down, 4 cm wide and 3 cm deep, with a notch going up into the nose on
# each side of the middle, 5 mm across, 10 long and 3 deep. The tip of the nose is 4 mm ahead of the plate, at its level, and a
# point 10 cm up stands for the top of the head.
STEP = 0.001
XS = np.round(np.arange(-0.02, 0.0201, STEP), 6)
YS = np.round(np.arange(0.0, 0.0301, STEP), 6)
LEVEL, NOTCH_AT, NOTCH_HALF, NOTCH_DEEP = 1.5, (0.009, 0.014), (0.0025, 0.005), 0.003
TIP, TOP = (0.0, -0.004, LEVEL), (0.0, 0.03, LEVEL + 0.1)


def nose_base(deep=NOTCH_DEEP, at=NOTCH_AT[0]):
    """The points and the triangles (outer side down), the triangles of one half the mirror image of those of the other."""
    roof = lambda x, y: LEVEL + deep * max(0.0, 1 - ((abs(x) - at) / NOTCH_HALF[0]) ** 2 - ((y - NOTCH_AT[1]) / NOTCH_HALF[1]) ** 2) ** 2
    points = [(x, y, roof(x, y)) for y in YS for x in XS] + [TIP, TOP]
    wide, triangles = len(XS), []
    for row in range(len(YS) - 1):
        for col in range(wide - 1):
            a, b, c, d = row * wide + col, row * wide + col + 1, (row + 1) * wide + col + 1, (row + 1) * wide + col
            triangles.extend([(a, c, b), (a, d, c)] if XS[col] >= 0 else [(a, d, b), (b, d, c)])
    return np.array(points), np.array(triangles)


POINTS, TRIANGLES = nose_base()
X, Y = POINTS[:-2, 0], POINTS[:-2, 1]
from_notch = np.minimum(np.hypot(X - NOTCH_AT[0], Y - NOTCH_AT[1]), np.hypot(X + NOTCH_AT[0], Y - NOTCH_AT[1]))
looks_down = lambda points: np.cross(points[TRIANGLES[:, 1]] - points[TRIANGLES[:, 0]], points[TRIANGLES[:, 2]] - points[TRIANGLES[:, 0]])[:, 2] < 0
assert looks_down(POINTS).all(), "the skin of the test does not look down"

# First the measure itself, on the points.
moved, dents = parle_nose.closed_and_dented(POINTS, TRIANGLES)
up = (moved[:-2, 2] - LEVEL) * 1000
assert sorted(dents) == [-1, 1], f"the nose of the test has {len(dents)} nostrils"
for side, dent in dents.items():
    assert dent["points"] > 20 and 0.0015 < dent["was_deep"] < NOTCH_DEEP, f"the notch of the test: {dent['points']} points, {dent['was_deep'] * 1000:.2f} mm under the ball"
    assert abs(dent["centre"][0] - side * NOTCH_AT[0]) < 0.001 and abs(dent["centre"][1] - NOTCH_AT[1]) < 0.0015, f"the dent is at {dent['centre'] * 1000} mm"
    # The dent goes up into the nose, and its long side lies along the notch, away from the tip.
    assert dent["into"][2] > 0.9 and dent["along"][1] > 0.9, f"the dent goes in along {dent['into']} and lies along {dent['along']}"
# The hole is closed: the roof of each notch stood 3 mm up, with walls of 60 degrees; what is left is a dent, between the
# depth it is given and that with the hollow of the ball added, and no skin stands out of the plate.
middle_of_notch = from_notch < 0.0006
assert 2.9 < (POINTS[:-2, 2] - LEVEL)[middle_of_notch].max() * 1000 <= 3.0, "the notch of the test is not 3 mm deep"
assert parle_nose.DENT_DEEP_M * 1000 - 0.2 < up[middle_of_notch].max() < 2.0 and up.max() < 2.0 and up.min() > -0.05, f"the dent is {up[middle_of_notch].max():.2f} mm deep at its middle, the skin goes from {up.min():.2f} to {up.max():.2f} mm"
slope = lambda points: np.degrees(np.arccos(np.clip(-(lambda n: n[:, 2] / np.linalg.norm(n, axis=1))(np.cross(points[TRIANGLES[:, 1]] - points[TRIANGLES[:, 0]], points[TRIANGLES[:, 2]] - points[TRIANGLES[:, 0]])), -1, 1)))
assert slope(POINTS).max() > 55 and slope(moved).max() < 40, f"the steepest wall was {slope(POINTS).max():.0f} degrees and is {slope(moved).max():.0f}"
# No triangle is turned over: laid on the ball one by one, the points of the two walls crossed.
assert looks_down(moved).all(), f"{(~looks_down(moved)).sum()} triangles turned over"
# What stays: the tip, the middle line of the nose (its outline in profile), and the skin away from the notches.
step = np.linalg.norm(moved - POINTS, axis=1)
for name, still in (("the tip and the top of the head", np.arange(len(POINTS)) >= len(POINTS) - 2), ("the middle line", np.r_[np.abs(X) < parle_nose.PROFILE_HALF_M[0], False, False]),
                    ("skin 1 cm from the notches", np.r_[from_notch > 0.01, False, False])):
    assert still.any() and step[still].max() < 1e-9, f"{name} moved {step[still].max() * 1000:.3f} mm"
assert 0.001 < step.max() < NOTCH_DEEP, f"the point that moved most went {step.max() * 1000:.2f} mm"
# The two sides are the same: each point ends where its twin across the middle ends, seen in the mirror.
grid = moved[:-2].reshape(len(YS), len(XS), 3)
twin = grid[:, ::-1] * (-1, 1, 1)
assert np.abs(grid - twin).max() < 2e-5, f"the two sides end {np.abs(grid - twin).max() * 1000:.3f} mm apart"
# Run again on its own result, nothing moves: the dent is not a notch.
again, none = parle_nose.closed_and_dented(moved, TRIANGLES)
assert not none and np.array_equal(again, moved), "the step took its own dent for a notch"
# A nose with no notch is left as it is, and so is a hole on the middle line of the face, which is not a nostril.
for name, (points, triangles) in (("a nose with no notch", nose_base(0.0)), ("a hole on the middle line", nose_base(at=0.0))):
    same, none = parle_nose.closed_and_dented(points, triangles)
    assert not none and np.array_equal(same, points), f"{name} was moved"

# Then on a mounted skin: an object half the size of its mesh, so the step in the world is not the step in the mesh, with a
# shape key that opens the jaw (the whole plate 2 mm down, the left notch 1 mm more) and one that does nothing here.
bpy.ops.wm.read_factory_settings(use_empty=True)


def mounted(points, keys=True):
    mesh = bpy.data.meshes.new("Body")
    mesh.from_pydata([tuple(point * 2) for point in points], [], [tuple(int(corner) for corner in triangle) for triangle in TRIANGLES])
    obj = bpy.data.objects.new("Body", mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.scale = (0.5, 0.5, 0.5)
    bpy.context.view_layer.update()
    if keys:
        obj.shape_key_add(name="Basis")
        jaw = obj.shape_key_add(name="jawOpen")
        for index in range(len(points)):
            jaw.data[index].co.z -= 0.004 + (0.002 if points[index][0] > 0.004 else 0.0)
        obj.shape_key_add(name="eyeBlinkLeft")
    return obj


in_world = lambda obj, block: parle_nose.read(block) @ np.array(obj.matrix_world)[:3, :3].T + np.array(obj.matrix_world)[:3, 3]
body = mounted(POINTS)
keys = body.data.shape_keys.key_blocks
# The mesh keeps its points in single precision: the measure is taken again on the points as the object has them.
rest = in_world(body, keys[0])
assert np.allclose(rest, POINTS, atol=1e-6), "the mounted skin of the test is not where its points are"
wanted, _ = parle_nose.closed_and_dented(rest, TRIANGLES)
wanted_step = np.linalg.norm(wanted - rest, axis=1)
before = {key.name: parle_nose.read(key).copy() for key in keys}
count, far = parle_nose.shallow_nostrils(body)
assert count == (wanted_step > 1e-5).sum() > 60 and abs(far - wanted_step.max()) < 1e-6, f"{count} points moved, the furthest {far * 1000:.2f} mm ({wanted_step.max() * 1000:.2f} mm by the measure)"
assert np.allclose(in_world(body, keys[0]), wanted, atol=1e-6), f"the skin at rest ends {np.abs(in_world(body, keys[0]) - wanted).max() * 1000:.4f} mm from where the measure puts it"
# Every shape key takes the step of the rest shape: the jaw still opens as it did, and a key that did nothing does nothing.
for key in keys[1:]:
    assert np.allclose(parle_nose.read(key) - parle_nose.read(keys[0]), before[key.name] - before["Basis"], atol=2e-6), f"{key.name} lost its move"
assert np.abs(before["jawOpen"] - before["Basis"]).max() > 0.005 and np.abs(before["eyeBlinkLeft"] - before["Basis"]).max() == 0, "the shape keys of the test do not move as said"
assert np.allclose(parle_nose.read(body.data.vertices), parle_nose.read(keys[0]), atol=1e-7), "the mesh and its rest shape differ"
assert parle_nose.shallow_nostrils(body) == (0, 0.0), "run again, the step moved the skin"
bpy.data.objects.remove(body)
# A skin with no shape keys is closed all the same, and one with no notch is left alone.
bare = mounted(POINTS, keys=False)
assert parle_nose.shallow_nostrils(bare)[0] == count and np.allclose(in_world(bare, bare.data.vertices), wanted, atol=1e-6), "a skin with no shape keys was not closed"
assert parle_nose.shallow_nostrils(mounted(nose_base(0.0)[0])) == (0, 0.0), "a nose with no notch was moved"
print("ok")
