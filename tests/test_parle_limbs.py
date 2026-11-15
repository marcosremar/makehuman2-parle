"""Regression for the arms (2026-11-15): run with `Blender -b --python-exit-code 1 -P tests/test_parle_limbs.py`.

The women's wrists went 10.1 to 10.6 cm round and their forearms tapered to them like a cone; every wrist and elbow pinched when it bent,
the player blending two turns linearly (Seu Jorge's wrist lost 14.7% of its girth, his elbow 19.8%). The forearm grows to the men's
taper on a wrist of `WRIST_SHARE` of the height, and a helper bone at each joint turns half the lower bone's turn and takes the middle of
the blend.
"""
import math
import sys
from pathlib import Path

import mathutils
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import parle_limbs  # noqa: E402

# A circle of 24 points, 3 cm across: its hull goes pi * 3 cm round.
ring = np.array([[0.015 * math.cos(a), 0.015 * math.sin(a)] for a in np.linspace(0, 2 * math.pi, 24, endpoint=False)])
assert abs(parle_limbs.hull_perimeter(ring) - math.pi * 0.03) < 0.001, parle_limbs.hull_perimeter(ring)

# A tube along x, 2 cm in radius, cut across at its middle: 12.6 cm round.
angles = np.linspace(0, 2 * math.pi, 16, endpoint=False)
tube = np.array([[x, 0.02 * math.cos(a), 0.02 * math.sin(a)] for x in (0.0, 0.1) for a in angles])
edges = np.array([[i, i + 16] for i in range(16)])
girth = parle_limbs.section_girth(tube, edges, np.array([0.05, 0, 0]), np.array([1.0, 0, 0]))
assert abs(girth - 2 * 16 * 0.02 * math.sin(math.pi / 16)) < 1e-6, girth

# The cast's florist: 1.62 m, her forearm 21.4 cm round at the elbow and 11.1 at the wrist. It grows to a wrist of 14.1 cm and to the
# men's taper above it, never shrinks, and grows no more than FOREARM_GROW_MOST.
florist = [0.214, 0.203, 0.196, 0.171, 0.144, 0.116, 0.111]
grow = parle_limbs.forearm_growth(florist, 1.62, "female")
assert grow[0] == 1.0 and grow[1] == 1.0, grow
assert abs(grow[-1] * florist[-1] - parle_limbs.WRIST_SHARE["female"] * 1.62) < 1e-9, grow
assert all(g >= 1.0 for g in grow) and max(grow) <= parle_limbs.FOREARM_GROW_MOST
# A wrist far too thin grows only as far as allowed.
assert parle_limbs.forearm_growth([0.2] * 6 + [0.05], 1.62, "female")[-1] == parle_limbs.FOREARM_GROW_MOST
# Seu Jorge's forearm, a man's, stays as it was.
assert np.allclose(parle_limbs.forearm_growth([0.237, 0.243, 0.252, 0.244, 0.215, 0.186, 0.177], 1.74, "male"), 1.0)

# Along the forearm: nothing above the elbow, the wrist's growth into the palm fading to none, nothing past that.
shares = np.array([share for share, _ in parle_limbs.TAPER])
along = parle_limbs.growth_along(np.array([-0.2, 0.0, 1.0, 1.05, 1.2]), shares, grow, 0.15)
assert along[0] == 1.0 and along[1] == grow[0] and abs(along[2] - grow[-1]) < 1e-9, along
assert 1.0 < along[3] < grow[-1] and along[4] == 1.0, along

# The helper takes the middle of the blend: a point half on each bone goes whole to it, a point on one bone stays there, and the weight
# a point holds is kept.
assert parle_limbs.split_weight(0.5, 0.5) == (0.0, 0.0, 1.0)
assert parle_limbs.split_weight(1.0, 0.0) == (1.0, 0.0, 0.0) and parle_limbs.split_weight(0.0, 0.8) == (0.0, 0.8, 0.0)
upper, lower, helper = parle_limbs.split_weight(0.6, 0.2)
assert abs(upper + lower + helper - 0.8) < 1e-12 and lower == 0.0 and abs(helper - 0.4) < 1e-12, (upper, lower, helper)

# And turns half of the lower bone's turn: 90 degrees about x gives 45.
turn = parle_limbs.helper_turn(mathutils.Quaternion((1, 0, 0), math.radians(90)))
assert abs(math.degrees(turn.angle) - 45) < 1e-4 and abs(turn.axis.x - 1) < 1e-6, (turn.angle, turn.axis)

# Evening out a joint: a strip of 12 points along x, the last two at the same place (a seam), weights stepping from the upper bone to
# the lower one in a single step at the joint (x = 0.05). After, the step is spread, each point keeps the sum it held, the two points
# at the same place carry the same weights, and points out of reach do not move.
import bpy  # noqa: E402

xs = [0.01 * i for i in range(11)] + [0.10]
mesh = bpy.data.meshes.new("strip")
mesh.from_pydata([(x, 0, 0) for x in xs], [(i, i + 1) for i in range(10)] + [(9, 11)], [])
strip = bpy.data.objects.new("strip", mesh)
bpy.context.scene.collection.objects.link(strip)
names = ("LeftArm", "LeftElbowHelper", "LeftForeArm")
groups = [strip.vertex_groups.new(name=name) for name in names]
for i, x in enumerate(xs):
    groups[0 if x < 0.05 else 2].add([i], 0.9, "REPLACE")
assert parle_limbs.even_joint(strip, (0.05, 0, 0), names, reach=0.035) > 0
held = lambda i: {name: strip.vertex_groups[name].weight(i) for name in names if i in [v.index for v in mesh.vertices if any(g.group == strip.vertex_groups[name].index for g in v.groups)]}
assert abs(sum(held(5).values()) - 0.9) < 1e-6 and 0.05 < held(5).get("LeftArm", 0) < 0.85, held(5)
assert held(10) == held(11), (held(10), held(11))
assert held(0) == {"LeftArm": 0.9} or abs(held(0)["LeftArm"] - 0.9) < 1e-6, held(0)
print("test_parle_limbs: ok")
