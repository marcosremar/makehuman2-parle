"""Regression for the lengths of the body (2026-11-14): run with `python tests/test_parle_lengths.py` (no Blender).

The owner picked a man for his body, dressed in a T-shirt and shorts: "o homem tem bom corpo, copia toda a estrutura e formato". The
widths of the normal man followed him first; his structure is other too: the joints of his hips stand at 50% of his stature
barefoot and his knees at 27%, where the kit, whose skeleton had been set by the game's own man, has them at 56% and 31%. A profile
may now say how many times the kit's own the thigh and the shank are long. This checks the field (each bone of the leg keeps its
heading, the foot its shape and its height, the hips come down by what the legs lose and the trunk takes it up under shoulders that
stay), the hips of a clip set again over the soles, and the lengths a reference's joints ask of a kit.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import body_joints  # noqa: E402
import parle_build  # noqa: E402

# The kit's own joints, at a stature of 1.74 m (as in tests/test_parle_build.py).
HEIGHT, UNIT, FLOOR = 1.74, 0.87, 0.14
RAW = dict(Hips=(0, 0, 1.008), Spine=(0, -0.026, 1.094), Spine1=(0, 0, 1.199), Spine2=(0, 0.069, 1.324), Neck=(0, -0.012, 1.534), Head=(0, -0.026, 1.59),
           LeftArm=(0.197, -0.019, 1.428), LeftForeArm=(0.407, -0.017, 1.189), LeftHand=(0.531, -0.202, 1.069),
           LeftUpLeg=(0.106, 0.007, 0.967), LeftLeg=(0.166, -0.027, 0.471), LeftFoot=(0.222, -0.014, -0.067))
JOINTS = {name: (np.array(point) + (0, 0, FLOOR)) * UNIT for name, point in RAW.items()}
JOINTS.update({name.replace("Left", "Right"): point * (-1, 1, 1) for name, point in JOINTS.items() if name.startswith("Left")})
plain = parle_build.identity("test")
short = dict(plain, lengths=dict(thigh=0.9, shank=0.8))

# No lengths, or the kit's own: nothing to do.
assert parle_build.lengths_field(plain, JOINTS, HEIGHT) is None and parle_build.lengths_field(dict(plain, lengths=dict(thigh=1.0, shank=1.0)), JOINTS, HEIGHT) is None

field, ride, drop = parle_build.lengths_field(short, JOINTS, HEIGHT)
hip, knee, foot = (JOINTS["Left" + part] for part in parle_build.LEG)
new_hip, new_knee, new_foot = (point + ride("Left", point)[0] for point in (hip, knee, foot))
# Each bone of the leg is as long as its ratio says and points the way it did: the clips only turn the bones.
assert abs(np.linalg.norm(new_knee - new_hip) - 0.9 * np.linalg.norm(knee - hip)) < 1e-9, "the thigh is not its ratio long"
assert abs(np.linalg.norm(new_foot - new_knee) - 0.8 * np.linalg.norm(foot - knee)) < 1e-9, "the shank is not its ratio long"
assert np.linalg.norm(np.cross(new_knee - new_hip, knee - hip)) < 1e-12 and np.linalg.norm(np.cross(new_foot - new_knee, foot - knee)) < 1e-12, "a bone of the leg points another way"
# The foot stays at its height, the hips come down by what the two bones lose, straight down.
assert abs(drop - (0.1 * (hip - knee)[2] + 0.2 * (knee - foot)[2])) < 1e-12 and drop > 0.1, f"the hips do not come down by what the legs lose: {drop}"
assert abs(new_foot[2] - foot[2]) < 1e-12 and np.allclose(new_hip, hip - (0, 0, drop)), f"the ankle left its height, or the hip went sideways: {new_foot}, {new_hip}"
# The foot goes whole: the sole, the heel and the toes move as the ankle does.
sole = foot + np.array([[0.02, -0.15, -0.05], [-0.02, 0.05, -0.055], [0.0, -0.2, -0.05]])
assert np.abs(ride("Left", sole) - ride("Left", foot)).max() < 1e-12 and np.abs(ride("Left", sole)[:, 2]).max() < 1e-12, "the foot changed shape, or left the floor"
# From the joints of the arms up nothing moves: the stature stays.
above = np.array([JOINTS["LeftArm"], JOINTS["Neck"], JOINTS["Head"], [0.0, 0.0, HEIGHT]])
assert np.abs(ride("trunk", above)).max() == 0, "the shoulders, the neck or the head moved"
# The trunk takes the difference up between the joints of the hips and of the arms: no level overtakes another, none is squeezed,
# the stretch comes in and goes out little by little and is nowhere more than its even share over the eased span.
levels = np.linspace(hip[2], JOINTS["LeftArm"][2], 401)
column = np.stack([np.zeros(401), np.zeros(401), levels], axis=1)
moved = column + ride("trunk", column)
gaps = np.diff(moved[:, 2]) / np.diff(levels)
most = 1 + drop / (JOINTS["LeftArm"][2] - hip[2]) / (1 - parle_build.TRUNK_EASE)
assert abs(moved[0, 2] - (hip[2] - drop)) < 1e-12 and abs(moved[-1, 2] - levels[-1]) < 1e-12 and np.abs(moved[:, :2]).max() == 0, "the trunk does not go from the hips come down to the shoulders that stay"
assert gaps.min() > 1 - 1e-9 and gaps.max() < most + 1e-6, f"the trunk is squeezed somewhere, or stretched past its share: {gaps.min()}, {gaps.max()}"
assert gaps[0] < 1 + 0.05 * (most - 1) and gaps[-1] < 1 + 0.05 * (most - 1) and abs(gaps[200] - most) < 1e-6, f"the stretch does not come in and go out little by little: {gaps[0]}, {gaps[200]}, {gaps[-1]}"
below = np.array([[0.0, 0.05, hip[2] - 0.03], [0.05, -0.05, hip[2] - 0.2]])
assert np.allclose(ride("trunk", below), (0, 0, -drop)), "under the joints of the hips the trunk does not come down whole"
# `eased`: from 0 to 1, the same from either end.
t = np.linspace(0, 1, 101)
share = parle_build.eased(t, 0.1)
assert share[0] == 0 and abs(share[-1] - 1) < 1e-12 and (np.diff(share) > 0).all() and np.allclose(share + share[::-1], 1), "the eased share is not even from end to end"

# The field: what follows an arm or the head stays with its joint, hanging beside the hips as it may; left mirrors right.
rng = np.random.default_rng(3)
cloud = rng.uniform((-0.3, -0.15, 0.0), (0.3, 0.15, HEIGHT), (400, 3))
none, ones, zero = {side: np.zeros(400) for side in parle_build.SIDES}, {side: np.ones(400) for side in parle_build.SIDES}, np.zeros(400)
assert np.abs(field(cloud, ones, zero)).max() < 1e-12 and np.abs(field(cloud, none, np.ones(400))).max() < 1e-12, "an arm or the head moved"
assert np.allclose(field(cloud, none, zero) * (-1, 1, 1), field(cloud * (-1, 1, 1), none, zero), atol=1e-12), "left does not mirror right"
half = field(cloud, {side: np.full(400, 0.5) for side in parle_build.SIDES}, zero)
assert np.allclose(half, field(cloud, none, zero) * 0.25), "skin shared between the trunk and an arm does not go part of the way"
# A point of a leg goes by where it is along its bone, and the leg keeps its thickness: beside the bone as on it.
middle = hip + 0.5 * (knee - hip)
beside = middle + np.cross(knee - hip, (0, 1, 0)) / np.linalg.norm(knee - hip) * 0.05
low = np.array([middle, beside])
got = field(low, {side: np.zeros(2) for side in parle_build.SIDES}, np.zeros(2))
assert np.allclose(got[0], 0.5 * (0.9 - 1) * (knee - hip) - (0, 0, drop)) and np.allclose(got[0], got[1]), f"the thigh does not shorten evenly along its bone, or it got thinner: {got}"

# The bones: what hangs from a foot, from an arm or from the head goes whole with that joint, a bone of a leg by its leg, the rest by the trunk.
heads = dict(JOINTS, LeftToeBase=JOINTS["LeftFoot"] + (0, -0.13, -0.035), LeftToe_End=JOINTS["LeftFoot"] + (0, -0.2, -0.035), LeftHandIndex1=JOINTS["LeftHand"] + (0.05, 0, -0.05),
             HeadTop_End=JOINTS["Head"] + (0, 0, 0.2), LeftShoulder=JOINTS["Neck"] + (0.05, 0, -0.03), LeftEye=JOINTS["Head"] + (0.03, -0.08, 0.05))
parents = dict(Hips=None, Spine="Hips", Spine1="Spine", Spine2="Spine1", Neck="Spine2", Head="Neck", HeadTop_End="Head", LeftEye="Head", LeftShoulder="Spine2", LeftArm="LeftShoulder",
               LeftForeArm="LeftArm", LeftHand="LeftForeArm", LeftHandIndex1="LeftHand", LeftUpLeg="Hips", LeftLeg="LeftUpLeg", LeftFoot="LeftLeg", LeftToeBase="LeftFoot", LeftToe_End="LeftToeBase",
               RightArm="Spine2", RightForeArm="RightArm", RightHand="RightForeArm", RightUpLeg="Hips", RightLeg="RightUpLeg", RightFoot="RightLeg")
rides = parle_build.bone_rides(ride, heads, parents)
assert set(rides) == set(heads) and all(np.allclose(rides[name], ride("Left", foot)[0]) for name in ("LeftFoot", "LeftToeBase", "LeftToe_End")), "the toes do not go whole with the foot"
assert all(np.abs(rides[name]).max() == 0 for name in ("LeftArm", "LeftForeArm", "LeftHand", "LeftHandIndex1", "Head", "HeadTop_End", "LeftEye", "Neck", "LeftShoulder")), "an arm, the head or the collar bone moved"
assert np.allclose(rides["LeftUpLeg"], (0, 0, -drop)) and np.allclose(rides["LeftLeg"], ride("Left", knee)[0]) and np.allclose(rides["RightLeg"], ride("Right", JOINTS["RightLeg"])[0]), "a bone of a leg does not go by its leg"
assert -drop < rides["Hips"][2] < rides["Spine"][2] < rides["Spine1"][2] < rides["Spine2"][2] < 0 and all(np.abs(rides[name][:2]).max() == 0 for name in ("Hips", "Spine", "Spine1", "Spine2")), "the spine does not go by the trunk, each joint by its level"

# The hips of a clip. A pose: the hips turned and each leg bent its own way.
quat = lambda axis, degrees: np.array([np.cos(np.radians(degrees) / 2), *(np.sin(np.radians(degrees) / 2) * np.asarray(axis, float) / np.linalg.norm(axis))])
matrix = lambda q: np.array([parle_build.quat_turn(q, axis) for axis in np.eye(3)]).T
assert np.allclose(parle_build.quat_turn(quat((0, 0, 1), 90), (1, 0, 0)), (0, 1, 0)) and np.allclose(parle_build.quat_times(quat((1, 0, 0), 30), quat((1, 0, 0), 60)), quat((1, 0, 0), 90))
assert np.allclose(parle_build.quat_turn(quat((0, 0, 1), 90), np.eye(3)), [(0, 1, 0), (-1, 0, 0), (0, 0, 1)]), "several points are not turned as one is"
chain = ["Hips"] + [side + part for side in parle_build.SIDES for part in parle_build.LEG + ("ToeBase",)]
toes = {side + "ToeBase": JOINTS[side + "Foot"] + (0, -0.13, -0.035) for side in parle_build.SIDES}
rests = {name: quat(rng.normal(size=3), rng.uniform(0, 180)) for name in chain}
still = {name: np.array([1.0, 0.0, 0.0, 0.0]) for name in chain}
before = {name: dict(JOINTS, **toes)[name] for name in chain}
part = lambda name: "trunk" if name == "Hips" else "Left" if name.startswith("Left") else "Right"
# What hangs from a foot goes whole with it: the joint of the toes moves as the ankle does.
after = {name: before[name] + ride(part(name), JOINTS[name.replace("ToeBase", "Foot")])[0] for name in chain}
# The soles. With no skin to read, a foot has its ball under the joint of the toes and its heel behind the ankle, on the floor.
ball, heel = parle_build.sole_points(before["LeftFoot"], before["LeftToeBase"], 0.0)
assert np.allclose(ball, (before["LeftToeBase"][0], before["LeftToeBase"][1], 0)) and heel[2] == 0 and abs(heel[1] - before["LeftFoot"][1] - parle_build.HEEL_BEHIND * 0.13) < 1e-12, f"the ball and the heel of a foot: {ball}, {heel}"
# The rim of a sole: of a plate of points, the ones as far out as any in eight headings; what is inside is left out.
plate = np.array([[x, y, 0.001 * abs(x)] for x in np.linspace(-0.04, 0.04, 9) for y in np.linspace(-0.12, 0.12, 13)])
rim = parle_build.sole_rim(plate)
assert 4 <= len(rim) <= parle_build.SOLE_WAYS and all(abs(point[0]) == 0.04 or abs(point[1]) == 0.12 for point in rim), f"the rim of a sole is not its edge: {rim}"
assert len(parle_build.sole_rim([])) == 0
# Here each sole is its ball and its heel, with the foot, and the tip of the shoe, with the toes.
soles = {side: {"Foot": parle_build.sole_points(before[side + "Foot"], before[side + "ToeBase"], 0.0), "ToeBase": np.array([before[side + "ToeBase"] * (1, 1, 0) + (0, -0.07, 0.01)])} for side in parle_build.SIDES}
soles_now = {side: {name: points + (after[side + "Foot"] - before[side + "Foot"]) for name, points in soles[side].items()} for side in parle_build.SIDES}
# Standing as it rests, the soles are as far under the hips as the floor is, whatever way the bones head.
assert abs(parle_build.soles_under(still, rests, before, soles) + before["Hips"][2]) < 1e-12 and abs(parle_build.soles_under(still, rests, after, soles_now) + after["Hips"][2]) < 1e-12
# A turn in the frame of the skeleton, written in the axes of a bone: the bone's own heading does not change what the pose is.
local = lambda name, turn: parle_build.quat_times(parle_build.quat_times(rests[name] * (1, -1, -1, -1), turn), rests[name])
world = dict(Hips=quat((0, 0, 1), 40), LeftUpLeg=quat((1, 0, 0), -35), LeftLeg=quat((1, 0, 0), 50), LeftFoot=quat((1, 0, 0), 25), RightUpLeg=quat((1, 0, 0), 20), RightLeg=quat((1, 0, 0), 10),
             RightFoot=quat((1, 0, 0), -40), RightToeBase=quat((1, 0, 0), 30))
turns = dict(still, **{name: local(name, turn) for name, turn in world.items()})
# The same pose by plain matrices: the hips turn all, each thigh its leg, each shank what is under the knee, each foot its sole
# and the toes the tip of the shoe.
by_hand = []
for side in parle_build.SIDES:
    turned = lambda name: matrix(world.get(side + name, still["Hips"]))
    top, mid = matrix(world["Hips"]), matrix(world["Hips"]) @ turned("UpLeg")
    low = mid @ turned("Leg")
    foot = low @ turned("Foot")
    ankle = top @ (before[side + "UpLeg"] - before["Hips"]) + mid @ (before[side + "Leg"] - before[side + "UpLeg"]) + low @ (before[side + "Foot"] - before[side + "Leg"])
    by_hand += [(ankle + foot @ (point - before[side + "Foot"]))[2] for point in soles[side]["Foot"]]
    by_hand += [(ankle + foot @ (before[side + "ToeBase"] - before[side + "Foot"]) + foot @ turned("ToeBase") @ (point - before[side + "ToeBase"]))[2] for point in soles[side]["ToeBase"]]
assert abs(parle_build.soles_under(turns, rests, before, soles) - min(by_hand)) < 1e-12, "the soles are not where the turns of the clip put them"
# Set again, the hips stand so that the lowest point of the soles is as high as it was, and go across by the ratio given.
turn, place = matrix(rests["Hips"]), np.array([0.07, -0.12, 0.95])
kept = turn.T @ (place - before["Hips"])
assert np.allclose(parle_build.hips_place(kept, before["Hips"], turn, before["Hips"], 1.0, 0.0), kept), "hips that did not change were moved"
lift = parle_build.soles_under(turns, rests, before, soles) - parle_build.soles_under(turns, rests, after, soles_now)
now = after["Hips"] + turn @ parle_build.hips_place(kept, before["Hips"], turn, after["Hips"], 0.87, lift)
assert np.allclose(now[:2], 0.87 * place[:2]) and lift < 0, f"the hips do not go across by the ratio of the legs, or shorter legs lift them: {now}, {lift}"
assert abs((now[2] + parle_build.soles_under(turns, rests, after, soles_now)) - (place[2] + parle_build.soles_under(turns, rests, before, soles))) < 1e-12, "the lowest point of the soles is not as high over the floor as it was"

# The lengths a reference asks: its knees and its hips, as shares of the stature barefoot, on the kit's own foot.
mine, theirs = dict(hip=0.5645, knee=0.3108, ankle=0.0377), dict(hip=0.5003, knee=0.266, shoulder=0.8115)
asked = body_joints.lengths(None, mine, theirs)
assert abs(asked["thigh"] - 0.9235) < 1e-3 and abs(asked["shank"] - 0.836) < 1e-3, f"the lengths the reference asks of the kit: {asked}"
# Measured again with those lengths the kit stands where the reference does, and the lengths stay.
there = dict(hip=theirs["hip"], knee=theirs["knee"], ankle=mine["ankle"])
again = body_joints.lengths(asked, there, theirs)
assert abs(again["thigh"] - asked["thigh"]) < 1e-3 and abs(again["shank"] - asked["shank"]) < 1e-3, f"a kit already there was moved again: {again}"
# On a kit, with the field: the joints of the knees and of the hips land on the reference's shares.
kit = {name: JOINTS[name] for name in JOINTS}
tall = HEIGHT
shares = dict(hip=kit["LeftUpLeg"][2] / tall, knee=kit["LeftLeg"][2] / tall, ankle=kit["LeftFoot"][2] / tall)
_, ride_kit, _ = parle_build.lengths_field(dict(plain, lengths=body_joints.lengths(None, shares, theirs)), kit, tall)
landed = [(kit[name] + ride_kit("Left", kit[name])[0])[2] / tall for name in ("LeftUpLeg", "LeftLeg")]
assert np.allclose(landed, [theirs["hip"], theirs["knee"]], atol=2e-4), f"the kit's hips and knees do not land where the reference's stand: {landed}"

# The reference says how high his joints stand barefoot, and the normal man took the lengths they ask.
dressed_man = json.load(open(ROOT / "data" / "parle_build" / "reference-dressed-man.json"))
ups = dressed_man["joints_up"]
assert 0.25 < ups["knee"] < 0.28 and 0.49 < ups["hip"] < 0.51 and 0.80 < ups["shoulder"] < 0.82 and 0.01 < dressed_man["sole_m"] < 0.03, f"how high the reference's joints stand: {ups}"
normal = json.load(open(ROOT / "data" / "parle_build" / "male-normal.json"))
assert 0.90 < normal["lengths"]["thigh"] < 0.95 and 0.81 < normal["lengths"]["shank"] < 0.86, f"the lengths of the normal man: {normal['lengths']}"
for kit_name in ("male-slim", "male-muscled", "female-normal", "female-slim", "female-muscled"):
    assert "lengths" not in json.load(open(ROOT / "data" / "parle_build" / f"{kit_name}.json")), f"{kit_name} has lengths nobody asked for"
polish = (ROOT / "tools" / "parle_polish.py").read_text()
assert polish.index("lengths = natural_lengths(meshes, rig, height, kit)") < polish.index("built = natural_build(meshes, rig, height, kit)"), "the lengths do not run before the build"
print("ok")
