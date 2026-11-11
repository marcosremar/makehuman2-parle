"""Regression for the build of the body (2026-11-10): run with `Blender -b --python-exit-code 1 -P tests/test_parle_build.py`.

Unclothed and in A pose, the cast bodies were 80 to 89% like the reference bodies: trunk too shallow, legs and arms thin, the women's neck thin.
A profile per kit now scales trunk, arms and legs. This checks the field: no profile moves nothing, each curve moves only its own part, the head
and the feet keep their shape, the arm goes as a whole with its joint, and left mirrors right.
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import parle_build  # noqa: E402

# The kit's own joints, at a stature of 1.74 m.
HEIGHT, UNIT, FLOOR = 1.74, 0.87, 0.14
RAW = dict(Hips=(0, 0, 1.008), Spine=(0, -0.026, 1.094), Spine1=(0, 0, 1.199), Spine2=(0, 0.069, 1.324), Neck=(0, -0.012, 1.534), Head=(0, -0.026, 1.59),
           LeftArm=(0.197, -0.019, 1.428), LeftForeArm=(0.407, -0.017, 1.189), LeftHand=(0.531, -0.202, 1.069),
           LeftUpLeg=(0.106, 0.007, 0.967), LeftLeg=(0.166, -0.027, 0.471), LeftFoot=(0.222, -0.014, -0.067))
JOINTS = {name: (np.array(point) + (0, 0, FLOOR)) * UNIT for name, point in RAW.items()}
JOINTS.update({name.replace("Left", "Right"): point * (-1, 1, 1) for name, point in JOINTS.items() if name.startswith("Left")})
rng = np.random.default_rng(7)
cloud = rng.uniform((-0.5, -0.2, 0.0), (0.5, 0.2, HEIGHT), (400, 3))
none = {side: np.zeros(len(cloud)) for side in parle_build.SIDES}
zero = np.zeros(len(cloud))


def shift(profile, points, arms=None, head=None, neck=None):
    field, _ = parle_build.build_field(profile, JOINTS, HEIGHT)
    return field(points, arms or {side: np.zeros(len(points)) for side in parle_build.SIDES}, np.zeros(len(points)) if head is None else head, neck)


plain = parle_build.identity("test")
assert np.abs(shift(plain, cloud)).max() == 0, "the profile that moves nothing moved something"
assert np.abs(shift(plain, cloud, {side: np.ones(len(cloud)) for side in parle_build.SIDES})).max() < 1e-12, "the arms moved with no profile"

# One trunk level wider: only that level widens, and nothing moves up, down, forward or back.
wide = parle_build.identity("test")
station = 16
wide["trunk_x"][station] = 1.2
hip, neck = JOINTS["LeftUpLeg"][2], JOINTS["Neck"][2]
level = hip + parle_build.TRUNK_W[station] * (neck - hip)
row = np.array([[0.1, 0.05, level], [-0.1, 0.05, level], [0.1, 0.05, level + 0.2], [0.1, 0.05, level - 0.2]])
moved = shift(wide, row)
assert abs(moved[0, 0] - 0.02) < 1e-9 and abs(moved[1, 0] + 0.02) < 1e-9, f"the level did not widen by its ratio: {moved[:2, 0]}"
assert np.abs(moved[2:]).max() == 0 and np.abs(moved[:, 1:]).max() == 0, "another level, or another axis, moved"

# The trunk deepens about the spine, so the spine's own points stay.
deep = parle_build.identity("test")
deep["trunk_y"] = [1.3] * len(parle_build.TRUNK_W)
spine = np.array([JOINTS[name] for name in ("Spine", "Spine1", "Spine2")])
assert np.abs(shift(deep, spine)).max() < 1e-12, "the spine moved"
front = shift(deep, spine + (0, -0.1, 0))
assert np.allclose(front[:, 1], -0.03) and np.abs(front[:, (0, 2)]).max() == 0, f"the chest did not come forward by its ratio: {front}"

# The head keeps its shape whatever the profile says.
assert np.abs(shift(deep, cloud, head=np.ones(len(cloud)))).max() == 0, "the head changed shape"

# An arm goes as a whole with its joint: the trunk's shift at the shoulder plus the way out; the bone's own line gets no more than that.
broad = parle_build.identity("test")
broad["trunk_x"], broad["arm_out"] = [1.1] * len(parle_build.TRUNK_W), 0.01
field, rides = parle_build.build_field(broad, JOINTS, HEIGHT)
want = JOINTS["LeftArm"][0] * 0.1 + 0.01 * HEIGHT
assert abs(rides["LeftArm"][0] - want) < 1e-9 and abs(rides["RightArm"][0] + want) < 1e-9, f"the arm does not ride with the shoulder: {rides}"
chain = np.array([JOINTS[name] for name in ("LeftArm", "LeftForeArm", "LeftHand")])
whole = field(chain, {"Left": np.ones(3), "Right": np.zeros(3)}, np.zeros(3))
assert np.allclose(whole, rides["LeftArm"]), f"the arm's bones do not go as a whole: {whole}"

# A thicker upper arm: a point beside the bone moves away from it, the hand and the forearm stay.
thick = parle_build.identity("test")
thick["arm_thick"][2], thick["arm_depth"][2] = 1.5, 1.5
middle = JOINTS["LeftArm"] + 0.4 * (JOINTS["LeftForeArm"] - JOINTS["LeftArm"])
spots = np.array([middle, middle + (0, 0.03, 0), JOINTS["LeftHand"] + (0, 0.02, 0), JOINTS["LeftForeArm"] + 0.6 * (JOINTS["LeftHand"] - JOINTS["LeftForeArm"]) + (0, 0.02, 0)])
grown = shift(thick, spots, {"Left": np.ones(4), "Right": np.zeros(4)})
assert np.abs(grown[0]).max() < 1e-12 and grown[1, 1] > 0.01, f"the upper arm did not thicken about its bone: {grown[:2]}"
assert np.abs(grown[2:]).max() < 1e-12, f"the hand or the forearm changed: {grown[2:]}"

# Thicker legs: the knee widens about its joint, the foot keeps its shape, and left mirrors right.
legs = parle_build.identity("test")
legs["leg_x"] = [1.2] * (len(parle_build.LEG_Q) - 1) + [1.0]
knee = JOINTS["LeftLeg"]
low = np.array([knee + (0.03, 0, 0), knee, (knee[0], 0.0, 0.02)])
wider = shift(legs, low)
assert abs(wider[0, 0] - 0.006) < 1e-9 and np.abs(wider[1:]).max() < 1e-12, f"the knee did not widen about its joint, or the foot moved: {wider}"
both = shift(legs, cloud)
mirror = shift(legs, cloud * (-1, 1, 1))
assert np.allclose(both * (-1, 1, 1), mirror, atol=1e-12), "left does not mirror right"
assert np.abs(both[:, 2]).max() == 0, "something moved up or down"

# The shoulder line (2026-11-11): the muscled man's stood 16 to 40 mm lower than the reference body's and no ratio moved it. A lift takes the
# top of the shoulder up from the spine back, the skin that follows the arm included (most of the top of the shoulder does, and lifted with the
# trunk only the line rose next to the neck and nowhere else); the collarbones, the fold behind the armpit, the arm past the deltoid and the arm's bone stay.
high = parle_build.identity("test")
high["shoulder_rise"] = [0.02] * len(parle_build.RISE_S)
top, pit, reach = hip + 0.95 * (neck - hip), hip + 0.76 * (neck - hip), JOINTS["LeftArm"][0]
spots = np.array([[0.1, 0.1, top], [-0.1, 0.1, top], [0.1, -0.15, top], [0.1, 0.1, pit], [1.6 * reach, 0.1, top]])
lifted = shift(high, spots)
assert np.allclose(lifted[:2, 2], 0.02 * HEIGHT), f"the top of the shoulder did not go up by its share of the height: {lifted[:2, 2]}"
assert np.abs(lifted[2:]).max() == 0, f"the collarbones, the armpit or the arm went up: {lifted[2:, 2]}"
assert np.abs(lifted[:, :2]).max() == 0, "the lift moved something sideways"
sleeve = shift(high, spots[:1], {"Left": np.ones(1), "Right": np.zeros(1)})
assert abs(sleeve[0, 2] - 0.02 * HEIGHT) < 1e-9, f"the skin of the arm on top of the shoulder did not go up: {sleeve}"
_, still = parle_build.build_field(high, JOINTS, HEIGHT)
assert all(np.abs(ride).max() == 0 for ride in still.values()), "the lift moved a bone"
older = {key: value for key, value in high.items() if key != "shoulder_rise"}
assert np.abs(shift(older, spots)).max() == 0, "a profile saved before the shoulder line moved something"

# The shoulders (2026-11-11): the women stood 2 to 3% of their stature broader across the shoulders than the reference women, and the curve
# of the trunk, smoothed every round, could not hold shoulders narrower than both the chest and the neck. One ratio is the width of the trunk
# at the level of the joints of the arms, whatever the curve says there, and the arms ride with it; from the armpit down and at the neck the
# curve stays.
slender = parle_build.identity("test")
slender["trunk_x"], slender["girdle"] = [1.1] * len(parle_build.TRUNK_W), 0.9
field, rides = parle_build.build_field(slender, JOINTS, HEIGHT)
reach, joint_w = JOINTS["LeftArm"][0], (JOINTS["LeftArm"][2] - hip) / (neck - hip)
assert abs(rides["LeftArm"][0] + reach * 0.1) < 1e-9 and abs(rides["RightArm"][0] - reach * 0.1) < 1e-9, f"the arms do not come in with the shoulders: {rides}"
levels = [joint_w, joint_w - parle_build.GIRDLE_BELOW, joint_w - 0.3, 1.0, 1.1, joint_w - parle_build.GIRDLE_BELOW / 2]
spots = np.array([[0.1, 0.05, hip + w * (neck - hip)] for w in levels])
moved = shift(slender, spots)
assert abs(moved[0, 0] + 0.1 * 0.1) < 1e-9, f"the trunk at the level of the joints is not as wide as the ratio of the shoulders says: {moved[0]}"
assert np.allclose(moved[1:5, 0], 0.1 * 0.1), f"the chest from the armpit down, or the neck, moved with the shoulders: {moved[1:5, 0]}"
assert -0.1 * 0.1 < moved[5, 0] < 0.1 * 0.1, f"the trunk does not pass from the chest to the shoulders little by little: {moved[5]}"
assert np.abs(moved[:, 1:]).max() == 0, "the shoulders moved something up, down, forward or back"
older = {key: value for key, value in slender.items() if key != "girdle"}
assert abs(parle_build.build_field(older, JOINTS, HEIGHT)[1]["LeftArm"][0] - reach * 0.1) < 1e-9, "a profile saved before the shoulders had a ratio moved them"
assert parle_build.identity("test")["girdle"] is None and np.abs(shift(dict(plain, girdle=1.0), cloud)).max() == 0, "the kit's own shoulders moved something"

# Cloth goes the way of the skin under it (2026-11-11): with a share of the arm of its own, the back of a shirt's armhole went a centimetre
# further in than the skin when the shoulders came in, and the skin showed under the sleeve. On the skin a point of cloth takes the skin's
# share of the arm, and far from it its own.
own, under = np.array([0.6, 0.6, 0.6, 0.6]), np.array([0.1, 0.1, 0.1, 0.1])
taken = parle_build.share_over_skin(own, under, np.array([0.0, 0.015, 0.0275, 0.04]), 0.015, 0.04)
assert np.allclose(taken[:2], 0.1) and abs(taken[3] - 0.6) < 1e-9 and 0.1 < taken[2] < 0.6, f"cloth does not take the share of the skin it lies on: {taken}"
polish = (Path(__file__).parent.parent / "tools" / "parle_polish.py").read_text()
assert "arms = over_skin(obj, arms, skin, skin_arms)" in polish and "parle_build.share_over_skin(" in polish, "the build moves cloth by its own share of the arm again"

# The head against the body (2026-11-11). The aim is avatars like Rafa and Sofia, "mais estilo Disney", and the women of the cast had a head
# a quarter to a third smaller than Sofia's and the game's women, where the men's is the game's. On the kits of women the head, and what
# hangs on it, grows about its joint; the men's and the body under any head stay.
assert parle_build.head_ratio("female-slim") == parle_build.head_ratio("female-muscled") == parle_build.HEAD["female"] > 1.15, "the women's head is the kit's again, or no bigger than the 12% that was too little"
assert parle_build.head_ratio("male-slim") == 1.0 == parle_build.head_ratio("test"), "the men's head, which is the game's, changed size"
lady, grown = parle_build.identity("female-test"), parle_build.head_ratio("female-test") - 1
spots = JOINTS["Head"] + np.array([[0.05, -0.08, 0.1], [0.0, -0.06, -0.04], [0.05, -0.08, 0.1], [0.0, 0.0, 0.0]])
moved = shift(lady, spots, head=np.array([1.0, 1.0, 0.5, 1.0]))
# The neck (2026-11-11, the same owner, later: "a cabeça das mulheres poderia ter um pouco maior para ter mais aparência de cartoon e o
# pescoço um pouco menor"): shorter from its joint to the head's on the women, so the head, joint and all, comes down by what it lost.
lost = (1 - parle_build.neck_ratio("female-test")) * (JOINTS["Head"] - JOINTS["Neck"])
assert parle_build.neck_ratio("female-slim") == parle_build.NECK["female"] < 1 and parle_build.neck_ratio("male-slim") == 1.0 == parle_build.neck_ratio("test"), "whose neck is shorter"
assert np.allclose(moved[0], grown * np.array([0.05, -0.08, 0.1]) - lost) and np.allclose(moved[1], grown * np.array([0.0, -0.06, -0.04]) - lost), f"the head does not grow about its joint and come down with it: {moved[:2]}"
assert np.allclose(moved[2], moved[0] / 2) and np.allclose(moved[3], -lost), "the skin between the head and the neck does not go half way, or the joint of the head stayed where the long neck had it"
assert np.abs(shift(lady, cloud)).max() == 0, "the body under the head moved with it"
up = JOINTS["Neck"] + np.outer([0.0, 0.5, 1.0, 1.4, -0.3], JOINTS["Head"] - JOINTS["Neck"]) + (0.03, 0.0, 0.0)
closed = shift(lady, up, neck=np.ones(5))
assert np.allclose(closed, -np.outer([0.0, 0.5, 1.0, 1.0, 0.0], lost)), f"the skin of the neck does not close up by how far up the neck it is: {closed}"
assert np.abs(shift(lady, up, neck=np.zeros(5))).max() == 0 and np.abs(shift(parle_build.identity("male-test"), up, neck=np.ones(5))).max() == 0, "what does not follow the neck moved, or a man's neck"
polish = (Path(__file__).parent.parent / "tools" / "parle_polish.py").read_text()
assert "parle_build.head_ratio(profile" in polish and "rig.scale = [rig.scale[0] * stature /" in polish, "the bones of the head do not grow with it, or the body does not keep its stature"
assert "parle_build.neck_ratio(profile" in polish and 'rig.data.edit_bones["Neck"], "tail", base + down' in polish, "the bones do not come down with the head where the neck is shorter"

# A profile comes back as it was saved; a kit without one has none.
folder = tempfile.mkdtemp()
parle_build.save_profile(broad, os.path.join(folder, "kit.json"))
assert parle_build.load_profile(os.path.join(folder, "kit.json"))["arm_out"] == 0.01 and parle_build.load_profile(os.path.join(folder, "none.json")) is None
assert parle_build.load_profile("no-such-kit") is None
print("ok")
