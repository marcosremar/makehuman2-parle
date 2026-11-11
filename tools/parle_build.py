"""
    License information: data/licenses/makehuman_license.txt

    The build of the body, per kit (owner, 10/11/2026: the unclothed cast body, in A pose, has to look like the reference bodies).
    A profile (`data/parle_build/<kit>.json`) holds only ratios: how much wider and deeper the trunk is at each level, each ring of the arm
    and of the leg, and how far out the arms sit. `tools/body_match.py` finds those ratios by measuring the cast body against the measures of
    a reference body; no mesh and no texture of a reference comes in here. This file has no bpy: `parle_polish.natural_build` applies the field.
"""
import json
import os

import numpy as np

SIDES = ("Left", "Right")
SPINE = ("Hips", "Spine", "Spine1", "Spine2", "Neck", "Head")
ARM = ("Arm", "ForeArm", "Hand")
LEG = ("UpLeg", "Leg", "Foot")
BONES = SPINE + tuple(side + part for side in SIDES for part in ARM + LEG)
# Trunk levels: 0 at the hip joints, 1 at the base of the neck. They start below the crotch and end above the neck, so both ends have a station.
TRUNK_W = np.linspace(-0.25, 1.15, 29)
# Arm rings: 0 shoulder, 1 elbow, 2 wrist. Leg rings: 0 hip, 1 knee, 2 ankle (the last one stays 1: the foot keeps its shape).
ARM_T = np.linspace(0, 2, 11)
LEG_Q = np.linspace(0, 2, 21)
CURVES = {"trunk_x": TRUNK_W, "trunk_y": TRUNK_W, "arm_depth": ARM_T, "arm_thick": ARM_T, "leg_x": LEG_Q, "leg_y": LEG_Q}
# Shoulder line: how far up the top of the shoulder goes, in shares of the height, at shares of the way from the midline to the shoulder joint.
# It is the trapezius: nothing at the armpit, all of it from the top of the shoulder up, and only from the spine back, so the collarbones and
# the pit of the neck stay. No ratio moved this line: the muscled man's stood 16 to 40 mm lower than the reference body's (2026-11-11). The
# skin on top of the shoulder follows the arm bone from well before the joint (40 to 100% of it): lifted with the trunk only, the line rose
# next to the neck and nowhere else. So the lift is added after the arm takes its skin, and fades over the deltoid, past the joint.
# It starts above the fold behind the armpit: from 0.7 it reached the back of the arm, which stood out as a flap with the arm down.
RISE_S = np.linspace(0, 1.6, 9)
RISE_W, RISE_FRONT, RISE_ARM = (0.78, 0.92), (0.05, 0.012), (1.2, 1.6)
# The shoulders: one ratio, `girdle`, that the trunk takes at the level of the joints of the arms in place of its curve's, and the arms ride
# with their joints: 1 is the kit's own shoulders. It is whole at that level and gives way to the curve little by little, up to the base
# of the neck above and, below, down to the armpit, GIRDLE_BELOW of hip to neck under the joints (0.11 to 0.12 on the kits): the chest
# and the neck keep the widths their own measures gave them. The curve could not hold this alone: it is smoothed every round of the
# loop, and shoulders narrower than both the chest under them and the neck over them are a dip three stations wide that the smoothing
# fills again. Left to the curve, the women stayed 0.4 to 1% of the stature broader across the shoulders than the reference women,
# however many rounds the loop ran (2026-11-11). A profile with no ratio (None) has the shoulders of its curve, as before.
GIRDLE_BELOW = 0.12
# The head, against the body. The aim of the cast is avatars like Rafa and Sofia, "mais estilo Disney" (owner, 2026-11-11), and theirs
# have a big head: Sofia is 6.4 heads tall and the women of the game 6.2 to 6.7. The men of the cast have the head of the game's men
# (Carlos: 6.6 cm between the eyes against 6.5, 10.5 cm from the eyes to the top against 10.7; Rafa 6.6 and 10.7). The women came out
# 7.6 to 7.7 heads tall, with a head a quarter to a third smaller than Sofia's and the game's women (5.3 cm between the eyes against
# 7.5 to 7.8) and small next to the man of the cast too. The owner saw it at 12 and at 25% more and took 12: the head, and all that
# hangs on it, grows by that about its joint, and the body keeps its stature (`parle_polish.natural_build`). Later the same day, with
# the faces and the hair seen to: "a cabeça das mulheres poderia ter um pouco maior para ter mais aparência de cartoon e o pescoço um
# pouco menor". A little more is 18%, between the two he had seen.
HEAD = {"female": 1.18}
# And the neck, from its joint to the head's, is 15% shorter on the women: the head comes down that much, and the skin of the neck
# closes up with it, each point by how far up the neck it is. A choice by eye on a sheet of three (as it was, this, and 25% of each).
NECK = {"female": 0.85}
# Widths of the blends, as shares of the height or in metres: trunk to legs, left leg to right leg, upper arm to forearm, wrist to hand.
# The trunk hands over to the legs across 10 cm: with a short band, a pelvis made deeper than the thighs left a ledge at the groin.
CROTCH_BAND, MIDLINE_BAND, ELBOW_BAND, HAND_FADE = 0.03, 0.02, 0.025, 0.15
FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "parle_build")


def smooth(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def identity(kit, crotch=0.2):
    """The profile that moves nothing. `crotch` is how far down the thigh the legs part, as a share of hip to knee."""
    curves = {name: [1.0] * len(stations) for name, stations in CURVES.items()}
    return dict(kit=kit, reference="", similarity=None, crotch=crotch, arm_out=0.0, girdle=None, shoulder_rise=[0.0] * len(RISE_S), **curves)


def load_profile(kit):
    """A kit name or the path of a profile; None when that kit has no profile."""
    path = kit if str(kit).endswith(".json") else os.path.join(FOLDER, f"{kit}.json")
    return json.load(open(path)) if os.path.exists(path) else None


def save_profile(profile, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tidy = {key: [round(float(item), 4) for item in value] if isinstance(value, (list, np.ndarray)) else value for key, value in profile.items()}
    json.dump(tidy, open(path, "w"), indent=1)


def trunk_param(z, joints):
    hip = (joints["LeftUpLeg"][2] + joints["RightUpLeg"][2]) / 2
    return (z - hip) / (joints["Neck"][2] - hip)


def head_ratio(kit):
    """How many times the size of the kit's own the head of a kit is, against its body."""
    return next((ratio for sex, ratio in HEAD.items() if str(kit).startswith(sex)), 1.0)


def neck_ratio(kit):
    """How many times the length of the kit's own the neck of a kit is, from its joint to the head's."""
    return next((ratio for sex, ratio in NECK.items() if str(kit).startswith(sex)), 1.0)


def share_over_skin(own, skin, away, near, far):
    """The share of an arm for points of cloth: the share of the skin under each where the cloth lies on it, within `near` metres, and
    its `own` from `far` on. `skin` is the share of the nearest point of the skin and `away` how far that point is."""
    return own + (skin - own) * smooth((far - away) / (far - near))


def girdle_share(w, joint_w):
    """How much a trunk level takes the ratio of the shoulders in place of its curve's: wholly at the level of the joints of the arms,
    `joint_w`, not at all from the base of the neck up nor from the armpit down."""
    return smooth((w - (joint_w - GIRDLE_BELOW)) / GIRDLE_BELOW) * smooth((1 - w) / (1 - joint_w))


def leg_param(z, top, knee, foot):
    return np.clip(np.where(z > knee, (top - z) / (top - knee), 1 + (knee - z) / (knee - foot)), 0, 2)


def arm_param(points, shoulder, elbow, wrist):
    """Where each point is along the arm (0 shoulder, 1 elbow, 2 wrist, more on the hand) and how much of it is forearm."""
    upper, fore = elbow - shoulder, wrist - elbow
    long_u, long_f = np.linalg.norm(upper), np.linalg.norm(fore)
    fore_share = smooth(0.5 + (points - elbow) @ (upper / long_u + fore / long_f) / (2 * ELBOW_BAND))
    along_u = np.clip((points - shoulder) @ upper / long_u ** 2, 0, 1)
    along_f = np.clip((points - elbow) @ fore / long_f ** 2, 0, None)
    return (1 - fore_share) * along_u + fore_share * (1 + along_f), fore_share


def ring(points, start, end, depth, thick):
    """Scales the ring about the bone: `depth` front to back, `thick` across."""
    way = (end - start) / np.linalg.norm(end - start)
    front = np.array([0.0, 1.0, 0.0]) - way * way[1]
    front /= np.linalg.norm(front)
    across = np.cross(way, front)
    rel = points - start
    return ((rel @ front) * (depth - 1))[:, None] * front + ((rel @ across) * (thick - 1))[:, None] * across


def build_field(profile, joints, height):
    """`field(points, arms, head)` gives the shift of each point, in metres; `rides` is how far each arm and each leg goes as a whole, for the bones.
    `joints` are the heads of BONES in the world, `arms` the share of each point that follows each arm, `head` the share that follows the head."""
    joints = {name: np.asarray(joints[name], float) for name in BONES}
    mid = (joints["LeftUpLeg"][0] + joints["RightUpLeg"][0]) / 2
    hip, knee = (joints["LeftUpLeg"][2] + joints["RightUpLeg"][2]) / 2, (joints["LeftLeg"][2] + joints["RightLeg"][2]) / 2
    spine = np.array(sorted((joints[name] for name in SPINE), key=lambda joint: joint[2]))
    cut, band = hip - profile["crotch"] * (hip - knee), CROTCH_BAND * height
    side_sign = 1.0 if joints["LeftUpLeg"][0] > mid else -1.0

    girdle, joint_w = profile.get("girdle"), trunk_param((joints["LeftArm"][2] + joints["RightArm"][2]) / 2, joints)

    def trunk(points):
        w = trunk_param(points[:, 2], joints)
        wide = np.interp(w, TRUNK_W, profile["trunk_x"])
        if girdle is not None:
            wide = wide + (girdle - wide) * girdle_share(w, joint_w)
        out = np.zeros_like(points)
        out[:, 0] = (points[:, 0] - mid) * (wide - 1)
        out[:, 1] = (points[:, 1] - np.interp(points[:, 2], spine[:, 2], spine[:, 1])) * (np.interp(w, TRUNK_W, profile["trunk_y"]) - 1)
        return out

    lift, reach = profile.get("shoulder_rise") or [0.0] * len(RISE_S), abs(joints["LeftArm"][0] - mid)

    def rise(points):
        w, s = trunk_param(points[:, 2], joints), np.abs(points[:, 0] - mid) / reach
        ahead = (np.interp(points[:, 2], spine[:, 2], spine[:, 1]) - points[:, 1]) / height
        share = smooth((w - RISE_W[0]) / (RISE_W[1] - RISE_W[0])) * smooth((RISE_FRONT[0] - ahead) / (RISE_FRONT[0] - RISE_FRONT[1]))
        out = np.zeros_like(points)
        out[:, 2] = np.interp(s, RISE_S, lift) * height * share * (1 - smooth((s - RISE_ARM[0]) / (RISE_ARM[1] - RISE_ARM[0])))
        return out

    rides = {}
    for side in SIDES:
        rides[side + "UpLeg"] = trunk(joints[side + "UpLeg"][None])[0]
        out = np.array([np.sign(joints[side + "Arm"][0] - mid) * profile["arm_out"] * height, 0.0, 0.0])
        rides[side + "Arm"] = trunk(joints[side + "Arm"][None])[0] + out

    def leg(points, side):
        top, mid_joint, foot = (joints[side + part] for part in LEG)
        q = leg_param(points[:, 2], top[2], mid_joint[2], foot[2])
        out = np.tile(rides[side + "UpLeg"], (len(points), 1))
        for axis, curve in ((0, "leg_x"), (1, "leg_y")):
            centre = np.interp(q, (0, 1, 2), (top[axis], mid_joint[axis], foot[axis]))
            out[:, axis] += (points[:, axis] - centre) * (np.interp(q, LEG_Q, profile[curve]) - 1)
        return out

    def body(points):
        lower = smooth((cut + band - points[:, 2]) / (2 * band))[:, None]
        left = smooth(0.5 + side_sign * (points[:, 0] - mid) / (2 * MIDLINE_BAND * height))[:, None]
        return (1 - lower) * trunk(points) + lower * (left * leg(points, "Left") + (1 - left) * leg(points, "Right"))

    def arm(points, side):
        shoulder, elbow, wrist = (joints[side + part] for part in ARM)
        t, fore = arm_param(points, shoulder, elbow, wrist)
        fade = 1 - smooth((t - 2) / HAND_FADE)
        depth = 1 + (np.interp(t, ARM_T, profile["arm_depth"]) - 1) * fade
        thick = 1 + (np.interp(t, ARM_T, profile["arm_thick"]) - 1) * fade
        return rides[side + "Arm"] + (1 - fore)[:, None] * ring(points, shoulder, elbow, depth, thick) + fore[:, None] * ring(points, elbow, wrist, depth, thick)

    grow = head_ratio(profile.get("kit", ""))
    neck_way = joints["Head"] - joints["Neck"]
    drop = (1 - neck_ratio(profile.get("kit", ""))) * neck_way

    def field(points, arms, head, neck=None):
        out = body(points)
        for side in SIDES:
            share = np.asarray(arms[side])[:, None]
            out = (1 - share) * out + share * arm(points, side)
        top = np.asarray(head)[:, None]
        mid = np.zeros_like(top) if neck is None else np.asarray(neck)[:, None]
        # How far up the neck each point is, from its joint to the head's: where the neck is shorter, the head comes down whole and
        # what follows the bone of the neck by that share of the way.
        along = np.clip((points - joints["Neck"]) @ neck_way / (neck_way @ neck_way), 0, 1)[:, None]
        # The head keeps its shape and, on the kits whose head grows, grows about its joint.
        return (out + rise(points)) * (1 - top) + top * (grow - 1) * (points - joints["Head"]) - np.clip(top + mid * along, 0, 1) * drop

    return field, rides
