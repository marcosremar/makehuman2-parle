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
# Widths of the blends, as shares of the height or in metres: trunk to legs, left leg to right leg, upper arm to forearm, wrist to hand.
# The trunk hands over to the legs across 10 cm: with a short band, a pelvis made deeper than the thighs left a ledge at the groin.
CROTCH_BAND, MIDLINE_BAND, ELBOW_BAND, HAND_FADE = 0.03, 0.02, 0.025, 0.15
FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "parle_build")


def smooth(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def identity(kit, crotch=0.2):
    """The profile that moves nothing. `crotch` is how far down the thigh the legs part, as a share of hip to knee."""
    return dict(kit=kit, reference="", similarity=None, crotch=crotch, arm_out=0.0, **{name: [1.0] * len(stations) for name, stations in CURVES.items()})


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

    def trunk(points):
        w = trunk_param(points[:, 2], joints)
        out = np.zeros_like(points)
        out[:, 0] = (points[:, 0] - mid) * (np.interp(w, TRUNK_W, profile["trunk_x"]) - 1)
        out[:, 1] = (points[:, 1] - np.interp(points[:, 2], spine[:, 2], spine[:, 1])) * (np.interp(w, TRUNK_W, profile["trunk_y"]) - 1)
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

    def field(points, arms, head):
        out = body(points)
        for side in SIDES:
            share = np.asarray(arms[side])[:, None]
            out = (1 - share) * out + share * arm(points, side)
        return out * (1 - np.asarray(head))[:, None]

    return field, rides
