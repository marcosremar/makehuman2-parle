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
# pouco menor". Shown a sheet of three (as it was; 18% and the neck 15% shorter; 25% and 25%), with the middle one built, he took the
# strongest: "Mais forte". So 25%, which was Sofia's proportion from the start.
HEAD = {"female": 1.25}
# And the neck, from its joint to the head's, is a quarter shorter on the women: the head comes down that much, and the skin of the
# neck closes up with it, each point by how far up the neck it is.
NECK = {"female": 0.75}
# Widths of the blends, as shares of the height or in metres: trunk to legs, left leg to right leg, upper arm to forearm, wrist to hand.
# The trunk hands over to the legs across 10 cm: with a short band, a pelvis made deeper than the thighs left a ledge at the groin.
CROTCH_BAND, MIDLINE_BAND, ELBOW_BAND, HAND_FADE = 0.03, 0.02, 0.025, 0.15
# The lengths (owner, 2026-11-14, of a man he had picked for his body: "copia toda a estrutura e formato"). A profile may say how
# many times the kit's own the thigh and the shank are long, joint to joint (`lengths`). Each bone of the leg keeps its heading and
# the foot its shape and its place on the floor, so the hips come down by what the legs lose, and the trunk takes that up between
# the joints of the hips and the joints of the arms: the stature stays, and so does all that is above the shoulders, the arms with
# it. The stretch of the trunk comes in and goes out over TRUNK_EASE of that span at each end: evenly spread from end to end, the
# skin would go from no stretch to all of it on one level, across the buttocks and under the collarbones.
# The kit's own legs are Rafa's: on 2026-11-03 the owner had the skeleton of the cast's men set by his ("modifica o nosso para
# ficar idêntico"), the joints of the hips at 55 to 56% of the stature, where the reference man has them at 50%. Shown the normal
# man both ways, standing beside the reference and beside Rafa, and walking, he took the reference's ("A do Avaturn", 2026-11-14):
# the lengths of `male-normal` are his choice over that earlier one. The slim and the muscled men have the skeleton of the normal
# one (the joints of the hips and of the knees within 0.2% of the stature on the three kits); shown one of each with the same
# lengths, he took them for all the men ("Sim, todos os homens", the same day). Their widths stay their own references' work.
LENGTHS = ("thigh", "shank")
TRUNK_EASE = 0.1
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


def neck_follow(points, head, neck, joints):
    """How much of what a shorter neck loses each point comes down by, from its weights alone: the head whole, and what follows the
    bone of the neck by how far up the neck it is, from its joint to the head's. On a skin the two meet in a ring or two, and all the
    loss closed up there, in a fold (the women at 25%, 2026-11-11): `parle_polish.even_follow` spreads it."""
    way = joints["Head"] - joints["Neck"]
    along = np.clip((np.asarray(points) - joints["Neck"]) @ way / (way @ way), 0, 1)
    return np.clip(np.asarray(head) + np.asarray(neck) * along, 0, 1)


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


def halves(profile, joints, height):
    """`shares(points)`: how much of each point is leg and not trunk, and how much of that is the left leg's. The legs part
    `profile["crotch"]` of the way down the thigh; trunk and legs, and one leg and the other, hand over across their bands."""
    mid = (joints["LeftUpLeg"][0] + joints["RightUpLeg"][0]) / 2
    hip, knee = (joints["LeftUpLeg"][2] + joints["RightUpLeg"][2]) / 2, (joints["LeftLeg"][2] + joints["RightLeg"][2]) / 2
    cut, band = hip - profile["crotch"] * (hip - knee), CROTCH_BAND * height
    side_sign = 1.0 if joints["LeftUpLeg"][0] > mid else -1.0

    def shares(points):
        lower = smooth((cut + band - points[:, 2]) / (2 * band))[:, None]
        left = smooth(0.5 + side_sign * (points[:, 0] - mid) / (2 * MIDLINE_BAND * height))[:, None]
        return lower, left

    return shares


def eased(t, ease):
    """The share of a stretch that lies below `t` of its span (0 to 1), when the stretch is even in the middle and comes in and goes
    out in a straight ramp over `ease` of the span at each end."""
    t = np.clip(np.asarray(t, float), 0, 1)
    below = lambda s: np.where(s < ease, s * s / (2 * ease), s - ease / 2)
    return np.where(t < 0.5, below(t), (1 - ease) - below(1 - t)) / (1 - ease)


def lengths_field(profile, joints, height):
    """The lengths of a profile (LENGTHS), or None when it has none: `field(points, arms, head)`, the shift of each point in metres,
    `ride(part, points)`, the shift of the bones of a part at those points ("trunk", or the side of a leg), and how far the hips
    come down. `joints`, `arms` and `head` as in `build_field`. The thigh and the shank shorten or lengthen along their own bones,
    the foot goes whole and stays at its height, the trunk stretches level by level, and what follows an arm or the head goes with
    the joint of that arm or of the head."""
    lengths = profile.get("lengths") or {}
    thigh, shank = float(lengths.get("thigh", 1.0)), float(lengths.get("shank", 1.0))
    if thigh == 1.0 and shank == 1.0:
        return None
    joints = {name: np.asarray(joints[name], float) for name in BONES}
    legs = {}
    for side in SIDES:
        top, knee, foot = (joints[side + part] for part in LEG)
        legs[side] = (top, knee, foot, (thigh - 1) * (knee - top), (shank - 1) * (foot - knee))
    # What the two bones of a leg lose in height (how far the ankle would come up), the hips come down by: the soles stay on the floor.
    drop = float(np.mean([upper[2] + lower[2] for *_, upper, lower in legs.values()]))
    hip = (joints["LeftUpLeg"][2] + joints["RightUpLeg"][2]) / 2
    top = (joints["LeftArm"][2] + joints["RightArm"][2]) / 2
    shares = halves(profile, joints, height)

    def trunk(points):
        out = np.zeros_like(points)
        out[:, 2] = -drop * (1 - eased((points[:, 2] - hip) / (top - hip), TRUNK_EASE))
        return out

    def leg(points, side):
        start, knee, foot, upper, lower = legs[side]
        on_upper = np.clip((points - start) @ (knee - start) / ((knee - start) @ (knee - start)), 0, 1)
        on_lower = np.clip((points - knee) @ (foot - knee) / ((foot - knee) @ (foot - knee)), 0, 1)
        out = np.outer(on_upper, upper) + np.outer(on_lower, lower)
        out[:, 2] -= drop
        return out

    def ride(part, points):
        points = np.atleast_2d(np.asarray(points, float))
        return trunk(points) if part == "trunk" else leg(points, part)

    def field(points, arms, head):
        lower, left = shares(points)
        out = (1 - lower) * trunk(points) + lower * (left * leg(points, "Left") + (1 - left) * leg(points, "Right"))
        for side in SIDES:
            share = np.asarray(arms[side])[:, None]
            out = (1 - share) * out + share * trunk(joints[side + "Arm"][None])
        top_share = np.asarray(head)[:, None]
        return (1 - top_share) * out + top_share * trunk(joints["Head"][None])

    return field, ride, drop


def quat_times(first, second):
    """The product of two turns, each as w, x, y, z."""
    a, b = np.asarray(first, float), np.asarray(second, float)
    return np.array([a[0] * b[0] - a[1:] @ b[1:], *(a[0] * b[1:] + b[0] * a[1:] + np.cross(a[1:], b[1:]))])


def quat_turn(turn, vector):
    """A vector turned by a turn (w, x, y, z)."""
    turn, vector = np.asarray(turn, float), np.asarray(vector, float)
    side = np.cross(turn[1:], vector)
    return vector + 2 * (turn[0] * side + np.cross(turn[1:], side))


# The soles. A foot meets the floor by its heel, by the ball or by the tip of the toes, as it turns: what is watched is the rim of
# the sole, the points of the feet and of the shoes that rest within SOLE_BAND_M of the floor, as far out as any in each of SOLE_WAYS
# headings, each with the bone it follows more, the foot's or the toes'. A body with no such points has the ball, under the joint
# of the toes, and the heel, HEEL_BEHIND of the way from the ankle to the ball behind the ankle.
SOLE_BAND_M = 0.025
SOLE_WAYS = 8
HEEL_BEHIND = 0.35
SOLE = ("Foot", "ToeBase")


def sole_points(ankle, toes, floor):
    """The ball and the heel of a foot at rest (HEEL_BEHIND), on the level of its sole: `ankle` and `toes` are the joints."""
    ankle, toes = np.asarray(ankle, float), np.asarray(toes, float)
    heel = ankle - HEEL_BEHIND * (toes - ankle)
    return np.array([[toes[0], toes[1], floor], [heel[0], heel[1], floor]])


def sole_rim(points, ways=SOLE_WAYS):
    """The points of a sole that are as far out as any, level with the floor, in each of `ways` headings."""
    points = np.asarray(points, float).reshape(-1, 3)
    if not len(points):
        return points
    angles = np.arange(ways) * 2 * np.pi / ways
    return points[np.unique(np.argmax(points[:, :2] @ np.stack([np.cos(angles), np.sin(angles)]), axis=0))]


def soles_under(turns, rests, joints, soles):
    """How far over the head of the hips the lowest point of the two soles is in a pose (under it: negative). `turns` are the turns
    the clip gives the hips and each thigh, shank, foot and toes, `rests` the headings those bones rest with and `joints` where they
    rest; `soles` are the points of each sole where they rest, by side and by the bone they go with (SOLE). All in the frame of
    the skeleton, the turns as w, x, y, z."""
    back = lambda turn: np.asarray(turn, float) * (1, -1, -1, -1)
    posed = lambda name, base: quat_times(base, quat_times(quat_times(rests[name], turns[name]), back(rests[name])))
    hips, lows = posed("Hips", (1.0, 0.0, 0.0, 0.0)), []
    for side in SIDES:
        upper = posed(side + "UpLeg", hips)
        lower = posed(side + "Leg", upper)
        foot = posed(side + "Foot", lower)
        toes = posed(side + "ToeBase", foot)
        ankle = quat_turn(hips, joints[side + "UpLeg"] - joints["Hips"]) + quat_turn(upper, joints[side + "Leg"] - joints[side + "UpLeg"]) + quat_turn(lower, joints[side + "Foot"] - joints[side + "Leg"])
        ball = ankle + quat_turn(foot, joints[side + "ToeBase"] - joints[side + "Foot"])
        for name, start, turned in (("Foot", ankle, foot), ("ToeBase", ball, toes)):
            mine = np.asarray(soles[side].get(name, ()), float).reshape(-1, 3)
            if len(mine):
                lows.append(float((start + quat_turn(turned, mine - joints[side + name]))[:, 2].min()))
    return min(lows)


def bone_rides(ride, heads, parents):
    """How far the head of each bone goes with the lengths (`lengths_field`): `ride` is the field's, `heads` where the bones rest
    and `parents` the parent of each, by name. What hangs from a foot, from an arm or from the head goes whole with that joint; a
    bone of a leg goes by the field of its leg and every other by the trunk's, each by where its own joint is."""
    whole = {**{side + "Foot": side for side in SIDES}, **{side + "Arm": "trunk" for side in SIDES}, "Head": "trunk"}
    legs = {side + "UpLeg": side for side in SIDES}

    def under(name, tops):
        while name is not None and name not in tops:
            name = parents.get(name)
        return name

    out = {}
    for name, head in heads.items():
        top = under(name, whole)
        out[name] = ride(whole[top], heads[top])[0] if top is not None else ride(legs.get(under(name, legs), "trunk"), head)[0]
    return out


def hips_place(local, head, turn, head_new, across, lift):
    """A place of the hips in a clip, for a body whose legs changed length. `local` is the place as the clip keeps it, in the axes of
    the bone and counted from its head at rest, `head`; `turn` takes those axes to the skeleton's, and `head_new` is where the hips
    rest now. Level with the floor the hips go `across` times as far from the origin of the skeleton, as the feet now do under them;
    and they stand `lift` higher (lower, when it is negative): what keeps the lowest point of the two soles as high over the floor
    as the clip had it (`soles_under`), so nothing goes further into the floor than it did. Three rules were tried before this one
    on a man with legs an eighth shorter (2026-11-14), each read on his 58 clips as the lowest point of his feet and shoes. By the
    height of the hips alone, the rule the clips are first set by (`parle_clips.retarget`), the standing clips came out right and
    the walking ones sank a foot 8 to 18 mm further into the floor. By the lower of the two ankles the foot that stands flat stayed
    where it was and the one on its toes still sank, up to 19 mm: a shank that slants loses less height than one that stands, and
    the foot none. By the ball and the heel of each foot, up to 16 mm: on its toes a shoe reaches the floor with its tip."""
    turn = np.asarray(turn, float)
    place = np.asarray(head, float) + turn @ np.asarray(local, float)
    return turn.T @ (np.array([across * place[0], across * place[1], place[2] + lift]) - np.asarray(head_new, float))


def build_field(profile, joints, height):
    """`field(points, arms, head)` gives the shift of each point, in metres; `rides` is how far each arm and each leg goes as a whole, for the bones.
    `joints` are the heads of BONES in the world, `arms` the share of each point that follows each arm, `head` the share that follows the head."""
    joints = {name: np.asarray(joints[name], float) for name in BONES}
    mid = (joints["LeftUpLeg"][0] + joints["RightUpLeg"][0]) / 2
    spine = np.array(sorted((joints[name] for name in SPINE), key=lambda joint: joint[2]))
    shares = halves(profile, joints, height)

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
        lower, left = shares(points)
        return (1 - lower) * trunk(points) + lower * (left * leg(points, "Left") + (1 - left) * leg(points, "Right"))

    def arm(points, side):
        shoulder, elbow, wrist = (joints[side + part] for part in ARM)
        t, fore = arm_param(points, shoulder, elbow, wrist)
        fade = 1 - smooth((t - 2) / HAND_FADE)
        depth = 1 + (np.interp(t, ARM_T, profile["arm_depth"]) - 1) * fade
        thick = 1 + (np.interp(t, ARM_T, profile["arm_thick"]) - 1) * fade
        return rides[side + "Arm"] + (1 - fore)[:, None] * ring(points, shoulder, elbow, depth, thick) + fore[:, None] * ring(points, elbow, wrist, depth, thick)

    grow = head_ratio(profile.get("kit", ""))
    drop = (1 - neck_ratio(profile.get("kit", ""))) * (joints["Head"] - joints["Neck"])

    def field(points, arms, head, follow=None):
        """`follow` is how much of the neck's loss each point comes down by (`neck_follow`); the head alone, when it is not given."""
        out = body(points)
        for side in SIDES:
            share = np.asarray(arms[side])[:, None]
            out = (1 - share) * out + share * arm(points, side)
        top = np.asarray(head)[:, None]
        down = top if follow is None else np.asarray(follow)[:, None]
        # The head keeps its shape and, on the kits whose head grows, grows about its joint; where the neck is shorter it comes down whole.
        return (out + rise(points)) * (1 - top) + top * (grow - 1) * (points - joints["Head"]) - down * drop

    return field, rides
