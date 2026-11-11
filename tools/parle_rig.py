"""
    License information: data/licenses/makehuman_license.txt

    How a MakeHuman body takes the Parle game's motion clips. The two skeletons rest differently, so each
    bone is first pointed the way the game's bone points, and only then does the clip turn it.

    - Arms and legs point exactly as in the game: otherwise the clip opened the legs and crossed the hands.
    - The clavicle points front to back as in the game: the MakeHuman one points forward, and standing
      still the shoulders sat 2 cm in front of the neck (owner, 2026-11-07). In height it goes only half
      way, because all the way the shoulders came up 2.5 cm and the person looked hunched. It also turns back a little more than the
      game's: seen from above the tips of the shoulders still curved forward, standing still, on a body that has a deeper chest (owner, 2026-11-09).
      With the arms copied from Rafa and Sofia (2026-11-11) that turn was left over: standing still it held the clavicle 7 to 16 degrees
      back and up from where it was modelled, and the skin around it went along. The shoulder stood as a high square mass with a deep
      groove between the arm and the chest, where at rest it is round (owner: "os ombros do homem estão jogados para frente e a dobra
      entre o braço e o ombro está estranha"). Now the clavicle is pointed so that, when the body it copies stands still, ours is exactly
      where it was modelled, and the clip moves it from there.
    - Standing, the head sits over the ankles (owner, 2026-11-11: "os ombros do homem estão jogados para frente"). The spine keeps its own
      rest, and with the game's clips on it the cast stooped: in the idle clip the middle of the head stood 2.7 to 2.9% of the height in
      front of the ankles (5 cm), where Rafa holds his 0.8% behind them and Sofia hers 1.9% behind. On the bodies the owner gave as
      reference, standing at rest, it is between 1.5% behind and 2.7% in front. The three bones of the spine share a lean back that
      brings the head to HEAD_AHEAD, measured on each body in its own idle clip: the chest opens and the shoulders come back over the
      hips. The neck, the head and the arms keep their own direction. Leaning by the game's old body instead (its hips-to-neck line)
      worked on the men and sent the women further forward: their old bodies are not Rafa's.
    - On the slim man the front of the chest follows the spine alone (owner, 2026-11-11: "o peitoral está com tetas"). At rest his chest is
      flat. The skin of the chest follows the arm bone a little (4% of its weight 5 cm from the nipple, none at the nipple), and with the
      arms hanging, far from the A pose it was modelled in, the skin around the nipple sank a centimetre and left the middle standing: a
      mound 9.5 mm tall under each nipple, measured against the rim of the patch in the idle clip. With the arm's and the clavicle's
      weight taken off the front of the chest it is 0.0 mm. The weight goes to the spine and comes back towards the armpit, where the fold
      of the arm belongs. Skin and cloth alike, by where the point is. The other kits wait for the owner's word.
    - Each finger phalanx points as in the game: MakeHuman rests with the fingers half curled and spread, the
      game (Mixamo) with them straight, and the clip's curl came on top of ours. Pointing, the hand closed into a
      claw: 64 degrees of curl against 38, and the index and little finger 52 degrees apart against 18 (2026-11-08,
      measured in the Unity player on Dona Rosa). The tip (Thumb4, Index4...) has no child and follows its parent.
    - The arms move as Rafa's on the men and as Sofia's on the women (owner, 2026-11-11). The game fitted each clip to each of its old
      bodies: standing, talking and walking it
      opened the arms by an angle tuned by hand to that body (0.03 rad on Rafa, 0.05 to 0.18 on the men, another on the women), and the
      clip of each old body came with it. On our bodies that tuning is for a body that is no longer there: next to Rafa in the same clip,
      Carlos held the upper arm 3 to 6 degrees further out and the elbow 10 degrees more bent, and Lúcia, breathing, stood with the arms 20
      degrees further out, like wings. The gestures (agreeing, counting, pointing) were the same within 1 degree. So the clavicle, the arm,
      the forearm, the hand and the fingers take Rafa's turn in each clip, counted from the chest, and the rest of the body keeps its own.
      A woman takes Sofia's: in the gestures (agreeing, talking, pointing) she carries the arms 10 to 20 degrees further from the body than
      he does, and with his arms Lúcia's gestures ended 19 degrees from hers. A clip one of them lacks comes from the other.
    - Each body keeps the gap Rafa keeps between the arm and the body. Standing, his forearm and hand stay 2.6 to 3.3 cm clear of the trunk and
      the thigh, skin to skin. His motion on a body of another girth lands the arm elsewhere, and so did the game's own tuning: the muscled
      man's forearm and hand sank into the trunk (0.8 cm clear standing, 0.7 cm inside talking) and Lúcia's hand brushed the hip (0.6 cm). The
      Unity player has no step that takes the hand out of the body, so the clip has to bring it out: the arms open about the shoulder until
      the standing clips have his gap.
"""
import math

from mathutils import Quaternion, Vector

FINGERS = tuple(f"Hand{finger}{joint}" for finger in ("Thumb", "Index", "Middle", "Ring", "Pinky") for joint in (1, 2, 3))
ALIGNED = ("Shoulder", "Arm", "ForeArm", "UpLeg", "Leg") + FINGERS
SHOULDER_HEIGHT_SHARE = 0.5
# Added to the clavicle's direction towards the back (+y): about 9 degrees, 2 cm at the tip of a 14 cm clavicle.
SHOULDER_BACK = 0.16


def clavicle_as_modelled(kit):
    """Whether the clavicle of this kit rests where it was modelled when the body stands still. The men's does: the turn left over from the
    old clips made their shoulder a square mass with a groove in front. The women's keeps that turn: with the clavicle as modelled their
    breasts sat higher and flatter, and the owner liked them better before (2026-11-11)."""
    return str(kit).startswith("male")


def rest_turn(name, old_dir, new_dir, standing=None):
    """The turn that points our bone like the game's, or None when this bone keeps its own rest. `standing` is the turn the clavicle gets
    from the clip when the body it copies the arms from stands still, in the space of the skeleton, when there is such a body."""
    if name not in old_dir or name not in new_dir or not name.endswith(ALIGNED):
        return None
    wanted = old_dir[name]
    if name.endswith("Shoulder") and standing is not None:
        # Standing still, our clavicle rests where it was modelled: the clip's own turn, undone.
        wanted = standing.inverted() @ new_dir[name]
    elif name.endswith("Shoulder"):
        wanted = Vector((wanted.x, wanted.y + SHOULDER_BACK, new_dir[name].z + (wanted.z - new_dir[name].z) * SHOULDER_HEIGHT_SHARE)).normalized()
    return new_dir[name].rotation_difference(wanted)


SPINE = ("Spine", "Spine1", "Spine2")
# Where the middle of the head stands in front of the ankle joints, standing still, as a share of the height: the plumb line of a
# standing body runs from the ear to just in front of the ankle bone (Kendall, McCreary & Provance, 1993, Muscles: Testing and Function,
# 4th ed., Williams & Wilkins), about 1.5 cm on an adult. Between Rafa's and Sofia's (behind the ankle) and the reference bodies at rest.
HEAD_AHEAD = 0.008
# Past this the difference is between two rigs, not between two postures.
LEAN_MOST = math.radians(9)


def trunk_lean(ahead, reach):
    """The turn of each bone of the spine that brings the head over the ankles. `ahead` is how far in front of the ankles the middle of the
    head stands in the idle clip with no lean, and `reach` the mean height of the head over the three bones of the spine, both as shares of
    the height. The turn is about the axis that runs from side to side (z up, the front of the body at -y), counted from the bottom, so the
    last bone holds all of it. A body that already stands behind the mark is left as it is."""
    angle = max(0.0, min(LEAN_MOST, math.atan2(ahead - HEAD_AHEAD, reach)))
    return {name: Quaternion((1.0, 0.0, 0.0), -angle * (index + 1) / len(SPINE)) for index, name in enumerate(SPINE)}


ARM_ROOT = "Spine2"
ARM_PARTS = ("Shoulder", "Arm", "ForeArm")
# Rafa's file holds the few clips he does not have as copies of his idle (thinking, waving): a clip of another length is another motion.
SAME_CLIP_FRAMES = 3


def moves_with_arm(name):
    """The clavicle, the arm, the forearm, the hand and the fingers of either side."""
    return name.startswith(("Left", "Right")) and (name.endswith(ARM_PARTS) or "Hand" in name)


def same_clip(own, reference):
    """Whether Rafa has this clip: `own` and `reference` are the first and the last frame of each."""
    return abs((own[1] - own[0]) - (reference[1] - reference[0])) <= SAME_CLIP_FRAMES


def arm_turn(chest, reference_chest, reference_bone, rest_gap=None):
    """How far an arm bone of ours turns from its rest: as far as Rafa's turned from his, seen from his chest and set on our chest.
    The three are turns from rest in the space of each skeleton. `rest_gap` turns the rest direction of the game body's bone into Rafa's,
    for a body that does not rest as he does."""
    turn = chest @ reference_chest.inverted() @ reference_bone
    return turn @ rest_gap if rest_gap is not None else turn


# The gap of the standing clips, skin to skin, in metres: Rafa's own is 0.026 to 0.033 (idle, acknowledging, weight shift; 2026-11-11).
ARM_GAP_M = 0.025
# The game opened the arms of its widest body by 0.18 rad by hand; past this the body stands like a gunslinger.
ARM_OPENING_MOST = 0.2
STANDING = ("idle", "idleBreathing", "idleWeightShift", "idleFemale")
# The side of a surface is read from the face nearest to the point. Far from the body that face may look the other way; nothing sinks this deep.
INSIDE_MOST_M = 0.08
# The knuckle sits under the skin of the back of the hand, and the palm is on the other side of it: no part of the arm is thinner than this around its bone.
ARM_SKIN_LEAST_M = 0.01
# Where the arm is watched, from the elbow (0) to the wrist (1); the wrist and the knuckles come on top.
FOREARM_SPOTS = (0.0, 0.35, 0.65)


def arm_spots(elbow, wrist, knuckle):
    """The points on the centre line of a forearm and hand whose gap to the body is watched. The upper arm is left out: it always runs along the trunk."""
    return [elbow.lerp(wrist, share) for share in FOREARM_SPOTS] + [wrist, knuckle]


def skin_gap(point, body_hit, arm_distance):
    """Gap from the skin of the arm around `point` to the body: `body_hit` is (where, normal, distance) of the nearest face of the body, and
    `arm_distance` is how far the arm's own skin is from the point. Negative when the arm is inside."""
    where, normal, distance = body_hit
    outside = (point - where).dot(normal) > 0 or distance > INSIDE_MOST_M
    return (distance if outside else -distance) - max(arm_distance, ARM_SKIN_LEAST_M)


def arm_opening(gap, reach):
    """How much more the arms open, in radians, for a standing gap of `gap` metres on an arm `reach` metres long from the shoulder to the wrist."""
    return float(min(max((ARM_GAP_M - gap) / reach, 0.0), ARM_OPENING_MOST))


def opened(turn, rest, outward, radians):
    """The turn of an upper arm (a pose bone's own rotation), opened by `radians` about the axis that runs from the front to the back of the
    body. `rest` is the bone's rest in the space where the body stands upright and that axis is y; `outward` is +1 for the arm on +x, -1 for the other."""
    swing = Quaternion((0.0, 1.0, 0.0), -outward * radians)
    return rest.inverted() @ swing @ rest @ turn


# The front of the chest that follows the spine alone. Heights are shares of the way from the hip joints to the shoulder joints (the bottom
# of the ribs to over the collarbones), sides are shares of the way from the midline to the shoulder joint, and `ahead` is how far in
# front of the spine the point is, as a share of the height: the back and the flank keep the arm.
CHEST_KITS = ("male-slim",)
CHEST_LEVELS = (0.3, 0.45, 1.0, 1.12)
CHEST_SIDES = (0.55, 0.9)
CHEST_AHEAD = (0.03, 0.06)
CHEST_BONES = ("Shoulder", "Arm", "ForeArm")


def _ramp(value, low, high):
    t = max(0.0, min(1.0, (value - low) / (high - low)))
    return t * t * (3 - 2 * t)


def chest_share(level, side, ahead):
    """How much of the weight of the arm and of the clavicle a point gives to the spine: all of it on the front of the chest, none past it."""
    rise, full, top, gone = CHEST_LEVELS
    return _ramp(level, rise, full) * (1 - _ramp(level, top, gone)) * (1 - _ramp(side, *CHEST_SIDES)) * _ramp(ahead, *CHEST_AHEAD)


def chest_weights(weights, share):
    """The weights of one point, `{bone: weight}`, with `share` of what the arm and the clavicle hold given to the bones of the spine, in the
    measure each already holds it (to the top one when none does)."""
    taken = {name: weight * share for name, weight in weights.items() if name.startswith(("Left", "Right")) and name.endswith(CHEST_BONES)}
    total = sum(taken.values())
    if total <= 0:
        return dict(weights)
    spine = {name: weights.get(name, 0.0) for name in SPINE}
    held = sum(spine.values())
    out = {name: weight - taken.get(name, 0.0) for name, weight in weights.items()}
    for name in SPINE:
        out[name] = out.get(name, 0.0) + total * (spine[name] / held if held > 0 else float(name == SPINE[-1]))
    return {name: weight for name, weight in out.items() if weight > 1e-6}
