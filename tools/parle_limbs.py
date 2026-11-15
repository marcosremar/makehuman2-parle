"""
    License information: data/licenses/makehuman_license.txt

    The arms of the Parle cast, as near perfect as the game's skinning lets them be (owner, 2026-11-15: "melhora os braços para serem
    os mais perfeitos possível").

    Two things were wrong, both measured on the cast in every clip (babylon-cinema out/cabeca-desenho/src/braco_medida.py, sections of
    the skin across the bone):

    - The joints shrank. The player blends the turns of two bones linearly (no dual quaternions in Unity), and the skin that follows two
      bones loses girth in the middle of a bend: Seu Jorge's wrist lost 14.7% of its girth bent back (`shrugging`) and his elbow 19.8%
      bent shut (`terrified`), the wrist pinched to a waist with a crease across it. Spreading the weights wider or narrower did not help
      (two, three and a half and five centimetres either side of the wrist: 17 to 19%; the elbow 20 to 25%). What does is a helper bone
      at each joint (`helpers`): child of the bone above, at rest where the bone below is, and turning in every frame half of the bone
      below's turn. The skin in the middle of the bend goes to it, and each pair of neighbours only blends half a turn: the wrist then
      lost 4.8% at worst, the elbow 13.5%, and the crease went. The clips carry the helper's keys, worked out from the bone below's, so
      the player needs no code: whatever plays a clip plays the helper.
    - The women's forearms were sticks. Their wrists went 10.1 to 10.6 cm round, where the slimmest woman in twenty measured by ANSUR II
      goes 14.2 (Gordon et al. 2014, NATICK/TR-15/007; 5th percentile, median 15.4, men 17.6), and the forearm tapered from the elbow
      to it like a cone; the men's went down like a tube. `natural_forearms` gives every forearm the taper the owner picked (`TAPER`) on a
      wrist of `WRIST_SHARE` of the height: a woman's fills out near the hand, a man's narrows.
"""
import os

import bpy
import mathutils
import numpy as np

# The girth of the forearm, from the elbow (0) to the wrist (1), over the girth at the wrist, and the wrist as a share of the height.
# First the forearm took the men's own taper on a wrist of the ANSUR II median (176 mm on 1756 mm) and of a woman's 5th percentile
# (142 mm on 1628 mm): the women's filled out, the men's stayed. Seen on the cast the owner said "os antebraços ainda estão muito
# gordos, melhora os braços em geral" (2026-11-15): the forearm went down to the wrist like a tube. He picked, of three, the one whose
# forearm narrows from the elbow to a slimmer wrist (sheet 131, "B"): 8.8% of the height at a man's wrist (15.4 cm on 1.75 m) and 8% at
# a woman's (13 cm on 1.62 m), slimmer than most people as the drawn cast is, with the belly of the forearm a tenth of the way down.
WRIST_SHARE = {"female": 0.080, "male": 0.088}
TAPER = ((0.0, 1.50), (0.1, 1.55), (0.3, 1.50), (0.5, 1.36), (0.7, 1.20), (0.9, 1.05), (1.0, 1.0))
FOREARM_GROW_MOST = 1.45
FOREARM_GROW_LEAST = 0.75
# Above the elbow the growth fades out over this much, so the upper arm meets the forearm with no step.
ELBOW_FADE_M = 0.04
# Into the palm the growth of the wrist fades out over this much, so the hand meets the wrist with no step and the fingers do not move.
WRIST_FADE_M = 0.035
SECTION_REACH_M = 0.09
ARM_BONES = ("Arm", "ForeArm", "Hand")
# Upper bone, lower bone, helper: the helper turns HELPER_SHARE of the lower bone's local turn. The shoulder took one too (2026-11-15,
# "melhora o corpo do avatar"): with the arm raised or forward, the armpit lost 13 to 20% of its girth (Dona Rosa pointing, the gym man
# jogging) and showed a crease behind it; the skin there is on the collarbone and the arm, as at the other joints.
HELPERS = (("Shoulder", "Arm", "ShoulderHelper"), ("Arm", "ForeArm", "ElbowHelper"), ("ForeArm", "Hand", "WristHelper"))
HELPER_SHARE = 0.5
# After the split, the weights near each joint are evened out over the mesh (`even_joint`): this many rounds, within this far of the
# joint. Sharp steps of weight between neighbours folded faces outwards in a deep bend: on the strong man of the gym, bits of the
# inner elbow stood out of the skin bent shut (41 faces of the elbow crossing others at worst, 629 in all the clips); evened out, 21
# and 337, and nothing stands out. Points at the same place (the seam of the hand) are one point: evened apart, they parted, and the
# wrist showed a ring like a bracelet.
EVEN_ROUNDS = 12
EVEN_REACH_M = {"ShoulderHelper": 0.09, "ElbowHelper": 0.10, "WristHelper": 0.07}
SAME_PLACE_M = 1e-4


def hull_perimeter(flat):
    """The length round the convex hull of 2D points (Andrew's monotone chain)."""
    points = sorted(map(tuple, np.round(flat, 7)))
    if len(points) < 3:
        return 0.0

    def chain(seq):
        out = []
        for point in seq:
            while len(out) >= 2 and (out[-1][0] - out[-2][0]) * (point[1] - out[-2][1]) - (out[-1][1] - out[-2][1]) * (point[0] - out[-2][0]) <= 0:
                out.pop()
            out.append(point)
        return out

    hull = chain(points)[:-1] + chain(points[::-1])[:-1]
    ring = np.array(hull + hull[:1])
    return float(np.linalg.norm(np.diff(ring, axis=0), axis=1).sum())


def section_girth(points, edges, centre, axis, reach=SECTION_REACH_M):
    """The girth of a mesh across `axis` at `centre`: where its edges cross the plane, round their hull."""
    a, b = points[edges[:, 0]], points[edges[:, 1]]
    da, db = (a - centre) @ axis, (b - centre) @ axis
    cut = da * db < 0
    if not cut.any():
        return 0.0
    at = a[cut] + (b[cut] - a[cut]) * (da[cut] / (da[cut] - db[cut]))[:, None]
    off = at - centre
    off = off - np.outer(off @ axis, axis)
    off = off[np.linalg.norm(off, axis=1) < reach]
    if len(off) < 3:
        return 0.0
    side = np.cross(axis, [0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.cross(axis, [1.0, 0.0, 0.0])
    side /= np.linalg.norm(side)
    other = np.cross(axis, side)
    return hull_perimeter(np.stack([off @ side, off @ other], axis=1))


def forearm_growth(girths, height, sex):
    """How much the forearm grows (or thins) across its bone at each share of it, to the girths of `TAPER` on a wrist of `WRIST_SHARE`
    of the height: between FOREARM_GROW_LEAST and FOREARM_GROW_MOST."""
    wrist = WRIST_SHARE[sex] * height
    grow = [wrist * ratio / girth if girth > 0 else 1.0 for (_, ratio), girth in zip(TAPER, girths)]
    return np.clip(grow, FOREARM_GROW_LEAST, FOREARM_GROW_MOST)


def growth_along(along, shares, grow, fade, elbow_fade=None):
    """The growth at each point by its share of the forearm (`along`, 0 at the elbow, 1 at the wrist): the elbow's fading to none
    `elbow_fade` (a share of the forearm) up the upper arm, the wrist's fading to none `fade` into the hand."""
    out = np.interp(np.clip(along, 0.0, 1.0), shares, grow)
    past = np.clip((along - 1.0) / fade, 0.0, 1.0)
    out = 1.0 + (out - 1.0) * (1.0 - past * past * (3 - 2 * past))
    if not elbow_fade:
        return np.where(along < 0.0, 1.0, out)
    above = np.clip(-along / elbow_fade, 0.0, 1.0)
    return 1.0 + (out - 1.0) * (1.0 - above * above * (3 - 2 * above))


def arm_share(obj, side):
    groups = {group.index for group in obj.vertex_groups if group.name in [side + name for name in ARM_BONES]}
    share = np.zeros(len(obj.data.vertices))
    for vertex in obj.data.vertices if groups else ():
        share[vertex.index] = sum(item.weight for item in vertex.groups if item.group in groups)
    return np.clip(share * 2, 0, 1)


# Trial door (like PARLE_HEAD): PARLE_FOREARM="<woman's wrist share>,<man's wrist share>,<least growth>,<taper>" with the taper
# "fine" (TAPER) or "men" (TAPER_MEN, the cast's men's own: the first try), to build the same body with another forearm.
TRIAL = "PARLE_FOREARM"
TAPER_MEN = ((0.0, 1.34), (0.1, 1.37), (0.3, 1.42), (0.5, 1.38), (0.7, 1.21), (0.9, 1.05), (1.0, 1.0))


def trial():
    global TAPER, FOREARM_GROW_LEAST
    text = os.environ.get(TRIAL)
    if not text:
        return
    woman, man, least, taper = text.split(",")
    WRIST_SHARE.update(female=float(woman), male=float(man))
    FOREARM_GROW_LEAST, TAPER = float(least), {"men": TAPER_MEN, "fine": TAPER}[taper]


def natural_forearms(meshes, rig, height, sex, warp, rest_points):
    """Run at rest, the bones in their final place. The skin's forearm is measured across its bone; skin and cloth then grow out from the
    bone by `forearm_growth`, as much as each point follows the arm. The bones do not move. Gives the growth of each side."""
    trial()
    body = next((obj for obj in meshes if obj.name == "Body"), None)
    if body is None:
        return {}
    bpy.context.view_layer.update()
    joint = lambda name: np.array(rig.matrix_world @ rig.data.bones[name].head_local)
    skin = rest_points(body)
    shares, done = np.array([share for share, _ in TAPER]), {}
    for side in ("Left", "Right"):
        elbow, wrist = joint(side + "ForeArm"), joint(side + "Hand")
        length = float(np.linalg.norm(wrist - elbow))
        axis = (wrist - elbow) / length
        on_arm = arm_share(body, side) > 0.99
        edges = np.array([edge.vertices[:] for edge in body.data.edges if on_arm[edge.vertices[0]] and on_arm[edge.vertices[1]]], int).reshape(-1, 2)
        girths = [section_girth(skin, edges, elbow + axis * length * share, axis) for share in shares]
        grow = forearm_growth(girths, height, sex)
        done[side] = {"girth_cm": [round(g * 100, 1) for g in girths], "grow": [round(float(g), 3) for g in grow]}
        if np.allclose(grow, 1.0):
            continue
        fade = WRIST_FADE_M / length

        def field(points, elbow=elbow, axis=axis, length=length, grow=grow, fade=fade):
            off = points - elbow
            along = off @ axis / length
            radial = off - np.outer(off @ axis, axis)
            return (growth_along(along, shares, grow, fade, ELBOW_FADE_M / length) - 1.0)[:, None] * radial

        for obj in meshes:
            if not obj.vertex_groups:
                continue
            share = arm_share(obj, side)
            if share.any():
                warp(obj, lambda points, share=share, field=field: points + share[:, None] * field(points))
    return done


def split_weight(upper, lower):
    """The weights of one point on the upper bone, the lower bone and the helper: the helper takes the middle of the blend."""
    total = upper + lower
    if upper <= 0 or lower <= 0:
        return upper, lower, 0.0
    share = lower / total
    return total * max(0.0, 1 - 2 * share), total * max(0.0, 2 * share - 1), total * (1 - abs(2 * share - 1))


def helper_turn(quaternion, share=HELPER_SHARE):
    """The helper's local turn for the lower bone's: `share` of it, by slerp from rest."""
    return mathutils.Quaternion().slerp(mathutils.Quaternion(quaternion).normalized(), share)


def action_curves(action):
    try:
        return list(action.fcurves)
    except AttributeError:
        return [curve for layer in action.layers for strip in layer.strips for slot in action.slots
                for curve in (strip.channelbag(slot).fcurves if strip.channelbag(slot) else ())]


def new_curve(action, path, index, group):
    try:
        return action.fcurves.new(path, index=index, action_group=group)
    except AttributeError:
        bag = action.layers[0].strips[0].channelbag(action.slots[0], ensure=True)
        return bag.fcurves.new(path, index=index, group_name=group)


def even_joint(obj, joint, names, rounds=EVEN_ROUNDS, reach=0.1):
    """The shares of `names` (upper, helper, lower) in each point within `reach` of `joint` (world) made the mean of their neighbours',
    half a step a round, the sum each point holds on them kept. Points at the same place move as one."""
    if any(name not in obj.vertex_groups for name in names) or not len(obj.data.vertices):
        return 0
    mw = np.array(obj.matrix_world)
    rest = np.array([v.co[:] for v in obj.data.vertices]) @ mw[:3, :3].T + mw[:3, 3]
    index = [obj.vertex_groups[name].index for name in names]
    weights = np.zeros((len(rest), 3))
    for vertex in obj.data.vertices:
        for item in vertex.groups:
            if item.group in index:
                weights[vertex.index, index.index(item.group)] = item.weight
    total = weights.sum(axis=1)
    place, group = {}, np.empty(len(rest), int)
    for number, point in enumerate(rest):
        group[number] = place.setdefault(tuple(np.round(point / SAME_PLACE_M).astype(int)), len(place))
    count_places = len(place)
    edges = np.array([edge.vertices[:] for edge in obj.data.edges], int).reshape(-1, 2)
    edges = np.unique(np.sort(group[edges], axis=1), axis=0)
    edges = edges[edges[:, 0] != edges[:, 1]]
    members = np.bincount(group, minlength=count_places).astype(float)
    held = np.bincount(group, weights=total, minlength=count_places)
    shares = np.stack([np.bincount(group, weights=weights[:, k], minlength=count_places) for k in range(3)], axis=1) / np.maximum(held, 1e-9)[:, None]
    near_joint = (np.linalg.norm(rest - np.asarray(joint), axis=1) < reach) & (total > 0.05)
    free = np.bincount(group, weights=near_joint.astype(float), minlength=count_places) > 0
    if not free.any() or not len(edges):
        return 0
    count = np.bincount(edges.ravel(), minlength=count_places).astype(float)
    for _ in range(rounds):
        around = np.stack([np.bincount(edges[:, 0], weights=shares[edges[:, 1], k], minlength=count_places)
                           + np.bincount(edges[:, 1], weights=shares[edges[:, 0], k], minlength=count_places) for k in range(3)], axis=1)
        around /= np.maximum(count[:, None], 1)
        shares = np.where(free[:, None], 0.5 * shares + 0.5 * around, shares)
    moved = 0
    for number in np.where(free[group] & (total > 0))[0]:
        moved += 1
        for k, name in enumerate(names):
            weight = float(shares[group[number], k] * total[number])
            if weight > 1e-5:
                obj.vertex_groups[name].add([int(number)], weight, "REPLACE")
            else:
                obj.vertex_groups[name].remove([int(number)])
    return moved


def helpers(rig, meshes, actions):
    """Adds the helper bones of the elbows and wrists, gives them the middle of each joint's blend in every mesh, and keys them in every
    clip. Run last, once the clips are final: the keys are worked out from the lower bone's. Gives the names of the helpers."""
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    made = []
    for side in ("Left", "Right"):
        for upper, lower, helper in HELPERS:
            up, low = rig.data.edit_bones.get(side + upper), rig.data.edit_bones.get(side + lower)
            if up is None or low is None or side + helper in rig.data.edit_bones:
                continue
            bone = rig.data.edit_bones.new(side + helper)
            bone.head, bone.tail, bone.roll = low.head.copy(), low.tail.copy(), low.roll
            bone.parent, bone.use_connect, bone.use_deform = up, False, True
            made.append((side + upper, side + lower, side + helper))
    bpy.ops.object.mode_set(mode="OBJECT")
    for _, _, helper in made:
        rig.pose.bones[helper].rotation_mode = "QUATERNION"
    for obj in meshes:
        if not obj.vertex_groups:
            continue
        for upper, lower, helper in made:
            if upper not in obj.vertex_groups or lower not in obj.vertex_groups:
                continue
            gu, gl = obj.vertex_groups[upper], obj.vertex_groups[lower]
            gh = obj.vertex_groups.get(helper) or obj.vertex_groups.new(name=helper)
            for vertex in obj.data.vertices:
                own = {item.group: item.weight for item in vertex.groups}
                a, b = own.get(gu.index, 0.0), own.get(gl.index, 0.0)
                if a <= 0 or b <= 0:
                    continue
                for group, weight in zip((gu, gl, gh), split_weight(a, b)):
                    if weight > 1e-6:
                        group.add([vertex.index], weight, "REPLACE")
                    else:
                        group.remove([vertex.index])
            joint = rig.matrix_world @ rig.data.bones[lower].head_local
            even_joint(obj, joint, (upper, helper, lower), reach=EVEN_REACH_M[helper.removeprefix("Left").removeprefix("Right")])
    for action in actions:
        curves = action_curves(action)
        for _, lower, helper in made:
            turn = {c.array_index: c for c in curves if c.data_path == f'pose.bones["{lower}"].rotation_quaternion'}
            if len(turn) < 4:
                continue
            frames = sorted({round(key.co[0], 4) for curve in turn.values() for key in curve.keyframe_points})
            out = [new_curve(action, f'pose.bones["{helper}"].rotation_quaternion', index, helper) for index in range(4)]
            for curve in out:
                curve.keyframe_points.add(len(frames))
            for number, frame in enumerate(frames):
                half = helper_turn([turn[index].evaluate(frame) for index in range(4)])
                for index in range(4):
                    out[index].keyframe_points[number].co = (frame, half[index])
                    out[index].keyframe_points[number].interpolation = "LINEAR"
            for curve in out:
                curve.update()
            for kind in ("location", "scale"):
                for curve in [c for c in curves if c.data_path == f'pose.bones["{lower}"].{kind}']:
                    copy = new_curve(action, f'pose.bones["{helper}"].{kind}', curve.array_index, helper)
                    copy.keyframe_points.add(len(curve.keyframe_points))
                    for number, key in enumerate(curve.keyframe_points):
                        copy.keyframe_points[number].co = tuple(key.co)
                        copy.keyframe_points[number].interpolation = key.interpolation
                    copy.update()
    bpy.context.view_layer.update()
    return [helper for _, _, helper in made]
