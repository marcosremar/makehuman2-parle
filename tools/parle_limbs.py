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
      to it like a cone. The men's were right: 17.7 cm at the wrist, 25 at the belly of the forearm. `natural_forearms` gives every arm
      at least the men's taper on a wrist of `WRIST_SHARE` of the height, so a man's arm stays as it was and a woman's forearm fills out.
"""
import bpy
import mathutils
import numpy as np

# The girth of the wrist as a share of the height, at the least: a man's is the ANSUR II median (176 mm on 1756 mm), a woman's the 5th
# percentile (142 mm on 1628 mm), slimmer than most women as the drawn cast is, but no stick.
WRIST_SHARE = {"female": 0.087, "male": 0.100}
# The girth down the forearm over the girth at the wrist, from the elbow (0) to the wrist (1): Seu Jorge's, the cast's men.
TAPER = ((0.0, 1.34), (0.1, 1.37), (0.3, 1.42), (0.5, 1.38), (0.7, 1.21), (0.9, 1.05), (1.0, 1.0))
FOREARM_GROW_MOST = 1.45
# Into the palm the growth of the wrist fades out over this much, so the hand meets the wrist with no step and the fingers do not move.
WRIST_FADE_M = 0.035
SECTION_REACH_M = 0.09
ARM_BONES = ("Arm", "ForeArm", "Hand")
# Upper bone, lower bone, helper: the helper turns HELPER_SHARE of the lower bone's local turn.
HELPERS = (("Arm", "ForeArm", "ElbowHelper"), ("ForeArm", "Hand", "WristHelper"))
HELPER_SHARE = 0.5


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
    """How much the forearm grows across its bone at each share of it, from the girths measured at the shares of `TAPER`: never less than
    1, never more than FOREARM_GROW_MOST."""
    wrist = WRIST_SHARE[sex] * height
    grow = [max(1.0, wrist * ratio / girth) if girth > 0 else 1.0 for (_, ratio), girth in zip(TAPER, girths)]
    return np.clip(grow, 1.0, FOREARM_GROW_MOST)


def growth_along(along, shares, grow, fade):
    """The growth at each point by its share of the forearm (`along`, 0 at the elbow, 1 at the wrist): none above the elbow, the wrist's
    fading to none `fade` (a share of the forearm) into the hand."""
    out = np.interp(np.clip(along, 0.0, 1.0), shares, grow)
    past = np.clip((along - 1.0) / fade, 0.0, 1.0)
    out = 1.0 + (out - 1.0) * (1.0 - past * past * (3 - 2 * past))
    return np.where(along < 0.0, 1.0, out)


def arm_share(obj, side):
    groups = {group.index for group in obj.vertex_groups if group.name in [side + name for name in ARM_BONES]}
    share = np.zeros(len(obj.data.vertices))
    for vertex in obj.data.vertices if groups else ():
        share[vertex.index] = sum(item.weight for item in vertex.groups if item.group in groups)
    return np.clip(share * 2, 0, 1)


def natural_forearms(meshes, rig, height, sex, warp, rest_points):
    """Run at rest, the bones in their final place. The skin's forearm is measured across its bone; skin and cloth then grow out from the
    bone by `forearm_growth`, as much as each point follows the arm. The bones do not move. Gives the growth of each side."""
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
            return (growth_along(along, shares, grow, fade) - 1.0)[:, None] * radial

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
