"""
    License information: data/licenses/makehuman_license.txt

    The Parle game's motion clips on a MakeHuman body, in Blender: `clips_on` takes every clip of a body of the game to our skeleton, and
    `clear_arms` keeps the arms off the finished body. What each bone does and every number are in `parle_rig`; this file reads the two
    skeletons, measures the body in its clips and writes the actions.

    A clip holds the local rotation of the bones of the game's body. Each one goes to our skeleton by the rotation of each bone in the
    space of the skeleton, counted from the rest of each, and the height of the hips follows the ratio between the two bodies.
"""
import math
from collections import Counter

import bpy
import numpy as np
from mathutils import Quaternion, Vector
from mathutils.bvhtree import BVHTree

import parle_rig

MAIN_CHILD = {"Hips": ("Spine",), "Spine2": ("Neck",), "LeftHand": ("LeftHandMiddle1",), "RightHand": ("RightHandMiddle1",)}
# Three moments of each clip let the trouble in the middle of a gesture through: Paulo's sleeve opened at 80% of `reachingOut`.
CLIP_SHARES = tuple(step / 8 for step in range(8))
# The gap of the arm is read at ten moments of each standing clip, and the opening is set in up to three rounds (`parle_rig`). With five,
# the muscled man's hand passed 0.4 cm from his shorts at a moment of `idleBreathing` that fell between two of them.
GAP_SHARES = tuple((step + 0.5) / 10 for step in range(10))
GAP_ROUNDS = 3
GAP_LEFT_OUT = ("Coif", "Acc_", "Prop_", "Human.", "Footwear")

def rests(rig):
    turn = rig.matrix_world.to_quaternion()
    return {bone.name: turn @ bone.matrix_local.to_quaternion() for bone in rig.data.bones}


def order(rig):
    done, out = set(), []
    for bone in sorted(rig.data.bones, key=lambda item: len(item.parent_recursive)):
        out.append(bone.name)
    return out


def directions(rig):
    """Where each bone points at rest: from its head to the head of its main child."""
    table = {}
    for bone in rig.data.bones:
        kids = [child for child in bone.children if child.name in MAIN_CHILD.get(bone.name, (child.name,))]
        if kids:
            table[bone.name] = (rig.matrix_world.to_3x3() @ (kids[0].head_local - bone.head_local)).normalized()
    return table


def aligned_rest(old, new, new_rest, still=None, lean=None):
    """The rest of our skeleton with each bone pointed as the game's: the two rest differently (arms and legs open at other angles), and
    without it the clip opened the legs and crossed the hands. `still` is the turn each clavicle gets from the clip when the body stands
    still, and `lean` the turn of each bone of the spine that brings the head over the ankles (`standing_lean`)."""
    old_dir, new_dir = directions(old), directions(new)
    table, turns = {}, {}
    for bone in sorted(new.data.bones, key=lambda item: len(item.parent_recursive)):
        name = bone.name
        # Which bones point as in the game, and how far, is in `parle_rig`.
        turn = (lean or {}).get(name) or parle_rig.rest_turn(name, old_dir, new_dir, (still or {}).get(name))
        turns[name] = turn if turn is not None else turns[bone.parent.name] if bone.parent and "Hand" in name else Quaternion()
        table[name] = turns[name] @ new_rest[name]
    return table


def curves(action):
    table = {}
    for curve in action.fcurves:
        if '"' not in curve.data_path:
            continue
        bone = curve.data_path.split('"')[1]
        table.setdefault(bone, {}).setdefault(curve.data_path.rsplit(".", 1)[-1], {})[curve.array_index] = curve
    return table


def clip_name(action):
    """The name of a clip without what Blender adds to it: `idle.001` from a second file and `idle_Armature` from the importer."""
    name = action.name.rsplit(".", 1)[0] if action.name.rsplit(".", 1)[-1].isdigit() else action.name
    return name.split("_")[0]


def posed(names, parents, rest, table, frame):
    """The rotation of each bone in the space of the skeleton, at that frame of the clip."""
    world = {}
    for name in names:
        rotation = table.get(name, {}).get("rotation_quaternion")
        local = Quaternion([rotation[i].evaluate(frame) if i in rotation else (1, 0, 0, 0)[i] for i in range(4)]) if rotation else Quaternion()
        parent = parents[name]
        base = world[parent] @ rest[parent].inverted() if parent else Quaternion()
        world[name] = base @ rest[name] @ local
    return world


def arm_source(path, old):
    """A body the arms copy their motion from (Rafa or Sofia, `parle_rig`): its skeleton, its rest and its clips. Its mesh leaves the scene at once."""
    known = set(bpy.data.actions)
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    lead = [obj for obj in bpy.data.objects if obj not in before]
    rig = next(obj for obj in lead if obj.type == "ARMATURE")
    theirs, ours = directions(rig), directions(old)
    source = dict(names=order(rig), parents={bone.name: bone.parent.name if bone.parent else None for bone in rig.data.bones}, rest=rests(rig),
                  gap={name: ours[name].rotation_difference(theirs[name]) for name in ours if name in theirs and parle_rig.moves_with_arm(name)},
                  actions={clip_name(action): action for action in bpy.data.actions if action not in known})
    for action in source["actions"].values():
        action.use_fake_user = True
    for obj in lead:
        bpy.data.objects.remove(obj, do_unlink=True)
    return source


def still_turns(sources):
    """The turn each clavicle gets from the clip when the body the arms are copied from stands still, in the middle of its `idle`: undone
    in the rest, our clavicle stands where it was modelled (`parle_rig.rest_turn`)."""
    lead = next((source for source in sources if "idle" in source["actions"]), None)
    if lead is None:
        return {}
    action = lead["actions"]["idle"]
    seen = posed(lead["names"], lead["parents"], lead["rest"], curves(action), sum(action.frame_range) / 2)
    return {name: seen[name] @ lead["rest"][name].inverted() @ lead["gap"].get(name, Quaternion()) for name in seen if name.endswith("Shoulder")}


def standing_lean(old, new, body, idle, old_rest, new_rest, posed_rest, ratio, arms):
    """How far the spine leans for the head to sit over the ankles (`parle_rig.trunk_lean`): the body plays the idle clip with no lean and
    the middle of the head is measured. With the MakeHuman spine the cast stooped, the head 5 cm in front of the ankles."""
    made, slot = retarget(idle, old, new, old_rest, new_rest, posed_rest, ratio, arms)
    new.animation_data.action = made
    new.animation_data.action_slot = slot
    bpy.context.scene.frame_set(int(sum(idle.frame_range) / 2))
    bpy.context.view_layer.update()
    at = lambda name: new.matrix_world @ new.pose.bones[name].head
    points = [body.matrix_world @ vertex.co for vertex in body.data.vertices]
    tall = max(point.z for point in points) - min(point.z for point in points)
    group = body.vertex_groups["Head"].index
    skull = [points[vertex.index].y for vertex in body.data.vertices if any(item.group == group and item.weight > 0.8 for item in vertex.groups)]
    # The middle of the head follows its joint: the way from one to the other is the rest's.
    middle = (min(skull) + max(skull)) / 2 - (new.matrix_world @ new.data.bones["Head"].head_local).y
    ahead = -(at("Head").y + middle - (at("LeftFoot").y + at("RightFoot").y) / 2) / tall
    reach = sum(at("Head").z - at(name).z for name in parle_rig.SPINE) / len(parle_rig.SPINE) / tall
    new.animation_data.action = None
    bpy.data.actions.remove(made)
    lean = parle_rig.trunk_lean(ahead, reach)
    turned = round(math.degrees(lean[parle_rig.SPINE[-1]].angle), 1)
    print("POSTURA cabeça à frente do tornozelo", round(ahead * 100, 2), "% da altura; a coluna inclina", turned, "graus")
    return lean


def arm_lead(sources, action):
    """Whom the arms copy this clip from: the first body of the list that has it, with the same length (`parle_rig.same_clip`)."""
    for source in sources or []:
        theirs = source["actions"].get(clip_name(action))
        if theirs is not None and parle_rig.same_clip(tuple(action.frame_range), tuple(theirs.frame_range)):
            return source
    return None


def retarget(action, old, new, old_rest, new_rest, posed_rest, hip_ratio, arms=None):
    table = curves(action)
    frames = sorted({point.co.x for curve in action.fcurves for point in curve.keyframe_points})
    old_parent = {bone.name: bone.parent.name if bone.parent else None for bone in old.data.bones}
    new_parent = {bone.name: bone.parent.name if bone.parent else None for bone in new.data.bones}
    old_names, root = order(old), parle_rig.ARM_ROOT
    # The arms move as Rafa's or Sofia's (owner, 2026-11-11): the clip of each old body came with the opening the game had tuned to it.
    arms = arm_lead(arms, action)
    lead = arms["actions"][clip_name(action)] if arms else None
    lead_table = curves(lead) if lead else None
    (first, last), (lead_first, lead_last) = action.frame_range, lead.frame_range if lead else (0, 0)
    keys = {}
    for frame in frames:
        old_world, new_world = posed(old_names, old_parent, old_rest, table, frame), {}
        at = lead_first + (frame - first) * (lead_last - lead_first) / max(last - first, 1e-6)
        seen = posed(arms["names"], arms["parents"], arms["rest"], lead_table, at) if lead else {}
        for name in order(new):
            parent = new_parent[name]
            turn = old_world[name] @ old_rest[name].inverted() if name in old_world else None
            if turn is not None and name in seen and root in seen and root in old_world and parle_rig.moves_with_arm(name):
                turn = parle_rig.arm_turn(old_world[root] @ old_rest[root].inverted(), seen[root] @ arms["rest"][root].inverted(),
                                          seen[name] @ arms["rest"][name].inverted(), arms["gap"].get(name))
            want = turn @ posed_rest[name] if turn is not None else None
            base = new_world[parent] @ new_rest[parent].inverted() if parent else Quaternion()
            new_world[name] = want if want is not None else base @ new_rest[name]
            local = new_rest[name].inverted() @ base.inverted() @ new_world[name]
            keys.setdefault(name, []).append((frame, local))
    made = bpy.data.actions.new(clip_name(action))
    made.use_fake_user = True
    slot = made.slots.new(id_type="OBJECT", name=new.name)
    layer = made.layers.new("clipe")
    strip = layer.strips.new(type="KEYFRAME")
    bag = strip.channelbag(slot, ensure=True)
    for name, samples in keys.items():
        for axis in range(4):
            curve = bag.fcurves.new(f'pose.bones["{name}"].rotation_quaternion', index=axis)
            curve.keyframe_points.add(len(samples))
            curve.keyframe_points.foreach_set("co", [value for frame, quat in samples for value in (frame, quat[axis])])
            curve.update()
    hips = table.get("Hips", {}).get("location")
    if hips:
        positions = [hip_position(old, hips, frame) for frame in frames]
        head = new.data.bones["Hips"].head_local
        to_bone = new.data.bones["Hips"].matrix_local.to_quaternion().inverted()
        local = [to_bone @ (position * hip_ratio - head) for position in positions]
        for axis in range(3):
            curve = bag.fcurves.new('pose.bones["Hips"].location', index=axis)
            curve.keyframe_points.add(len(frames))
            curve.keyframe_points.foreach_set("co", [value for frame, shift in zip(frames, local) for value in (frame, shift[axis])])
            curve.update()
    return made, slot


def hip_position(rig, hips, frame):
    """Where the hips of the game's body are at that frame: its rest stands 13 cm from the floor, and the clip is what lifts it."""
    bone = rig.data.bones["Hips"]
    shift = Vector([hips[axis].evaluate(frame) if axis in hips else 0.0 for axis in range(3)])
    return rig.matrix_world @ (bone.head_local + bone.matrix_local.to_quaternion() @ shift)


def clip_poses(rig, actions, shares=CLIP_SHARES):
    """Takes the body through every clip, at `shares` of each: what is wrong with a pose shows in the middle of a gesture, never at rest."""
    rig.animation_data.use_nla = False
    for action in actions:
        rig.animation_data.action = action
        slots = rig.animation_data.action_suitable_slots
        if slots:
            rig.animation_data.action_slot = slots[0]
        first, last = action.frame_range
        for share in shares:
            bpy.context.scene.frame_set(int(first + (last - first) * share))
            bpy.context.view_layer.update()
            yield
    rig.animation_data.action = None
    rig.animation_data.use_nla = True


def arm_parts(rig, skirted):
    """The faces of the body and the faces of the arm of each mesh: the gap is measured from the skin of the arm to the body's. A full skirt
    is left out, the hand brushes it."""
    body, arm = [], []
    for obj in bpy.data.objects:
        if obj.type != "MESH" or obj.find_armature() is not rig or obj.name.startswith(GAP_LEFT_OUT) or (skirted and obj.name.startswith("Outfit_Bottom")):
            continue
        groups = {group.index for group in obj.vertex_groups if parle_rig.moves_with_arm(group.name) and not group.name.endswith("Shoulder")}
        share = [sum(item.weight for item in vertex.groups if item.group in groups) for vertex in obj.data.vertices]
        for parts, wanted in ((body, lambda value: value < 0.3), (arm, lambda value: value > 0.7)):
            faces = {face.index for face in obj.data.polygons if all(wanted(share[index]) for index in face.vertices)}
            if len(faces) > 20:
                parts.append((obj, faces))
    return body, arm


def posed_trees(parts):
    graph = bpy.context.evaluated_depsgraph_get()
    trees = []
    for obj, faces in parts:
        shown = obj.evaluated_get(graph)
        mesh = shown.to_mesh()
        kept = [tuple(face.vertices) for face in mesh.polygons if face.index in faces]
        trees.append(BVHTree.FromPolygons([shown.matrix_world @ vertex.co for vertex in mesh.vertices], kept))
        shown.to_mesh_clear()
    return trees


def standing_gap(rig, actions, body, arm):
    """The least gap, skin to skin, between the forearm or the hand and the body in the standing clips."""
    least = None
    at = lambda name: rig.matrix_world @ rig.pose.bones[name].head
    for _ in clip_poses(rig, actions, GAP_SHARES):
        solid, skins = posed_trees(body), posed_trees(arm)
        for side in ("Left", "Right"):
            for point in parle_rig.arm_spots(at(side + "ForeArm"), at(side + "Hand"), at(side + "HandMiddle1")):
                hits = [hit for hit in (tree.find_nearest(point) for tree in solid) if hit[0] is not None]
                if not hits:
                    continue
                where, normal, _, distance = min(hits, key=lambda hit: hit[3])
                thick = min((hit[3] for hit in (tree.find_nearest(point) for tree in skins) if hit[0] is not None), default=0.0)
                gap = parle_rig.skin_gap(point, (where, normal, distance), thick)
                least = gap if least is None else min(least, gap)
    return least


def open_arms(rig, actions, radians):
    """Opens both arms in every clip, about the shoulder: the forearm and the hand go along."""
    turn = rig.matrix_world.to_quaternion()
    middle = rig.data.bones["Hips"].head_local.x
    for side in ("Left", "Right"):
        bone = rig.data.bones[side + "Arm"]
        rest, outward = turn @ bone.matrix_local.to_quaternion(), 1.0 if (turn @ (bone.head_local - Vector((middle, 0, 0)))).x > 0 else -1.0
        path = f'pose.bones["{bone.name}"].rotation_quaternion'
        for action in actions:
            found = sorted((curve for layer in action.layers for strip in layer.strips for bag in strip.channelbags for curve in bag.fcurves if curve.data_path == path),
                           key=lambda curve: curve.array_index)
            if len(found) != 4:
                continue
            for index in range(len(found[0].keyframe_points)):
                keys = [curve.keyframe_points[index] for curve in found]
                opened = parle_rig.opened(Quaternion([key.co.y for key in keys]), rest, outward, radians)
                for key, value in zip(keys, opened):
                    key.co.y = value
            for curve in found:
                curve.update()


def clear_arms(rig, actions, skirted):
    """Each body keeps the gap Rafa has between the arm and the body (`parle_rig`): with his motion on a wider body the hand went into the thigh.
    Run on the finished body: before `parle_polish.natural_body` the trunk does not have the width of its kit yet."""
    standing = [action for action in actions if clip_name(action) in parle_rig.STANDING]
    body, arm = arm_parts(rig, skirted)
    if not standing or not body or not arm:
        return
    joint = lambda name: rig.matrix_world @ rig.data.bones[name].head_local
    reach = (joint("LeftForeArm") - joint("LeftArm")).length + (joint("LeftHand") - joint("LeftForeArm")).length
    total, gap = 0.0, standing_gap(rig, standing, body, arm)
    first = gap
    for _ in range(GAP_ROUNDS):
        more = parle_rig.arm_opening(gap, reach) if gap is not None else 0.0
        more = min(more, parle_rig.ARM_OPENING_MOST - total)
        if more < 0.004:
            break
        open_arms(rig, actions, more)
        total += more
        gap = standing_gap(rig, standing, body, arm)
    print("BRAÇO vão de pé", None if first is None else round(first, 4), "->", None if gap is None else round(gap, 4), "abertura", round(total, 3), "rad")


def lead_files(text):
    """`rafa=<file>,sofia=<file>` as a table: the bodies the arms may copy their motion from."""
    return dict(item.split("=", 1) for item in text.split(",") if "=" in item)


def clips_on(old, new, body, actions, kit, leads=None):
    """Every clip in `actions`, of the game's body `old`, on our skeleton `new`: one action each, on its own NLA track, and the list of them.
    `body` is our skin, `kit` the kit of the body, and `leads` the files of the bodies the arms copy their motion from, by name; which of
    them a kit copies first is in `parle_rig.lead_order`."""
    leads = leads or {}
    arms = [arm_source(leads[name], old) for name in parle_rig.lead_order(kit) if leads.get(name)]
    old_rest, new_rest = rests(old), rests(new)
    idle = next((action for action in actions if action.name.startswith("idle")), actions[0])
    standing = hip_position(old, curves(idle)["Hips"]["location"], idle.frame_range[0]).z
    # MakeHuman leaves the sole under the origin: measured to the origin only, the hips came down and the foot went 13 cm into the floor.
    grounded = [body, *(obj for obj in new.children if obj.name.startswith("Footwear"))]
    sole = min((new.matrix_world.inverted() @ obj.matrix_world @ vertex.co).z for obj in grounded for vertex in obj.data.vertices)
    ratio = (new.data.bones["Hips"].head_local.z - sole) / standing
    new.animation_data_create()
    # The clavicle standing where it was modelled is for the men only: on the women the breasts sat higher and flatter (`parle_rig`).
    still = still_turns(arms) if parle_rig.clavicle_as_modelled(kit) else None
    lean = standing_lean(old, new, body, idle, old_rest, new_rest, aligned_rest(old, new, new_rest, still), ratio, arms)
    posed_rest = aligned_rest(old, new, new_rest, still, lean)
    made_actions = []
    for action in actions:
        made, slot = retarget(action, old, new, old_rest, new_rest, posed_rest, ratio, arms)
        made_actions.append(made)
        track = new.animation_data.nla_tracks.new()
        track.name = action.name.split("_")[0]
        strip = track.strips.new(made.name, int(min(curve.range()[0] for curve in action.fcurves)), made)
        strip.action_slot = slot
    taken = Counter(arms.index(lead) for lead in (arm_lead(arms, action) for action in actions) if lead is not None)
    print("CLIPES", len(actions), "razão do quadril", round(ratio, 3), "braço copiado em", [taken[index] for index in range(len(arms))])
    return made_actions
