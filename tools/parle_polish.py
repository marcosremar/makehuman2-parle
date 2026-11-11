"""
    License information: data/licenses/makehuman_license.txt

    Finishing touches of the Parle cast's look, run inside Blender on a mounted character, what was judged an improvement
    next to the player and Rafa (owner, 2026-11-08). Every shape change moves all shape keys, so blinking and speaking keep working.
"""
import glob
import os

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

import parle_build
import parle_bust
import parle_rig

CLOTHES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".local/mhd/data/clothes/hm08")

# 12% with a staring look all round; 10% reads as awake next to the player.
EYE_SCALE = 1.1
EYE_CORE_M = 0.017
EYE_REACH_M = 0.032
NORMAL_PX = 512
# Arms (owner, 2026-11-10), as a share of the height (body without hair or shoes) from joint centre to joint centre. The yardstick that hair, hat and
# heels do not move is the arm against the leg (upper arm + forearm over thigh + shank): 0.58, 0.58 and 0.60 on the three rigged whole-body models the
# owner gave as reference, 0.51 to 0.56 on the cast, whose forearm was short in both sexes (12.4 to 12.9% of the height) and whose upper arm was short
# in the women (13.6%). De Leva (1996, Journal of Biomechanics 29(9), 1223-1230, DOI 10.1016/0021-9290(95)00178-6) measures between joint centres
# 16.2 and 15.4% in men and 15.9 and 15.2% in women, 0.64 of the leg in men. The shares below sit between the references and that measure and put the
# cast at 0.58 of the leg. One bone stretches 16% at most, which is what the cast needs.
ARM_UPPER_SHARE = {"female": 0.157, "male": 0.162}
ARM_FORE_SHARE = {"female": 0.142, "male": 0.148}
ARM_SCALE_LIMITS = (0.94, 1.16)
ARM_STILL_SHARE = 0.1
ARM_BONES = ("Arm", "ForeArm", "Hand")
# The top hangs outside the bottom. The build of the kit moves the trunk one way and the legs another, and the two meet at the hip, where
# the hem of a shirt is: after it the front of the hem of Carlos's shirt stood up to 11 mm inside his trousers, and the waistband showed
# through the plaid in two slits beside the last button (2026-11-11).
OVER_GAP_M = 0.005
OVER_REACH_M = 0.03
# Cloth goes the way of the skin under it. The build moves a point by where it is and by how much of it follows each arm, and a garment
# has weights of its own: where its share of the arm is not the skin's beneath, the two part. It showed once the shoulders of the slim
# woman came in while her chest stayed (2026-11-11): at the back of the armhole the shirt went a centimetre further in than the skin,
# and the skin of the trunk showed under the sleeve when the arm swung back. Lying on the skin, a point of cloth takes the skin's share;
# from SKIN_FAR_M on it keeps its own, as a loose sleeve does.
SKIN_NEAR_M = 0.015
SKIN_FAR_M = 0.04
OVER_STEP_M = 0.01
OVER_ROUNDS = 6
OVER_SHARED = 2


def smooth(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def read(block):
    """Points of a shape key, or of `mesh.vertices`."""
    items = block.data if hasattr(block, "relative_key") else block
    points = np.empty(len(items) * 3)
    items.foreach_get("co", points)
    return points.reshape(-1, 3)


def to_world(obj, points):
    matrix = np.array(obj.matrix_world)
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def to_local(obj, points):
    matrix = np.array(obj.matrix_world.inverted())
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def rest_points(obj):
    return to_world(obj, read(obj.data.vertices))


def warp(obj, field):
    """`field` takes world points; every key is absolute, so a stretched lid still closes all the way."""
    blocks = obj.data.shape_keys.key_blocks if obj.data.shape_keys else []
    for block in blocks:
        block.data.foreach_set("co", to_local(obj, field(to_world(obj, read(block)))).ravel())
    obj.data.vertices.foreach_set("co", to_local(obj, field(rest_points(obj))).ravel())
    obj.data.update()


def eye_radius(eye):
    points = rest_points(eye)
    return float(np.ptp(points[points[:, 0] < points[:, 0].mean(), 0]) / 2)


def eye_centres(eye):
    points = rest_points(eye)
    middle = points[:, 0].mean()
    centres = []
    for side in (points[:, 0] < middle, points[:, 0] >= middle):
        part = points[side]
        radius = np.ptp(part[:, 0]) / 2
        centres.append(np.array([(part[:, 0].min() + part[:, 0].max()) / 2, part[:, 1].min() + radius, (part[:, 2].min() + part[:, 2].max()) / 2]))
    return centres


def bigger_eyes(meshes, eye, scale=EYE_SCALE):
    """Globe and lids grow together around each eye, fading out before the nose and the temple."""
    centres = eye_centres(eye)

    def field(points):
        out = points.copy()
        for centre in centres:
            away = points - centre
            weight = smooth((EYE_REACH_M - np.linalg.norm(away, axis=1)) / (EYE_REACH_M - EYE_CORE_M))
            out += away * (scale - 1) * weight[:, None]
        return out

    for obj in meshes:
        warp(obj, field)


def arm_scales(height, upper_m, fore_m, sex):
    """How much the upper arm and the forearm stretch to reach the natural share of this height."""
    low, high = ARM_SCALE_LIMITS
    return float(np.clip(ARM_UPPER_SHARE[sex] * height / upper_m, low, high)), float(np.clip(ARM_FORE_SHARE[sex] * height / fore_m, low, high))


def arm_shift(joint, elbow, wrist, upper, fore):
    """Takes world points, gives how far each moves: nothing at the shoulder, an even stretch down each bone, the hand carried whole."""
    upper_way, fore_way = elbow - joint, wrist - elbow
    upper_m, fore_m = np.linalg.norm(upper_way), np.linalg.norm(fore_way)

    def shift(points):
        on_upper = np.clip(((points - joint) @ upper_way / upper_m**2 - ARM_STILL_SHARE) / (1 - ARM_STILL_SHARE), 0, 1)
        on_fore = np.clip((points - elbow) @ fore_way / fore_m**2, 0, 1)
        return np.outer(on_upper, (upper - 1) * upper_way) + np.outer(on_fore, (fore - 1) * fore_way)

    return shift


def arm_share(obj, side):
    """How much of each vertex follows that arm, from its weights: the flank beside a hanging arm stays, the sleeve goes."""
    groups = {group.index for group in obj.vertex_groups if group.name.startswith(side) and any(part in group.name for part in ARM_BONES)}
    share = np.zeros(len(obj.data.vertices))
    for vertex in obj.data.vertices if groups else ():
        share[vertex.index] = sum(item.weight for item in vertex.groups if item.group in groups)
    return np.clip(share * 2, 0, 1)


def chest_follows_spine(meshes, rig, kit):
    """On the kits of `parle_rig.CHEST_KITS`, the front of the chest gives the weight of the arm and of the clavicle to the spine
    (`parle_rig.chest_share`): skin and cloth alike, by where each point is. Run before anything reads the poses of the body."""
    if str(kit) not in parle_rig.CHEST_KITS:
        return 0
    bpy.context.view_layer.update()
    joint = lambda name: np.array(rig.matrix_world @ rig.data.bones[name].head_local)
    shoulder, hip = (joint("LeftArm") + joint("RightArm")) / 2, (joint("LeftUpLeg") + joint("RightUpLeg")) / 2
    half = abs(joint("LeftArm")[0] - shoulder[0])
    spine = np.array(sorted((joint(name) for name in ("Hips", "Spine", "Spine1", "Spine2", "Neck")), key=lambda point: point[2]))
    height = joint("Head")[2] * 1.08 - min(joint("LeftToeBase")[2], joint("RightToeBase")[2])
    changed = 0
    for obj in meshes:
        if not obj.vertex_groups:
            continue
        names = [group.name for group in obj.vertex_groups]
        points = rest_points(obj)
        level = (points[:, 2] - hip[2]) / (shoulder[2] - hip[2])
        side = np.abs(points[:, 0] - shoulder[0]) / half
        ahead = (np.interp(points[:, 2], spine[:, 2], spine[:, 1]) - points[:, 1]) / height
        for vertex in obj.data.vertices:
            share = parle_rig.chest_share(level[vertex.index], side[vertex.index], ahead[vertex.index])
            if share <= 0:
                continue
            before = {names[item.group]: item.weight for item in vertex.groups}
            after = parle_rig.chest_weights(before, share)
            if after == before:
                continue
            changed += 1
            for name in set(before) | set(after):
                if name not in after:
                    obj.vertex_groups[name].remove([vertex.index])
                elif name in names:
                    obj.vertex_groups[name].add([vertex.index], after[name], "REPLACE")
                else:
                    obj.vertex_groups.new(name=name).add([vertex.index], after[name], "REPLACE")
                    names.append(name)
    return changed


def natural_arms(meshes, rig, height, sex):
    """Run on a finished character, in its final size. Skin, cloth and bones move by the same field, and the clips only turn the bones, so every clip still plays."""
    bpy.context.view_layer.update()
    world, back = np.array(rig.matrix_world), np.array(rig.matrix_world.inverted())
    joint = lambda name: np.array(rig.matrix_world @ rig.data.bones[name].head_local)
    shifts = {}
    for side in ("Left", "Right"):
        ends = [joint(side + part) for part in ARM_BONES]
        shifts[side] = arm_shift(*ends, *arm_scales(height, np.linalg.norm(ends[1] - ends[0]), np.linalg.norm(ends[2] - ends[1]), sex))
    for obj in meshes:
        shares = {side: arm_share(obj, side) for side in shifts}
        if not any(share.any() for share in shares.values()):
            continue
        warp(obj, lambda points: points + sum(shares[side][:, None] * shifts[side](points) for side in shifts))
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    for side, shift in shifts.items():
        upper = rig.data.edit_bones[side + "Arm"]
        for bone, ends in [(upper, ("tail",))] + [(child, ("head", "tail")) for child in upper.children_recursive]:
            for end in ends:
                point = np.array(getattr(bone, end)) @ world[:3, :3].T + world[:3, 3]
                setattr(bone, end, (point + shift(point[None])[0]) @ back[:3, :3].T + back[:3, 3])
    bpy.ops.object.mode_set(mode="OBJECT")


def group_share(obj, names):
    """How much of each vertex follows those bones, from its weights."""
    groups = {group.index for group in obj.vertex_groups if group.name in names}
    share = np.zeros(len(obj.data.vertices))
    for vertex in obj.data.vertices if groups else ():
        share[vertex.index] = sum(item.weight for item in vertex.groups if item.group in groups)
    return np.clip(share, 0, 1)


def over_skin(obj, own, skin, skin_arms):
    """The share of each arm for the points of a garment: the skin's where the cloth lies on it, its own away from it (SKIN_NEAR_M,
    SKIN_FAR_M). `skin` is the tree of the points of the skin at rest and `skin_arms` their share of each arm."""
    found = [skin.find(Vector(point)) for point in rest_points(obj)]
    # A piece with weights but no points (the short hair of the cast) gives empty lists, which numpy would read as floats.
    nearest, away = np.array([index for _, index, _ in found], dtype=int), np.array([distance for _, _, distance in found], dtype=float)
    return {side: parle_build.share_over_skin(own[side], skin_arms[side][nearest], away, SKIN_NEAR_M, SKIN_FAR_M) for side in own}


def natural_build(meshes, rig, height, kit):
    """The build of the kit's profile (`parle_build`): trunk, arms and legs get the widths found against a reference body. Run last, after the arms.
    The head keeps its shape, the spine stays, and each arm and leg goes as a whole with its bones, so every clip still plays. Cloth goes
    the way of the skin under it (`over_skin`). On the kits whose head grows against the body (`parle_build.head_ratio`), its bones grow
    with it and the body comes back to the stature it had. No profile, no change."""
    profile = parle_build.load_profile(kit)
    if profile is None:
        return False
    bpy.context.view_layer.update()
    back = np.array(rig.matrix_world.inverted())[:3, :3]
    joints = {name: np.array(rig.matrix_world @ rig.data.bones[name].head_local) for name in parle_build.BONES}
    field, rides = parle_build.build_field(profile, joints, height)
    head = {"Head"} | {bone.name for bone in rig.data.bones["Head"].children_recursive}
    body = next((obj for obj in meshes if obj.name.split(".")[0] == "Body" and obj.vertex_groups), None)
    grow, stature = parle_build.head_ratio(profile.get("kit", "")), float(np.ptp(rest_points(body)[:, 2])) if body else 0.0
    skin, skin_arms = KDTree(len(body.data.vertices) if body else 0), {side: arm_share(body, side) for side in parle_build.SIDES} if body else {}
    for index, point in enumerate(rest_points(body) if body else ()):
        skin.insert(point, index)
    skin.balance()
    for obj in meshes:
        if not obj.vertex_groups:
            continue
        arms, top = {side: arm_share(obj, side) for side in parle_build.SIDES}, group_share(obj, head)
        if body is not None and obj is not body:
            arms = over_skin(obj, arms, skin, skin_arms)
        warp(obj, lambda points: points + field(points, arms, top))
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    moves = []
    for name, ride in rides.items():
        first, local = rig.data.edit_bones[name], ride @ back.T
        # The collar bone ends where the arm starts.
        moves += [(first.parent, "tail", np.array(first.parent.tail) + local)] if name.endswith("Arm") else []
        for bone in [first] + list(first.children_recursive):
            moves += [(bone, end, np.array(getattr(bone, end)) + local) for end in ("head", "tail")]
    # The bones of the head grow with it, about its joint.
    crown = rig.data.edit_bones["Head"]
    base = np.array(crown.head)
    for bone, ends in [(crown, ("tail",))] + [(child, ("head", "tail")) for child in crown.children_recursive]:
        moves += [(bone, end, base + grow * (np.array(getattr(bone, end)) - base)) for end in ends]
    # Every end is read before any is written: a bone joined to its parent would otherwise move twice.
    for bone, end, point in moves:
        setattr(bone, end, point)
    bpy.ops.object.mode_set(mode="OBJECT")
    if grow != 1 and body is not None:
        # A bigger head on the same body stands taller: the whole comes back to the stature it had, so it is the body that is smaller.
        rig.scale = [rig.scale[0] * stature / float(np.ptp(rest_points(body)[:, 2]))] * 3
        bpy.context.view_layer.update()
    return True


def top_over_bottom(tops, bottoms):
    """At rest, moves out the points of the top that stand inside the bottom, or closer to it than OVER_GAP_M. In each round the push is
    shared with the points around, so the hem keeps its line, and the halves of a point that a seam of the picture splits go together;
    a last pass sets whatever is still short of the gap. Returns how many points moved."""
    if not tops or not bottoms:
        return 0
    points, faces = [], []
    for obj in bottoms:
        obj.data.calc_loop_triangles()
        start = len(points)
        points.extend(map(tuple, rest_points(obj)))
        faces.extend(tuple(int(index) + start for index in triangle.vertices) for triangle in obj.data.loop_triangles)
    tree = BVHTree.FromPolygons(points, faces)

    def short(point):
        """How far a point of the top has to go, and which way, to stand OVER_GAP_M outside the bottom."""
        location, normal, _, _ = tree.find_nearest(Vector(point), OVER_REACH_M)
        need = 0.0 if location is None else OVER_GAP_M - (Vector(point) - location).dot(normal)
        return (np.array(normal) * need) if need > 0 else np.zeros(3)

    count = 0
    for obj in tops:
        rest = rest_points(obj)
        around = [[] for _ in rest]
        for edge in obj.data.edges:
            first, second = edge.vertices
            around[first].append(second)
            around[second].append(first)
        spots = {}
        twin = np.array([spots.setdefault(tuple(np.round(point, 4)), len(spots)) for point in rest])
        halves = np.bincount(twin, minlength=len(spots))[:, None]
        moved = np.zeros_like(rest)
        for _ in range(OVER_ROUNDS):
            step = np.array([short(point) for point in rest + moved])
            lengths = np.linalg.norm(step, axis=1, keepdims=True)
            step = step * np.minimum(1.0, OVER_STEP_M / np.maximum(lengths, 1e-9))
            if not lengths.any():
                break
            for _ in range(OVER_SHARED):
                step = np.array([(step[index] + step[near].mean(axis=0)) / 2 if near else step[index] for index, near in enumerate(around)])
            together = np.zeros((len(spots), 3))
            np.add.at(together, twin, step)
            moved += (together / halves)[twin]
        last = np.array([short(point) for point in rest + moved])
        together = np.zeros((len(spots), 3))
        np.add.at(together, twin, last)
        moved += (together / halves)[twin]
        gone = np.linalg.norm(moved, axis=1) > 1e-5
        if gone.any():
            count += int(gone.sum())
            warp(obj, lambda where, moved=moved: where + moved)
    return count


def seamless_normals(obj):
    """Gives the points of a mesh that stand at the same place the same normal, and returns how many places had more than one. The skin
    comes from a glTF, which splits a point wherever the picture has a seam: two rings of points at the same place, each lit by its own
    faces alone. Where the head meets the neck the two normals stood 4 to 20 degrees apart, and the light drew the seam as a line across
    the nape and down the side of the neck, a collar of another skin (Carlos, 2026-11-11); the colour was the same on both sides."""
    mesh = obj.data
    points = np.round(read(mesh.vertices), 5)
    _, place, count = np.unique(points, axis=0, return_inverse=True, return_counts=True)
    place = place.ravel()
    point_of = np.empty(len(mesh.loops), dtype=np.int64)
    mesh.loops.foreach_get("vertex_index", point_of)
    corners = np.empty(len(mesh.loops) * 3)
    mesh.corner_normals.foreach_get("vector", corners)
    corners = corners.reshape(-1, 3)
    # Each corner of a face brings the normal of its point: a place with more faces around it on one side weighs more on that side.
    total = np.zeros((len(count), 3))
    np.add.at(total, place[point_of], corners)
    total /= np.maximum(np.linalg.norm(total, axis=1, keepdims=True), 1e-9)
    shared = count[place[point_of]] > 1
    corners[shared] = total[place[point_of]][shared]
    mesh.normals_split_custom_set(corners.tolist())
    mesh.update()
    return int((count > 1).sum())


def round_bust(meshes, rig):
    """Under a top each breast of a woman takes the shape of a cup, as in a bra (`parle_bust`): the pieces of the top, and the skin
    with them. Run at rest on the finished body. Returns how far each tip came back, in metres; nothing when no top covers the bust."""
    tops = [obj for obj in meshes if obj.name.startswith("Outfit_Top")]
    if not tops:
        return []
    joint = lambda name: np.array(rig.matrix_world @ rig.data.bones[name].head_local)
    hip, shoulder = (joint("LeftUpLeg")[2] + joint("RightUpLeg")[2]) / 2, (joint("LeftArm")[2] + joint("RightArm")[2]) / 2
    middle = (joint("LeftArm")[0] + joint("RightArm")[0]) / 2
    cloth = np.concatenate([rest_points(obj) for obj in tops])
    reach, cups = parle_bust.REACH * float(np.ptp(np.concatenate([rest_points(obj) for obj in meshes if obj.name.split(".")[0] == "Body"])[:, 2])), []
    for side in (1, -1):
        tip = parle_bust.tip_of(cloth, hip, shoulder, middle, abs(joint("LeftArm")[0] - middle), side)
        found = parle_bust.rim_of(cloth, tip, reach) if tip is not None else None
        if found is not None and found[1] >= parle_bust.TALL_LEAST_M:
            cups.append((tip, *found))
    for obj in [obj for obj in meshes if obj in tops or obj.name.split(".")[0] == "Body"] if cups else ():
        warp(obj, lambda points: points + sum(parle_bust.cupped(points, tip, axis, tall, reach) for tip, axis, tall in cups))
    return [(1 - parle_bust.CUP) * tall for _, _, tall in cups]


def natural_body(meshes, rig, height, sex, kit):
    """The body of a finished character: the arms, then the build of the kit. The cast and the loop that finds the build (`body_match`, through
    `body_dump`) both call this and nothing else, so the body under the clothes is the body that was matched to the reference. A shoulder polish
    once ran before these steps in the cast alone: dressed, a man's shoulders came out 6 to 8 cm narrower on each side, at the levels above the
    armpit, than the body the loop had measured (owner, 2026-11-10: the dressed character was not the silhouette that had been shown). Last,
    the top goes back outside the bottom (`top_over_bottom`): that moves cloth alone, never the body. And under a top the bust of a
    woman is a bra's (`round_bust`): with no top on, as in the loop, nothing moves."""
    natural_arms(meshes, rig, height, sex)
    built = natural_build(meshes, rig, height, kit)
    over = top_over_bottom(*([obj for obj in meshes if obj.name.startswith(prefix)] for prefix in ("Outfit_Top", "Outfit_Bottom")))
    if over:
        print("BARRA", over, "pontos da peça de cima voltaram para fora da de baixo")
    cups = round_bust(meshes, rig) if sex == "female" else []
    if cups:
        print("BUSTO em taça debaixo da roupa: a ponta de cada seio voltou", [round(back * 1000, 1) for back in cups], "mm")
    return built


def garment_normal(name):
    """The garment's own normal map in the MakeHuman assets, or None when its author drew none."""
    base = name.split(".")[0]
    for folder in (base, base.split("-")[0]):
        found = sorted(glob.glob(os.path.join(CLOTHES, folder, "*[nN]ormal*")))
        if found:
            return found[0]
    return None


def cloth_relief(obj, path):
    """The folds drawn by the garment's own author: its normal map, at a size the phone can pay for."""
    image = bpy.data.images.load(path, check_existing=True)
    image.colorspace_settings.name = "Non-Color"
    if image.size[0] > NORMAL_PX:
        image.scale(NORMAL_PX, NORMAL_PX)
    image.pack()
    for material in obj.data.materials:
        tree = material.node_tree
        texture = tree.nodes.new("ShaderNodeTexImage")
        texture.image = image
        normal = tree.nodes.new("ShaderNodeNormalMap")
        tree.links.new(texture.outputs["Color"], normal.inputs["Color"])
        shader = next(node for node in tree.nodes if node.type == "BSDF_PRINCIPLED")
        tree.links.new(normal.outputs["Normal"], shader.inputs["Normal"])
