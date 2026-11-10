"""
    License information: data/licenses/makehuman_license.txt

    Finishing touches of the Parle cast's look, run inside Blender on a mounted character, what was judged an improvement
    next to the player and Rafa (owner, 2026-11-08). Every shape change moves all shape keys, so blinking and speaking keep working.
"""
import glob
import os

import bpy
import numpy as np

import parle_build

CLOTHES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".local/mhd/data/clothes/hm08")

# 12% with a staring look all round; 10% reads as awake next to the player.
EYE_SCALE = 1.1
EYE_CORE_M = 0.017
EYE_REACH_M = 0.032
NORMAL_PX = 512
# Shoulders, measured beside two whole-body scans in the same camera (owner, 2026-11-09). The body binds with the arm half raised: lowered, the skin over the
# joint stands out as a ball, and the slope from the neck is flat. Past 2 cm on the cap the sleeve and arm show a stepped crease.
SHOULDER_CAP_M = {"female": 0.014, "male": 0.02}
SHOULDER_RISE_M = {"female": 0.024, "male": 0.02}
CAP_CORE_M = (0.015, 0.03)
CAP_ALONG_M = (-0.07, 0.05, 0.17, 0.10)
CAP_INSIDE_M = (0.07, 0.05)
SLOPE_FROM_NECK_M = (0.025, 0.03)
SLOPE_PAST_ARM_M = 0.015
SLOPE_ABOVE_ARM_M = (0.005, 0.035)
SLOPE_ABOVE_NECK_M = (0.06, 0.04)
SLOPE_DEPTH_M = (0.11, 0.05)
# Men (owner, 2026-11-10): the end of the shoulder stood as high as its middle, a square corner under a sleeve. The whole end comes down and a little in,
# from inside the joint to the top of the arm; 3 cm down with 2 cm in already reads as a drooping shoulder narrower than the arm.
SHOULDER_DROP_M = {"female": 0.0, "male": 0.02}
SHOULDER_IN_M = {"female": 0.0, "male": 0.012}
DROP_INSIDE_M = 0.07
DROP_BELOW_M = 0.10
DROP_ALONG_M = (0.22, 0.16)
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


def shoulder_field(neck, arms, elbows, cap, rise, drop=0.0, inward=0.0):
    """Takes world points. A continuous field, so skin, sleeve, strap and necklace stay together without being fitted again."""
    up = np.array((0.0, 0.0, 1.0))

    def field(points):
        out = points.copy()
        for joint, elbow in zip(arms, elbows):
            arm = (elbow - joint) / np.linalg.norm(elbow - joint)
            top = up - up.dot(arm) * arm
            top /= np.linalg.norm(top)
            away = points - joint
            along = away @ arm
            ring = away - along[:, None] * arm
            radius = np.maximum(np.linalg.norm(ring, axis=1), 1e-6)
            start, start_fade, end, end_fade = CAP_ALONG_M
            on_arm = smooth((along - start) / start_fade) * smooth((end - along) / end_fade) * smooth((radius - CAP_CORE_M[0]) / CAP_CORE_M[1])
            outer = np.sign(joint[0] - neck[0])
            off_chest = smooth(((points[:, 0] - joint[0]) * outer + CAP_INSIDE_M[0]) / CAP_INSIDE_M[1])
            out -= (ring / radius[:, None]) * (cap * np.clip((ring @ top) / radius, 0, 1) * on_arm * off_chest)[:, None]
            side = (points[:, 0] - neck[0]) * outer
            reach = (joint[0] - neck[0]) * outer
            along_slope = smooth((side - SLOPE_FROM_NECK_M[0]) / SLOPE_FROM_NECK_M[1]) * smooth((reach + SLOPE_PAST_ARM_M - side) / (reach - SLOPE_FROM_NECK_M[1]))
            above = smooth((points[:, 2] - joint[2] - SLOPE_ABOVE_ARM_M[0]) / SLOPE_ABOVE_ARM_M[1]) * smooth((neck[2] + SLOPE_ABOVE_NECK_M[0] - points[:, 2]) / SLOPE_ABOVE_NECK_M[1])
            depth = smooth((SLOPE_DEPTH_M[0] - np.abs(points[:, 1] - neck[1])) / SLOPE_DEPTH_M[1])
            out[:, 2] += rise * along_slope * above * depth
            end = smooth((side - reach + DROP_INSIDE_M) / DROP_INSIDE_M) * smooth((DROP_ALONG_M[0] - along) / DROP_ALONG_M[1])
            out[:, 2] -= drop * end * smooth((points[:, 2] - joint[2] + DROP_BELOW_M) / DROP_BELOW_M)
            out[:, 0] -= outer * inward * end
        return out

    return field


def natural_shoulders(meshes, rig, sex):
    """Run last on a finished character, in its final size: the clips and the cloth were solved on the same skeleton, which does not move."""
    bpy.context.view_layer.update()
    joint = lambda name: np.array(rig.matrix_world @ rig.data.bones[name].head_local)
    field = shoulder_field(joint("Neck"), [joint("LeftArm"), joint("RightArm")], [joint("LeftForeArm"), joint("RightForeArm")], SHOULDER_CAP_M[sex], SHOULDER_RISE_M[sex], SHOULDER_DROP_M[sex], SHOULDER_IN_M[sex])
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


def natural_arms(meshes, rig, height, sex):
    """Run last, like the shoulders. Skin, cloth and bones move by the same field, and the clips only turn the bones, so every clip still plays."""
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


def natural_build(meshes, rig, height, kit):
    """The build of the kit's profile (`parle_build`): trunk, arms and legs get the widths found against a reference body. Run last, after the arms.
    The head keeps its shape, the spine stays, and each arm and leg goes as a whole with its bones, so every clip still plays. No profile, no change."""
    profile = parle_build.load_profile(kit)
    if profile is None:
        return False
    bpy.context.view_layer.update()
    back = np.array(rig.matrix_world.inverted())[:3, :3]
    joints = {name: np.array(rig.matrix_world @ rig.data.bones[name].head_local) for name in parle_build.BONES}
    field, rides = parle_build.build_field(profile, joints, height)
    head = {"Head"} | {bone.name for bone in rig.data.bones["Head"].children_recursive}
    for obj in meshes:
        if not obj.vertex_groups:
            continue
        arms, top = {side: arm_share(obj, side) for side in parle_build.SIDES}, group_share(obj, head)
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
    # Every end is read before any is written: a bone joined to its parent would otherwise move twice.
    for bone, end, point in moves:
        setattr(bone, end, point)
    bpy.ops.object.mode_set(mode="OBJECT")
    return True


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
