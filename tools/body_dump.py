"""
    License information: data/licenses/makehuman_license.txt

    Unclothed body in a common pose, dumped for `tools/body_measure.py`:
    `Blender -b --factory-startup -noaudio -P tools/body_dump.py -- <out.npz> <model.glb|fbx> kind=<kit|cc|slots|plain> [key=value ...]`.
    kit: a cast kit (`Body` mesh). `height=` scales it, `chest=<npy>` adds the chest shift, `shoulders=1` runs `natural_shoulders` and `arms=1` runs
    `natural_arms` (both need `sex=`), and `build=<kit name or profile.json>` runs `natural_build`. cc: a Character Creator body (`CC_Base_Body`); `flat=1` takes it off the tiptoes.
    slots: one skinned mesh, keeping the faces of the materials in `slots=`; `flip=1` when it faces +y. plain: a mesh with no rig, `keep=<prefix>`.
    `arm=<degrees>` puts the upper arms that far below the horizontal. `photos=<folder>` with `name=` adds clay photos (front, side, back).
    A reference model is only measured here: neither its mesh nor its texture goes into the cast.
"""
import math
import os
import sys

import bmesh
import bpy
import mathutils
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parle_polish  # noqa: E402

JOINTS = {
    "kit": dict(shoulder="LeftArm", elbow="LeftForeArm", wrist="LeftHand", hip="LeftUpLeg", knee="LeftLeg", ankle="LeftFoot", toe="LeftToeBase", neck="Neck", head="Head",
                shoulder_r="RightArm", elbow_r="RightForeArm", wrist_r="RightHand", ankle_r="RightFoot", toe_r="RightToeBase"),
    "cc": dict(shoulder="CC_Base_L_Upperarm", elbow="CC_Base_L_Forearm", wrist="CC_Base_L_Hand", hip="CC_Base_L_Thigh", knee="CC_Base_L_Calf", ankle="CC_Base_L_Foot",
               toe="CC_Base_L_ToeBase", neck="CC_Base_NeckTwist01", head="CC_Base_Head", shoulder_r="CC_Base_R_Upperarm", elbow_r="CC_Base_R_Forearm", wrist_r="CC_Base_R_Hand",
               ankle_r="CC_Base_R_Foot", toe_r="CC_Base_R_ToeBase"),
    "slots": dict(shoulder="upperarm_l", elbow="lowerarm_l", wrist="hand_l", hip="thigh_l", knee="calf_l", ankle="foot_l", toe="ball_l", neck="neck_01", head="head",
                  shoulder_r="upperarm_r", elbow_r="lowerarm_r", wrist_r="hand_r", ankle_r="foot_r", toe_r="ball_r"),
}
# The kit's foot at rest: the line from the ankle to the base of the toes goes down 16.6 degrees. A body modelled on high heels is put on the same foot before it is measured.
FOOT_DECLINE_DEG = 16.6

args = sys.argv[sys.argv.index("--") + 1:]
out, source = args[:2]
opts = dict(item.split("=", 1) for item in args[2:])
kind = opts.get("kind", "plain")
bpy.ops.wm.read_factory_settings(use_empty=True)
if source.lower().endswith(".fbx"):
    bpy.ops.import_scene.fbx(filepath=source)
else:
    bpy.ops.import_scene.gltf(filepath=source)
for obj in list(bpy.data.objects):
    if obj.type == "MESH" and obj.name.split(".")[0] in ("Cube", "Icosphere"):
        bpy.data.objects.remove(obj, do_unlink=True)
rigs = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]
rig = max(rigs, key=lambda obj: len(obj.data.bones)) if rigs else None
meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
names = {key: bone for key, bone in JOINTS.get(kind, {}).items() if rig is not None and bone in rig.pose.bones}


def joint(key):
    return np.array(rig.matrix_world @ rig.pose.bones[names[key]].head)


def declined(way, degrees):
    """The same heading as `way`, that many degrees below the horizontal."""
    flat = mathutils.Vector((way.x, way.y, 0)).normalized()
    return flat * math.cos(math.radians(degrees)) + mathutils.Vector((0, 0, -math.sin(math.radians(degrees))))


def aim(bone, way, want):
    """Turns a pose bone about its head until the world direction `way` becomes `want`."""
    turn = rig.matrix_world.to_3x3().inverted()
    spin = (turn @ way).normalized().rotation_difference((turn @ want).normalized()).to_matrix().to_4x4()
    bone.matrix = mathutils.Matrix.Translation(bone.head) @ spin @ mathutils.Matrix.Translation(-bone.head) @ bone.matrix
    bpy.context.view_layer.update()


keep = meshes
if kind == "kit":
    body = bpy.data.objects["Body"]
    if os.path.exists(opts.get("chest", "")):
        chest = np.load(opts["chest"])
        parle_polish.warp(body, lambda points: points + chest)
    if "height" in opts:
        rig.scale = [float(opts["height"]) / float(np.ptp(parle_polish.rest_points(body)[:, 2]))] * 3
        bpy.context.view_layer.update()
    # The same order as the cast: shoulders, arms, build.
    if opts.get("shoulders") == "1":
        parle_polish.natural_shoulders(meshes, rig, opts["sex"])
    if opts.get("arms") == "1":
        parle_polish.natural_arms(meshes, rig, float(opts["height"]), opts["sex"])
    if "build" in opts:
        parle_polish.natural_build(meshes, rig, float(opts["height"]), opts["build"])
    keep = [body]
elif kind == "cc":
    keep = [obj for obj in meshes if obj.name.split(".")[0] == "CC_Base_Body"]
elif kind == "slots":
    slots = {int(value) for value in opts["slots"].split(",")}
    keep = [max(meshes, key=lambda obj: len(obj.data.vertices))]
    bm = bmesh.new()
    bm.from_mesh(keep[0].data)
    bmesh.ops.delete(bm, geom=[face for face in bm.faces if face.material_index not in slots], context="FACES")
    bm.to_mesh(keep[0].data)
    bm.free()
elif "keep" in opts:
    keep = [obj for obj in meshes if obj.name.startswith(opts["keep"])]
for obj in meshes:
    obj.hide_render = obj not in keep
if rig is not None:
    if kind == "kit" and "arm" not in opts:
        rig.data.pose_position = "REST"
    else:
        rig.animation_data_clear()
        for bone in rig.pose.bones:
            bone.matrix_basis = mathutils.Matrix.Identity(4)
        rig.data.pose_position = "POSE"
    bpy.context.view_layer.update()
    if "arm" in opts:
        for a, b in (("shoulder", "elbow"), ("shoulder_r", "elbow_r")):
            way = mathutils.Vector((joint(b) - joint(a)).tolist())
            aim(rig.pose.bones[names[a]], way, declined(way, float(opts["arm"])))
    if opts.get("flat") == "1":
        for side in ("", "_r"):
            way = mathutils.Vector((joint("toe" + side) - joint("ankle" + side)).tolist())
            aim(rig.pose.bones[names["ankle" + side]], way, declined(way, FOOT_DECLINE_DEG))
            toe = rig.pose.bones[names["toe" + side]]
            way = rig.matrix_world.to_3x3() @ (toe.tail - toe.head)
            aim(toe, way, declined(way, 0))
deps = bpy.context.evaluated_depsgraph_get()
all_points, all_faces, base = [], [], 0
for obj in keep:
    done = obj.evaluated_get(deps)
    mesh = done.to_mesh()
    mesh.calc_loop_triangles()
    flat_points = np.empty(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", flat_points)
    triangles = np.empty(len(mesh.loop_triangles) * 3, np.int32)
    mesh.loop_triangles.foreach_get("vertices", triangles)
    world = np.array(done.matrix_world)
    all_points.append(flat_points.reshape(-1, 3) @ world[:3, :3].T + world[:3, 3])
    all_faces.append(triangles.reshape(-1, 3) + base)
    base += len(mesh.vertices)
    done.to_mesh_clear()
points, faces = np.concatenate(all_points), np.concatenate(all_faces)
joints = {key: joint(key) for key in names}
# The face looks to -y when the toes point there: the feet are ahead of the shins.
low, tall = float(points[:, 2].min()), float(np.ptp(points[:, 2]))
feet = points[points[:, 2] < low + 0.03 * tall]
shin = points[(points[:, 2] > low + 0.08 * tall) & (points[:, 2] < low + 0.2 * tall)]
flip = bool(int(opts["flip"])) if "flip" in opts else bool(feet[:, 1].mean() > shin[:, 1].mean())
arm_deg = None
if "elbow" in joints:
    arm_deg = round(math.degrees(math.atan2(-(joints["elbow"] - joints["shoulder"])[2], abs((joints["elbow"] - joints["shoulder"])[0]))), 1)
print("DUMP", opts.get("name", os.path.basename(out)), "points", len(points), "faces", len(faces), "tall", round(tall, 4), "flip", flip, "arm", arm_deg, flush=True)
np.savez(out, points=points, faces=faces, flip=flip, **{"j_" + key: value for key, value in joints.items()})
if "photos" in opts:
    scene = bpy.context.scene
    scene.render.engine, scene.view_settings.view_transform = "BLENDER_EEVEE_NEXT", "AgX"
    scene.render.resolution_x, scene.render.resolution_y = 660, 900
    scene.world = bpy.data.worlds.new("world")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = (0.75, 0.75, 0.75, 1)
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy = 2.5
    camera = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
    camera.data.type, camera.data.ortho_scale = "ORTHO", 1.1 * tall
    scene.collection.objects.link(sun)
    scene.collection.objects.link(camera)
    scene.camera = camera
    clay = bpy.data.materials.new("clay")
    clay.use_nodes = True
    clay.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.42, 0.34, 0.29, 1)
    clay.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.55
    for obj in keep:
        obj.data.materials.clear()
        obj.data.materials.append(clay)
    centre = mathutils.Vector((float(np.median(points[:, 0])), float(np.median(points[:, 1])), low + tall / 2))
    sign = -1 if flip else 1
    for label, yaw in (("front", 0), ("side", 90), ("back", 180)):
        camera.location = centre + mathutils.Vector((math.sin(math.radians(yaw)) * 5 * sign, -math.cos(math.radians(yaw)) * 5 * sign, 0))
        camera.rotation_euler = (centre - camera.location).to_track_quat("-Z", "Y").to_euler()
        sun.rotation_euler = camera.rotation_euler.copy()
        sun.rotation_euler.rotate_axis("X", math.radians(35))
        sun.rotation_euler.rotate_axis("Y", math.radians(25))
        scene.render.filepath = os.path.join(opts["photos"], f"{opts.get('name', 'body')}.{label}.png")
        bpy.ops.render.render(write_still=True)
