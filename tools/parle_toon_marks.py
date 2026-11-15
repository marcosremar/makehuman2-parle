"""
    License information: data/licenses/makehuman_license.txt

    Landmarks of a head, for `parle_toon_fit` (2026-11-15): placed by its crown and nose alone, a generated head's eyes and mouth stood
    elsewhere than the base's, and the walk laid the cast's lids over the eyes and tore the mouth open. Each head is shot from the front
    (orthographic), a face landmarker reads the picture, and each mark goes back to 3D along the camera's ray onto the surface.

      Blender -b -P tools/parle_toon_marks.py -- shoot <mesh.glb|base> <out-stem>     (writes <stem>.png and <stem>.camera.json)
      FACE_LANDMARKER=<face_landmarker.task> python tools/parle_toon_marks.py read <stem>   (writes <stem>.marks2d.json; needs mediapipe)
    The 3D marks come from `marks_3d` inside Blender, in `parle_toon_fit`.
"""
import json
import sys
from pathlib import Path

import numpy as np

# Face Mesh points used: eye corners and lids, pupils, nose tip and wings, mouth corners and lips, chin, brows, sides of the face.
MARKS = {"eye_l_out": 33, "eye_l_in": 133, "eye_l_up": 159, "eye_l_low": 145, "eye_r_out": 263, "eye_r_in": 362, "eye_r_up": 386, "eye_r_low": 374,
         "nose_tip": 1, "nose_l": 48, "nose_r": 278, "mouth_l": 61, "mouth_r": 291, "lip_up": 13, "lip_low": 14, "chin": 152,
         "brow_l": 105, "brow_r": 334, "side_l": 234, "side_r": 454, "forehead": 10}
SIZE = 1024


def shoot(source, stem):
    import bpy
    import mathutils

    bpy.ops.wm.read_factory_settings(use_empty=True)
    if source == "base":
        sys.path.insert(0, str(Path(__file__).parent))
        import parle_mouth_arkit as arkit

        points, faces, group = [], [], None
        for line in arkit.BASE_MESH.read_text().splitlines():
            if line.startswith("v "):
                points.append([float(value) for value in line.split()[1:4]])
            elif line.startswith("g "):
                group = line.split()[1]
            elif line.startswith("f ") and group in ("body", "helper-l-eye", "helper-r-eye"):
                faces.append([int(item.split("/")[0]) - 1 for item in line.split()[1:]])
        mesh = bpy.data.meshes.new("base")
        mesh.from_pydata(arkit.mh_to_blender(np.array(points)).tolist(), [], faces)
        obj = bpy.data.objects.new("base", mesh)
        bpy.context.scene.collection.objects.link(obj)
        objs = [obj]
    else:
        bpy.ops.import_scene.gltf(filepath=source)
        objs = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    pts = np.array([list(obj.matrix_world @ v.co) for obj in objs for v in obj.data.vertices])
    top = pts[:, 2].max()
    head = pts[pts[:, 2] > top - (0.28 if source == "base" else np.ptp(pts[:, 2]) * 0.8)]
    centre = (head.min(axis=0) + head.max(axis=0)) / 2
    span = float(max(np.ptp(head[:, 0]), np.ptp(head[:, 2]))) * 1.1
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "SINGLE"
    scene.display.shading.single_color = (0.85, 0.66, 0.55)
    scene.display.shading.show_cavity = True
    scene.render.resolution_x = scene.render.resolution_y = SIZE
    camera = bpy.data.objects.new("c", bpy.data.cameras.new("c"))
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = span
    camera.location = mathutils.Vector((centre[0], centre[1] - 5, centre[2]))
    camera.rotation_euler = mathutils.Vector((0, 1, 0)).to_track_quat("-Z", "Y").to_euler()
    scene.collection.objects.link(camera)
    scene.camera = camera
    scene.render.filepath = f"{stem}.png"
    bpy.ops.render.render(write_still=True)
    json.dump({"centre": centre.tolist(), "span": span, "size": SIZE}, open(f"{stem}.camera.json", "w"))
    print("FOTO", stem)


def read(stem):
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision

    import os

    model = os.environ.get("FACE_LANDMARKER", "face_landmarker.task")
    options = vision.FaceLandmarkerOptions(base_options=mpt.BaseOptions(model_asset_path=model), num_faces=1)
    marker = vision.FaceLandmarker.create_from_options(options)
    image = cv2.imread(f"{stem}.png")
    found = marker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
    if not found.face_landmarks:
        raise ValueError(f"no face found on {stem}.png")
    points = found.face_landmarks[0]
    json.dump({name: [points[index].x, points[index].y] for name, index in MARKS.items()}, open(f"{stem}.marks2d.json", "w"))
    print("MARCOS", stem, len(MARKS))


def marks_3d(stem, tree, points=None):
    """The marks of a shot back in 3D: each along the camera's ray (looking +y) onto the surface of `tree` (a BVHTree). A ray that
    goes through a hole of the surface (the eye of a generated head is a cavity) and lands behind the middle of the head takes, with
    `points` (the surface's points), the frontmost point within REACH of the ray instead."""
    camera = json.load(open(f"{stem}.camera.json"))
    centre, span = np.array(camera["centre"]), camera["span"]
    out = {}
    for name, (u, v) in json.load(open(f"{stem}.marks2d.json")).items():
        x, z = centre[0] + (u - 0.5) * span, centre[2] - (v - 0.5) * span
        hit = tree.ray_cast((x, centre[1] - 5, z), (0.0, 1.0, 0.0))
        spot = np.array(hit[0]) if hit[0] is not None else None
        if (spot is None or spot[1] > centre[1]) and points is not None:
            near = np.hypot(points[:, 0] - x, points[:, 2] - z) < REACH * span
            if near.any():
                front = points[near][np.argmin(points[near][:, 1])]
                spot = np.array([x, front[1], z])
        if spot is not None:
            out[name] = spot
    return out


REACH = 0.02


if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    if args[0] == "shoot":
        shoot(args[1], args[2])
    elif args[0] == "read":
        read(args[1])
