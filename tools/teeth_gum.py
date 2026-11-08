"""
    License information: data/licenses/makehuman_license.txt

    Pink gums on the Parle teeth map. The stock map draws teeth and gums alike in its alpha, so
    `teeth_texture.py` made both ivory, and an open mouth showed a white band of gum above the
    teeth (owner, 2026-11-08: the open mouth looked aggressive). Lower teeth are the points the
    kit's `jawOpen` moves; a face is gum when it lies more than a crown away from the tip of its
    row, measured on the kit's rest shape.

      blender -b -P tools/teeth_gum.py -- <kit-body.glb> <gum.json>   # the gum faces, in UV
      python tools/teeth_gum.py <gum.json> <teeth.png>                # paints them, in place
"""
import json
import sys

import numpy as np

CROWN_M = 0.0095
GUM = np.array((112, 116, 196))  # BGR, the colour of a gum in daylight
EDGE_PX = 2


def gum_faces(kit, out):
    import bpy
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=kit)
    teeth = next(obj for obj in bpy.data.objects if obj.type == "MESH" and "teeth" in obj.name)
    keys = teeth.data.shape_keys.key_blocks
    rest = np.array([point.co[:] for point in keys[0].data])
    moved = np.linalg.norm(np.array([point.co[:] for point in keys["jawOpen"].data]) - rest, axis=1)
    lower = moved > moved.max() * 0.3
    tip_up = np.percentile(rest[~lower, 2], 2)
    top_low = np.percentile(rest[lower, 2], 98)
    gum = np.where(lower, rest[:, 2] < top_low - CROWN_M, rest[:, 2] > tip_up + CROWN_M)
    uv = teeth.data.uv_layers.active.data
    faces = [[tuple(uv[loop].uv) for loop in polygon.loop_indices] for polygon in teeth.data.polygons if all(gum[index] for index in polygon.vertices)]
    with open(out, "w") as file:
        json.dump(faces, file)
    print("GENGIVA", len(faces), "faces de", len(teeth.data.polygons))


def paint(faces_file, out):
    import cv2
    image = cv2.imread(out)
    size = image.shape[0]
    mask = np.zeros(image.shape[:2], np.uint8)
    with open(faces_file) as file:
        for face in json.load(file):
            cv2.fillPoly(mask, [np.array([(u * size, (1 - v) * size) for u, v in face], np.int32)], 255)
    share = cv2.GaussianBlur(cv2.dilate(mask, np.ones((3, 3), np.uint8)), (0, 0), EDGE_PX).astype(np.float32)[:, :, None] / 255
    # The dark of the inside of the mouth stays dark: only the ivory turns pink.
    share *= (image.mean(axis=2, keepdims=True) > 128).astype(np.float32)
    cv2.imwrite(out, (image * (1 - share) + GUM * share).round().astype(np.uint8))
    print("PINTADA", out)


if __name__ == "__main__":
    if "--" in sys.argv:
        gum_faces(*sys.argv[sys.argv.index("--") + 1:][:2])
    else:
        paint(*sys.argv[1:3])
