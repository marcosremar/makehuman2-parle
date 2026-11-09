"""
    License information: data/licenses/makehuman_license.txt

    Pink gums on the Parle teeth map. The stock map draws teeth and gums alike in its alpha, so
    `teeth_texture.py` made both ivory, and an open mouth showed a white band of gum above the
    teeth (owner, 2026-11-08: the open mouth looked aggressive). Lower teeth are the points the
    kit's `jawOpen` moves; a face is gum when it lies more than a crown away from the tip of its
    row, measured on the kit's rest shape.

      blender -b -P tools/teeth_gum.py -- <kit-body.glb> <gum.json>   # the gum faces, in UV
      python tools/teeth_gum.py <gum.json> <teeth.png>                # paints them, in place
      blender -b -P tools/teeth_gum.py -- <kit-body.glb> <rear.json> rear  # how far back each face is
      python tools/teeth_gum.py --rear <rear.json> <teeth.png>        # darkens the back of the mouth, in place
"""
import json
import sys

import numpy as np

CROWN_M = 0.0095
GUM = np.array((112, 116, 196))  # BGR, the colour of a gum in daylight
EDGE_PX = 2
# No light reaches the back of a mouth, and the player has no occlusion there: the molars shone as white blocks by the corners of the mouth (owner, 2026-11-09).
REAR_FROM_M = 0.006
REAR_TO_M = 0.035
REAR_DARK = 0.18
REAR_SOFT_PX = 4
LOWER_SIDE_FROM_M = 0.01
LOWER_SIDE_FADE_M = 0.012
LOWER_SIDE_MOST = 0.7


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


def rear_faces(kit, out):
    import bpy
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=kit)
    teeth = next(obj for obj in bpy.data.objects if obj.type == "MESH" and "teeth" in obj.name)
    rest = np.array([point.co[:] for point in teeth.data.shape_keys.key_blocks[0].data])
    front = rest[:, 1].min()
    middle = rest[:, 0].mean()
    keys = teeth.data.shape_keys.key_blocks
    moved = np.linalg.norm(np.array([point.co[:] for point in keys["jawOpen"].data]) - rest, axis=1)
    lower = moved > moved.max() * 0.3
    uv = teeth.data.uv_layers.active.data

    def share(points):
        depth = (rest[points, 1].mean() - front - REAR_FROM_M) / (REAR_TO_M - REAR_FROM_M)
        # The upper lip shades the lower arch: its side teeth showed as loose white blocks.
        side = min(LOWER_SIDE_MOST, (abs(rest[points, 0].mean() - middle) - LOWER_SIDE_FROM_M) / LOWER_SIDE_FADE_M) if lower[points].all() else 0.0
        return float(np.clip(max(depth, side), 0, 1))

    faces = [[[tuple(uv[loop].uv) for loop in polygon.loop_indices], share(list(polygon.vertices))] for polygon in teeth.data.polygons]
    with open(out, "w") as file:
        json.dump(faces, file)
    print("FUNDO", len(faces), "faces")


def shade_rear(faces_file, out):
    import cv2
    image = cv2.imread(out)
    size = image.shape[0]
    shade = np.zeros(image.shape[:2], np.float32)
    with open(faces_file) as file:
        for face, share in json.load(file):
            cv2.fillPoly(shade, [np.array([(u * size, (1 - v) * size) for u, v in face], np.int32)], float(share))
    shade = cv2.GaussianBlur(shade, (0, 0), REAR_SOFT_PX)[:, :, None]
    cv2.imwrite(out, (image * (1 - shade * (1 - REAR_DARK))).round().astype(np.uint8))
    print("FUNDO PINTADO", out)


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
        args = sys.argv[sys.argv.index("--") + 1:]
        (rear_faces if args[2:] == ["rear"] else gum_faces)(*args[:2])
    elif sys.argv[1] == "--rear":
        shade_rear(*sys.argv[2:4])
    else:
        paint(*sys.argv[1:3])
