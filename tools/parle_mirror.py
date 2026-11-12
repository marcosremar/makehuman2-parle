"""
    License information: data/licenses/makehuman_license.txt

    How far a face is from its own mirror image, on the mesh (2026-11-12). Run in Blender:

      Blender -b --factory-startup -P tools/parle_mirror.py -- <out.json> <name>=<body.glb> [...]

    Owner, 2026-11-12: "deve adicionar cálculos de simetria de beleza. busca na internet teorias sobre isso e implementa".

    The measure is the one taken on scans of real faces (read 2026-11-12, each DOI checked against its record): the face and its
    mirror image laid on each other, turned and moved only, and the root of the mean square of what is left between them. Claes,
    Walters, Vandermeulen & Clement (2011), "Spatially-dense 3D facial asymmetry assessment in both typical and disordered growth",
    Journal of Anatomy 219(4):444-455, DOI 10.1111/j.1469-7580.2011.01411.x (read): "the magnitude of spatial discrepancies between
    corresponding quasi-landmarks in the original and mirror configurations after superimposition ... summarized by a RMSE ... as an
    error in mm", with medians of 0.94 mm on the women and 1.05 mm on the men of 359 people. Taylor et al. (2014), "Quantitative facial
    asymmetry: using three-dimensional photogrammetry to measure baseline facial surface symmetry", Journal of Craniofacial Surgery
    25(1):124-128, DOI 10.1097/SCS.0b013e3182a2e99d (abstract): the "root mean square deviation (RMSD) between the native and
    reflected faces", 0.80 mm (SD 0.24) on 100 people. Theirs is between points that answer to each other; here it is from each
    point of the image to the skin nearest to it, which can only read less. What a face of the cast reads is to be set against
    those: a real face is some 0.8 to 1 mm from its mirror image, and a face made of one half and its mirror is found less
    attractive than the face it came from (`parle_symmetry`).

    On the 52 faces the owner judged (2026-11-12) it reads 0.002 to 0.17 mm (0.07 the middle one), with the worst point of a face
    up to 2.8 mm off: built by mirroring, all of them, the ones he faults no less than the ones he likes. The measure is a watch
    on a body that comes out crooked from a step that should not have told left from right, not a mark of beauty. It found one:
    on three of five people of the new cast one upper lid stands 2.2 to 2.8 mm from the other's mirror image.

    The face of a body is mirrored across the plane down its middle, the mirror image is laid back on the face (turned and moved as
    a whole until it fits best, so a head that is a little turned or off the middle does not read as uneven), and what is measured
    is how far each point of the image then stands from the skin. The points of the Face Mesh on a portrait cannot tell this on a
    drawn face: from one portrait to the next they move by more than the two halves of these faces differ (`parle_symmetry`).

    The face is the skin of the head from the chin up, no further back than FACE_DEEP of the width of the head behind the tip of the
    nose: the ears, the neck and the back of the head are not face, and the hair is another mesh.
"""
import json
import sys

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

# The head is the top share of the height of the body; the chin is the lowest point of the skin no further behind the tip of the
# nose than CHIN_DEEP of the width of the head; the face goes back FACE_DEEP of that width.
HEAD_SHARE = 0.16
CHIN_DEEP = 0.15
FACE_DEEP = 0.4
# The mirror image is laid back on the face this many times (each time every point goes to the skin nearest to it and the whole is
# turned and moved to fit); it has settled when it moves less than the second, in metres.
FIT_ROUNDS = 12
FIT_SETTLED_M = 1e-6
# The thirds of the face, top to bottom, for where it is uneven: brow and eyes, nose and cheeks, mouth and jaw.
THIRDS = ("upper", "middle", "lower")


def skin_of(meshes):
    """The mesh that is the skin of the face: the one with the jaw or the visemes among its shapes, the biggest of them; with no
    such shape on any mesh, the biggest mesh."""
    keyed = [obj for obj in meshes if obj.data.shape_keys and any(key.name in ("jawOpen", "viseme_aa") for key in obj.data.shape_keys.key_blocks)]
    return max(keyed or meshes, key=lambda obj: len(obj.data.vertices))


def face_of(points):
    """Which points of the skin (world, z up, the face looking at -y) are face, and the width of the head."""
    top, tall = points[:, 2].max(), points[:, 2].max() - points[:, 2].min()
    head = points[:, 2] > top - HEAD_SHARE * tall
    wide = float(points[head, 0].max() - points[head, 0].min())
    front = float(points[head, 1].min())
    chin = float(points[head & (points[:, 1] < front + CHIN_DEEP * wide), 2].min())
    return head & (points[:, 2] >= chin) & (points[:, 1] <= front + FACE_DEEP * wide), wide


def fitted(moving, target):
    """`moving` turned and moved as a whole to lie on `target`, point for point, as near as it can (least squares)."""
    centre, aim = moving.mean(axis=0), target.mean(axis=0)
    left, _, right = np.linalg.svd((moving - centre).T @ (target - aim))
    flip = np.sign(np.linalg.det(right.T @ left.T))
    turn = right.T @ np.diag([1.0, 1.0, flip]) @ left.T
    return (moving - centre) @ turn.T + aim


def mirror_distance(obj):
    """How far the mirror image of the face of that skin stands from the face, in metres: for each point of the image once it is
    laid back on the face, its distance to the skin. Returns the distances, the height of each point as a share of the face (0 at
    the chin, 1 at the top), how far it is to one side as a share of half the width (-1 to 1), and the width of the face."""
    points = np.array([tuple(obj.matrix_world @ vertex.co) for vertex in obj.data.vertices])
    face, _ = face_of(points)
    obj.data.calc_loop_triangles()
    skin = BVHTree.FromPolygons([tuple(point) for point in points], [tuple(triangle.vertices) for triangle in obj.data.loop_triangles])
    own = points[face]
    image = own * np.array([-1.0, 1.0, 1.0]) + np.array([2 * own[:, 0].mean(), 0.0, 0.0])
    near = image
    for _ in range(FIT_ROUNDS):
        near = np.array([tuple(skin.find_nearest(Vector(point))[0]) for point in image])
        moved = fitted(image, near)
        settled = float(np.abs(moved - image).max()) < FIT_SETTLED_M
        image = moved
        if settled:
            break
    near = np.array([tuple(skin.find_nearest(Vector(point))[0]) for point in image])
    low, high = own[:, 2].min(), own[:, 2].max()
    wide = float(own[:, 0].max() - own[:, 0].min())
    aside = (image[:, 0] - (own[:, 0].max() + own[:, 0].min()) / 2) / max(wide / 2, 1e-9)
    return np.linalg.norm(image - near, axis=1), (image[:, 2] - low) / max(high - low, 1e-9), aside, wide


def measures(obj):
    """The measures of one face: the root of the mean square of the distance of its mirror image, as a share of the width of the
    face (`sym_mirror`) and in millimetres (`mirror_rms_mm`, the number the studies give), the same share in each third of its
    height, and in millimetres the mean, the 95th hundredth and the most, with where on the face the most is (its height from
    the chin, 0, to the top, 1, and how far to the side, 0 at the middle to 1 at the edge)."""
    apart, height, aside, wide = mirror_distance(obj)
    spread = lambda part: float(np.sqrt(np.mean(np.square(part))))
    found = {"sym_mirror": spread(apart) / wide, "mirror_rms_mm": spread(apart) * 1000, "mirror_mean_mm": float(apart.mean() * 1000),
             "mirror_p95_mm": float(np.percentile(apart, 95) * 1000), "mirror_most_mm": float(apart.max() * 1000), "face_wide_mm": wide * 1000, "points": int(len(apart))}
    for index, name in enumerate(THIRDS):
        part = apart[(height >= 1 - (index + 1) / 3) & (height <= 1 - index / 3)]
        found[f"sym_mirror_{name}"] = spread(part) / wide if len(part) else None
    # Where it is most uneven: the place of the worst point, to go and look at it.
    worst = int(apart.argmax())
    found["mirror_most_at"] = {"height": float(height[worst]), "aside": float(abs(aside[worst]))}
    return found


def read(path):
    """The skin of the body in a glTF file, at rest."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=path)
    for rig in [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]:
        rig.animation_data_clear()
        rig.data.pose_position = "REST"
    bpy.context.view_layer.update()
    return skin_of([obj for obj in bpy.context.scene.objects if obj.type == "MESH" and len(obj.data.polygons)])


if __name__ == "__main__":
    out, *items = sys.argv[sys.argv.index("--") + 1:]
    table = {}
    for item in items:
        name, path = item.split("=", 1)
        try:
            table[name] = measures(read(path))
            print("ESPELHO %-26s %.4f da largura (%.2f mm de erro quadrático médio, %.2f no pior ponto, a %.2f da altura e %.2f para o lado, %d pontos)" % (
                name, table[name]["sym_mirror"], table[name]["mirror_rms_mm"], table[name]["mirror_most_mm"], table[name]["mirror_most_at"]["height"],
                table[name]["mirror_most_at"]["aside"], table[name]["points"]), flush=True)
        except Exception as error:  # one body that cannot be read must not lose the others
            table[name] = None
            print("ESPELHO", name, "sem leitura:", error, flush=True)
    with open(out, "w") as file:
        json.dump(table, file, indent=1)
