"""
    License information: data/licenses/makehuman_license.txt

    Face rules of the Parle cast, run inside Blender while a character is assembled. Each one
    removes something that looked wrong on the final avatar. The owner's examples are Rafa and the
    player: no eyelash cards, one eyebrow, no hair through the forehead, no eyeball through a lid.

      blender -b -P tools/parle_face.py -- <kit-body.glb> <out.json>   # writes the eye points the lids show
"""
import json
import os
import sys
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

EYE_TUCK_REACH_M = 0.02
EYE_TUCK_BEHIND_M = 0.001
FOREHEAD_HAIR_M = 0.085
FOREHEAD_HALF_M = 0.06
FOREHEAD_REACH_M = 0.06
# The kit's own morphs, baked into the rest shape: eyes open all around and brows pulled down read as anger on everybody (owner's review, 2026-11-07).
# The kit rests with the lips a little apart, and the teeth showed through in a white line (owner, 2026-11-07): the jaw rests a quarter closed. Teeth and tongue take it too.
CALM_LOOK = {"faceEyeOpen": -0.45, "faceBrowAngle": 0.6, "browInnerUp": 0.5, "jawOpen": -0.25}
BROW_THIN = 0.75
# Halves of a point split on a UV seam drift apart in the head fit and the background shows through, in a line along the jaw.
SEAM_REACH_M = 0.008
SEAM_SAME_MOVE_M = 0.001
SEAM_TOP_M = 0.05
SEAM_LOW_M = 0.4
# Earrings: one pair, smaller, and not the same game pair on every woman (owner, 2026-11-07).
EARRING_SCALE = 0.6
GAME_EARRINGS_KEPT_EVERY = 3


def skin_tree(body):
    return BVHTree.FromPolygons([tuple(body.matrix_world @ vertex.co) for vertex in body.data.vertices], [tuple(polygon.vertices) for polygon in body.data.polygons])


def wears_from_game(name, earrings, glasses=True):
    """Men of the cast wear no earrings (owner, 2026-11-06). Glasses of the game body stay unless the look says otherwise (Renata, owner, 2026-11-07)."""
    return (earrings or "Earring" not in name) and (glasses or not name.startswith(("Glasses", "Aviator")))


def trim_forehead_hair(obj, body, mine, band_z=None):
    """O crânio novo é menor: o afro que no jogo fica debaixo do turbante saía pela testa, numa mancha preta acima das sobrancelhas. Com lenço, o corte para no meio dele: acima, a testa aparecia entre o lenço e o cabelo."""
    skin = skin_tree(body)
    back = Vector((0, 1, 0))
    mesh = bmesh.new()
    mesh.from_mesh(obj.data)
    out = [vert for vert in mesh.verts if vert.co.z < (band_z if band_z is not None else mine["eye"] + FOREHEAD_HAIR_M) and abs(vert.co.x - mine["x"]) < FOREHEAD_HALF_M and skin.ray_cast(vert.co, back, FOREHEAD_REACH_M)[0] is not None]
    bmesh.ops.delete(mesh, geom=out, context="VERTS")
    mesh.to_mesh(obj.data)
    mesh.free()
    obj.data.update()
    print("TESTA", len(out), "pontos de cabelo")


def tuck_eyes(body, added, opening):
    """O globo anda inteiro com o rosto e a pele anda ponto a ponto: onde o rosto do jogo é mais estreito, o branco furava a pálpebra em duas marcas."""
    shown = set(json.loads(Path(opening).read_text()))
    skin = skin_tree(body)
    front, back = Vector((0, -1, 0)), Vector((0, 1, 0))
    tucked = 0
    for eye in (obj for obj in added if obj.type == "MESH" and "low-poly" in obj.name):
        keys = eye.data.shape_keys.key_blocks if eye.data.shape_keys else []
        inverse = eye.matrix_world.inverted()
        for vertex in eye.data.vertices:
            at = eye.matrix_world @ vertex.co
            if vertex.index in shown or skin.ray_cast(at + front * 1e-4, front, 0.05)[0] is not None:
                continue
            hit = skin.ray_cast(at, back, EYE_TUCK_REACH_M)[0]
            if hit is None:
                continue
            step = inverse.to_3x3() @ (back * ((hit - at).length + EYE_TUCK_BEHIND_M))
            vertex.co += step
            for key in keys:
                key.data[vertex.index].co += step
            tucked += 1
        eye.data.update()
    print("OLHO", tucked, "pontos furando a pele")


def key_points(block):
    points = np.zeros(len(block.data) * 3)
    block.data.foreach_get("co", points)
    return points.reshape(-1, 3)


def move_rest(obj, move):
    """Every key stores absolute points: the rest shape only moves if all of them move with it."""
    for block in obj.data.shape_keys.key_blocks:
        block.data.foreach_set("co", (key_points(block) + move).ravel())
    for vertex, shift in zip(obj.data.vertices, move):
        vertex.co += Vector(shift)
    obj.data.update()


def brow_line_move(points):
    """Thinner around its middle line, side by side."""
    middle = points[:, 0].mean()
    move = np.zeros_like(points)
    for side in (points[:, 0] < middle, points[:, 0] >= middle):
        line = np.polyval(np.polyfit(points[side, 0], points[side, 2], 3), points[side, 0])
        move[side, 2] = (line - points[side, 2]) * (1 - BROW_THIN)
    return move


def calm_look(meshes):
    for obj in meshes:
        keys = obj.data.shape_keys.key_blocks if obj.data.shape_keys else None
        if not keys:
            continue
        rest = key_points(keys[0])
        move = sum(((key_points(keys[name]) - rest) * value for name, value in CALM_LOOK.items() if name in keys), np.zeros_like(rest))
        if "eyebrow" in obj.name:
            move = move + brow_line_move(rest + move)
        if move.any():
            move_rest(obj, move)


def weld_seams(body):
    """Mutual nearest open points on the head and neck that move together when the jaw and the lids move: those are one point, and lips and eyelids never qualify."""
    keys = body.data.shape_keys.key_blocks
    rest = key_points(keys[0])
    top = rest[:, 2].max()
    mesh = bmesh.new()
    mesh.from_mesh(body.data)
    mesh.verts.ensure_lookup_table()
    open_points = [vert.index for vert in mesh.verts if any(len(edge.link_faces) == 1 for edge in vert.link_edges) and top - SEAM_LOW_M < rest[vert.index, 2] < top - SEAM_TOP_M]
    mesh.free()
    tree = KDTree(len(open_points))
    for index in open_points:
        tree.insert(rest[index], index)
    tree.balance()

    def nearest(index):
        return next((found for found in tree.find_n(rest[index], 2) if found[1] != index), None)

    movers = [key_points(keys[name]) - rest for name in ("jawOpen", "eyeBlinkLeft", "eyeBlinkRight") if name in keys]
    pairs = []
    for index in open_points:
        found = nearest(index)
        if found and found[2] < SEAM_REACH_M and index < found[1] and nearest(found[1])[1] == index and all(np.linalg.norm(mover[index] - mover[found[1]]) < SEAM_SAME_MOVE_M for mover in movers):
            pairs.append((index, found[1]))
    if not pairs:
        print("EMENDA 0")
        return
    first, second = np.array(pairs).T
    for block in keys:
        points = key_points(block)
        middle = (points[first] + points[second]) / 2
        points[first], points[second] = middle, middle
        block.data.foreach_set("co", points.ravel())
    for one, other in pairs:
        middle = (body.data.vertices[one].co + body.data.vertices[other].co) / 2
        body.data.vertices[one].co = middle
        body.data.vertices[other].co = middle
    body.data.update()
    print("EMENDA", len(pairs), "pares, maior", round(float(np.linalg.norm(rest[first] - rest[second], axis=1).max()) * 1000, 1), "mm")


def shrink_earring(obj):
    """Each side shrinks around the point that hangs from the lobe."""
    points = np.array([tuple(obj.matrix_world @ vertex.co) for vertex in obj.data.vertices])
    middle = points[:, 0].mean()
    moved = points.copy()
    for side in (points[:, 0] < middle, points[:, 0] >= middle):
        if side.any():
            top = points[side][points[side][:, 2].argmax()]
            moved[side] = top + (points[side] - top) * EARRING_SCALE
    inverse = obj.matrix_world.inverted()
    for vertex, point in zip(obj.data.vertices, moved):
        vertex.co = inverse @ Vector(point)
    obj.data.update()


def tidy_earrings(meshes, seed):
    """The look's own earrings win over the game body's; the game pair stays on one woman in three, chosen by name, so it is not on everybody."""
    chosen = [obj for obj in meshes if not obj.name.startswith(("Acc_", "Earring")) and any(word in obj.name.lower() for word in ("earring", "hoops"))]
    game = [obj for obj in meshes if obj.name.startswith(("Acc_Earring", "Earring"))]
    if chosen or sum(map(ord, seed)) % GAME_EARRINGS_KEPT_EVERY:
        for obj in game:
            bpy.data.objects.remove(obj, do_unlink=True)
        game = []
    for obj in chosen + game:
        shrink_earring(obj)
    print("BRINCOS", [obj.name for obj in chosen + game])


def set_map(obj, image):
    for node in obj.active_material.node_tree.nodes:
        if node.type == "TEX_IMAGE":
            node.image = image


def clean_face(body, added, job, env):
    for obj in [item for item in added if item.type == "MESH" and "eyelash" in item.name.lower()]:
        added.remove(obj)
        bpy.data.objects.remove(obj, do_unlink=True)
    weld_seams(body)
    calm_look([body, *(obj for obj in added if obj.type == "MESH" and any(word in obj.name for word in ("eyebrow", "teeth", "tongue")))])
    opening = os.path.join(os.path.dirname(job["out"]), f"eye-open-{job.get('kit')}.json")
    if os.path.exists(opening):
        tuck_eyes(body, added, opening)
    if env.get("CAST_TEETH"):
        teeth_map = bpy.data.images.load(env["CAST_TEETH"])
        teeth_map.pack()
        for obj in added:
            if obj.type == "MESH" and "teeth" in obj.name:
                material = obj.active_material
                material.blend_method = "OPAQUE"
                for link in [link for link in material.node_tree.links if link.to_socket.name == "Alpha"]:
                    material.node_tree.links.remove(link)
                set_map(obj, teeth_map)
    if env.get("CAST_EYE"):
        eye_map = bpy.data.images.load(env["CAST_EYE"])
        eye_map.pack()
        for obj in added:
            if obj.type == "MESH" and "low-poly" in obj.name:
                set_map(obj, eye_map)
                for polygon in obj.data.polygons:
                    polygon.use_smooth = True


def write_eye_opening(kit, out):
    """Which eye points the kit shows through the lids: any other one must not come out in front of the skin."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=kit)
    meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    body = next(obj for obj in meshes if obj.name == "Body")
    eye = next(obj for obj in meshes if "low-poly" in obj.name)
    skin = skin_tree(body)
    front = Vector((0, -1, 0))
    shown = [vertex.index for vertex in eye.data.vertices if skin.ray_cast(eye.matrix_world @ vertex.co + front * 1e-4, front, 0.05)[0] is None]
    Path(out).write_text(json.dumps(shown))
    print("ABERTURA", len(shown), "de", len(eye.data.vertices))


if __name__ == "__main__":
    write_eye_opening(*sys.argv[sys.argv.index("--") + 1:][:2])
