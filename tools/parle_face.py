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
# With the jaw a quarter closed, the upper teeth sat wholly behind the upper lip: speaking, the mouth showed the lower teeth and the
# tongue under a bare lip, and read as a snarl (owner, 2026-11-08). Measured on the pharmacy's attendant at the player's widest "a"
# (`viseme_aa` 0.55): incisor tip 1 mm above the lip's edge. Teeth go 3.5 mm down and the tongue 3 mm more, behind the lower teeth.
# The women's kit has full lips that bulged like a sausage in the player's perspective and light (owner, 2026-11-08: lower lip too
# big). The kit's own `faceLipVolume`, on the lip that follows the jaw and fading 4 mm around it: -1 on women, -0.5 on men; -2 creased.
LIP_THIN = {"female": -1.0, "male": -0.5}
LIP_FADE_M = 0.004
# Thinner, the women's lips still stood out in profile like a beak and the lower one hung forward (owner, 2026-11-08, pharmacy). The
# whole mouth goes 3 mm back: lips and what lies 6 mm around them fully, fading over 20 mm more, teeth and tongue with them. Only the
# lower lip back opened a gap between the lips; 4 mm flattened the face.
MOUTH_BACK = {"female": 0.003, "male": 0.0}
# Speaking, the rounded vowels pushed the full lips forward into a big beak, and the player mostly shows her mid-sentence: the
# forward part (-y) of the rounded visemes keeps 40% on women. The vowel still reads as round.
POUT_FORWARD = {"female": 0.4, "male": 1.0}
POUT_VISEMES = ("viseme_U", "viseme_O", "viseme_CH", "viseme_RR")
MOUTH_CORE_M = 0.006
MOUTH_FADE_M = 0.02
TEETH_DOWN_M = 0.0035
TONGUE_DOWN_M = 0.0065
# Halves of a point split on a UV seam drift apart in the head fit and the background shows through, in a line along the jaw.
SEAM_REACH_M = 0.008
SEAM_SAME_MOVE_M = 0.001
SEAM_TOP_M = 0.05
SEAM_LOW_M = 0.4
# Brows were one flat dark colour with a hard edge and read as painted on (owner, 2026-11-08): a little way to the skin, and the
# strands' edge fades. A third of the way made the restaurant attendant's brows almost vanish.
BROW_TOWARD_SKIN = 0.15
BROW_ALPHA = 0.95
# A matte eye (roughness 0.7) had no catchlight and looked dead; the cornea is wet.
EYE_ROUGHNESS = 0.15
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


def lip_thinning(keys, rest, amount):
    """The lip is what moves in the kiss and the closed-lips visemes; only the part the jaw carries thins."""
    lip_moves = [np.linalg.norm(key_points(keys[name]) - rest, axis=1) for name in ("viseme_U", "viseme_PP") if name in keys]
    if not lip_moves or "faceLipVolume" not in keys or "jawOpen" not in keys:
        return np.zeros_like(rest)
    moved = np.maximum.reduce(lip_moves)
    lip = np.where((moved >= moved.max() * 0.45) & ((key_points(keys["jawOpen"]) - rest)[:, 2] < -0.001))[0]
    tree = KDTree(len(lip))
    for index in lip:
        tree.insert(rest[index], int(index))
    tree.balance()
    weight = np.zeros(len(rest))
    near = np.where(np.linalg.norm(rest - rest[lip].mean(axis=0), axis=1) < 0.05)[0]
    for index in near:
        weight[index] = max(0.0, 1 - tree.find(rest[index])[2] / LIP_FADE_M)
    return (key_points(keys["faceLipVolume"]) - rest) * (weight * amount)[:, None]


def mouth_back(keys, rest, amount):
    """The lips and the skin around them, back (+y, the face looks to -y), with a smooth edge."""
    lip_moves = [np.linalg.norm(key_points(keys[name]) - rest, axis=1) for name in ("viseme_U", "viseme_PP") if name in keys]
    if not lip_moves or not amount:
        return np.zeros_like(rest)
    moved = np.maximum.reduce(lip_moves)
    lip = np.where(moved >= moved.max() * 0.45)[0]
    tree = KDTree(len(lip))
    for index in lip:
        tree.insert(rest[index], int(index))
    tree.balance()
    move = np.zeros_like(rest)
    reach = MOUTH_CORE_M + MOUTH_FADE_M
    for index in np.where(np.linalg.norm(rest - rest[lip].mean(axis=0), axis=1) < 0.08)[0]:
        t = min(1.0, max(0.0, 1 - (tree.find(rest[index])[2] - MOUTH_CORE_M) / MOUTH_FADE_M)) if tree.find(rest[index])[2] < reach else 0.0
        move[index, 1] = amount * t * t * (3 - 2 * t)
    return move


def soften_pout(body, forward):
    keys = body.data.shape_keys.key_blocks if body.data.shape_keys else None
    if not keys or forward == 1.0:
        return
    rest = key_points(keys[0])
    for name in (name for name in POUT_VISEMES if name in keys):
        delta = key_points(keys[name]) - rest
        delta[:, 1] = np.where(delta[:, 1] < 0, delta[:, 1] * forward, delta[:, 1])
        keys[name].data.foreach_set("co", (rest + delta).ravel())
    body.data.update()


def calm_look(meshes, lip_thin=0.0, back=0.0):
    for obj in meshes:
        keys = obj.data.shape_keys.key_blocks if obj.data.shape_keys else None
        if not keys:
            continue
        rest = key_points(keys[0])
        move = sum(((key_points(keys[name]) - rest) * value for name, value in CALM_LOOK.items() if name in keys), np.zeros_like(rest))
        if obj.name == "Body" and lip_thin:
            move = move + lip_thinning(keys, rest, lip_thin)
        if obj.name == "Body" and back:
            move = move + mouth_back(keys, rest, back)
        if back and ("teeth" in obj.name or "tongue" in obj.name):
            move[:, 1] += back
        if "eyebrow" in obj.name:
            move = move + brow_line_move(rest + move)
        if "teeth" in obj.name or "tongue" in obj.name:
            move[:, 2] -= TEETH_DOWN_M if "teeth" in obj.name else TONGUE_DOWN_M
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


def soften_brows(obj, skin):
    """`skin` is the skin's median colour in the map's own (sRGB) values."""
    for material in obj.data.materials:
        for node in material.node_tree.nodes:
            if node.type != "TEX_IMAGE" or not node.image:
                continue
            width, height = node.image.size
            pixels = np.array(node.image.pixels[:], dtype=np.float32).reshape(height, width, 4)
            pixels[:, :, :3] = pixels[:, :, :3] * (1 - BROW_TOWARD_SKIN) + np.asarray(skin[:3], np.float32) * BROW_TOWARD_SKIN
            alpha = pixels[:, :, 3]
            soft = sum(np.roll(np.roll(alpha, dy, 0), dx, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)) / 9
            pixels[:, :, 3] = np.minimum(alpha, soft) * BROW_ALPHA
            node.image.pixels[:] = pixels.ravel()
            node.image.pack()


def set_map(obj, image):
    for node in obj.active_material.node_tree.nodes:
        if node.type == "TEX_IMAGE":
            node.image = image


def clean_face(body, added, job, env):
    for obj in [item for item in added if item.type == "MESH" and "eyelash" in item.name.lower()]:
        added.remove(obj)
        bpy.data.objects.remove(obj, do_unlink=True)
    weld_seams(body)
    sex = "female" if str(job.get("kit", "")).startswith("female") else "male"
    calm_look([body, *(obj for obj in added if obj.type == "MESH" and any(word in obj.name for word in ("eyebrow", "teeth", "tongue")))], LIP_THIN[sex], MOUTH_BACK[sex])
    soften_pout(body, POUT_FORWARD[sex])
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
                shader = obj.active_material.node_tree.nodes.get("Principled BSDF")
                if shader:
                    shader.inputs["Roughness"].default_value = EYE_ROUGHNESS
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
