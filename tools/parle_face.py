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

sys.path.insert(0, str(Path(__file__).parent))
import parle_mouth  # noqa: E402
import parle_tongue  # noqa: E402

EYE_TUCK_REACH_M = 0.02
EYE_TUCK_BEHIND_M = 0.001
FOREHEAD_HAIR_M = 0.085
FOREHEAD_HALF_M = 0.06
FOREHEAD_REACH_M = 0.06
# The kit's own morphs, baked into the rest shape: eyes open all around and brows pulled down read as anger on everybody (owner's review, 2026-11-07).
# The kit rests with the lips a little apart, and the teeth showed through in a white line (owner, 2026-11-07): the jaw rests a quarter closed. Teeth and tongue take it too.
CALM_LOOK = {"faceEyeOpen": -0.45, "faceBrowAngle": 0.6, "browInnerUp": 0.5, "jawOpen": -0.25}
# The reference women have about twice the height from the lips to the chin as from the nose to the lips; ours was nearly one to one (owner, 2026-11-09).
CHIN_LONGER = {"female": 0.3, "male": 0.0}
BROW_THIN = 0.75
# With the jaw a quarter closed, the upper teeth sat wholly behind the upper lip: speaking, the mouth showed the lower teeth and the
# tongue under a bare lip, and read as a snarl (owner, 2026-11-08). Measured on the pharmacy's attendant at the player's widest "a"
# (`viseme_aa` 0.55): incisor tip 1 mm above the lip's edge. Teeth go 3.5 mm down and the tongue 3 mm more, behind the lower teeth.
# The women's kit has full lips that bulged like a sausage in the player's perspective and light (owner, 2026-11-08: lower lip too
# big). The kit's own `faceLipVolume`, on the lip that follows the jaw and fading 4 mm around it: -1 on women, -0.5 on men; -2 creased.
# Next to four reference heads (owner, 2026-11-09) thinning was the wrong way: their lips are two full forms and hers a thin band. The shape comes from `full_lips`.
LIP_THIN = {"female": 0.0, "male": -0.5}
# The lower lip read as two, a crumpled ridge over the hollow under it: three lips with the upper one (owner, 2026-11-09, pharmacy). Thinner
# only creased it and filling the hollow made a band of its own. The outer skin of its lower half and of the hollow relaxes into one form.
UNDER_LIP_ROUNDS = {"female": 0, "male": 0}
UNDER_LIP_NEAR_M = 0.01
UNDER_LIP_FAR_M = 0.018
UNDER_LIP_SHARE = 0.45
# What lies this far behind the lip is the inside of the mouth, which the relaxing would pull out of place.
UNDER_LIP_BEHIND_M = 0.012
# The mouth went 3 mm back and the chin stayed: in profile it stuck out (owner, 2026-11-09, pharmacy). It goes back as much, fading towards the jaw and under it.
# In three reference women the chin sits about 3% of the head's height (6 mm) behind the lower lip; on the women's kit it was 2 mm ahead of it.
CHIN_BACK = {"female": 0.0035, "male": 0.0}
# Each lip a form of its own, as on the reference heads: forward at the middle of its height, the lower one taller, and a soft hollow under it. All in metres; the lips' meeting line stays.
LIPS_FULL = {"female": {"upper_forward": 0.0, "upper_at": 0.0045, "upper_wide": 0.0035, "upper_grow": 0.0012, "lower_forward": 0.0005, "lower_at": 0.0055, "lower_wide": 0.0042, "lower_grow": 0.002, "hollow_back": 0.003, "hollow_at": 0.0135, "hollow_wide": 0.004, "roll_back": 0.0032, "roll_wide": 0.0028}, "male": None}
LIPS_KEEP_LINE_M = 0.0012
LIPS_CUPID = {"female": 0.8, "male": 0.0}
# In the wide-open vowels the lining of the mouth came out through the cheek in two cuts (owner, 2026-11-09).
MOUTH_LINING_REACH_M = 0.08
GUM_TOOTH_MOST_POINTS = 60
GUM_KEEP_M = 0.009
GUM_TORN_M = 0.005
# Held still, the cheek's skin still closed over it when the jaw dropped: it also sits further in, past 2 cm from the middle.
MOUTH_LINING_INSET_M = 0.009
MOUTH_LINING_INSET_FROM_M = 0.02
MOUTH_LINING_INSET_FADE_M = 0.02
LIP_FADE_M = 0.004
# Thinner, the women's lips still stood out in profile like a beak and the lower one hung forward (owner, 2026-11-08, pharmacy). The
# whole mouth goes 3 mm back: lips and what lies 6 mm around them fully, fading over 20 mm more, teeth and tongue with them. Only the
# lower lip back opened a gap between the lips; 4 mm flattened the face.
# Next to photographs in profile, at 3 mm the lips no longer stood out from the face (owner, 2026-11-09): 1.5 mm.
MOUTH_BACK = {"female": 0.0052, "male": 0.0}
# Speaking, the rounded vowels pushed the full lips forward into a big beak, and the player mostly shows her mid-sentence: the
# forward part (-y) of the rounded visemes keeps 40% on women. The vowel still reads as round.
POUT_FORWARD = {"female": 0.4, "male": 1.0}
POUT_VISEMES = ("viseme_U", "viseme_O", "viseme_CH", "viseme_RR")
# Seen as the player frames her (front, eye level), the women's mouth still read as a pout: it was short and the lips round. The
# kit's own faceMouthWidth +0.8 lengthens it; thinning the lips further opened a gap between them (owner, 2026-11-08, pharmacy).
# At +0.8 it read as a long flat band next to photographs (owner, 2026-11-09): +0.4.
MOUTH_WIDE = {"female": 0.4, "male": 0.0}
MOUTH_CORE_M = 0.006
MOUTH_FADE_M = 0.02
TEETH_DOWN_M = 0.0035
TONGUE_DOWN_M = 0.009
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


def full_lips(keys, rest, shape):
    """Sculpts both lips and the hollow under the lower one on the outer skin."""
    move = np.zeros_like(rest)
    if not shape:
        return move
    moves = [np.linalg.norm(key_points(keys[name]) - rest, axis=1) for name in ("viseme_U", "viseme_PP") if name in keys]
    moved = np.maximum.reduce(moves)
    lips = np.where(moved >= moved.max() * 0.45)[0]
    jaw = (key_points(keys["jawOpen"]) - rest)[:, 2] < -0.001
    lower, upper = lips[jaw[lips]], lips[~jaw[lips]]
    middle = rest[lips, 0].mean()
    half = np.ptp(rest[lips, 0]) / 2
    x_lower, x_upper = rest[lower, 0] - middle, rest[upper, 0] - middle
    edges = np.linspace(-half, half, 15)
    centres, line = [], []
    for start, stop in zip(edges[:-1], edges[1:]):
        low, up = rest[lower[(x_lower >= start) & (x_lower < stop)], 2], rest[upper[(x_upper >= start) & (x_upper < stop)], 2]
        if len(low) and len(up):
            centres.append((start + stop) / 2)
            line.append((low.max() + up.min()) / 2)
    curve = np.polyfit(centres, line, 4)
    x = rest[:, 0] - middle
    height = rest[:, 2] - np.polyval(curve, np.clip(x, -half, half))
    across = np.sqrt(np.clip(1 - (x / (half * 1.05)) ** 2, 0, 1))
    outer = rest[:, 1] < rest[lips, 1].mean() + 0.012
    gauss = lambda value, at, wide: np.exp(-(((value - at) / wide) ** 2))
    keep = np.clip(np.abs(height) / LIPS_KEEP_LINE_M, 0, 1)
    up = height > 0
    move[:, 1] -= (shape["upper_forward"] * gauss(height, shape["upper_at"] * across, shape["upper_wide"]) * across * keep * up + shape["lower_forward"] * gauss(height, -shape["lower_at"] * across, shape["lower_wide"]) * across * keep * ~up)
    move[:, 1] += shape["hollow_back"] * gauss(height, -shape["hollow_at"], shape["hollow_wide"]) * np.clip(across * 1.6, 0, 1) * ~up
    # The kit's lips are furthest forward right at the line where they meet, a beak in profile; real ones roll in towards it. Both sides move alike, so they still touch.
    move[:, 1] += shape["roll_back"] * gauss(height, 0.0, shape["roll_wide"]) * across ** 2
    move[:, 2] += shape["upper_grow"] * np.clip(height / 0.008, 0, 1) * gauss(height, 0.008, 0.008) * across * up
    move[:, 2] -= shape["lower_grow"] * np.clip(-height / 0.01, 0, 1) * gauss(height, -0.01, 0.01) * across * ~up
    move[~outer] = 0
    return move


def chin_back(rest, lip, amount):
    """The move, backwards (+y), of the chin: starts under the lip, fades out to the sides, behind the jawbone's front and on the neck."""
    move = np.zeros_like(rest)
    if not amount or not len(lip):
        return move
    low, middle = rest[lip, 2].min(), rest[lip, 0].mean()
    below = rest[:, 2] < low
    front = rest[below & (np.abs(rest[:, 0] - middle) < 0.01), 1].min()
    start = np.clip((low - 0.004 - rest[:, 2]) / 0.012, 0, 1)
    across = np.clip((0.06 - np.abs(rest[:, 0] - middle)) / 0.03, 0, 1)
    deep = np.clip((front + 0.075 - rest[:, 1]) / 0.03, 0, 1)
    floor = np.clip((rest[:, 2] - (low - 0.075)) / 0.02, 0, 1)
    move[:, 1] = amount * start * start * (3 - 2 * start) * across * deep * floor
    return move


def relaxed_under_lip(obj, rest, move, lip, rounds):
    """`move` after relaxing the outer skin from the middle of the lower lip down through the hollow under it. The lips' meeting line and the inside of the mouth stay."""
    if not rounds or not len(lip):
        return move
    tree = KDTree(len(lip))
    for index in lip:
        tree.insert(rest[index], int(index))
    tree.balance()
    share = np.zeros(len(rest))
    for index in np.where(np.linalg.norm(rest - rest[lip].mean(axis=0), axis=1) < 0.07)[0]:
        share[index] = np.clip((UNDER_LIP_FAR_M - tree.find(rest[index])[2]) / (UNDER_LIP_FAR_M - UNDER_LIP_NEAR_M), 0, 1)
    low, high = rest[lip, 2].min(), rest[lip, 2].max()
    share *= np.clip((low + (high - low) * UNDER_LIP_SHARE - rest[:, 2]) / 0.002, 0, 1)
    share *= np.clip((rest[lip, 1].mean() + UNDER_LIP_BEHIND_M - rest[:, 1]) / 0.004, 0, 1)
    edges = np.empty(len(obj.data.edges) * 2, int)
    obj.data.edges.foreach_get("vertices", edges)
    edges = edges.reshape(-1, 2)
    placed = rest + move
    count = np.maximum(np.bincount(edges.ravel(), minlength=len(placed)), 1)[:, None]
    for _ in range(rounds):
        total = np.zeros_like(placed)
        np.add.at(total, edges[:, 0], placed[edges[:, 1]])
        np.add.at(total, edges[:, 1], placed[edges[:, 0]])
        placed = placed + (total / count - placed) * (share * 0.5)[:, None]
    return placed - rest


def lining_points(rest, polygons, lip):
    """The points that line the inside of the mouth: there is skin ahead of them and skin beside them. Only ahead also caught the jaw and the neck, under the chin."""
    tree = BVHTree.FromPolygons(rest.tolist(), polygons)
    middle = rest[lip, 0].mean()
    near = np.where((np.linalg.norm(rest - rest[lip].mean(axis=0), axis=1) < MOUTH_LINING_REACH_M) & (rest[:, 1] > rest[lip, 1].mean()))[0]

    def hidden(index, direction):
        return tree.ray_cast(Vector(rest[index]) + direction * 0.0005, direction, MOUTH_LINING_REACH_M * 2)[0] is not None

    return np.array([index for index in near if hidden(index, Vector((0, -1, 0))) and hidden(index, Vector((1 if rest[index, 0] >= middle else -1, 0, 0)))], int)


def held_in(rest, delta, lining, middle):
    """`delta` with the lining never going forward or out to the side, the two ways it can come through the cheek."""
    delta = delta.copy()
    delta[lining, 1] = np.maximum(delta[lining, 1], 0.0)
    outward = np.sign(rest[lining, 0] - middle) * delta[lining, 0] > 0
    delta[lining[outward], 0] = 0.0
    return delta


def keep_lining_inside(body):
    keys = body.data.shape_keys.key_blocks
    rest = key_points(keys[0])
    lip = carried_lip(keys, rest)
    if not len(lip):
        return
    lining = lining_points(rest, [list(polygon.vertices) for polygon in body.data.polygons], lip)
    for block in keys[1:]:
        if parle_tongue.is_mouth_key(block.name):
            block.data.foreach_set("co", (rest + held_in(rest, key_points(block) - rest, lining, rest[lip, 0].mean())).ravel())
    body.data.update()
    side = rest[lining, 0] - rest[lip, 0].mean()
    move = np.zeros_like(rest)
    move[lining, 0] = -np.sign(side) * MOUTH_LINING_INSET_M * np.clip((np.abs(side) - MOUTH_LINING_INSET_FROM_M) / MOUTH_LINING_INSET_FADE_M, 0, 1)
    move_rest(body, move)
    print("FORRO DA BOCA", len(lining), "pontos")


def trim_gums(teeth):
    """Drops the parts of the denture nobody should see: gum far from any tooth (the broken palate and floor) and the faces stretched between the two arches."""
    keys = teeth.data.shape_keys.key_blocks
    rest = key_points(keys[0])
    jaw = np.linalg.norm(key_points(keys["jawOpen"]) - rest, axis=1)
    mesh = bmesh.new()
    mesh.from_mesh(teeth.data)
    mesh.verts.ensure_lookup_table()
    seen, crowns = set(), []
    for start in mesh.verts:
        if start.index in seen:
            continue
        part, stack = [], [start]
        seen.add(start.index)
        while stack:
            vertex = stack.pop()
            part.append(vertex.index)
            for edge in vertex.link_edges:
                other = edge.other_vert(vertex)
                if other.index not in seen:
                    seen.add(other.index)
                    stack.append(other)
        if len(part) <= GUM_TOOTH_MOST_POINTS:
            crowns += part
    tree = KDTree(len(crowns))
    for index in crowns:
        tree.insert(rest[index], index)
    tree.balance()
    crown = set(crowns)
    def useless(face):
        points = [vertex.index for vertex in face.verts]
        if any(index in crown for index in points):
            return False
        moves = jaw[points]
        return moves.max() - moves.min() > GUM_TORN_M or all(tree.find(rest[index])[2] > GUM_KEEP_M for index in points)
    doomed = [face for face in mesh.faces if useless(face)]
    bmesh.ops.delete(mesh, geom=doomed, context="FACES_ONLY")
    bmesh.ops.delete(mesh, geom=[vertex for vertex in mesh.verts if not vertex.link_faces], context="VERTS")
    mesh.to_mesh(teeth.data)
    mesh.free()
    teeth.data.update()
    print("GENGIVA aparada:", len(doomed), "faces")


def carried_lip(keys, rest):
    moves = [np.linalg.norm(key_points(keys[name]) - rest, axis=1) for name in ("viseme_U", "viseme_PP") if name in keys]
    if not moves or "jawOpen" not in keys:
        return np.array([], int)
    moved = np.maximum.reduce(moves)
    return np.where((moved >= moved.max() * 0.45) & ((key_points(keys["jawOpen"]) - rest)[:, 2] < -0.001))[0]


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


def calm_look(meshes, lip_thin=0.0, back=0.0, wide=0.0, chin=0.0, rounds=0, lips=None, bow=0.0, longer=0.0):
    look = {**CALM_LOOK, **({"faceMouthWidth": wide} if wide else {}), **({"faceCupidBow": bow} if bow else {}), **({"faceVChin": longer} if longer else {})}
    for obj in meshes:
        keys = obj.data.shape_keys.key_blocks if obj.data.shape_keys else None
        if not keys:
            continue
        rest = key_points(keys[0])
        move = sum(((key_points(keys[name]) - rest) * value for name, value in look.items() if name in keys), np.zeros_like(rest))
        if obj.name == "Body" and lip_thin:
            move = move + lip_thinning(keys, rest, lip_thin)
        if obj.name == "Body" and chin:
            move = move + chin_back(rest, carried_lip(keys, rest), chin)
        if obj.name == "Body" and lips:
            move = move + full_lips(keys, rest, lips)
        if obj.name == "Body" and back:
            move = move + mouth_back(keys, rest, back)
        if obj.name == "Body":
            move = relaxed_under_lip(obj, rest, move, carried_lip(keys, rest), rounds)
        if back and ("teeth" in obj.name or "tongue" in obj.name):
            move[:, 1] += back
        if "eyebrow" in obj.name:
            move = move + brow_line_move(rest + move)
        if "teeth" in obj.name or "tongue" in obj.name:
            move[:, 2] -= TEETH_DOWN_M if "teeth" in obj.name else TONGUE_DOWN_M
        if move.any():
            move_rest(obj, move)


def tongue_follows_jaw(meshes):
    """Runs after the rest-shape rules, so the moves it reads are the ones the game plays."""
    teeth = next((obj for obj in meshes if "teeth" in obj.name), None)
    tongue = next((obj for obj in meshes if "tongue" in obj.name), None)
    if teeth is None or tongue is None or "jawOpen" not in teeth.data.shape_keys.key_blocks:
        return
    teeth_keys, tongue_keys = teeth.data.shape_keys.key_blocks, tongue.data.shape_keys.key_blocks
    teeth_rest, tongue_rest = key_points(teeth_keys[0]), key_points(tongue_keys[0])
    lower = parle_tongue.lower_teeth(key_points(teeth_keys["jawOpen"]) - teeth_rest)
    carried = 0
    for block in tongue_keys[1:]:
        if not parle_tongue.is_mouth_key(block.name) or block.name not in teeth_keys:
            continue
        move = parle_tongue.carried(tongue_rest, key_points(teeth_keys[block.name]) - teeth_rest, lower)
        if move is not None:
            block.data.foreach_set("co", (key_points(block) + move).ravel())
            carried += 1
    print("LINGUA segue a mandibula em", carried, "formas")


def shade_map_of(obj, shade, gums=False, only=None, wine=False):
    """Multiplies the object's colour map by `shade`, one value per point, through its UV islands. `only`: a map the mesh shares; see parle_mouth."""
    image = next(node.image for node in obj.active_material.node_tree.nodes if node.type == "TEX_IMAGE" and node.image)
    width, height = image.size
    loops = obj.data.uv_layers.active.data
    triangles = [(polygon.loop_indices[0], second, third) for polygon in obj.data.polygons for second, third in zip(polygon.loop_indices[1:], polygon.loop_indices[2:])]
    uv = np.array([[loops[index].uv[:] for index in triangle] for triangle in triangles], np.float64)
    points = np.array([[obj.data.loops[index].vertex_index for index in triangle] for triangle in triangles])
    if only is not None:
        vertex_uv = np.zeros((len(obj.data.vertices), 2))
        for corner, point in zip(uv.reshape(-1, 2), points.ravel()):
            vertex_uv[point] = corner[0], 1 - corner[1]
        kept = parle_mouth.body_triangles(points, only, vertex_uv)
        uv, points = uv[kept], points[kept]
    canvas = parle_mouth.grow(parle_mouth.shade_map(width, height, uv, shade[points]), fill=None if only is None else 1.0)
    pixels = np.empty(width * height * 4, np.float32)
    image.pixels.foreach_get(pixels)
    image.pixels.foreach_set(parle_mouth.darken(pixels.reshape(height, width, 4), canvas, gums, wine).ravel())
    image.pack()


def shade_mouth(meshes):
    """Runs on the maps the export takes, after the teeth map went in, and on the rest shape the game plays."""
    teeth = next((obj for obj in meshes if "teeth" in obj.name), None)
    tongue = next((obj for obj in meshes if "tongue" in obj.name), None)
    if teeth is not None and teeth.data.shape_keys:
        rest = key_points(teeth.data.shape_keys.key_blocks[0])
        shade_map_of(teeth, parle_mouth.teeth_shade(rest), gums=True, wine=True)
    if tongue is not None:
        shade_map_of(tongue, parle_mouth.tongue_shade(key_points(tongue.data.shape_keys.key_blocks[0])))
    print("BOCA escurecida", bool(teeth), bool(tongue))


def shade_inside_of_mouth(body, teeth_rest):
    """The body's own inside of the mouth, on the skin map."""
    rest = key_points(body.data.shape_keys.key_blocks[0])
    loops = body.data.uv_layers.active.data
    uv = np.zeros((len(body.data.vertices), 2))
    for loop in body.data.loops:
        uv[loop.vertex_index] = loops[loop.index].uv[0], 1 - loops[loop.index].uv[1]
    triangles = np.array([(polygon.vertices[0], second, third) for polygon in body.data.polygons for second, third in zip(polygon.vertices[1:], polygon.vertices[2:])])
    shade_map_of(body, np.full(len(rest), parle_mouth.BODY_INSIDE), only=parle_mouth.body_inside(rest, uv, teeth_rest, triangles), wine=True)


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
    calm_look([body, *(obj for obj in added if obj.type == "MESH" and any(word in obj.name for word in ("eyebrow", "teeth", "tongue")))], LIP_THIN[sex], MOUTH_BACK[sex], MOUTH_WIDE[sex], CHIN_BACK[sex], UNDER_LIP_ROUNDS[sex], LIPS_FULL[sex], LIPS_CUPID[sex], CHIN_LONGER[sex])
    soften_pout(body, POUT_FORWARD[sex])
    keep_lining_inside(body)
    tongue_follows_jaw([obj for obj in added if obj.type == "MESH" and obj.data.shape_keys])
    for obj in [item for item in added if item.type == "MESH" and "teeth" in item.name]:
        trim_gums(obj)
    opening = os.path.join(os.path.dirname(job["out"]), f"eye-open-{job.get('kit')}.json")
    if os.path.exists(opening):
        tuck_eyes(body, added, opening)
    if env.get("CAST_TEETH"):
        teeth_map = bpy.data.images.load(parle_mouth.clean_teeth(env["CAST_TEETH"]))
        teeth_map.pack()
        for obj in added:
            if obj.type == "MESH" and "teeth" in obj.name:
                material = obj.active_material
                material.blend_method = "OPAQUE"
                for link in [link for link in material.node_tree.links if link.to_socket.name == "Alpha"]:
                    material.node_tree.links.remove(link)
                set_map(obj, teeth_map)
    shade_mouth([obj for obj in added if obj.type == "MESH" and obj.data.shape_keys])
    teeth_for_inside = next((obj for obj in added if obj.type == "MESH" and "teeth" in obj.name), None)
    if teeth_for_inside is not None and body.data.shape_keys:
        shade_inside_of_mouth(body, key_points(teeth_for_inside.data.shape_keys.key_blocks[0]))
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
