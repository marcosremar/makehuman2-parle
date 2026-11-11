"""
    License information: data/licenses/makehuman_license.txt

    Hair rules of the Parle cast, run inside Blender while a character is assembled.

    - MakeHuman hair is cards cut by the alpha of its texture: without that alpha each card shows
      whole, in hard steps.
    - Game hair goes from the game skull to ours through one smooth stretch per direction. A stretch
      per point tears it apart, because two neighbours can hit the ear and the skull behind it.
    - Hair that ended up under the shirt or under the skin comes out to the surface.
"""
import os

import bmesh
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from parle_bald import bare_crown, line_planes

MH_HAIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".local/mhd/data/hair/hm08")
# Strand cards keep their edges solid and let the player sort them as opaque.
COIF_CUTOFF = 0.4
# Part of the strand shade stays in the dyed map: flat colour turned the hair into plastic.
COIF_SHADE = 0.6

SKULL_DIRECTIONS = 1500
SKULL_REACH_M = 0.4
SKULL_LEAST = 0.6
SKULL_MOST = 1.3
SKULL_BELOW_MOUTH_M = 0.03
SKULL_SMOOTH = 0.12
SKULL_SAMPLE = 0.04

# Rigid: a little bigger than the skull, and raised so that the rim does not cut the eyebrow.
HAT_GROW = 1.03
HAT_LIFT_M = 0.008
# Through a hole in the forehead the inside of the far wall of the hat shows: pale shards below the rim. Skin only leaves from well inside the crown.
CROWN_DEEP_M = 0.025
# Under a cap the hair that goes past the crown's rim pokes through the cloth at the back.
CAP_RIM_M = 0.012
HAT_HAIR_FRONT_M = 0.06
HAT_NAMES = ("Acc_Cap", "Acc_Straw")
# Crown and brim are two pieces: the forehead shows between them unless the brim goes up to meet the crown.
BRIM_OVERLAP_M = 0.003
BRIM_ROOT_PERCENTILE = 25
BRIM_REACH_M = 0.02
BRIM_GAP_PERCENTILE = 90
# A rigid hat around a head of another shape: the skin came through under the rim, in jagged shards on the forehead.
HAT_MOST = 1.25
HAT_ENCLOSE = 99.8
# Rim standing a little off the skin: on the skin, the low-poly edge and the forehead cross each other in a zigzag.
HAT_GAP = (1.03, 1.05)
HAIR_ROUGH = 0.8

CLEAR_REACH_M = 0.09
CLEAR_GAP_M = 0.012
CLEAR_ROUNDS = 3
# Pushed out point by point along the skin's own normal, the long hair of the pharmacy's attendant crumpled round each ear into a knot
# of spikes: an ear looks every way at once (2026-11-11, seen from the front as a ragged edge down each cheek). Above the mouth a point
# that is too near the skin goes out from the middle of the head, level, to just outside the skin that stands furthest out that way,
# found by looking in from this far; and what was pushed carries its neighbours with it, each ring by this share of the one before,
# this many rings out, so the hair swells where it was crumpled. (Going out that way by as much as the normal asked ran away: where the normal lies across it, no step out
# is ever enough, and the hair stood out 10 cm in two points.)
CLEAR_FROM_M = 0.3
CLEAR_SPREAD = 6
CLEAR_CARRIED = 0.7
# A cap of hair lies on the skull: this share of its points no further than this from the skin. Measured on four people on 2026-11-11:
# the caps have 98 to 100% of their points within 2 cm, long hair 55%, curls 47%, an afro 34%, a ponytail 15%.
CAP_NEAR_M = 0.02
CAP_SHARE = 0.9
# The skull is ours, the hair came from another: where it sinks into the forehead, the skin cuts it in a serrated line.
SCALP_GAP_M = 0.006
# Strand tips on the collar pile up as shards: in front, they stop on the neck, with room to spare.
CHEST_ABOVE_NECK_M = 0.05
CHEST_FRONT_M = 0.03
LOOSE_MARGIN_M = 0.01
# Beside the neck it reaches further back, over the shoulder; the tail in the middle of the back stays whole.
SHOULDER_SIDE_M = 0.05
SHOULDER_BACK_M = 0.07
# The short game cut has pale patches in its map at the sideburn: a grey plate on the temple.
CALM_PERCENTILE = 60
CALM_MOST = 1.1
# Grey hair in this map is static on a screen: strand against strand, a smaller difference.
CALM_CONTRAST = 0.4
COARSE_HAIR_POINTS = 300
SIDEBURN_ABOVE_EYE_M = 0.045


def coif_map(asset):
    """The .mhclo names its material and the material names its map: neither has to follow the asset."""
    folder = os.path.join(MH_HAIR, asset)

    def value(path, key):
        with open(path, encoding="utf-8", errors="replace") as lines:
            return next(line.split(None, 1)[1].strip() for line in lines if line.startswith(key + " "))

    return os.path.join(folder, value(os.path.join(folder, value(os.path.join(folder, f"{asset}.mhclo"), "material")), "diffuseTexture"))


def dye_coif(obj, asset, colour, roughness):
    """Colour from the cast, shade and cut from the MakeHuman map."""
    image = bpy.data.images.load(coif_map(asset))
    width, height = image.size
    pixels = np.array(image.pixels[:], dtype=np.float32).reshape(-1, 4)
    seen = pixels[pixels[:, 3] > COIF_CUTOFF, :3].mean(axis=1)
    middle = max(float(seen.mean()) if len(seen) else 1.0, 1e-3)
    shade = np.clip(1 + (pixels[:, :3].mean(axis=1) / middle - 1) * COIF_SHADE, 0.5, 1.4)[:, None]
    pixels[:, :3] = np.clip(np.array(colour[:3], dtype=np.float32) * shade, 0, 1)
    dyed = bpy.data.images.new(f"{asset}_cabelo", width, height, alpha=True)
    dyed.pixels[:] = pixels.ravel()
    dyed.pack()
    bpy.data.images.remove(image)

    # `Coif` is the player's strand surface (`NpcLook.Lash`).
    material = bpy.data.materials.new("Coif_Cabelo")
    material.use_nodes = True
    tree = material.node_tree
    shader = next(node for node in tree.nodes if node.type == "BSDF_PRINCIPLED")
    shader.inputs["Roughness"].default_value = roughness
    texture = tree.nodes.new("ShaderNodeTexImage")
    texture.image = dyed
    tree.links.new(texture.outputs["Color"], shader.inputs["Base Color"])
    cut = tree.nodes.new("ShaderNodeMath")
    cut.operation = "GREATER_THAN"
    cut.inputs[1].default_value = COIF_CUTOFF
    tree.links.new(texture.outputs["Alpha"], cut.inputs[0])
    tree.links.new(cut.outputs[0], shader.inputs["Alpha"])
    material.use_backface_culling = False
    obj.data.materials.clear()
    obj.data.materials.append(material)


def sphere_directions(count):
    golden = np.pi * (3 - np.sqrt(5))
    height = 1 - 2 * (np.arange(count) + 0.5) / count
    ring = np.sqrt(1 - height * height)
    angle = golden * np.arange(count)
    return np.stack([ring * np.cos(angle), ring * np.sin(angle), height], 1)


def spread(ways, values, wanted, reach):
    """Weighted mean over the sphere; dot products keep it free of angle wrap."""
    weights = np.exp((wanted @ ways.T - 1) / reach)
    return (weights * values).sum(axis=1) / np.maximum(weights.sum(axis=1), 1e-9)


def skull_field(theirs, mine, middle, mouth_z):
    """Stretch of the skull per direction, smooth; only rays that end on the head count, the neck says nothing about the skull."""
    ways, ratios = [], []
    for way in sphere_directions(SKULL_DIRECTIONS):
        direction = Vector(way)
        old, new = theirs.ray_cast(middle, direction, SKULL_REACH_M)[3], mine.ray_cast(middle, direction, SKULL_REACH_M)[3]
        if old and new and (middle + direction * new).z > mouth_z - SKULL_BELOW_MOUTH_M:
            ways.append(way)
            ratios.append(min(SKULL_MOST, max(SKULL_LEAST, new / old)))
    ways, ratios = np.array(ways), np.array(ratios)
    return ways, spread(ways, ratios, ways, SKULL_SMOOTH)


def fit_skull(points, skulls, mouth_z):
    theirs, mine, middle = skulls
    ways, ratios = skull_field(theirs, mine, middle, mouth_z)
    origin = np.array(middle)
    offsets = np.array([tuple(point) for point in points]) - origin
    length = np.maximum(np.linalg.norm(offsets, axis=1, keepdims=True), 1e-9)
    # Long hair is many points: per batch, or the weight table takes gigabytes.
    stretch = np.concatenate([spread(ways, ratios, part, SKULL_SAMPLE) for part in np.array_split(offsets / length, max(1, len(offsets) // 2000 + 1))])
    return [Vector(point) for point in origin + offsets * stretch[:, None]]


def fit_hat(points, skulls):
    """One ratio for a rigid piece, or its brim tears point by point."""
    theirs, mine, middle = skulls
    up = Vector((0, 0, 1))
    old, new = theirs.ray_cast(middle, up, SKULL_REACH_M)[3], mine.ray_cast(middle, up, SKULL_REACH_M)[3]
    ratio = min(SKULL_MOST, max(SKULL_LEAST, new / old)) * HAT_GROW if old and new else 1.0
    return [middle + (point - middle) * ratio + up * HAT_LIFT_M for point in points]


def close_brim(hats, middle):
    """Measured where they meet: over the root of the brim, the lowest point of the crown right above it."""
    brims = [obj for obj in hats if "Brim" in obj.name]
    crown = np.array([tuple(obj.matrix_world @ vertex.co) for obj in hats if "Brim" not in obj.name for vertex in obj.data.vertices])
    if not brims or not len(crown):
        return 0.0
    brim = np.array([tuple(obj.matrix_world @ vertex.co) for obj in brims for vertex in obj.data.vertices])
    root = brim[np.linalg.norm(brim[:, :2] - np.array(middle)[:2], axis=1) < np.percentile(np.linalg.norm(brim[:, :2] - np.array(middle)[:2], axis=1), BRIM_ROOT_PERCENTILE)]
    gaps = []
    for point in root:
        above = crown[np.linalg.norm(crown[:, :2] - point[:2], axis=1) < BRIM_REACH_M]
        if len(above):
            gaps.append(above[:, 2].min() - point[2])
    rise = max(0.0, float(np.percentile(gaps, BRIM_GAP_PERCENTILE)) + BRIM_OVERLAP_M) if gaps else 0.0
    for obj in brims:
        for vertex in obj.data.vertices:
            vertex.co.z += rise
        obj.data.update()
    return rise


def enclose_head(hats, body, middle):
    """Stretches the hat as one piece, sideways and front to back, until the head fits inside it. Only the crown counts: the skin always passes under a brim."""
    crown = [obj for obj in hats if "Brim" not in obj.name]
    if not crown:
        return
    tree = surface_tree(crown)
    rim = min((obj.matrix_world @ vertex.co).z for obj in crown for vertex in obj.data.vertices)
    need = {0: [1.0], 1: [1.0]}
    for vertex in body.data.vertices:
        point = body.matrix_world @ vertex.co
        if point.z > rim:
            way = point - middle
            reach = way.length
            hit = tree.ray_cast(middle, way.normalized(), SKULL_REACH_M)[3]
            if hit is not None and reach > hit:
                need[0 if abs(way.x) > abs(way.y) else 1].append(reach / hit)
    scale = [min(HAT_MOST, float(np.percentile(need[axis], HAT_ENCLOSE)) * HAT_GAP[axis]) for axis in (0, 1)]
    for obj in hats:
        for vertex in obj.data.vertices:
            vertex.co.x = middle.x + (vertex.co.x - middle.x) * scale[0]
            vertex.co.y = middle.y + (vertex.co.y - middle.y) * scale[1]
        obj.data.update()
    print("CHAPEU", [round(value, 3) for value in scale], [obj.name for obj in hats], "aba subiu", round(close_brim(hats, middle) * 1000, 1), "mm")


def clear_under_crown(body, hats):
    """Skin that a hat of another shape does not contain comes through its top and back. Looked at from the hat's own axis, what the crown covers, and still covers a little further down, is not seen."""
    crown = [obj for obj in hats if "Brim" not in obj.name]
    points = [obj.matrix_world @ vertex.co for obj in crown for vertex in obj.data.vertices]
    x, y = sum(point.x for point in points) / len(points), sum(point.y for point in points) / len(points)
    tree = surface_tree(crown)

    def covered(point):
        way = Vector((point.x - x, point.y - y, 0))
        if way.length < 1e-6:
            return True
        way.normalize()
        return all(tree.ray_cast(Vector((x, y, point.z - drop)), way, SKULL_REACH_M)[3] is not None for drop in (0, CROWN_DEEP_M))

    print("PELE SOB A COPA", delete_faces(body, covered))


def wear_hats(worn, body, middle, eye_y):
    """A game hat on our head: stretched to contain it, brim against the crown, and the hair that would come through it gone."""
    hats = [obj for obj in worn if obj.name.startswith(HAT_NAMES)]
    caps = [obj for obj in hats if obj.name in ("Acc_Cap", "Acc_Straw_Hat")]
    if not caps:
        return
    enclose_head(hats, body, middle)
    rim = min(vertex.co.z for vertex in caps[0].data.vertices) + CAP_RIM_M
    for obj in (item for item in worn if item.name.startswith("Acc_Hair_")):
        # Measured on the points as stored, like the hat's rim above.
        mesh = bmesh.new()
        mesh.from_mesh(obj.data)
        bmesh.ops.delete(mesh, geom=[face for face in mesh.faces if all(vert.co.z > rim for vert in face.verts)], context="FACES_ONLY")
        mesh.to_mesh(obj.data)
        mesh.free()
        trim_hair_under_hat(obj, eye_y + HAT_HAIR_FRONT_M)


def clear_skin_under_hats(body, meshes):
    hats = [obj for obj in meshes if obj.name.startswith(HAT_NAMES)]
    if not hats:
        return
    for brim in (obj for obj in hats if obj.name.startswith("Acc_Cap_Brim")):
        clear_under_cap(body, brim)
    clear_under_crown(body, hats)


def trim_hair_under_hat(obj, forward_y):
    """In front of the ears, hair under a hat is a patch on the forehead or nothing at all."""
    print("CABELO NA TESTA", obj.name, delete_faces(obj, lambda point: point.y < forward_y, "FACES"))


def matte_hair(obj):
    """Game hair came with the shine of a plastic toy next to the matte skin."""
    for slot in obj.material_slots:
        tree = slot.material.node_tree if slot.material and slot.material.use_nodes else None
        shader = next((node for node in tree.nodes if node.type == "BSDF_PRINCIPLED"), None) if tree else None
        if shader:
            for link in list(shader.inputs["Roughness"].links):
                tree.links.remove(link)
            shader.inputs["Roughness"].default_value = max(HAIR_ROUGH, shader.inputs["Roughness"].default_value)


def trim_sideburn(obj, eye_z):
    """The coarsest short cut ends in a flap beside the ear, which stuck out of the narrower head as a plate. The hair painted on the skin is the sideburn."""
    if len(obj.data.vertices) < COARSE_HAIR_POINTS:
        print("COSTELETA", obj.name, delete_faces(obj, lambda point: point.z < eye_z + SIDEBURN_ABOVE_EYE_M, "FACES"))


def calm_map(obj):
    """Caps how bright the map of a hair piece gets; the strands below that keep their shade."""
    for material in obj.data.materials:
        for node in material.node_tree.nodes:
            if node.type == "TEX_IMAGE" and node.image:
                pixels = np.array(node.image.pixels[:], dtype=np.float32).reshape(-1, 4)
                light = pixels[:, :3].mean(axis=1)
                most = float(np.percentile(light, CALM_PERCENTILE)) * CALM_MOST
                pixels[:, :3] *= np.minimum(1, most / np.maximum(light, 1e-4))[:, None]
                shown = pixels[:, 3] > COIF_CUTOFF
                centre = pixels[shown, :3].mean(axis=0) if shown.any() else pixels[:, :3].mean(axis=0)
                pixels[:, :3] = centre + (pixels[:, :3] - centre) * CALM_CONTRAST
                calm = bpy.data.images.new(f"{node.image.name}_calmo", *node.image.size, alpha=True)
                calm.pixels[:] = pixels.ravel()
                calm.pack()
                node.image = calm


def dye_scalp(obj, colour, roughness):
    """The braided scalp's own map is a grid made for a cap that does not come along: on our forehead it read as a pale stain."""
    material = bpy.data.materials.new("Penteado_Scalp_Liso")
    material.use_nodes = True
    shader = next(node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
    shader.inputs["Base Color"].default_value = colour
    shader.inputs["Roughness"].default_value = roughness
    obj.data.materials.clear()
    obj.data.materials.append(material)


# The game scalp is a few big faces: its edge on the forehead went up and down in square teeth (owner, 2026-11-08, restaurant). The
# open edge walks along itself towards the mean of its neighbours, keeping each point's distance from the skull's middle.
HAIRLINE_PASSES = 12
HAIRLINE_STEP = 0.5
HAIRLINE_SAME_M = 1e-5


def smooth_hairline(obj):
    """The open edge of a cap walks along itself towards the mean of its neighbours, each point keeping its distance from the
    skull's middle. Points at the same place are one point: a cap is cut along a seam of its picture, and smoothed as two edges the
    seam opened into a slit down the nape (Carlos, 2026-11-11)."""
    mesh = bmesh.new()
    mesh.from_mesh(obj.data)
    mesh.verts.ensure_lookup_table()
    place = lambda vert: tuple(round(value / HAIRLINE_SAME_M) for value in vert.co)
    shared = {}
    for vert in mesh.verts:
        shared.setdefault(place(vert), []).append(vert.index)
    sides = {}
    for edge in mesh.edges:
        key = tuple(sorted((place(edge.verts[0]), place(edge.verts[1]))))
        if key[0] != key[1]:
            sides[key] = sides.get(key, 0) + len(edge.link_faces)
    ring = {}
    for (first, second), faces in sides.items():
        if faces == 1:
            ring.setdefault(first, []).append(second)
            ring.setdefault(second, []).append(first)
    if not ring:
        mesh.free()
        return
    middle = sum((vert.co for vert in mesh.verts), Vector()) / len(mesh.verts)
    start = {key: mesh.verts[shared[key][0]].co.copy() for key in ring}
    points = dict(start)
    for _ in range(HAIRLINE_PASSES):
        points = {key: points[key].lerp(sum((points[other] for other in near), Vector()) / len(near), HAIRLINE_STEP) for key, near in ring.items()}
    keys = obj.data.shape_keys.key_blocks if obj.data.shape_keys else []
    for key, point in points.items():
        reach = (start[key] - middle).length
        step = middle + (point - middle).normalized() * reach - start[key]
        for index in shared[key]:
            obj.data.vertices[index].co += step
            for block in keys:
                block.data[index].co += step
    mesh.free()
    obj.data.update()
    print("LINHA DO CABELO", obj.name, len(points), "pontos")


def bald_crown(body, hairs):
    """Cuts the hair at the balding line of `parle_bald`, the one that clears the scalp on the skin map: first along the line, so the border is
    the line itself and not the teeth of this hair's few large faces, then whatever stayed above it."""
    head = np.array([tuple(body.matrix_world @ vert.co) for vert in body.data.vertices])
    for hair in hairs:
        back = hair.matrix_world.inverted()
        mesh = bmesh.new()
        mesh.from_mesh(hair.data)
        for point, normal in line_planes(head):
            geometry = list(mesh.verts) + list(mesh.edges) + list(mesh.faces)
            bmesh.ops.bisect_plane(mesh, geom=geometry, plane_co=back @ Vector(point), plane_no=hair.matrix_world.to_3x3().transposed() @ Vector(normal))
        mesh.to_mesh(hair.data)
        mesh.free()
        hair.data.update()
        delete_faces(hair, lambda point: bool(bare_crown(np.array([tuple(point)]), head)[0]), "FACES")


def hugs_scalp(obj, body):
    """Whether a piece of hair is a cap on the skull (the scalp under a coif, a cropped cut) and not hair that stands off it: nearly
    all of its points lie right on the skin. The edge of such a cap is a hairline, and takes `smooth_hairline`: the game's caps are a
    few big faces, and on Carlos the edge went round the nape and the temple in square steps (2026-11-11), as the restaurant's had on
    the forehead, because only the piece called Hair_Scalp was smoothed."""
    tree = surface_tree([body])
    world = obj.matrix_world
    near = sum(1 for vertex in obj.data.vertices if tree.find_nearest(world @ vertex.co, CAP_NEAR_M)[0] is not None)
    return bool(len(obj.data.vertices)) and near >= CAP_SHARE * len(obj.data.vertices)


def cut_at(obj, planes):
    """Splits the faces of a piece along each plane (a point and a normal, in the world), so that a cut made after it ends on the
    plane and not on the teeth of the faces that crossed it."""
    mesh = bmesh.new()
    mesh.from_mesh(obj.data)
    inverse, turn = obj.matrix_world.inverted(), obj.matrix_world.to_3x3().transposed()
    for point, normal in planes:
        bmesh.ops.bisect_plane(mesh, geom=mesh.verts[:] + mesh.edges[:] + mesh.faces[:], dist=1e-6, plane_co=inverse @ Vector(point),
                               plane_no=(turn @ Vector(normal)).normalized(), clear_inner=False, clear_outer=False)
    mesh.to_mesh(obj.data)
    mesh.free()
    obj.data.update()


def delete_faces(obj, wanted, context="FACES_ONLY"):
    mesh = bmesh.new()
    mesh.from_mesh(obj.data)
    world = obj.matrix_world
    faces = [face for face in mesh.faces if wanted(world @ face.calc_center_median()) and (context == "FACES" or all(wanted(world @ vert.co) for vert in face.verts))]
    bmesh.ops.delete(mesh, geom=faces, context=context)
    mesh.to_mesh(obj.data)
    mesh.free()
    obj.data.update()
    return len(faces)


def trim_chest_hair(obj, neck):
    """A strand that reaches the chest has to lie flat on the collar, and there it looks like a pile of shards. It stops on the neck, and what is left loose below goes too.
    The faces are split along the borders of what goes before any is taken away: whole faces gone left the hem of the pharmacy
    attendant's hair in saw teeth (2026-11-11)."""
    def on_chest(point):
        reach = SHOULDER_BACK_M if abs(point.x - neck.x) > SHOULDER_SIDE_M else CHEST_FRONT_M
        return point.z < neck.z + CHEST_ABOVE_NECK_M and point.y < neck.y + reach

    cut_at(obj, [((0, 0, neck.z + CHEST_ABOVE_NECK_M), (0, 0, 1)), ((0, neck.y + CHEST_FRONT_M, 0), (0, 1, 0)), ((0, neck.y + SHOULDER_BACK_M, 0), (0, 1, 0)),
                 ((neck.x - SHOULDER_SIDE_M, 0, 0), (1, 0, 0)), ((neck.x + SHOULDER_SIDE_M, 0, 0), (1, 0, 0))])
    cut = delete_faces(obj, on_chest, "FACES")
    mesh = bmesh.new()
    mesh.from_mesh(obj.data)
    world = obj.matrix_world
    seen, loose = set(), []
    for start in mesh.verts:
        if start in seen:
            continue
        part, stack = [], [start]
        seen.add(start)
        while stack:
            vert = stack.pop()
            part.append(vert)
            for edge in vert.link_edges:
                other = edge.other_vert(vert)
                if other not in seen:
                    seen.add(other)
                    stack.append(other)
        # Anything that now starts below the cut hangs from nothing: a rope of hair floating on the back of the shirt.
        if max((world @ vert.co).z for vert in part) < neck.z + CHEST_ABOVE_NECK_M + LOOSE_MARGIN_M or not any(vert.link_faces for vert in part):
            loose += part
    bmesh.ops.delete(mesh, geom=loose, context="VERTS")
    mesh.to_mesh(obj.data)
    mesh.free()
    obj.data.update()
    print("CABELO NO PEITO", obj.name, cut, "faces", len(loose), "pontos soltos")


def clear_under_cap(body, brim):
    """The crown and the brim do not meet, and the forehead showed through the gap in a skin-coloured line. Under the crown there is nothing to see."""
    top = max((brim.matrix_world @ vertex.co).z for vertex in brim.data.vertices)
    print("PELE SOB O BONE", delete_faces(body, lambda point: point.z > top))


def surface_tree(objects):
    points, faces = [], []
    for obj in objects:
        obj.data.calc_loop_triangles()
        start = len(points)
        points += [obj.matrix_world @ vertex.co for vertex in obj.data.vertices]
        faces += [tuple(start + index for index in triangle.vertices) for triangle in obj.data.loop_triangles]
    return BVHTree.FromPolygons(points, faces)


def clear_of_body(obj, covers, below_z):
    """Hair below the mouth rests on the shirt and the skin, never through them. Above it, only the skin counts, with a thin gap: `covers[0]` is the body.
    Points at the same place are one point and move as one: a piece is cut along the seams of its picture, and pushed as two edges a
    seam opened (a slit down the back of Carlos's cap, a thread standing out of the pharmacy attendant's crown, 2026-11-11)."""
    tree = surface_tree(covers)
    scalp = surface_tree(covers[:1])
    world = obj.matrix_world
    spots = [world @ vertex.co for vertex in obj.data.vertices]
    place = {}
    for index, spot in enumerate(spots):
        place.setdefault(tuple(round(value / HAIRLINE_SAME_M) for value in spot), []).append(index)
    groups = list(place.values())
    group_of = np.empty(len(spots), int)
    for number, members in enumerate(groups):
        group_of[members] = number
    around = [set() for _ in groups]
    for edge in obj.data.edges:
        first, second = (int(group_of[index]) for index in edge.vertices)
        if first != second:
            around[first].add(second)
            around[second].add(first)
    around = [sorted(near) for near in around]
    head = [covers[0].matrix_world @ vertex.co for vertex in covers[0].data.vertices]
    head = [point for point in head if point.z > below_z] or head
    middle = Vector((sum(point.x for point in head) / len(head), sum(point.y for point in head) / len(head), 0))
    # A cap lies on the skull and stops at the ear; only hair that hangs goes out over it.
    hangs = not hugs_scalp(obj, covers[0])
    moved = np.zeros((len(groups), 3))
    pushed = 0
    for _ in range(CLEAR_ROUNDS):
        step = np.zeros_like(moved)
        for number, members in enumerate(groups):
            point = spots[members[0]] + Vector(moved[number])
            high = point.z > below_z
            location, normal, _, _ = (scalp if high else tree).find_nearest(point, CLEAR_REACH_M)
            if location is None:
                continue
            need = (SCALP_GAP_M if high else CLEAR_GAP_M) - (point - location).dot(normal)
            if need <= 0:
                continue
            out = Vector((point.x - middle.x, point.y - middle.y, 0))
            skin = None
            if high and hangs and out.length > 1e-6:
                # Seen from outside, level, towards the middle of the head: the first skin met is the furthest out that way, ear and all.
                axis, far = Vector((middle.x, middle.y, point.z)), out.length
                out.normalize()
                skin = scalp.ray_cast(axis + out * CLEAR_FROM_M, -out, CLEAR_FROM_M)[3]
            if skin is not None:
                step[number] = np.array(out) * max(0.0, CLEAR_FROM_M - skin + SCALP_GAP_M - far)
            else:
                step[number] = np.array(normal) * need
            pushed += len(members)
        # Neighbours follow, or a strand turns into a row of spikes.
        step = np.array([(step[i] + step[n].mean(axis=0)) / 2 if n and not step[i].any() else step[i] for i, n in enumerate(around)])
        moved += step
    for _ in range(CLEAR_SPREAD):
        far = np.linalg.norm(moved, axis=1)
        most = np.array([near[int(far[near].argmax())] if near else index for index, near in enumerate(around)])
        more = far[most] * CLEAR_CARRIED > far
        moved[more] = moved[most[more]] * CLEAR_CARRIED
    inverse = world.inverted().to_3x3()
    for vertex, number in zip(obj.data.vertices, group_of):
        vertex.co += inverse @ Vector(moved[number])
    obj.data.update()
    print("CABELO FORA DO PANO", obj.name, pushed)
