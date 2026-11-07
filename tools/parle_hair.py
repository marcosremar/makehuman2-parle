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
    """A strand that reaches the chest has to lie flat on the collar, and there it looks like a pile of shards. It stops on the neck, and what is left loose below goes too."""
    def on_chest(point):
        reach = SHOULDER_BACK_M if abs(point.x - neck.x) > SHOULDER_SIDE_M else CHEST_FRONT_M
        return point.z < neck.z + CHEST_ABOVE_NECK_M and point.y < neck.y + reach

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
    """Hair below the mouth rests on the shirt and the skin, never through them. Above it, only the skin counts, with a thin gap: `covers[0]` is the body."""
    tree = surface_tree(covers)
    scalp = surface_tree(covers[:1])
    around = [[] for _ in obj.data.vertices]
    for edge in obj.data.edges:
        first, second = edge.vertices
        around[first].append(second)
        around[second].append(first)
    world = obj.matrix_world
    moved = np.zeros((len(obj.data.vertices), 3))
    pushed = 0
    for _ in range(CLEAR_ROUNDS):
        step = np.zeros_like(moved)
        for index, vertex in enumerate(obj.data.vertices):
            point = world @ vertex.co + Vector(moved[index])
            high = point.z > below_z
            location, normal, _, _ = (scalp if high else tree).find_nearest(point, CLEAR_REACH_M)
            if location is None:
                continue
            need = (SCALP_GAP_M if high else CLEAR_GAP_M) - (point - location).dot(normal)
            if need > 0:
                step[index] = np.array(normal) * need
                pushed += 1
        # Neighbours follow, or a strand turns into a row of spikes.
        step = np.array([(step[i] + step[n].mean(axis=0)) / 2 if n and not step[i].any() else step[i] for i, n in enumerate(around)])
        moved += step
    inverse = world.inverted().to_3x3()
    for vertex, shift in zip(obj.data.vertices, moved):
        vertex.co += inverse @ Vector(shift)
    obj.data.update()
    print("CABELO FORA DO PANO", obj.name, pushed)
