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
CAP_SEAM_M = 0.015

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
    print("PELE SOB O BONE", delete_faces(body, lambda point: point.z > top - CAP_SEAM_M))


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
