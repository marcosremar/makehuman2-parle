"""
    License information: data/licenses/makehuman_license.txt

    Paints the print of a new garment where the game's garment carries a print that repeats (the rules are in `parle_print`):

        blender -b -P tools/print_cloth.py -- <cast.glb> <game.glb> <out prefix>

    For the top and for the bottom it writes `<out prefix>-<kind>.print.png`, laid out as the garment's own picture, when a piece of that
    garment in the game has such a print, and removes an old one when none has. `stamp_tile.py` joins it to the picture baked from the
    game.
"""
import os
import sys

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parle_print  # noqa: E402
import parle_rig  # noqa: E402
from parle_stamp import GAME_BOTTOM, GAME_TOP, stamp_sources  # noqa: E402

SIZE = 1024
# Each point of the picture is the mean of this many points of the print: a plaid shrunk to half its size without it comes out in dots.
SAMPLES = 16
MARGIN_PX = 16
UP = (0.0, 0.0, 1.0)
# A piece is a sleeve when the arm holds this much of it; the clavicle holds the shoulder of the trunk and does not count.
SLEEVE_SHARE = 0.5
OURS = {"top": "Outfit_Top", "bottom": "Outfit_Bottom"}
GAME = {"top": GAME_TOP, "bottom": GAME_BOTTOM}


def read(obj, layer=None):
    """A piece as arrays: its points in space, the points and the corners of its triangles, and where each corner lies on a UV layer."""
    mesh = obj.data
    mesh.calc_loop_triangles()
    world = np.array(obj.matrix_world)
    points = np.empty(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", points)
    points = points.reshape(-1, 3) @ world[:3, :3].T + world[:3, 3]
    vertices = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int32)
    mesh.loop_triangles.foreach_get("vertices", vertices)
    loops = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int32)
    mesh.loop_triangles.foreach_get("loops", loops)
    uv = np.empty(len(mesh.loops) * 2)
    (layer or mesh.uv_layers.active).data.foreach_get("uv", uv)
    return points, vertices.reshape(-1, 3), loops.reshape(-1, 3), uv.reshape(-1, 2)


def islands(vertices, loops, uv):
    """The triangles of each island of a piece: the ones that share a point at the same place of the picture."""
    parent = list(range(len(vertices)))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    seen = {}
    for index, (points, corners) in enumerate(zip(vertices, loops)):
        for point, corner in zip(points, corners):
            key = (int(point), round(float(uv[corner][0]), 5), round(float(uv[corner][1]), 5))
            if key in seen:
                parent[root(index)] = root(seen[key])
            else:
                seen[key] = index
    roots = np.array([root(index) for index in range(len(vertices))])
    return [np.nonzero(roots == value)[0] for value in sorted(set(roots.tolist()))]


def grain_of(obj, rig, points):
    """The way the cloth of an island hangs, and whether it is of the trunk: along the arm when the arm holds it (a sleeve), up
    otherwise. `points` are the points of the island."""
    held = {}
    for side in ("Left", "Right"):
        groups = {group.index for group in obj.vertex_groups if group.name.startswith(side) and parle_rig.moves_with_arm(group.name)
                  and not group.name.endswith("Shoulder")}
        held[side] = float(np.mean([sum(item.weight for item in obj.data.vertices[int(point)].groups if item.group in groups) for point in points]))
    side = max(held, key=held.get)
    if rig is None or held[side] < SLEEVE_SHARE:
        return UP, True
    bones = rig.data.bones
    along = rig.matrix_world @ bones[side + "Arm"].head_local - rig.matrix_world @ bones[side + "ForeArm"].head_local
    return tuple(along.normalized()), False


def picture_of(obj):
    """The picture a piece wears, or None: the one that gives its colour, when nothing moves or scales it on the way."""
    for material in obj.data.materials:
        if not material or not material.use_nodes:
            continue
        for node in material.node_tree.nodes:
            if node.type != "TEX_IMAGE" or not node.image or not any(link.to_socket.name == "Base Color" for link in node.outputs[0].links):
                continue
            if node.inputs["Vector"].is_linked and node.inputs["Vector"].links[0].from_node.type == "MAPPING":
                print("ESTAMPA", obj.name, "tem a figura movida no material: não tratada")
                return None
            return node.image
    return None


def game_print(pieces, rig):
    """The print that repeats of a game garment, from the piece with the most cloth that has one: the picture to lay edge to edge, the
    matrix that takes a step on the cloth, in metres, to a step on that picture (`parle_print.as_in_game`), and the place of the
    picture that goes on the middle of each piece (`parle_print.mirror_line`). None when no piece has one."""
    best = None
    for obj in pieces:
        image = picture_of(obj)
        if image is None or not obj.data.uv_layers.active:
            continue
        points, vertices, loops, uv = read(obj)
        parts = islands(vertices, loops, uv)
        wide, high = image.size
        pixels = np.array(image.pixels[:], dtype=np.float32).reshape(high, wide, -1)[:, :, :3]
        keep = parle_print.tile_of(pixels, uv[loops.ravel()], [uv[loops[part]] for part in parts])
        if keep is None:
            continue
        corners = points[vertices]
        area = float(np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1).sum() / 2)
        if best is not None and area <= best[0]:
            continue
        biggest = max(parts, key=len)
        grain, _ = grain_of(obj, rig, np.unique(vertices[biggest]))
        step = parle_print.as_in_game(corners[biggest], uv[loops[biggest]], grain)
        # The picture cut to what repeats is that much smaller: a step on it is that much longer.
        step = step / np.array([[keep[1] / wide], [keep[0] / high]])
        tile = bpy.data.images.new("print-tile", keep[1], keep[0], alpha=True)
        # Before the pixels: a new colour space throws them away.
        tile.colorspace_settings.name = image.colorspace_settings.name
        cut = pixels[:keep[0], :keep[1]]
        tile.pixels.foreach_set(np.concatenate([cut, np.ones((keep[0], keep[1], 1), np.float32)], axis=2).ravel())
        tile.pack()
        best = (area, tile, step, np.array(parle_print.mirror_line(cut)), obj.name.split(".")[0], image.name)
    return best


def lay(obj, rig, step, start, middle, level):
    """Gives a piece of ours a second UV layer, `print`, with where each of its corners falls on the print: the piece laid flat anew
    from its shape (its own picture is laid out for another body, and stretched by ours), each island turned to its grain and counted
    in metres (`parle_print.laid`), and from metres to pictures as in the game, with `start` of the picture on the middle of it."""
    mesh = obj.data
    own = mesh.uv_layers.active.name
    points, vertices, loops, uv = read(obj)
    if "print" not in mesh.uv_layers:
        mesh.uv_layers.new(name="print")
    mesh.uv_layers.active = mesh.uv_layers["print"]
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.unwrap(method="ANGLE_BASED", margin=0.001)
    bpy.ops.object.mode_set(mode="OBJECT")
    lain = np.empty(len(mesh.loops) * 2)
    mesh.uv_layers["print"].data.foreach_get("uv", lain)
    lain = lain.reshape(-1, 2)
    for part in islands(vertices, loops, uv):
        grain, trunk = grain_of(obj, rig, np.unique(vertices[part]))
        cloth = parle_print.laid(points[vertices[part]], lain[loops[part]], grain, middle if trunk else None, level)
        lain[loops[part].ravel()] = (cloth @ step.T + start).reshape(-1, 2)
    mesh.uv_layers["print"].data.foreach_set("uv", lain.ravel())
    mesh.uv_layers.active = mesh.uv_layers[own]
    mesh.update()
    return own


def paint(parts, own, tile, path):
    """The print, read through the `print` layer of each piece, painted on one picture laid out by the garment's own UV."""
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = SAMPLES
    scene.view_settings.view_transform = "Standard"
    target = bpy.data.images.new("print", SIZE, SIZE, alpha=False)
    for obj, name in zip(parts, own):
        material = bpy.data.materials.new("print")
        material.use_nodes = True
        tree = material.node_tree
        tree.nodes.clear()
        where = tree.nodes.new("ShaderNodeUVMap")
        where.uv_map = "print"
        picture = tree.nodes.new("ShaderNodeTexImage")
        picture.image = tile
        picture.extension = "REPEAT"
        glow = tree.nodes.new("ShaderNodeEmission")
        out = tree.nodes.new("ShaderNodeOutputMaterial")
        tree.links.new(where.outputs["UV"], picture.inputs["Vector"])
        tree.links.new(picture.outputs["Color"], glow.inputs["Color"])
        tree.links.new(glow.outputs["Emission"], out.inputs["Surface"])
        goal = tree.nodes.new("ShaderNodeTexImage")
        goal.image = target
        tree.nodes.active = goal
        obj.data.materials.clear()
        obj.data.materials.append(material)
        bpy.ops.object.select_all(action="DESELECT")
        obj.hide_render = False
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.bake(type="EMIT", uv_layer=name, margin=MARGIN_PX, use_clear=False)
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    target.save_render(path)


def main():
    cast, game_path, prefix = sys.argv[sys.argv.index("--") + 1:][:3]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=cast)
    rig = next(obj for obj in bpy.data.objects if obj.type == "ARMATURE")
    mine = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=game_path)
    got = [obj for obj in bpy.data.objects if obj not in before]
    game_rig = next((obj for obj in got if obj.type == "ARMATURE"), None)
    for skeleton in (rig, game_rig):
        if skeleton is not None:
            skeleton.data.pose_position = "REST"
    bpy.context.view_layer.update()
    hips = rig.matrix_world @ rig.data.bones["Hips"].head_local
    for kind in ("top", "bottom"):
        path = f"{prefix}-{kind}.print.png"
        if os.path.exists(path):
            os.remove(path)
        parts = [obj for obj in mine if obj.name.startswith(OURS[kind])]
        pieces = [obj for obj in got if obj.type == "MESH"]
        chosen = stamp_sources([obj.name.split(".")[0] for obj in pieces], GAME[kind])
        found = game_print([obj for obj in pieces if obj.name.split(".")[0] in chosen], game_rig) if parts else None
        if found is None:
            print("ESTAMPA", kind, "sem estampa que se repete")
            continue
        _, tile, step, start, piece, name = found
        own = [lay(obj, rig, step, start, hips.x, hips.z) for obj in parts]
        paint(parts, own, tile, path)
        metres = 1 / float(np.sqrt(abs(np.linalg.det(step))))
        print("ESTAMPA", kind, "repete:", piece, name, tuple(tile.size), "px,", round(metres, 3), "m de pano por figura, passo",
              np.round(step * metres, 2).tolist())


main()
