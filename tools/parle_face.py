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
from mathutils import Vector
from mathutils.bvhtree import BVHTree

EYE_TUCK_REACH_M = 0.02
EYE_TUCK_BEHIND_M = 0.001
FOREHEAD_HAIR_M = 0.085
FOREHEAD_HALF_M = 0.06
FOREHEAD_REACH_M = 0.06


def skin_tree(body):
    return BVHTree.FromPolygons([tuple(body.matrix_world @ vertex.co) for vertex in body.data.vertices], [tuple(polygon.vertices) for polygon in body.data.polygons])


def wears_from_game(name, earrings):
    """Men of the cast wear no earrings (owner, 2026-11-06)."""
    return earrings or "Earring" not in name


def trim_forehead_hair(obj, body, mine):
    """O crânio novo é menor: o afro que no jogo fica debaixo do turbante saía pela testa, numa mancha preta acima das sobrancelhas."""
    skin = skin_tree(body)
    back = Vector((0, 1, 0))
    mesh = bmesh.new()
    mesh.from_mesh(obj.data)
    out = [vert for vert in mesh.verts if vert.co.z < mine["eye"] + FOREHEAD_HAIR_M and abs(vert.co.x - mine["x"]) < FOREHEAD_HALF_M and skin.ray_cast(vert.co, back, FOREHEAD_REACH_M)[0] is not None]
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


def set_map(obj, image):
    for node in obj.active_material.node_tree.nodes:
        if node.type == "TEX_IMAGE":
            node.image = image


def clean_face(body, added, job, env):
    for obj in [item for item in added if item.type == "MESH" and "eyelash" in item.name.lower()]:
        added.remove(obj)
        bpy.data.objects.remove(obj, do_unlink=True)
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
