"""The pieces of a garment back into one object with one material (2026-11-21): run with
`blender -b --factory-startup --python-exit-code 1 -P tests/test_join_garment_pieces.py`.

A suit is split into its loose islands to tell the shirt from the trousers, and each island kept its object and a copy of the material
(`Roupa_top`, `Roupa_top.001`, ...): in the game each one was a draw call, seven more for a shop clerk in a blouse than in a bikini.
`parle_polish.join_garment_pieces` joins the pieces of the top and of the bottom, each with the first piece's material, and leaves the
skin, the shoes and a garment of one piece as they were.
"""
import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import parle_polish  # noqa: E402

bpy.ops.wm.read_factory_settings(use_empty=True)


def piece(name, x0, material):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([(x0, 0, 0), (x0 + 1, 0, 0), (x0 + 1, 1, 0), (x0, 1, 0)], [], [(0, 1, 2, 3)])
    mesh.materials.append(bpy.data.materials.get(material) or bpy.data.materials.new(material))
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


tops = [piece("Outfit_Top" if i == 0 else f"Outfit_Top.00{i}", i * 2.0, "Roupa_top" if i == 0 else f"Roupa_top.00{i}") for i in range(4)]
bottom = piece("Outfit_Bottom", 20.0, "Roupa_bottom")
shoes, skin = piece("Footwear", 30.0, "Roupa_shoes"), piece("Body", 40.0, "Human.body")
left = parle_polish.join_garment_pieces([*tops, bottom, shoes, skin])
names = sorted(obj.name for obj in left)
assert names == ["Body", "Footwear", "Outfit_Bottom", "Outfit_Top"], names
top = bpy.data.objects["Outfit_Top"]
assert len(top.data.polygons) == 4, len(top.data.polygons)
assert [slot.material.name for slot in top.material_slots] == ["Roupa_top"], [slot.material.name for slot in top.material_slots]
assert {polygon.material_index for polygon in top.data.polygons} == {0}
assert len(bottom.data.polygons) == 1 and [slot.material.name for slot in bottom.material_slots] == ["Roupa_bottom"]
assert len([obj for obj in bpy.data.objects if obj.name.startswith("Outfit_Top")]) == 1
print("test_join_garment_pieces: ok")
