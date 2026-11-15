"""Regression for the clothes lit by their file's normals (2026-11-15): run with
`blender -b --factory-startup -P tests/test_garment_normals.py`.

The clothes come from a glTF with its normals; every step after moved the points and the normals stayed, so the light drew a hollow on
the back of the pharmacist's T-shirt and a lump on the gym man's shorts that were not in the mesh. Writing zero normals did not take
them away on Blender 4.5. `parle_polish.garment_normals` works them out again from the faces, the same where two pieces meet.
"""
import sys
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import parle_polish  # noqa: E402

bpy.ops.wm.read_factory_settings(use_empty=True)


def piece(name, x0):
    """A flat strip of two squares in the plane z = 0, facing up, from x0 to x0 + 2."""
    mesh = bpy.data.meshes.new(name)
    points = [(x0, 0, 0), (x0 + 1, 0, 0), (x0 + 2, 0, 0), (x0, 1, 0), (x0 + 1, 1, 0), (x0 + 2, 1, 0)]
    mesh.from_pydata(points, [], [(0, 1, 4, 3), (1, 2, 5, 4)])
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    # The file's normals, wrong for the shape: leaning 60 degrees to the side.
    mesh.normals_split_custom_set([(0.87, 0.0, 0.5)] * len(mesh.loops))
    mesh.update()
    return obj


left, right = piece("Outfit_Top", 0.0), piece("Outfit_Top.001", 2.0)
skin = piece("Body", 10.0)
assert parle_polish.garment_normals([left, right, skin]) == 2
for obj in (left, right):
    normals = np.empty(len(obj.data.loops) * 3)
    obj.data.corner_normals.foreach_get("vector", normals)
    assert np.allclose(normals.reshape(-1, 3), (0, 0, 1), atol=1e-4), normals.reshape(-1, 3)[:3]
# The skin is not the clothes': its normals stay as they were.
normals = np.empty(len(skin.data.loops) * 3)
skin.data.corner_normals.foreach_get("vector", normals)
assert np.allclose(normals.reshape(-1, 3)[0], (0.87, 0.0, 0.5), atol=0.02), normals[:3]
print("test_garment_normals: ok")
