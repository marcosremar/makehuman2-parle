"""Regression for the measure of a face against its mirror image (2026-11-12). Run with
`Blender -b --factory-startup --python-exit-code 1 -P tests/test_parle_mirror.py`.

A head whose halves mirror each other measures nothing, turned and off the middle too; with a swelling of 3 mm on one cheek it
measures that swelling, in the third of the face where it is."""
import math
import sys
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import parle_mirror  # noqa: E402

RADIUS, HEAD_Z, SWELL_M = 0.1, 1.55, 0.003


def body(turn=0.0, aside=0.0, swollen=False):
    """A stick with a ball for a head on top, 1.65 m tall, the face looking at -y."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.05, depth=1.45, location=(0, 0, 0.725))
    stick = bpy.context.active_object
    bpy.ops.mesh.primitive_uv_sphere_add(segments=96, ring_count=48, radius=RADIUS, location=(0, 0, HEAD_Z))
    head = bpy.context.active_object
    if swollen:
        spot = np.array((0.05, -0.08, HEAD_Z + 0.02))
        for vertex in head.data.vertices:
            place = np.array(head.matrix_world @ vertex.co)
            near = np.linalg.norm(place - spot)
            if near < 0.02:
                vertex.co += vertex.normal * SWELL_M * (1 - (near / 0.02) ** 2)
    stick.select_set(True)
    bpy.context.view_layer.objects.active = head
    bpy.ops.object.join()
    head.rotation_euler = (0, 0, turn)
    head.location.x += aside
    bpy.context.view_layer.update()
    return head


even = parle_mirror.measures(body())
assert even["points"] > 500 and 150 < even["face_wide_mm"] <= 201, f"the face of the test head: {even['points']} points, {even['face_wide_mm']:.0f} mm wide"
assert even["sym_mirror"] < 0.0005 and even["mirror_most_mm"] < 0.2, f"a head whose halves mirror each other is not even: {even}"
assert even["mirror_mean_mm"] <= even["mirror_rms_mm"] <= even["mirror_most_mm"] and abs(even["sym_mirror"] - even["mirror_rms_mm"] / even["face_wide_mm"]) < 1e-9, f"the mean, the root of the mean square and the most do not stand in that order: {even}"

turned = parle_mirror.measures(body(turn=math.radians(5), aside=0.01))
assert turned["sym_mirror"] < 0.001 and turned["mirror_most_mm"] < 0.4, f"the same head turned 5 degrees and 1 cm off the middle reads uneven: {turned}"

swollen = parle_mirror.measures(body(swollen=True))
assert 0.5 * SWELL_M * 1000 < swollen["mirror_most_mm"] < 1.2 * SWELL_M * 1000, f"a swelling of 3 mm on one cheek reads {swollen['mirror_most_mm']:.2f} mm at most"
assert swollen["sym_mirror"] > 5 * max(even["sym_mirror"], 1e-5), f"the swollen head is no more uneven than the even one: {swollen['sym_mirror']} against {even['sym_mirror']}"
thirds = {name: swollen[f"sym_mirror_{name}"] for name in parle_mirror.THIRDS}
assert max(thirds, key=thirds.get) == "middle", f"the swelling is in the middle third of the face and reads most in another: {thirds}"
print("ok")
