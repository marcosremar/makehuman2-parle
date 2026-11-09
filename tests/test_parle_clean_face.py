"""Regressions for the clean face (2026-11-08): run with `.local/venv/bin/python tests/test_parle_clean_face.py`.

The game face, enlarged, brought a red band across the cheeks (Carlos) and a loose dark sideburn the hair mesh does not reach.
Bare skin keeps only its broad light; a beard and the big painted scalp stay; the loose patch goes. The lip keeps less of its dark crease.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import skin_style  # noqa: E402
from parle_lips import painted  # noqa: E402

SIZE = skin_style.SIZE
TIP_Z = 1.5
# Two triangles filling the map, the upper-left one above the nose tip. The last point, used by no triangle, is the nose tip itself.
points = np.array([[0.0, -0.1, TIP_Z - 0.2], [0.0, -0.1, TIP_Z + 0.2], [0.001, -0.1, TIP_Z + 0.2], [0.001, -0.1, TIP_Z - 0.2], [0.0, -0.2, TIP_Z]])
uv = np.array([[[0, 0], [0, 1], [1, 1]], [[0, 0], [1, 1], [1, 0]]], float)
data = {"pos": points, "uv": uv, "tri": np.array([[0, 1, 2], [0, 2, 3]])}

head = np.full((SIZE, SIZE, 3), 170.0, np.float32)
head[700:760, 200:800] = (120, 120, 210)  # red band on the cheeks, low on the head
head[90:130, 310:335] = 40                # loose painted sideburn, high on the head
head[0:60, 0:SIZE] = 40                   # the big painted scalp, high on the head
head[820:900, 400:600] = 60               # beard
face = np.ones((SIZE, SIZE), np.float32)
beard = np.zeros((SIZE, SIZE), np.float32)
beard[820:900, 400:600] = 1
lips = np.zeros((SIZE, SIZE), np.float32)

out = skin_style.plain_face(head, face, beard, lips, data)
assert out[730, 500][2] - out[730, 500][0] < 0.65 * 90, out[730, 500]  # the band's red fades
assert abs(out[110, 322] - 170).max() < 35, out[110, 322]
assert out[20, 500].mean() < 70, out[20, 500]
assert out[860, 500].mean() < 70, out[860, 500]

# The dark crease is the lip's own shade: painted, it must stay within reach of the lip's colour.
skin = np.full((64, 64, 3), 170, np.uint8)
skin[30:34] = 60
share = np.ones((64, 64), np.float32)
lip = painted(skin, share).astype(np.float32).mean(axis=2)
assert lip[10, 10] - lip[32, 10] < 25, (lip[10, 10], lip[32, 10])
print("ok")
