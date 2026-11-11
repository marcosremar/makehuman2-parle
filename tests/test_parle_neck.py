"""Regression for the line where the head meets the neck (2026-11-11): from behind, Carlos showed a collar of paler skin under the hair.
Run with `.local/venv/bin/python tests/test_parle_neck.py`."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
from parle_neck import NECK_WIDE_M, behind_face, islands, neck_tone  # noqa: E402

SIZE = 400
HEAD, BODY, MOUTH = np.array([108.0, 139.0, 188.0]), np.array([108.0, 144.0, 198.0]), np.array([30.0, 20.0, 90.0])


def strip(columns, rows, origin, across, up, corner, wide, high):
    """A flat sheet of points with its triangles and where each lies on the picture."""
    points, uv, faces = [], [], []
    for row in range(rows):
        for column in range(columns):
            points.append(np.array(origin) + np.array(across) * column / (columns - 1) + np.array(up) * row / (rows - 1))
            uv.append((corner[0] + wide * column / (columns - 1), corner[1] + high * row / (rows - 1)))
    for row in range(rows - 1):
        for column in range(columns - 1):
            first = row * columns + column
            faces += [(first, first + 1, first + columns + 1), (first, first + columns + 1, first + columns)]
    return np.array(points), np.array(uv), np.array(faces)


# The back of a head, 20 cm wide and 12 cm high, on the neck, 8 cm high under it: two islands that meet at 1.50 m. The nose is on a
# third sheet in front, part of the head's island, and the inside of the mouth is an island of its own that touches the head.
back, back_uv, back_faces = strip(11, 7, (-0.1, 0.08, 1.50), (0.2, 0, 0), (0, 0, 0.12), (0.1, 0.5), 0.5, 0.3)
front, front_uv, front_faces = strip(3, 7, (-0.01, -0.10, 1.60), (0.02, 0, 0), (0, 0, 0.12), (0.7, 0.5), 0.05, 0.3)
neck, neck_uv, neck_faces = strip(11, 5, (-0.1, 0.08, 1.42), (0.2, 0, 0), (0, 0, 0.08), (0.1, 0.1), 0.5, 0.2)
mouth, mouth_uv, mouth_faces = strip(3, 3, (-0.01, -0.10, 1.47), (0.02, 0, 0), (0, 0, 0.03), (0.8, 0.1), 0.05, 0.05)
# The head is one island: its back and its front share a seam of the mesh.
bridge = np.array([(10, 77, 80), (10, 80, 21)])
points = np.concatenate([back, front, neck, mouth]).astype(np.float32)
uv = np.concatenate([back_uv, front_uv, neck_uv, mouth_uv])
triangles = np.concatenate([back_faces, bridge, front_faces + 77, neck_faces + 98, mouth_faces + 153])
data = {"tri": triangles, "uv": np.stack([uv[triangles][:, :, 0], 1 - uv[triangles][:, :, 1]], axis=2)}
label = islands(triangles, len(points))
assert label[0] == label[77] and label[98] != label[0] and label[153] not in (label[0], label[98]), "the islands are not the head, the body and the mouth"

# Behind the face is behind the ears and under them: nothing at the nose or 6 cm behind it, all of it 13 cm behind it and at the nape,
# and nothing on the ear itself, level with the nose, which keeps its relief.
where = behind_face(points)
assert where[77:98].max() == 0 and where[:44].min() == 1 and where[66:77].max() == 0, (where[77:98].max(), where[:44].min(), where[66:77].max())
ear = np.array([[0.0, -0.10, 1.5], [0.0, -0.04, 1.5], [0.0, 0.03, 1.5], [0.07, 0.03, 1.63]], np.float32)
assert np.allclose(behind_face(np.concatenate([points, ear]))[-4:], [0, 0, 1, 0])

image = np.zeros((SIZE, SIZE, 3), np.float32)
place = lambda corner, wide, high: (slice(int(corner[1] * SIZE) - 3, int((corner[1] + high) * SIZE) + 4), slice(int(corner[0] * SIZE) - 3, int((corner[0] + wide) * SIZE) + 4))
image[place((0.1, 0.5), 0.5, 0.3)] = HEAD
image[place((0.7, 0.5), 0.05, 0.3)] = HEAD
image[place((0.1, 0.1), 0.5, 0.2)] = BODY
image[place((0.8, 0.1), 0.05, 0.05)] = MOUTH
out, read = neck_tone(image, data, points, SIZE)
at = lambda u, v: out[int(v * SIZE), int(u * SIZE)]
assert read >= 9, read
# At the seam the head has the tone of the neck: no line.
assert np.abs(at(0.35, 0.505) - BODY).max() < 1.5, at(0.35, 0.505)
# It gives the difference back little by little: half way through the reach it is between the two, and past the reach it is itself.
reach = NECK_WIDE_M / 0.12 * 0.3
middle = at(0.35, 0.5 + reach / 2)
assert HEAD[2] + 2 < middle[2] < BODY[2] - 2, middle
assert np.abs(at(0.35, 0.5 + reach * 1.15) - HEAD).max() < 0.5, at(0.35, 0.5 + reach * 1.15)
# The body keeps its tone, and the inside of the mouth is not the neck: the head beside it stays as it is.
assert np.abs(at(0.35, 0.2) - BODY).max() < 1e-3 and np.abs(at(0.82, 0.12) - MOUTH).max() < 1e-3
assert np.abs(at(0.72, 0.51) - HEAD).max() < 0.5, "the face next to the mouth took the colour of the inside of the mouth"
# In front of the ears the seam is the face's, under the jaw: a neck that meets the head there leaves it as it is.
forward = points.copy()
forward[:, 1] -= 0.12
forward[77:98, 1] = -0.10
same, none = neck_tone(image, data, forward, SIZE)
assert none == 0 and np.abs(same - image).max() < 1e-6, "the seam under the jaw was touched"
print("ok")
