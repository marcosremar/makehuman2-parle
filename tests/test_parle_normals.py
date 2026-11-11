"""Regression for the grey band under the jaw (scoreboard of 2026-11-10): run with `.local/venv/bin/python tests/test_parle_normals.py`."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import parle_normals as normals  # noqa: E402

# A head that looks to -y: the top of the skull, the tip of the nose, and points along the jaw and the neck.
top = np.array([[0, 0.05, 1.7]], float)
nose = np.array([[0, -0.12, 1.55]], float)
under_jaw = np.array([[0.02, -0.08, 1.45]], float)
cheek = np.array([[0.04, -0.09, 1.5]], float)
neck = np.array([[0, -0.05, 1.3]], float)
mirror = np.array([[-0.04, -0.09, 1.5]], float)
points = np.vstack([top, nose, under_jaw, cheek, neck, mirror])
facing = np.array([[0, 0, 1], [0, -1, 0], [0, -0.2, -0.98], [0, -1, 0], [0, -1, 0], [0, -1, 0]], float)
facing /= np.linalg.norm(facing, axis=1, keepdims=True)
out = normals.softened(points, facing)
assert out[2, 2] > facing[2, 2] + 0.5, f"the under side of the jaw must turn forward: {out[2]}"
assert out[2, 1] < facing[2, 1] - 0.2, out[2]
for keep in (0, 1, 3, 4, 5):
    assert abs((out[keep] - facing[keep]).max()) < 1e-6, f"point {keep} is not under the jaw: {out[keep]}"
assert abs(np.linalg.norm(out, axis=1).max() - 1) < 1e-6
print("ok")
