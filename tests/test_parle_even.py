"""Regression for the teeth above a hem (2026-11-15): run with `.local/venv/bin/python tests/test_parle_even.py`.

The leg of the pharmacist's shorts was level at its hem and still went in and out in teeth just above it. `parle_even.evened` makes a
patch even: on a tube whose rings above the open rim go in and out by turns, the teeth go, the tube keeps its girth, and the rim stays
level (it is evened along itself, so it is not drawn up into the cloth).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import parle_even  # noqa: E402

around, rings, radius = 24, 8, 0.08
angle = np.linspace(0, 2 * np.pi, around, endpoint=False)
points = []
for ring in range(rings):
    for k, a in enumerate(angle):
        tooth = 0.012 * (1 if k % 2 else -1) if ring in (1, 2) else 0.0
        points.append(((radius + tooth) * np.cos(a), (radius + tooth) * np.sin(a), ring * 0.01))
points = np.array(points)
faces = [(r * around + k, r * around + (k + 1) % around, (r + 1) * around + (k + 1) % around, (r + 1) * around + k)
         for r in range(rings - 1) for k in range(around)]
weight = np.ones(len(points))
flat = parle_even.evened(points, faces, weight)
spread = lambda p, ring: np.ptp(np.linalg.norm(p[ring * around:(ring + 1) * around, :2], axis=1))
assert spread(points, 1) > 0.02 and spread(flat, 1) < 0.006, spread(flat, 1)
# The tube keeps its girth: the mean radius of the rings with teeth stays within 4 mm of the tube's.
mean = np.linalg.norm(flat[around:3 * around, :2], axis=1).mean()
assert abs(mean - radius) < 0.004, mean
# The open rim (ring 0) stays level and in its place along the tube.
assert np.ptp(flat[:around, 2]) < 1e-6 and abs(flat[:around, 2].mean()) < 0.002, flat[:around, 2]
# A point with no weight does not move.
still = parle_even.evened(points, faces, np.zeros(len(points)))
assert np.abs(still - points).max() < 1e-12
# A seam is a place that two meshes share.
places = parle_even.shared_places([points[:around], points[:2 * around]])
assert len(places) == around and all(len(group) == 2 for group in places)
print("test_parle_even: ok")
