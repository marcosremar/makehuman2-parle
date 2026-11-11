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

# Serrilhado no canto da mandíbula (placar de 11/11/2026, Ivone e Lúcia regravadas): a zona era uma caixa de borda dura e o peso vinha só da
# normal de cada ponto, então o vinco do pescoço alternava ponto virado e ponto não virado e o player mostrava dentes escuros no canto do queixo.
tip_z, tip_y = nose[0, 2], nose[0, 1]
skull = np.vstack([points, np.tile([[0.0, 0.04, 1.62]], (30, 1))])
skull_facing = np.vstack([facing, np.tile([[0.0, 1.0, 0.0]], (30, 1))])
edge = np.array([[x, tip_y + 0.04, tip_z - 0.09] for x in (0.0785, 0.0815)], float)
down = np.tile(np.array([[0, -0.2, -0.98]]) / np.linalg.norm([0, -0.2, -0.98]), (2, 1))
pair = normals.under_chin_share(np.vstack([skull, edge]), np.vstack([skull_facing, down]))[len(skull):]
assert abs(pair[0] - pair[1]) < 0.15, f"3 mm across the side of the zone must not switch the turn on and off: {pair}"
chain = np.array([[0.02 + 0.004 * k, tip_y + 0.04, tip_z - 0.09] for k in range(9)], float)
crease = np.array([[0, -0.2, -0.98] if k % 2 else [0, -0.98, -0.2] for k in range(9)], float)
crease /= np.linalg.norm(crease, axis=1, keepdims=True)
share = normals.under_chin_share(np.vstack([skull, chain]), np.vstack([skull_facing, crease]))[len(skull):]
links = np.array([(k, k + 1) for k in range(8)])
even = normals.smoothed_share(share, links)
assert np.abs(np.diff(even)).max() < 0.5 * np.abs(np.diff(share)).max(), f"neighbours along a crease must turn together: {share} -> {even}"
assert even.max() > 0.3, f"the crease under the jaw still turns: {even}"
print("ok serrilhado")
