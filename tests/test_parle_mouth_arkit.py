"""The ARKit lip shapes land on the right points of a cast body (2026-11-12): run with `.local/venv/bin/python tests/test_parle_mouth_arkit.py`.

The first try (2026-11-08) walked the base mesh onto the body with the lips touching: a point of one lip could take the id of the other,
which tore the corners and left a jagged edge round the teeth; and a point that took a neighbour's id left a dimple on the chin.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import parle_mouth_arkit as arkit  # noqa: E402


def grid(columns, rows, step):
    """A face strip: a grid of points in x and z (y = 0), with its edges."""
    points = np.array([[x * step, 0.0, z * step] for z in range(rows) for x in range(columns)])
    edges = [(z * columns + x, z * columns + x + 1) for z in range(rows) for x in range(columns - 1)]
    edges += [(z * columns + x, (z + 1) * columns + x) for z in range(rows - 1) for x in range(columns)]
    return points, np.array(edges)


def test_makehuman_axes_become_blender_axes_in_metres():
    assert np.allclose(arkit.mh_to_blender(np.array([[1.0, 2.0, 3.0]])), [[0.1, -0.3, 0.2]])


def test_the_pack_has_every_shape_and_drops_helper_points():
    for name in (*arkit.SHAPES, "jawOpen", "mouthSmileLeft"):
        ids, moves = arkit.read_target(arkit.PACK / f"{name}.target")
        assert len(ids) and ids.max() < arkit.BODY_POINTS, name
        assert np.linalg.norm(moves, axis=1).max() > 0.005, f"{name} moves the lips millimetres at full weight"


def test_a_stray_takes_its_neighbours_move_and_seam_copies_end_equal():
    rest, edges = grid(9, 9, 0.002)
    move = np.tile([0.0, -0.004, 0.0], (len(rest), 1))
    move[40] = 0.0  # a point that took a still id in the middle of the moving lip: the dimple
    # A seam copy of point 41: same place, its own index, joined only to point 42.
    rest = np.vstack([rest, rest[41]])
    move = np.vstack([move, [0.0, -0.001, 0.0]])
    edges = np.vstack([edges[(edges != 41).all(axis=1)], [[81, 42]]])
    cleaned = arkit.clean(move, rest, edges)
    assert np.allclose(cleaned[40], [0.0, -0.004, 0.0], atol=2e-4), "the dimple is gone"
    assert np.allclose(cleaned[41], cleaned[81]), "seam copies move together: no tear"
    assert np.allclose(cleaned[:81], np.tile([0.0, -0.004, 0.0], (81, 1)), atol=2e-4), "the rest of the field keeps its move"


def test_lips_walked_open_keep_their_own_ids():
    # The head: upper lip row at z = 0, lower lip row 0.4 mm under it; a column of skin below so the head share holds them.
    upper = np.array([[x * 0.004, 0.002 * (x % 2), 1.0] for x in range(8)])
    lower = upper - [0.0, 0.0, 0.0004]
    below = np.array([[x * 0.004, 0.0, z] for x in range(8) for z in (0.0, 0.5)])
    base = np.vstack([upper, lower, below])
    edges = np.array([(i, i + 1) for i in range(7)] + [(8 + i, 9 + i) for i in range(7)] + [(i, i + 8) for i in range(8)])
    jaw = np.zeros_like(base)
    jaw[8:16] = [0.0, 0.0, -0.02]
    # The body is the same head, reordered, with the lower lip nudged up to 0.1 mm under the upper one.
    order = np.random.default_rng(0).permutation(len(base))
    body = base[order] + np.where((order >= 8) & (order < 16), 0.0003, 0.0)[:, None] * [0.0, 0.0, 1.0]
    body_jaw = jaw[order]
    ids, _ = arkit.base_ids(body + body_jaw, base + jaw, edges)
    lips = (order < 16)
    assert np.array_equal(ids[lips], order[lips]), "open, every lip point finds its own base point"


if __name__ == "__main__":
    for test in [value for name, value in dict(globals()).items() if name.startswith("test_")]:
        test()
    print("ok")
