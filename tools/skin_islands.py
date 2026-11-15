"""Loose bits of skin left under the clothes.

The skin under a garment is cut away where the cloth covers it (the cast assembly, `trim_skin` and `settle_cloth`). What the cut leaves
between two covered stretches is a loose bit, a few centimetres across and wholly under the cloth. Nothing pulls a loose bit along with
the cloth, so in a gesture it comes out through it: the hollow seen on the back of the T-shirt of the pharmacist with the arm raised
(16 points on the shoulder blade) and the lump and hole on the front and the back of the gym man's shorts (two bits of 13 and 15 points
at the crotch), 2026-11-15. A body keeps its head, its arms and its legs as a few big pieces; a loose bit is a small one.

Numpy only; `parle_polish.loose_skin_under_cloth` runs it on the mounted body.
"""
import numpy as np

# Points closer than this are one point (the skin is split along its texture seams).
SAME_M = 1e-5
# A piece is loose below this many points: the bits found on the cast had 3 to 75, the smallest real piece (a foot under trousers,
# split from the leg at the hem) 1 081.
LOOSE_MOST = 400


def welded(points):
    """The index of each point among the points taken once."""
    _, index = np.unique(np.round(np.asarray(points, float) / SAME_M).astype(np.int64), axis=0, return_inverse=True)
    return index.ravel()


def pieces(points, edges):
    """The piece of skin each point belongs to (0 = the biggest) and how many points (taken once) each piece has."""
    weld = welded(points)
    count = int(weld.max()) + 1 if len(weld) else 0
    parent = np.arange(count)
    pairs = weld[np.asarray(edges, np.int64).reshape(-1, 2)]
    # Union by pointer jumping, all edges at once, until nothing moves.
    while True:
        a, b = parent[pairs[:, 0]], parent[pairs[:, 1]]
        low = np.minimum(a, b)
        before = parent.copy()
        np.minimum.at(parent, a, low)
        np.minimum.at(parent, b, low)
        parent = parent[parent]
        if np.array_equal(parent, before):
            break
    _, label, size = np.unique(parent, return_inverse=True, return_counts=True)
    rank = np.empty_like(size)
    rank[np.argsort(-size, kind="stable")] = np.arange(len(size))
    return rank[label][weld], size[np.argsort(-size, kind="stable")]


def loose(points, edges, most=LOOSE_MOST):
    """The points of the loose pieces: every piece with fewer than `most` points (taken once)."""
    piece, size = pieces(points, edges)
    return np.flatnonzero(size[piece] < most)
