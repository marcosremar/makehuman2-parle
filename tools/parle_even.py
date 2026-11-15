"""
    License information: data/licenses/makehuman_license.txt

    A patch of a mesh made even, without shrinking it.

    Taubin's two-step smoothing (Taubin 1995, A signal processing approach to fair surface design, SIGGRAPH '95, DOI
    10.1145/218380.218473): a step toward the mean of the neighbours (lambda 0.5) and one back (mu -0.53) each round, so the teeth go
    and the patch keeps its size. A point on an open edge (a hem) is evened along that edge alone: with its neighbours inside the
    cloth it would be drawn up off the edge. Points at the same place (a seam of the texture) move as one.

    Numpy alone; `parle_polish` reads the meshes.
"""
import numpy as np

STEP_IN, STEP_OUT = 0.5, -0.53
ROUNDS = 10
SAME_M = 1e-5


def evened(points, faces, weight, rounds=ROUNDS):
    """The points with each moved by its `weight` (0 to 1) of the evening; `faces` lists the points of each face."""
    points = np.asarray(points, float)
    _, weld = np.unique(np.round(points / SAME_M).astype(np.int64), axis=0, return_inverse=True)
    weld = weld.ravel()
    count = int(weld.max()) + 1
    merged = np.zeros((count, 3))
    merged[weld] = points
    share = np.zeros(count)
    np.maximum.at(share, weld, np.clip(np.asarray(weight, float), 0, 1))
    pairs = np.array([(face[k], face[(k + 1) % len(face)]) for face in faces for k in range(len(face))], np.int64).reshape(-1, 2)
    pairs = np.sort(weld[pairs], axis=1)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    edges, uses = np.unique(pairs, axis=0, return_counts=True)
    rim = edges[uses == 1]
    on_rim = np.zeros(count, bool)
    on_rim[rim.ravel()] = True
    # A point on the rim takes its rim neighbours alone; any other point all of its neighbours: an edge between two inner points and
    # an edge of the rim count both ways, an edge from the rim to the inside for the inner point only.
    shut = uses > 1
    links = np.concatenate([edges[shut & ~on_rim[edges[:, 0]] & ~on_rim[edges[:, 1]]], rim])
    cross = edges[shut & (on_rim[edges[:, 0]] != on_rim[edges[:, 1]])]
    for_inner = np.where(on_rim[cross[:, 0]][:, None], cross[:, ::-1], cross)
    for _ in range(rounds):
        for step in (STEP_IN, STEP_OUT):
            total = np.zeros_like(merged)
            many = np.zeros(count)
            np.add.at(total, links[:, 0], merged[links[:, 1]])
            np.add.at(total, links[:, 1], merged[links[:, 0]])
            np.add.at(many, links.ravel(), 1)
            np.add.at(total, for_inner[:, 0], merged[for_inner[:, 1]])
            np.add.at(many, for_inner[:, 0], 1)
            mean = np.where(many[:, None] > 0, total / np.maximum(many, 1)[:, None], merged)
            merged = merged + step * share[:, None] * (mean - merged)
    return merged[weld]


def shared_places(clouds, same=1e-4):
    """Groups of points at the same place, across meshes and within each: (mesh number, point number) pairs, one list per place
    that has more than one point."""
    keys = {}
    for number, cloud in enumerate(clouds):
        for index, key in enumerate(map(tuple, np.round(np.asarray(cloud, float) / same).astype(np.int64))):
            keys.setdefault(key, []).append((number, index))
    return [group for group in keys.values() if len(group) > 1]
