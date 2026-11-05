"""
    License information: data/licenses/makehuman_license.txt

    Garment weights from what the cloth would do, for makehuman2-parle.

    The caller hands over bone matrices sampled from the clips the character plays. For each pose the
    cloth leaves the skin and goes back to its own length by position based dynamics (Muller et al.
    2007, J. Vis. Commun. Image Represent. 18(2), DOI 10.1016/j.jvcir.2007.01.005). Each point then
    takes the weights that come closest to the simulated cloth in all poses, with the bones that
    already exist (Le & Deng 2012, Smooth Skinning Decomposition with Rigid Bones, ACM TOG 31(6),
    DOI 10.1145/2366145.2366218). The result is plain skinning: the player runs no simulation.

    Last, an edge that still stretches too far trades weight with its neighbour, so the step in the
    weights widens until it fits. On the rooftop shirt that took torn edges from 43 to 1.

    Works in metres with Z up, on the welded vertices of the asset.
"""
import numpy as np
from scipy.spatial import cKDTree

from core.garment_fit import edges_of
from core.garment_weights import project

ROUNDS = 12
# Shirt cloth stretches little and shrinks freely, because it folds: the limit that matters is the longest length.
STRETCH_MOST = 1.1
SQUASH_LEAST = 0.6
HANGING_LEAST = 0.92
GAP_M = 0.006
REACH_M = 0.05
SMOOTH_SHARE = 0.3
PULL_TO_START = 0.02
# Separate panels of one garment: the seam goes in as an edge, or the gap opens.
STITCH_M = 0.012
MOST_CANDIDATES = 6
MOST_BONES = 4
WEIGHT_FLOOR = 0.01
RIDGE = 0.002
AFTER_ROUNDS = 1
# The cast benchmark calls an edge torn at 3 times its length: weights spread before that, in poses the benchmark never sees.
# 2.0 let go of the sleeves: 238 skin pokes on the cast against 26. Higher tears more, lower shows more skin.
RELAX_RATIO = 2.5
RELAX_MIN_M = 0.02
RELAX_ROUNDS = 150
RELAX_SHARE = 0.5
LIMB_HOLDS = 0.6
# Cloth only hits what it dresses: a shirt does not back away from the arm pressed on it, nor a trouser leg from the other leg.
TOUCHES = {0: (0, 3, 4), 1: (1, 0), 2: (2, 0), 3: (3, 0), 4: (4, 0)}


def skinned(weights, turns, homogeneous):
    """Blend the matrices first: one product per vertex instead of one per bone."""
    blended = (weights @ turns.reshape(len(turns), 12)).reshape(-1, 3, 4)
    return np.einsum("vab,vb->va", blended, homogeneous)


def zone_of(name):
    side = 0 if name.startswith("Left") else 1
    if "Hand" in name or "ForeArm" in name:
        return 5
    if "Arm" in name:
        return 1 + side
    if "Leg" in name or "Foot" in name or "Toe" in name:
        return 3 + side
    return 0 if name.startswith(("Hips", "Spine", "Neck")) or "Shoulder" in name else 6


class Collider:
    """The triangles one zone of cloth can touch, with the triangles around each of their vertices."""

    def __init__(self, faces):
        self.faces = faces
        self.used = np.unique(faces)
        place = np.full(faces.max() + 1, -1, dtype=np.int64)
        place[self.used] = np.arange(len(self.used))
        rows = [[] for _ in self.used]
        for index, face in enumerate(faces):
            for vertex in face:
                rows[place[vertex]].append(index)
        self.around = np.full((len(rows), max(len(row) for row in rows)), -1, dtype=np.int64)
        for index, row in enumerate(rows):
            self.around[index, :len(row)] = row

    def nearest(self, points, skin):
        """Closest point on these triangles and the triangle normal, for points within reach."""
        gaps, near = cKDTree(skin[self.used]).query(points, distance_upper_bound=REACH_M * 2)
        alive = np.flatnonzero(np.isfinite(gaps))
        spot = np.zeros((len(alive), 3))
        normal = np.zeros((len(alive), 3))
        best = np.full(len(alive), np.inf)
        for slot in range(self.around.shape[1]):
            chosen = self.around[near[alive], slot]
            some = np.flatnonzero(chosen >= 0)
            if not len(some):
                continue
            a, b, c = (skin[self.faces[chosen[some], corner]] for corner in range(3))
            found, _ = project(points[alive[some]], a, b, c)
            gap = np.linalg.norm(found - points[alive[some]], axis=1)
            better = gap < best[some]
            where = some[better]
            best[where] = gap[better]
            spot[where] = found[better]
            normal[where] = np.cross(b - a, c - a)[better]
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
        close = best < REACH_M
        return alive[close], spot[close], normal[close]


class Cloth:
    """One garment as a single mesh, with the seam between panels as an edge."""

    def __init__(self, points, faces, weights, bones, hanging_below):
        self.rest = points
        self.count = len(points)
        self.weights = weights
        edges = {tuple(pair) for pair in edges_of(faces).tolist()}
        self.edges = np.array(sorted(edges | self.stitches(edges)), dtype=np.int64)
        self.lengths = np.linalg.norm(points[self.edges[:, 0]] - points[self.edges[:, 1]], axis=1)
        self.degree = np.maximum(np.bincount(self.edges.ravel(), minlength=self.count), 1)[:, None]
        self.least = np.full(len(self.edges), SQUASH_LEAST)
        if hanging_below is not None:
            self.least[(points[self.edges, 2] < hanging_below).all(axis=1)] = HANGING_LEAST
        zones = np.array([zone_of(name) for name in bones])
        self.zone = zones[weights.argmax(axis=1)]
        self.homogeneous = np.concatenate([points, np.ones((self.count, 1))], axis=1)

    def stitches(self, edges):
        root = list(range(self.count))

        def find(item):
            while root[item] != item:
                root[item] = root[root[item]]
                item = root[item]
            return item

        for first, second in edges:
            root[find(first)] = find(second)
        found = set()
        for first, second in cKDTree(self.rest).query_pairs(STITCH_M):
            if find(first) != find(second):
                found.add((min(first, second), max(first, second)))
        return found

    def around(self, values):
        total = np.zeros_like(values)
        np.add.at(total, self.edges[:, 0], values[self.edges[:, 1]])
        np.add.at(total, self.edges[:, 1], values[self.edges[:, 0]])
        return total / self.degree

    def pose(self, turns):
        return skinned(self.weights, turns, self.homogeneous)

    def stretch(self, points):
        return np.linalg.norm(points[self.edges[:, 0]] - points[self.edges[:, 1]], axis=1) / np.maximum(self.lengths, 1e-5)


def settle(cloth, start, skin, colliders):
    """One pose: the cloth leaves the skin and returns to its length, the shift smoothed so that it does not crumple."""
    points = start.copy()
    first, second = cloth.edges[:, 0], cloth.edges[:, 1]
    for round_ in range(ROUNDS):
        way = points[second] - points[first]
        length = np.maximum(np.linalg.norm(way, axis=1), 1e-6)
        wanted = np.clip(length, cloth.lengths * cloth.least, cloth.lengths * STRETCH_MOST)
        step = way * ((length - wanted) / length * 0.5)[:, None]
        moved = np.zeros_like(points)
        np.add.at(moved, first, step)
        np.add.at(moved, second, -step)
        points += moved / cloth.degree * 1.6
        shift = points - start
        points = start + shift * (1 - SMOOTH_SHARE - PULL_TO_START) + cloth.around(shift) * SMOOTH_SHARE
        if round_ % 2:
            continue
        for zone, collider in colliders.items():
            mine = np.flatnonzero(cloth.zone == zone)
            if not len(mine):
                continue
            index, spot, normal = collider.nearest(points[mine], skin)
            under = ((points[mine[index]] - spot) * normal).sum(axis=1) < GAP_M
            points[mine[index[under]]] = spot[under] + normal[under] * GAP_M
    return points


def on_simplex(gram, target):
    """Minimum of w'Gw - 2c'w with w >= 0 and sum 1: few bones, so solve the system and drop the negative until none is left."""
    alive = list(range(len(target)))
    while True:
        size = len(alive)
        system = np.zeros((size + 1, size + 1))
        system[:size, :size] = gram[np.ix_(alive, alive)]
        system[size, :size] = system[:size, size] = 1
        right = np.append(target[alive], 1)
        answer = np.linalg.lstsq(system, right, rcond=None)[0][:size]
        if size == 1 or answer.min() >= 0:
            found = np.zeros(len(target))
            found[alive] = answer
            return found
        alive.pop(int(answer.argmin()))


def fit(cloth, turns, targets):
    nearby = cloth.weights + cloth.around(cloth.weights)
    found = np.zeros_like(cloth.weights)
    poses = len(turns)
    for place in range(cloth.count):
        chosen = np.argsort(-nearby[place])[:MOST_CANDIDATES]
        chosen = chosen[nearby[place, chosen] > 0]
        if len(chosen) < 2:
            found[place] = cloth.weights[place]
            continue
        moved = (turns[:, chosen] @ cloth.homogeneous[place]).transpose(0, 2, 1).reshape(poses * 3, len(chosen))
        goal = targets[:, place].reshape(poses * 3)
        gram = moved.T @ moved / poses + RIDGE * np.eye(len(chosen))
        found[place, chosen] = on_simplex(gram, moved.T @ goal / poses + RIDGE * cloth.weights[place, chosen])
    return found


def finish(cloth, weights):
    for _ in range(AFTER_ROUNDS):
        weights = (weights + cloth.around(weights)) / 2
    weights[weights < WEIGHT_FLOOR] = 0
    order = np.argsort(-weights, axis=1)
    keep = np.zeros_like(weights, dtype=bool)
    np.put_along_axis(keep, order[:, :MOST_BONES], True, axis=1)
    weights = np.where(keep, weights, 0)
    return weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-6)


def relax(cloth, turns, limb):
    """`limb` marks the bones of arms and legs: a sleeve that lets go of the arm shows the skin, so the torso side gives way instead."""
    first, second = cloth.edges[:, 0], cloth.edges[:, 1]
    left = 0
    for _ in range(RELAX_ROUNDS):
        worst = np.zeros(len(cloth.edges))
        reach = np.zeros(len(cloth.edges))
        for pose in turns:
            points = cloth.pose(pose)
            length = np.linalg.norm(points[first] - points[second], axis=1)
            ratio = length / np.maximum(cloth.lengths, 1e-4)
            better = ratio > worst
            worst[better], reach[better] = ratio[better], length[better]
        bad = np.flatnonzero((worst > RELAX_RATIO) & (reach > RELAX_MIN_M))
        left = len(bad)
        if not left:
            break
        total = np.zeros_like(cloth.weights)
        count = np.zeros(cloth.count)
        np.add.at(total, first[bad], cloth.weights[second[bad]])
        np.add.at(total, second[bad], cloth.weights[first[bad]])
        np.add.at(count, first[bad], 1)
        np.add.at(count, second[bad], 1)
        moved = count > 0
        share = (RELAX_SHARE * (1 - LIMB_HOLDS * cloth.weights[moved][:, limb].sum(axis=1)))[:, None]
        cloth.weights[moved] = cloth.weights[moved] * (1 - share) + total[moved] / count[moved, None] * share
    return left


def read_turns(path, names):
    """Bone matrices of the sampled poses, in the order of the weight columns."""
    data = np.load(path)
    place = {str(name): index for index, name in enumerate(data["names"])}
    turns = np.zeros((len(data["turns"]), len(names), 3, 4))
    turns[:, :, 0, 0] = turns[:, :, 1, 1] = turns[:, :, 2, 2] = 1
    for column, name in enumerate(names):
        if name in place:
            turns[:, column] = data["turns"][:, place[name]]
    return turns


def solve(points, faces, weights, source, turns, hanging_below=None):
    """`turns` has one 3x4 matrix per weight column in each pose; returns the weights of the garment."""
    cloth = Cloth(points, faces, weights.copy(), source.names, hanging_below)
    zones = np.array([zone_of(name) for name in source.names])[source.weights.argmax(axis=1)]
    face_zones = zones[source.faces]
    colliders = {}
    for zone, touched in TOUCHES.items():
        chosen = np.isin(face_zones, touched).all(axis=1)
        if chosen.any():
            colliders[zone] = Collider(source.faces[chosen])
    homogeneous = np.concatenate([source.points, np.ones((len(source.points), 1))], axis=1)
    targets = []
    for pose in turns:
        skin = skinned(source.weights, pose, homogeneous)
        targets.append(settle(cloth, cloth.pose(pose), skin, colliders))
    targets = np.array(targets)
    cloth.weights = finish(cloth, fit(cloth, turns, targets))
    limb = np.array([zone_of(name) in (1, 2, 3, 4, 5) for name in source.names])
    left = relax(cloth, turns, np.zeros_like(limb) if hanging_below is not None else limb)
    cloth.weights = finish(cloth, cloth.weights)
    stretched = float(np.mean([np.percentile(cloth.stretch(cloth.pose(pose)), 99.5) for pose in turns]))
    print("PANO", len(turns), "poses", cloth.count, "pontos", "esticão", round(stretched, 2), "arestas sobrando", left)
    return cloth.weights
