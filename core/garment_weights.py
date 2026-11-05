"""
    License information: data/licenses/makehuman_license.txt

    Skin weights for garments in makehuman2-parle.

    Abdrashitov, Raichstat, Monsen & Hill 2023, Robust Skin Weights Transfer via Weight Inpainting,
    SIGGRAPH Asia Technical Communications, DOI 10.1145/3610543.3626180. Weights are copied from the
    body only where the closest skin point is near and faces the same way. Every other vertex is
    solved from those by minimising Dirichlet plus Laplacian energy, so nothing ties a sleeve to the
    ribs or one leg of a pair of shorts to the other.

    The paper leaves the two masks open to the artist (its section 3.4). The crotch is always solved
    here, and a skirt then hangs from the hips.

    Works in metres with Z up, on the welded vertices of the asset, like garment_fit.
"""
import os

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

from core.garment_fit import SKIRT_WORDS, edges_of, to_metres, vertex_normals

# Section 3.1 of the paper: 5% of the bounding box diagonal and 35 degrees.
MATCH_SHARE = 0.05
MATCH_COS = float(np.cos(np.radians(35)))
# Measured on the cast: the crotch fold reaches 18 cm above the crotch. Solving the armpit too made it worse on the Guia,
# 12 torn edges with no mask, 14 with 13 cm and 20 with 18 cm: there the body weights are better than the solve.
SEAT_HALF_WIDTH_M = 0.06
SEAT_HEIGHT_M = 0.18
CROTCH_REACH_M = 0.1
SMOOTH_ROUNDS = 5
# A skirt hangs from the hips: below the hip joints the leg weight becomes a mix of both legs, by the side the point is on.
# Measured on Dona Rosa with the 58 clips: side .54 tore 12 edges, .66 tore 2, .75 none; middle .5 tore 31, .3 none.
SKIRT_FADE_M = 0.25
SKIRT_HANGS_M = 0.05
SKIRT_MIDDLE = 0.3
SKIRT_SIDE = 0.75
REGULARISE = 1e-8
WEIGHT_FLOOR = 0.01
MOST_BONES = 8


def closest_on_triangles(points, skin):
    """Closest point among the triangles around the nearest skin vertex, with its barycentric weights."""
    _, near = skin.tree.query(points)
    best = np.full(len(points), np.inf)
    face = np.zeros(len(points), dtype=np.int64)
    bary = np.zeros((len(points), 3))
    for slot in range(skin.around.shape[1]):
        chosen = skin.around[near, slot]
        alive = chosen >= 0
        if not alive.any():
            continue
        triangle = skin.faces[chosen[alive]]
        spot, mix = project(points[alive], skin.points[triangle[:, 0]], skin.points[triangle[:, 1]], skin.points[triangle[:, 2]])
        gap = np.linalg.norm(spot - points[alive], axis=1)
        where = np.flatnonzero(alive)
        better = gap < best[where]
        best[where[better]] = gap[better]
        face[where[better]] = chosen[alive][better]
        bary[where[better]] = mix[better]
    return best, face, bary


def project(point, a, b, c):
    """Point on triangle abc: the plane projection with its barycentric coordinates clamped back inside."""
    first, second, third = b - a, c - a, point - a
    d00, d01, d11 = (first * first).sum(1), (first * second).sum(1), (second * second).sum(1)
    d20, d21 = (third * first).sum(1), (third * second).sum(1)
    under = np.maximum(d00 * d11 - d01 * d01, 1e-18)
    v = (d11 * d20 - d01 * d21) / under
    w = (d00 * d21 - d01 * d20) / under
    mix = np.clip(np.stack([1 - v - w, v, w], 1), 0, None)
    mix /= np.maximum(mix.sum(1, keepdims=True), 1e-12)
    return mix[:, :1] * a + mix[:, 1:2] * b + mix[:, 2:] * c, mix


class Source:
    """The rigged body, as the caller dumped it: points, triangles, weights, bone names and joints."""

    def __init__(self, path):
        data = np.load(path, allow_pickle=False)
        self.points = data["points"].astype(np.float64)
        self.faces = data["faces"].astype(np.int64)
        self.weights = data["weights"].astype(np.float64)
        self.names = [str(name) for name in data["names"]]
        self.joints = dict(zip([str(name) for name in data["joint_names"]], data["joints"].astype(np.float64)))
        self.normals = vertex_normals(self.points, self.faces)
        self.tree = cKDTree(self.points)
        rows = [[] for _ in range(len(self.points))]
        for place, face in enumerate(self.faces):
            for vertex in face:
                rows[vertex].append(place)
        self.around = np.full((len(rows), max(len(row) for row in rows)), -1, dtype=np.int64)
        for place, row in enumerate(rows):
            self.around[place, :len(row)] = row
        middle = self.points[np.abs(self.points[:, 0] - np.median(self.points[:, 0])) < 0.006]
        self.crotch = middle[middle[:, 2].argmin()]


def laplacian(points, faces):
    """Cotangent stiffness matrix and lumped mass, the L and M of the paper."""
    count = len(points)
    rows, columns, values = [], [], []
    area = np.zeros(count)
    for one, two, three in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        first, second = points[faces[:, two]] - points[faces[:, one]], points[faces[:, three]] - points[faces[:, one]]
        cross = np.linalg.norm(np.cross(first, second), axis=1)
        cotangent = (first * second).sum(1) / np.maximum(cross, 1e-12)
        # Sliver triangles of a folded hem give huge cotangents, and the solve then copies one vertex to the whole island.
        cotangent = np.clip(cotangent, -5, 5) / 2
        rows += [faces[:, two], faces[:, three]]
        columns += [faces[:, three], faces[:, two]]
        values += [cotangent, cotangent]
        np.add.at(area, faces[:, one], cross / 6)
    stiffness = sparse.coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))), shape=(count, count)).tocsr()
    stiffness = stiffness - sparse.diags(np.asarray(stiffness.sum(axis=1)).ravel())
    return stiffness, np.maximum(area, 1e-9)


def matched(points, normals, skin):
    gap, face, bary = closest_on_triangles(points, skin)
    corners = skin.faces[face]
    facing = (skin.normals[corners] * bary[:, :, None]).sum(axis=1)
    facing /= np.maximum(np.linalg.norm(facing, axis=1, keepdims=True), 1e-12)
    diagonal = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
    # The inner layer of a folded hem faces the body: the paper tries the flipped normal too.
    agree = np.abs((facing * normals).sum(axis=1)) > MATCH_COS
    copied = (skin.weights[corners] * bary[:, :, None]).sum(axis=1)
    return (gap < MATCH_SHARE * diagonal) & agree, copied


def hard_places(points, skin):
    """Vertices the body cannot answer for: between the two legs each side pulls its own way."""
    middle = np.abs(points[:, 0] - skin.crotch[0]) < SEAT_HALF_WIDTH_M
    return middle & (points[:, 2] > skin.crotch[2] - CROTCH_REACH_M) & (points[:, 2] < skin.crotch[2] + SEAT_HEIGHT_M)


def hang(points, values, skin):
    """The hem belongs to the hips and to the side it is on: copied from the leg, it opens up at every step."""
    left, right = skin.joints["LeftUpLeg"], skin.joints["RightUpLeg"]
    centre = (left[0] + right[0]) / 2
    wide = max(1e-3, float(np.percentile(np.abs(points[:, 0] - centre), 95)))
    share = np.clip((left[2] - points[:, 2]) / SKIRT_FADE_M, 0, 1)[:, None]
    side = 1 if left[0] > right[0] else -1
    lean = np.clip(side * (points[:, 0] - centre) / (wide * SKIRT_SIDE), -1, 1)
    hips = SKIRT_MIDDLE * (1 - np.abs(lean))
    target = np.zeros_like(values)
    target[:, skin.names.index("Hips")] = hips
    target[:, skin.names.index("LeftUpLeg")] = (1 - hips) * (1 + lean) / 2
    target[:, skin.names.index("RightUpLeg")] = (1 - hips) * (1 - lean) / 2
    return values * (1 - share) + target * share


def inpaint(points, faces, known, values):
    """Equation 6 of the paper, one solve for all bones."""
    stiffness, area = laplacian(points, faces)
    energy = (-stiffness + stiffness @ sparse.diags(1 / area) @ stiffness).tocsc()
    free = np.flatnonzero(~known)
    fixed = np.flatnonzero(known)
    result = values.copy()
    if not len(free):
        return result
    inner = energy[free][:, free] + REGULARISE * sparse.identity(len(free), format="csc")
    used = np.flatnonzero(values[fixed].sum(axis=0) > 0)
    right = -(energy[free][:, fixed] @ values[fixed][:, used])
    result[free] = 0
    result[np.ix_(free, used)] = splu(inner.tocsc()).solve(np.asarray(right))
    return result


def islands(edges, count):
    root = np.arange(count)
    moved = True
    while moved:
        low = np.minimum(root[edges[:, 0]], root[edges[:, 1]])
        before = root.copy()
        np.minimum.at(root, edges[:, 0], low)
        np.minimum.at(root, edges[:, 1], low)
        root = root[root]
        moved = (root != before).any()
    return root


def weigh(points, faces, skin, skirt):
    normals = vertex_normals(points, faces)
    good, copied = matched(points, normals, skin)
    known = good & ~hard_places(points, skin)
    edges = edges_of(faces)
    # A piece with no anchor has nothing to solve from: it keeps what the closest skin gives.
    root = islands(edges, len(points))
    for island in np.unique(root):
        if not known[root == island].any():
            known[root == island] = True
    values = inpaint(points, faces, known, copied)
    degree = np.maximum(np.bincount(edges.ravel(), minlength=len(points)), 1)[:, None]
    for _ in range(SMOOTH_ROUNDS):
        total = np.zeros_like(values)
        np.add.at(total, edges[:, 0], values[edges[:, 1]])
        np.add.at(total, edges[:, 1], values[edges[:, 0]])
        values[~known] = ((values + total / degree) / 2)[~known]
    return tidy(hang(points, values, skin) if skirt else values), int((~known).sum())


def tidy(values):
    values = np.clip(values, 0, None)
    values[values < WEIGHT_FLOOR] = 0
    order = np.argsort(-values, axis=1)
    keep = np.zeros_like(values, dtype=bool)
    np.put_along_axis(keep, order[:, :MOST_BONES], True, axis=1)
    values = np.where(keep, values, 0)
    return values / np.maximum(values.sum(axis=1, keepdims=True), 1e-9)


def welded_faces(obj):
    """Faces on the UV seam use the overflow copies of their vertices: dropped, the seams would be missing from the cloth."""
    faces = np.asarray(obj.fverts, dtype=np.int64)
    table = np.arange(faces.max() + 1)
    if obj.overflow is not None and len(obj.overflow):
        overflow = np.asarray(obj.overflow, dtype=np.int64)
        table[overflow[:, 1]] = overflow[:, 0]
    faces = table[faces]
    return faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])]


def weigh_garments(body, names, source, target, turns_path=None):
    """Writes one entry per garment: welded points and their weights, for whoever binds the asset to the rig.

    With the poses of the clips the weights also go through the cloth solve, and the file says so with a `cloth` entry.
    """
    from core.garment_cloth import read_turns, solve

    skin = Source(source)
    found = {"bones": np.array(skin.names)}
    turns = read_turns(turns_path, skin.names) if turns_path and os.path.exists(turns_path) else None
    if turns is not None:
        found["cloth"] = np.array(1)
    for asset in body.attachedAssets:
        stem = os.path.splitext(os.path.basename(asset.filename))[0]
        if stem not in names:
            continue
        obj = asset.obj
        count = obj.n_origverts
        points = to_metres(np.asarray(obj.gl_coord[:count * 3], dtype=np.float64).reshape(-1, 3))
        faces = welded_faces(obj)
        skirt = any(word in stem.lower() for word in SKIRT_WORDS)
        values, solved = weigh(points, faces, skin, skirt)
        print("PESOS", stem, count, "pontos", solved, "resolvidos")
        if turns is not None:
            values = solve(points, faces, values, skin, turns, skin.joints["LeftUpLeg"][2] - SKIRT_HANGS_M if skirt else None)
        found[stem + "/points"] = points.astype(np.float32)
        found[stem + "/weights"] = values.astype(np.float16)
    np.savez_compressed(target, **found)
