"""
    License information: data/licenses/makehuman_license.txt

    Pieces worn over the clothes of who serves at a counter: the apron and the name badge (`derive_garment.py --apron/--badge`).

    The vision critic of the shop interiors (cycle 3, 2026-11-18) saw the apron of the barista, the grocer, the baker and the
    barber with a torn, saw-toothed border where the T-shirt came through, holes over the belly and the lumps of modelling clay,
    and the badge of the shoe shop as a crumpled white blot. Three causes, measured on Carlos (café):

    - the apron was pushed off the shirt point by point (`cast-assemble.clear_skin`), so it copied every wrinkle of the shirt
      under it, and the cloth fit of the dress it was cut from left folds over the belly: the clay;
    - its points took the weights of the skin, the shirt those of its own cloth: in every gesture the two moved apart and the
      shirt crossed the apron where they were closest, at its border and over the belly;
    - the badge is a patch of a dress fitted point by point on the chest: it took the shirt's wrinkles too, and bent with them.

    So the apron is first smoothed (Taubin: smooth without shrinking), then lifted by a smooth field that is never less than the
    gap it needs over the shirt; above the waist it moves with the shirt under it; and one soft pleat runs down the front of its
    skirt. The badge is a flat plate that moves as one piece with the point of the shirt under its middle.
"""
import bmesh
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

# Taubin (1995), «A signal processing approach to fair surface design», SIGGRAPH '95, doi:10.1145/218380.218473: a step of
# LAMBDA and one of MU take the bumps away without the shrinking of plain Laplacian smoothing. ROUNDS chosen on Carlos's apron
# (edges of 16 mm): 8 rounds took the folds of the belly away and left the bib's shape.
TAUBIN_LAMBDA = 0.5
TAUBIN_MU = -0.53
TAUBIN_ROUNDS = 8
# The gap the piece needs is spread over this many rings of the mesh before it is smoothed: a lump of the shirt under the apron
# lifts a patch of 5 to 6 cm, not a point (a point lifted made the bump of clay).
LIFT_RINGS = 3
LIFT_ROUNDS = 6
LIFT_REACH_M = 0.06
# Above the waist the apron moves with the shirt under it, faded over FOLLOW_FADE_M: below, the skirt hangs over both legs with
# its own weights (a skirt that followed each leg tore in the middle of the step).
FOLLOW_NEIGHBOURS = 4
FOLLOW_FADE_M = 0.1
WEIGHT_FLOOR = 0.01
# One soft pleat down the front of the skirt (the critic: «uma prega vertical»): a ridge with a shallow valley on each side,
# wider than two edges of the cloth, fading in over the first centimetres under the waist.
PLEAT_DEPTH_M = 0.007
PLEAT_HALF_M = 0.035
PLEAT_FADE_M = 0.08
# The badge: a flat plate, this far over the cloth under it at its closest point.
BADGE_GAP_M = 0.006


def _twins(obj):
    """The two halves of a point split on a seam of the map move as one."""
    spots = {}
    twin = np.array([spots.setdefault(tuple(round(value, 4) for value in vertex.co), len(spots)) for vertex in obj.data.vertices])
    return twin, len(spots)


def _ring_edges(obj, twin):
    edges = np.array([tuple(edge.vertices) for edge in obj.data.edges], dtype=np.int64).reshape(-1, 2)
    edges = np.unique(np.sort(twin[edges], axis=1), axis=0)
    return edges[edges[:, 0] != edges[:, 1]]


def _mean_of_neighbours(values, edges, count):
    total = np.zeros((count,) + values.shape[1:])
    np.add.at(total, edges[:, 0], values[edges[:, 1]])
    np.add.at(total, edges[:, 1], values[edges[:, 0]])
    degree = np.bincount(edges.ravel(), minlength=count).reshape((count,) + (1,) * (values.ndim - 1))
    return np.where(degree > 0, total / np.maximum(degree, 1), values)


def _rest_tree(objects):
    points, faces = [], []
    for obj in objects:
        world = obj.matrix_world
        offset = len(points)
        points.extend(world @ vertex.co for vertex in obj.data.vertices)
        faces.extend(tuple(index + offset for index in polygon.vertices) for polygon in obj.data.polygons)
    return BVHTree.FromPolygons(points, faces) if faces else None


def _world(obj):
    turn = np.array(obj.matrix_world)
    local = np.array([tuple(vertex.co) for vertex in obj.data.vertices], dtype=np.float64)
    return local @ turn[:3, :3].T + turn[:3, 3]


def _store(obj, points):
    inverse = np.linalg.inv(np.array(obj.matrix_world))
    local = points @ inverse[:3, :3].T + inverse[:3, 3]
    for vertex, point in zip(obj.data.vertices, local):
        vertex.co = Vector(point)
    obj.data.update()


def taubin(points, edges, count, rounds=TAUBIN_ROUNDS):
    """Points (one row per twin) smoothed without shrinking."""
    for _ in range(rounds):
        for factor in (TAUBIN_LAMBDA, TAUBIN_MU):
            points = points + factor * (_mean_of_neighbours(points, edges, count) - points)
    return points


def lift_field(need, edges, count, rings=LIFT_RINGS, rounds=LIFT_ROUNDS):
    """The lift each point needs, spread over `rings` and smoothed, never under what the point itself needs."""
    spread = need.copy()
    for _ in range(rings):
        widest = spread.copy()
        np.maximum.at(widest, edges[:, 0], spread[edges[:, 1]])
        np.maximum.at(widest, edges[:, 1], spread[edges[:, 0]])
        spread = widest
    for _ in range(rounds):
        spread = (spread + _mean_of_neighbours(spread, edges, count)) / 2
    return np.maximum(spread, need)


def _normals(points, obj, twin, count):
    normals = np.zeros((count, 3))
    for polygon in obj.data.polygons:
        corners = twin[list(polygon.vertices)]
        first, second, third = points[corners[0]], points[corners[1]], points[corners[2]]
        normal = np.cross(second - first, third - first)
        for corner in corners:
            normals[corner] += normal
    return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)


def lift_over(piece, under, gap, smooth=True):
    """Smooths `piece` and lifts it off `under` (the clothes beneath, at rest) by a smooth field: at least `gap` everywhere."""
    twin, count = _twins(piece)
    edges = _ring_edges(piece, twin)
    world = _world(piece)
    points = np.zeros((count, 3))
    points[twin] = world
    if smooth:
        points = taubin(points, edges, count)
    tree = _rest_tree(under)
    if tree is None:
        _store(piece, points[twin])
        return 0.0
    normals = _normals(points, piece, twin, count)
    need = np.zeros(count)
    way = np.zeros((count, 3))
    for index, point in enumerate(points):
        location, normal, _, _ = tree.find_nearest(Vector(point), LIFT_REACH_M)
        if location is None:
            continue
        normal = np.array(normal)
        if normal.dot(normals[index]) < 0:
            normal = -normal
        way[index] = normal
        need[index] = max(0.0, gap - float((point - np.array(location)).dot(normal)))
    field = lift_field(need, edges, count)
    # The direction is the piece's own smooth normal, turned to the side of the clothes' normal: theirs alone is as wrinkled as the shirt.
    direction = normals * np.where((normals * way).sum(axis=1) < 0, -1.0, 1.0)[:, None]
    points = points + direction * field[:, None]
    _store(piece, points[twin])
    return float(field.max())


def pleat(piece, waist_z, middle_x):
    """One soft vertical pleat down the middle of the front of the skirt, under `waist_z`."""
    twin, count = _twins(piece)
    world = _world(piece)
    points = np.zeros((count, 3))
    points[twin] = world
    normals = _normals(points, piece, twin, count)
    across = (points[:, 0] - middle_x) / PLEAT_HALF_M
    shape = np.exp(-across ** 2) - 0.5 * np.exp(-(across - 1.6) ** 2) - 0.5 * np.exp(-(across + 1.6) ** 2)
    fade = np.clip((waist_z - points[:, 2]) / PLEAT_FADE_M, 0, 1)
    front = normals[:, 1] < 0  # the body looks to -y in the assembly scene
    points = points + normals * (PLEAT_DEPTH_M * shape * fade * front)[:, None]
    _store(piece, points[twin])
    return int((fade * front > 0).sum())


def _weights_of(obj):
    names = [group.name for group in obj.vertex_groups]
    weights = np.zeros((len(obj.data.vertices), len(names)), dtype=np.float32)
    for vertex in obj.data.vertices:
        for item in vertex.groups:
            weights[vertex.index, item.group] = item.weight
    return names, weights


def _inner_field(inner):
    names, rows, points = [], [], []
    for obj in inner:
        own, weights = _weights_of(obj)
        for name in own:
            if name not in names:
                names.append(name)
        mapped = np.zeros((len(weights), len(names)), dtype=np.float32)
        for column, name in enumerate(own):
            mapped[:, names.index(name)] = weights[:, column]
        rows.append(mapped)
        points.extend(map(tuple, _world(obj)))
    width = len(names)
    table = np.vstack([np.pad(row, ((0, 0), (0, width - row.shape[1]))) for row in rows]) if rows else np.zeros((0, 0))
    tree = KDTree(len(points))
    for index, point in enumerate(points):
        tree.insert(point, index)
    tree.balance()
    return names, table, tree


def _write_weights(obj, index, wanted):
    for group in obj.vertex_groups:
        group.remove([index])
    kept = dict(sorted(wanted.items(), key=lambda entry: -entry[1])[:4])
    total = sum(kept.values()) or 1.0
    for name, weight in kept.items():
        (obj.vertex_groups.get(name) or obj.vertex_groups.new(name=name)).add([index], weight / total, "REPLACE")


def follow_inner(piece, inner, waist_z):
    """Above `waist_z` the piece takes the weights of the clothes under it, so the two move together in every gesture."""
    if not inner:
        return 0
    names, table, tree = _inner_field(inner)
    own_names, own = _weights_of(piece)
    moved = 0
    for vertex, point in zip(piece.data.vertices, _world(piece)):
        share = float(np.clip((point[2] - waist_z) / FOLLOW_FADE_M + 1, 0, 1))
        if share <= 0:
            continue
        found = tree.find_n(Vector(point), FOLLOW_NEIGHBOURS)
        closeness = np.array([1 / max(distance, 1e-4) for _, _, distance in found], dtype=np.float32)
        mixed = (closeness / closeness.sum()) @ table[[index for _, index, _ in found]]
        wanted = {names[column]: float(mixed[column]) * share for column in np.flatnonzero(mixed > WEIGHT_FLOOR).tolist()}
        for column in np.flatnonzero(own[vertex.index] > 0).tolist():
            wanted[own_names[column]] = wanted.get(own_names[column], 0.0) + float(own[vertex.index, column]) * (1 - share)
        _write_weights(piece, vertex.index, wanted)
        moved += 1
    return moved


def rigid_badge(badge, under, inner):
    """The badge as a flat plate over the clothes, moving as one piece with the point of the clothes under its middle."""
    world = _world(badge)
    middle = world.mean(axis=0)
    _, _, turn = np.linalg.svd(world - middle)
    normal = turn[2]
    tree = _rest_tree(under)
    if tree is not None:
        location, outward, _, _ = tree.find_nearest(Vector(middle), LIFT_REACH_M)
        if location is not None and np.dot(normal, np.array(outward)) < 0:
            normal = -normal
    flat = world - np.outer((world - middle) @ normal, normal)
    if tree is not None:
        lift = 0.0
        for point in flat:
            location, outward, _, _ = tree.find_nearest(Vector(point), LIFT_REACH_M)
            if location is not None:
                lift = max(lift, BADGE_GAP_M - float((point - np.array(location)) @ normal))
        flat = flat + normal * lift
    _store(badge, flat)
    if inner:
        names, table, tree_inner = _inner_field(inner)
        _, index, _ = tree_inner.find(Vector(middle))
        row = table[index]
        wanted = {names[column]: float(row[column]) for column in np.flatnonzero(row > WEIGHT_FLOOR).tolist()}
        for vertex in badge.data.vertices:
            _write_weights(badge, vertex.index, wanted)
    return float(np.abs((world - middle) @ normal).max())
