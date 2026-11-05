"""
    License information: data/licenses/makehuman_license.txt

    Garment fit for makehuman2-parle.

    The .mhclo fit is three base vertices plus a fixed offset: it copies every hollow of the
    body and never looks at the skin, so on a sculpted body the skin comes through the cloth,
    a full skirt closes between the legs and shorts dive into the groin. This runs after that
    fit, on the welded vertices of each asset: the overflow table copies them to the UV seam
    twins afterwards, so a seam cannot open.

    Works in metres with Z up, the way the numbers below were measured on the game cast.
"""
import os

import numpy as np

SKIRT_WORDS = ("dress", "skirt")
CLEAR_GAP_M = 0.006
CLEAR_REACH_M = 0.06
CLEAR_STEP_M = 0.01
CLEAR_ROUNDS = 6
SKIN_CHUNK = 512
# A skirt hangs from here down; a fold deeper than this under the shell becomes a pleat.
SKIRT_ABOVE_CROTCH_M = 0.05
SKIRT_FADE_M = 0.15
SKIRT_FLARE_M = 0.03
FOLD_DEEPEST_M = 0.02
HEM_BINS = 36
SLICE_M = 0.02
ANGLES = 48
# Shorts: cloth goes straight from thigh to thigh and the crotch gets some slack.
LEGGED_BELOW_CROTCH_M = 0.1
HIP_FADE_M = 0.05
HIP_TOP_M = 0.16
HIP_DEEPEST_M = 0.012
GUSSET_DROP_M = 0.015
GUSSET_REACH_M = (0.1, 0.16, 0.1)
BRIDGE_ROUNDS = 10
BRIDGE_MOST_M = 0.03
EASE_ROUNDS = 8
UNFOLD_DOT = -0.5
UNFOLD_ROUNDS = 6


def to_metres(points):
    return np.stack([points[:, 0], -points[:, 2], points[:, 1]], 1).astype(np.float64) / 10


def from_metres(points):
    return (np.stack([points[:, 0], points[:, 2], -points[:, 1]], 1) * 10).astype(np.float32)


def vertex_normals(points, faces):
    normals = np.zeros_like(points)
    face = np.cross(points[faces[:, 1]] - points[faces[:, 0]], points[faces[:, 2]] - points[faces[:, 0]])
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face)
    return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)


def edges_of(faces):
    pairs = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    pairs = np.unique(np.sort(pairs, axis=1), axis=0)
    return pairs[pairs[:, 0] != pairs[:, 1]]


def around(values, edges, degree):
    total = np.zeros_like(values)
    np.add.at(total, edges[:, 0], values[edges[:, 1]])
    np.add.at(total, edges[:, 1], values[edges[:, 0]])
    return total / degree


class Skin:
    """The visible body: helper geometry shares the base mesh, and cloth must not collide with it."""

    def __init__(self, base):
        count = base.n_origverts
        points = to_metres(np.asarray(base.gl_coord[:count * 3], dtype=np.float64).reshape(-1, 3))
        # fverts holds the visible groups only, so helpers never get here.
        faces = np.asarray(base.fverts, dtype=np.int64)
        faces = faces[(faces < count).all(axis=1)]
        normals = vertex_normals(points, faces)
        used = np.unique(faces)
        self.points, self.normals = points[used], normals[used]
        self.crotch = self.find_crotch()

    def find_crotch(self):
        """The legs never reach the midline, so its lowest point is the crotch."""
        middle = self.points[np.abs(self.points[:, 0] - np.median(self.points[:, 0])) < 0.006]
        return middle[middle[:, 2].argmin()]

    def nearest(self, points):
        found = np.zeros(len(points), dtype=np.int64)
        gaps = np.zeros(len(points))
        for start in range(0, len(points), SKIN_CHUNK):
            part = points[start:start + SKIN_CHUNK]
            squares = ((part[:, None, :] - self.points[None, :, :]) ** 2).sum(axis=2)
            found[start:start + SKIN_CHUNK] = squares.argmin(axis=1)
            gaps[start:start + SKIN_CHUNK] = np.sqrt(squares.min(axis=1))
        return found, gaps


def clear(points, skin, edges, degree):
    """Takes the cloth off the skin, the push spread over neighbours so that it does not leave a dent."""
    for _ in range(CLEAR_ROUNDS):
        found, gaps = skin.nearest(points)
        normal = skin.normals[found]
        need = CLEAR_GAP_M - ((points - skin.points[found]) * normal).sum(axis=1)
        step = normal * np.clip(need, 0, CLEAR_STEP_M)[:, None] * (gaps < CLEAR_REACH_M)[:, None]
        for _ in range(2):
            step = (step + around(step, edges, degree)) / 2
        points = points + step
    return points


def turn(first, second, third):
    one, other = second - first, third - first
    return one[0] * other[1] - one[1] * other[0]


def hull_2d(points):
    """Andrew's monotone chain."""
    order = points[np.lexsort((points[:, 1], points[:, 0]))]

    def chain(rows):
        kept = []
        for row in rows:
            while len(kept) > 1 and turn(kept[-2], kept[-1], row) <= 0:
                kept.pop()
            kept.append(row)
        return kept[:-1]

    return np.array(chain(order) + chain(order[::-1]))


def rolling(values, size, axis, wrap):
    total = np.zeros_like(values)
    if wrap:
        for step in range(-size, size + 1):
            total += np.roll(values, step, axis=axis)
        return total / (2 * size + 1)
    pad = [(0, 0)] * values.ndim
    pad[axis] = (size, size)
    padded = np.pad(values, pad, mode="edge")
    for step in range(2 * size + 1):
        total += np.take(padded, range(step, step + values.shape[axis]), axis=axis)
    return total / (2 * size + 1)


def shell(points, top):
    """Convex outline of every slice of the piece: centre and radius per angle."""
    levels = np.arange(points[:, 2].min(), top + SLICE_M, SLICE_M)
    angles = (np.arange(ANGLES) + 0.5) / ANGLES * 2 * np.pi - np.pi
    centres = np.zeros((len(levels), 2))
    radii = np.zeros((len(levels), ANGLES))
    for row, level in enumerate(levels):
        near = points[np.abs(points[:, 2] - level) < SLICE_M * 1.5][:, :2]
        if len(near) < 6:
            centres[row], radii[row] = (centres[row - 1], radii[row - 1]) if row else (points[:, :2].mean(axis=0), 0)
            continue
        hull = hull_2d(near)
        centres[row] = hull.mean(axis=0)
        way = hull - centres[row]
        radii[row] = np.interp(angles, np.arctan2(way[:, 1], way[:, 0]), np.linalg.norm(way, axis=1), period=2 * np.pi)
    return levels, rolling(centres, 2, 0, False), rolling(rolling(radii, 2, 0, False), 2, 1, True)


def to_shell(points, chosen, top, share, deepest):
    """Moves what sits in a crease out to the shell of its slice, and returns the angle of every point."""
    levels, centres, radii = shell(points[chosen], top)
    height = np.clip((points[:, 2] - levels[0]) / SLICE_M, 0, len(levels) - 1.001)
    row = height.astype(int)
    rise = (height - row)[:, None]
    middle = centres[row] * (1 - rise) + centres[row + 1] * rise
    way = points[:, :2] - middle
    now = np.linalg.norm(way, axis=1)
    turn = np.arctan2(way[:, 1], way[:, 0])
    spot = (turn + np.pi) / (2 * np.pi) * ANGLES - 0.5
    column = np.floor(spot).astype(int)
    lean = spot - column
    rows = radii[row] * (1 - rise) + radii[row + 1] * rise
    places = np.arange(len(points))
    cover = rows[places, column % ANGLES] * (1 - lean) + rows[places, (column + 1) % ANGLES] * lean
    wanted = cover - deepest * (1 - np.exp(-np.maximum(cover - now, 0) / deepest))
    grown = now + share * chosen * np.maximum(wanted - now, 0)
    moved = points.copy()
    moved[:, :2] = middle + way / np.maximum(now, 1e-6)[:, None] * grown[:, None]
    return moved, turn


def ease(start, points, edges, degree):
    """Smooths the displacement, not the shape: neighbours move together and the cloth keeps its folds."""
    shift = points - start
    for _ in range(EASE_ROUNDS):
        shift = (shift + around(shift, edges, degree)) / 2
    return start + shift


def bell(points, skin, edges, degree):
    """A full skirt hangs as a bell with a level hem instead of following each leg."""
    waist = skin.crotch[2] + SKIRT_ABOVE_CROTCH_M
    share = np.clip((waist - points[:, 2]) / SKIRT_FADE_M, 0, 1)
    moved, turn = to_shell(points, np.ones(len(points), dtype=bool), waist, share, FOLD_DEEPEST_M)
    bins = ((turn + np.pi) / (2 * np.pi) * HEM_BINS).astype(int) % HEM_BINS
    hem = np.array([moved[bins == place, 2].min() if (bins == place).any() else np.nan for place in range(HEM_BINS)])
    hem = np.where(np.isnan(hem), np.nanmedian(hem), hem)
    level = float(np.median(hem))
    hem = rolling(hem, 1, 0, True)
    reach = np.clip((waist - moved[:, 2]) / np.maximum(waist - hem[bins], 1e-3), 0, 1)
    moved[:, 2] += (level - hem[bins]) * reach * (share > 0)
    moved = ease(points, moved, edges, degree)
    way = moved[:, :2] - skin.crotch[:2]
    moved[:, :2] += way / np.maximum(np.linalg.norm(way, axis=1, keepdims=True), 1e-6) * (SKIRT_FLARE_M * share)[:, None]
    return moved


def gusset(points, skin, edges, degree):
    """Cloth runs straight from thigh to thigh, and the crotch drops: the thigh no longer drags it into the groin."""
    crotch = skin.crotch
    height = points[:, 2] - crotch[2]
    above = (height > 0) & (height < HIP_TOP_M + SLICE_M)
    share = np.clip(height / HIP_FADE_M, 0, 1) * np.clip((HIP_TOP_M - height) / HIP_FADE_M, 0, 1)
    moved, _ = to_shell(points, above, crotch[2] + HIP_TOP_M, share, HIP_DEEPEST_M)
    near = np.clip(1 - np.linalg.norm((moved - crotch) / GUSSET_REACH_M, axis=1), 0, 1)
    moved[:, 2] -= GUSSET_DROP_M * near * near * (3 - 2 * near)
    return ease(points, moved, edges, degree)


def bridge(points, faces, edges, degree):
    """Every point in a valley rises to the mean of its neighbours, outwards only."""
    start = points.copy()
    normals = vertex_normals(points, faces)
    open_ = open_points(faces, len(points))
    for _ in range(BRIDGE_ROUNDS):
        rise = ((around(points, edges, degree) - points) * normals).sum(axis=1)
        rise = np.where((rise > 0) & ~open_, rise, 0)
        points = points + normals * (rise * 0.7)[:, None]
    shift = points - start
    length = np.maximum(np.linalg.norm(shift, axis=1, keepdims=True), 1e-9)
    return start + shift * np.minimum(1, BRIDGE_MOST_M / length)


def open_points(faces, count):
    pairs = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    unique, uses = np.unique(pairs, axis=0, return_counts=True)
    found = np.zeros(count, dtype=bool)
    found[unique[uses == 1].ravel()] = True
    return found


def unfold(points, faces, edges, degree):
    """A triangle turned over its neighbour goes back to the mean of its neighbours."""
    pairs = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    owners = np.tile(np.arange(len(faces)), 3)
    keys = np.sort(pairs, axis=1)
    order = np.lexsort((keys[:, 1], keys[:, 0]))
    same = (keys[order[:-1]] == keys[order[1:]]).all(axis=1)
    first, second = owners[order[:-1]][same], owners[order[1:]][same]
    shared = keys[order[:-1]][same]
    for _ in range(UNFOLD_ROUNDS):
        normal = np.cross(points[faces[:, 1]] - points[faces[:, 0]], points[faces[:, 2]] - points[faces[:, 0]])
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
        bad = np.unique(shared[(normal[first] * normal[second]).sum(axis=1) < UNFOLD_DOT])
        if not len(bad):
            break
        points = points.copy()
        points[bad] = around(points, edges, degree)[bad]
    return points


def fit_asset(obj, name, skin):
    count = obj.n_origverts
    points = to_metres(np.asarray(obj.gl_coord[:count * 3], dtype=np.float64).reshape(-1, 3))
    faces = np.asarray(obj.fverts, dtype=np.int64)
    faces = faces[(faces < count).all(axis=1)]
    edges = edges_of(faces)
    degree = np.maximum(np.bincount(edges.ravel(), minlength=count), 1)[:, None]
    skirt = any(word in name.lower() for word in SKIRT_WORDS)
    legged = points[:, 2].min() < skin.crotch[2] - LEGGED_BELOW_CROTCH_M < points[:, 2].max()
    if skirt:
        points = bell(points, skin, edges, degree)
    elif legged:
        points = gusset(points, skin, edges, degree)
    points = clear(points, skin, edges, degree)
    if legged and not skirt:
        points = bridge(points, faces, edges, degree)
    points = unfold(points, faces, edges, degree)
    obj.gl_coord[:count * 3] = from_metres(points).ravel()
    obj.overflowCorrection(obj.gl_coord)
    obj.calcNormals()


def fit_garments(body, names):
    """Call after updateAttachedAssets, with the names of what is worn on the body: shoes, hair and jewellery keep the plain .mhclo fit."""
    skin = Skin(body.baseMesh)
    for asset in body.attachedAssets:
        stem = os.path.splitext(os.path.basename(asset.filename))[0]
        if stem in names:
            fit_asset(asset.obj, stem, skin)
