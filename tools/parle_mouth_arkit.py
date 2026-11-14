"""
    License information: data/licenses/makehuman_license.txt

    The lip shapes the cast lacked for speech, with their ARKit names (2026-11-12): `mouthPucker`, `mouthFunnel`, `mouthPressLeft/Right`,
    `mouthRollLower/Upper` and `mouthStretchLeft/Right`. The kit had `jawOpen`, smile, frown and the fifteen visemes, and the player that
    plays human mouth trajectories from video (babylon-cinema #3029) could not show the lips rounding or pressing. `mouthClose` is the
    cast's own (`parle_mouth_close`), fitted to its jaw.

    The shapes are the CC0 pack faceunits01 (ARKit units by Mika Suominen, MakeHuman asset pack,
    https://static.makehumancommunity.org/assets/assetpacks/faceunits01.html), copied in `data/parle_faceunits/`. A pack target is indexed
    on the hm08 base mesh, and the cast body read back from glTF is not: points are reordered, split along the UV seams and some are gone
    under the clothes. The base head is walked onto the body head (a smooth field, sharper each round) and every body point takes the move
    of the base point on top of it.

    Both are walked with the jaw open (`jawOpen` of each at full weight): at rest the lips touch, and a point of the lower lip could take
    the id of the upper lip next to it, which tore the corners and left a jagged edge round the teeth in the first try (2026-11-08,
    Carlos). Open, the two lips are centimetres apart. The moves are scaled by the size of the head, so the women's bigger head (+25%,
    `parle_build.HEAD`) moves its lips as far for its size.

      Inside Blender, after the rest-shape rules: `add_arkit_mouth(body)` (called by `parle_face.clean_face`).
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE_MESH = ROOT / "data/base/hm08/base.obj"
PACK = ROOT / "data/parle_faceunits"
# Only the shapes the cast lacks; every shape key costs the GLB its vertices once more (about 0.1 MB a shape on a body).
SHAPES = (
    "mouthPucker", "mouthFunnel", "mouthPressLeft", "mouthPressRight", "mouthRollLower", "mouthRollUpper",
    "mouthStretchLeft", "mouthStretchRight",
)
# Seen in the Unity player at full weight (2026-11-12, docs/reports/2026-11-12-boca-formas-arkit/ in babylon-cinema): the pack's pucker
# folds the cupid's bow of the cast's lip into a V and parts its centre in a white slit, the funnel opens only a hole at the centre,
# and the upper roll folds a moustache into a dark blob. At 0.6, 0.6 and 0.5 of the pack they read clean on Carlos, Helena and
# Seu Manoel, so the shape at weight 1.0 is that share of the pack.
SHARE = {"mouthPucker": 0.6, "mouthFunnel": 0.6, "mouthRollUpper": 0.5}
# The pack's pucker narrows the mouth corners by 2.6 mm and pushes the lips 4.9 mm forward at the share above (Ivone, 2026-11-13): in the player
# at weight 1.0 it did not read as rounding on «v[o]cê» and «n[ã]o» (the lip protrusion of a real /o/ /u/ is 8 to 12 mm and the corners come in 8 to
# 10 mm). The cast's own `viseme_U` has that narrowing (10.5 mm in, 6.9 mm forward) on a skin that takes it without a fold, so the rounding
# shapes add this share of it to the pack's: pucker ends near 13 mm at full weight, funnel near 9 mm. Judged by eye, front and profile.
ROUND_FROM_U = {"mouthPucker": 0.6, "mouthFunnel": 0.4}
BODY_POINTS = 13380
SMOOTH_ROUNDS = (1600, 800, 400, 200, 100, 50, 25, 12, 6, 3, 2, 1)
# The walk has landed when 90% of the body head is this close to a base point (metres): 3.3 to 5.6 mm on the 42 of the cast (2026-11-12;
# the base is coarser than the shaped head round the ears and the scalp). Whether it landed on the right points is `MIN_CHECK`.
MAX_MISS_M = 0.007
# The cast's own jawOpen and left smile against the same pack shapes walked onto it: 0.91 to 0.93 and 0.77 to 0.83 on the 42 of the cast.
MIN_CHECK = 0.6
# The head is the top share of the height; the walk and the moves stay in it.
HEAD_SHARE = 0.16
# A point whose move is farther than this from the mean move of its neighbours is a stray: a point that took the id of a base point
# beside the one under it (a dimple on the chin and a pinch in the nostril in the first render, 2026-11-12). It takes the mean instead.
STRAY_M = 0.0008
STRAY_ROUNDS = 12
# Then the field is eased this many rounds over the skin, with the seam copies as one point: a base id is a whole face wide on the
# lip line, and a step from one id to the next showed as a crease.
EASE_ROUNDS = 2


def mh_to_blender(points):
    """MakeHuman is y-up in decimetres; Blender here is z-up in metres, facing -y."""
    return np.stack([points[:, 0], -points[:, 2], points[:, 1]], axis=1) * 0.1


def read_base(path=BASE_MESH):
    points, edges = [], set()
    for line in Path(path).read_text().splitlines():
        if line.startswith("v "):
            points.append([float(value) for value in line.split()[1:4]])
        elif line.startswith("f "):
            ring = [int(item.split("/")[0]) - 1 for item in line.split()[1:]]
            if max(ring) < BODY_POINTS:
                edges.update((min(a, b), max(a, b)) for a, b in zip(ring, ring[1:] + ring[:1]))
    return mh_to_blender(np.array(points[:BODY_POINTS])), np.array(sorted(edges))


def read_target(path):
    """Base ids and moves (Blender axes, metres) of a `.target`; helper points past the body are dropped."""
    ids, moves = [], []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[0].isdigit() and int(parts[0]) < BODY_POINTS:
            ids.append(int(parts[0]))
            moves.append([float(value) for value in parts[1:]])
    return np.array(ids, int), mh_to_blender(np.array(moves).reshape(-1, 3))


def dense(path, count=BODY_POINTS):
    """A target as one move per base point (zero where it does not move)."""
    ids, moves = read_target(path)
    full = np.zeros((count, 3))
    full[ids] = moves
    return full


def smooth(field, edges, rounds):
    degree = np.maximum(np.bincount(edges.ravel(), minlength=len(field)), 1)[:, None]
    for _ in range(rounds):
        total = np.zeros_like(field)
        np.add.at(total, edges[:, 0], field[edges[:, 1]])
        np.add.at(total, edges[:, 1], field[edges[:, 0]])
        field = (field + total / degree) / 2
    return field


def nearest(points, queries, chunk=512):
    """Index of and distance to the nearest of `points` for each query (brute force: a head is a few thousand points)."""
    hits, misses = np.empty(len(queries), int), np.empty(len(queries))
    for start in range(0, len(queries), chunk):
        block = queries[start:start + chunk]
        squared = ((block[:, None, :] - points[None, :, :]) ** 2).sum(axis=2)
        hits[start:start + chunk] = squared.argmin(axis=1)
        misses[start:start + chunk] = np.sqrt(squared.min(axis=1))
    return hits, misses


def head_of(points):
    return points[:, 2] > points[:, 2].max() - HEAD_SHARE * np.ptp(points[:, 2])


def walk(base, edges, body):
    """The base head walked onto the body head. Returns the base ids kept, their walked points and the head scale (body / base)."""
    keep = np.flatnonzero(head_of(base))
    renumber = np.full(len(base), -1)
    renumber[keep] = np.arange(len(keep))
    inner = renumber[edges]
    inner = inner[(inner >= 0).all(axis=1)]
    head = body[head_of(body)]
    ratio = np.ptp(head, axis=0) / np.ptp(base[keep], axis=0)
    walked = (base[keep] - base[keep].mean(axis=0)) * ratio + head.mean(axis=0)
    for rounds in SMOOTH_ROUNDS:
        hit, _ = nearest(head, walked)
        walked = walked + smooth(head[hit] - walked, inner, rounds)
    return keep, walked, float(np.mean(ratio))


def base_ids(body_open, base_open, edges):
    """For each body point, the hm08 id it sits on, or -1 (below the head, or nothing near). Seam copies share an id."""
    keep, walked, scale = walk(base_open, edges, body_open)
    inside = np.flatnonzero(head_of(body_open))
    hit, miss = nearest(walked, body_open[inside])
    if np.percentile(miss, 90) > MAX_MISS_M:
        raise ValueError(f"base mesh did not land on the head: 90% of the points within {np.percentile(miss, 90) * 1000:.1f} mm")
    print(f"BOCA ARKit: base no rosto, 90% a {np.percentile(miss, 90) * 1000:.1f} mm")
    ids = np.full(len(body_open), -1)
    ids[inside] = keep[hit]
    return ids, scale


def on_body(move, ids, scale):
    """A dense base move read by the body points (zero where the id is -1), scaled to the head."""
    padded = np.vstack([move, np.zeros((1, 3))])
    return padded[ids] * scale


def welded_edges(rest, edges):
    """Edges of the skin with each seam copy read as the first point at its place."""
    spots = {}
    twin = np.array([spots.setdefault(tuple(np.round(point, 5)), index) for index, point in enumerate(rest)])
    welded = np.unique(np.sort(twin[edges], axis=1), axis=0)
    return twin, welded[welded[:, 0] != welded[:, 1]]


def mean_of_neighbours(field, edges):
    total, degree = np.zeros_like(field), np.bincount(edges.ravel(), minlength=len(field))[:, None]
    np.add.at(total, edges[:, 0], field[edges[:, 1]])
    np.add.at(total, edges[:, 1], field[edges[:, 0]])
    return np.where(degree > 0, total / np.maximum(degree, 1), field)


def clean(move, rest, edges):
    """Strays take the mean of their neighbours, then the field is eased; seam copies end with one move."""
    twin, welded = welded_edges(rest, edges)
    field = move[twin].copy()
    for _ in range(STRAY_ROUNDS):
        around = mean_of_neighbours(field, welded)
        stray = np.linalg.norm(field - around, axis=1) > STRAY_M
        if not stray.any():
            break
        field[stray] = around[stray]
    for _ in range(EASE_ROUNDS):
        field = (field + mean_of_neighbours(field, welded)) / 2
    return field[twin]


def same_side(body_move, mapped):
    """Cosine between a body shape and the same pack shape on the body: checks the walk and the left/right naming."""
    a, b = body_move.ravel(), mapped.ravel()
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def arkit_moves(rest, body_jaw, body_edges, body_smile_left=None, pack=PACK, base=None, edges=None, body_round=None):
    """The move of every body point for each of `SHAPES`, and the check cosines (jaw, left smile)."""
    if base is None:
        base, edges = read_base()
    pack_jaw = dense(Path(pack) / "jawOpen.target")
    ids, scale = base_ids(rest + body_jaw, base + pack_jaw, edges)
    checks = {"jawOpen": same_side(body_jaw, on_body(pack_jaw, ids, scale))}
    if body_smile_left is not None:
        checks["mouthSmileLeft"] = same_side(body_smile_left, on_body(dense(Path(pack) / "mouthSmileLeft.target"), ids, scale))
    if min(checks.values()) < MIN_CHECK:
        raise ValueError(f"the pack shapes do not match the body's own: {checks}")
    moves = {name: SHARE.get(name, 1.0) * clean(on_body(dense(Path(pack) / f"{name}.target"), ids, scale), rest, body_edges) for name in SHAPES}
    if body_round is not None:
        for name, share in ROUND_FROM_U.items():
            moves[name] = moves[name] + share * body_round
    return moves, checks


def add_arkit_mouth(body):
    """Adds (or replaces) the `SHAPES` on the body; returns the check cosines."""
    keys = body.data.shape_keys.key_blocks if body.data.shape_keys else None
    if keys is None or "jawOpen" not in keys:
        return {}

    def points(block):
        flat = np.empty(len(block.data) * 3)
        block.data.foreach_get("co", flat)
        return flat.reshape(-1, 3)

    rest = points(keys[0])
    smile = points(keys["mouthSmileLeft"]) - rest if "mouthSmileLeft" in keys else None
    rounding = points(keys["viseme_U"]) - rest if "viseme_U" in keys else None
    edges = np.empty(len(body.data.edges) * 2, int)
    body.data.edges.foreach_get("vertices", edges)
    moves, checks = arkit_moves(rest, points(keys["jawOpen"]) - rest, edges.reshape(-1, 2), smile, body_round=rounding)
    for name, move in moves.items():
        if name in keys:
            body.shape_key_remove(keys[name])
        block = body.shape_key_add(name=name, from_mix=False)
        block.data.foreach_set("co", (rest + move).ravel())
    print("BOCA ARKit:", len(moves), "formas", " ".join(f"{name}={value:.2f}" for name, value in checks.items()))
    return checks
