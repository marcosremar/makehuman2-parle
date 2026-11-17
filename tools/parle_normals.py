"""
    License information: data/licenses/makehuman_license.txt

    The under side of the chin of a Parle cast member, soft. A surface that faces down gets no direct light and only the cool ambient of a room, so the skin
    there came out grey-lavender with a hard edge along the jaw (Ivone, Seu Nelson; scoreboard of 2026-11-10). Painted magenta on the skin map, the band stayed on those
    texels and only the shade changed, so the colour map is not the cause. The normals of that surface turn forward, where the face's direct light and warm ambient reach it.

    Calibrated by eye in the player, no source: a design choice to pilot.
"""
import numpy as np

CHIN_HALF_M = 0.08
BELOW_TIP_M = (0.055, 0.14)
FRONT_OF_NECK_M = 0.02
# The zone fades over 2 cm instead of stopping at a box wall, and each point's turn is averaged with its neighbours along the mesh edges: a box wall and a
# per-point weight let a neck crease alternate turned and plain points, and the player showed dark teeth at the corner of the jaw (2026-11-11).
FADE_M = 0.02
# Round 83 of the drawn heads (owner, 2026-11-17: "o maxilar do homem e da mulher ainda está com falhas"): drawn with the file's
# normals against the surface's own, the wavy line along the base of the jaw was this turn, not the form: at 0.85, from -0.15
# to -0.55 and 8 rounds, the normals under the jaw stood up to 90 degrees off the surface (the 95th percentile 59), a lit wedge
# under the chin whose edge followed each point's own normal and ended at a line across the throat. The turn is weaker, comes
# in over a wider span of the normal and is averaged over more neighbours: the under side keeps a soft shade of its own, as
# the series draws it, and the drawn heads paint that shade warm (`parle_paint.JAW_SHADE`). Not proved in the player.
SMOOTH_ROUNDS = 30
DOWN_FROM = 0.0
DOWN_FULL = -0.8
TURN = 0.6
AHEAD = np.array((0.0, -1.0, 0.05))


def smoothstep(low, high, value):
    t = np.clip((value - low) / (high - low), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def nose_tip(points):
    """The tip of the nose of a body that looks to -y, z up, its head the top 26 cm."""
    head = points[points[:, 2] > points[:, 2].max() - 0.26]
    middle = float(np.median(head[:, 0]))
    close = head[np.abs(head[:, 0] - middle) < 0.01]
    return close[close[:, 1].argmin()]


def under_chin_share(points, normals):
    """How far each point turns forward: facing down, under the jaw, in front of the neck."""
    tip = nose_tip(points)
    below = tip[2] - points[:, 2]
    side = 1 - smoothstep(CHIN_HALF_M - FADE_M, CHIN_HALF_M + FADE_M / 2, np.abs(points[:, 0] - tip[0]))
    height = smoothstep(BELOW_TIP_M[0] - FADE_M / 2, BELOW_TIP_M[0] + FADE_M / 2, below) * (1 - smoothstep(BELOW_TIP_M[1] - FADE_M, BELOW_TIP_M[1], below))
    depth = 1 - smoothstep(tip[1] + FRONT_OF_NECK_M + 0.09 - FADE_M, tip[1] + FRONT_OF_NECK_M + 0.09, points[:, 1])
    return smoothstep(DOWN_FROM, DOWN_FULL, normals[:, 2]) * side * height * depth


def smoothed_share(share, edges, rounds=SMOOTH_ROUNDS):
    """The turn averaged with the neighbours along `edges` (pairs of point indices), so a crease turns as one piece."""
    share = np.asarray(share, float).copy()
    edges = np.asarray(edges, int).reshape(-1, 2)
    count = np.bincount(edges.ravel(), minlength=len(share)).astype(float)
    for _ in range(rounds):
        total = np.zeros_like(share)
        np.add.at(total, edges[:, 0], share[edges[:, 1]])
        np.add.at(total, edges[:, 1], share[edges[:, 0]])
        share = np.where(count > 0, (share + total) / (1 + count), share)
    return share


# The fold between the under side of the jaw and the neck (owner, 2026-11-17: "o maxilar do homem e da mulher ainda está com falhas";
# round 83 left a small tooth in the line of every woman's jaw seen three quarters on). The form there is smooth, but the edges
# of the mesh cross the fold in a zigzag and each point's normal is its own faces': the edge of the shade under the jaw came
# out as a saw, in the kit's own head as much as in the drawn one. The normals there are averaged with their neighbours'
# (`eased`), FOLD_ROUNDS times, points at the same place (the seam of the neck) as one point.
FOLD_ROUNDS = 30


def eased(normals, edges, weight, same=None, rounds=FOLD_ROUNDS):
    """The normals (n x 3) averaged with their neighbours' along `edges` (m x 2), each point going `weight` (n) of half the way
    each round; `same` (n) gives the points that stand at one place the same number, and they keep one normal."""
    normals = np.asarray(normals, float)
    same = np.arange(len(normals)) if same is None else np.asarray(same, int)
    count = int(same.max()) + 1 if len(same) else 0
    members = np.maximum(np.bincount(same, minlength=count), 1)[:, None]
    one = np.zeros((count, 3))
    np.add.at(one, same, normals)
    one /= np.maximum(np.linalg.norm(one, axis=1, keepdims=True), 1e-9)
    share = np.zeros(count)
    np.add.at(share, same, np.asarray(weight, float))
    share = (share / members[:, 0])[:, None]
    links = np.unique(np.sort(same[np.asarray(edges, int).reshape(-1, 2)], axis=1), axis=0)
    links = links[links[:, 0] != links[:, 1]]
    near = np.maximum(np.bincount(links.ravel(), minlength=count), 1)[:, None]
    for _ in range(rounds):
        total = np.zeros_like(one)
        np.add.at(total, links[:, 0], one[links[:, 1]])
        np.add.at(total, links[:, 1], one[links[:, 0]])
        mean = total / near
        mean /= np.maximum(np.linalg.norm(mean, axis=1, keepdims=True), 1e-9)
        one = one + 0.5 * share * (mean - one)
        one /= np.maximum(np.linalg.norm(one, axis=1, keepdims=True), 1e-9)
    return one[same]


def softened(points, normals):
    """The normals with the under side of the chin turned forward."""
    share = (under_chin_share(points, normals) * TURN)[:, None]
    out = normals * (1 - share) + AHEAD[None] * share
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
