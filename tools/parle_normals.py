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
SMOOTH_ROUNDS = 8
DOWN_FROM = -0.15
DOWN_FULL = -0.55
TURN = 0.85
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


def softened(points, normals):
    """The normals with the under side of the chin turned forward."""
    share = (under_chin_share(points, normals) * TURN)[:, None]
    out = normals * (1 - share) + AHEAD[None] * share
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
