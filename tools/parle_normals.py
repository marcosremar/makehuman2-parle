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
    zone = (np.abs(points[:, 0] - tip[0]) < CHIN_HALF_M) & (below > BELOW_TIP_M[0]) & (below < BELOW_TIP_M[1]) & (points[:, 1] < tip[1] + FRONT_OF_NECK_M + 0.09)
    return smoothstep(DOWN_FROM, DOWN_FULL, normals[:, 2]) * zone


def softened(points, normals):
    """The normals with the under side of the chin turned forward."""
    share = (under_chin_share(points, normals) * TURN)[:, None]
    out = normals * (1 - share) + AHEAD[None] * share
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
