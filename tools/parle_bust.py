"""
    License information: data/licenses/makehuman_license.txt

    The bust under a top.

    The breast of the body comes to a point at the nipple, and the cloth fitted over it showed that point: seen from the side the bust
    of a T-shirt ended in a cone, and from the front in two tips with the folds of the cloth running to them (owner, 2026-11-11: "o bico
    do peito da mulher está muito pontudo na roupa", "com roupa deve parecer que está com sutiã"). A bra holds the breast in a cup: the
    tip is round and the sides are full. So where a top covers it, each breast takes the shape of a cup along its own axis: a point that
    stood a share of the way from the rim to the tip stands at `cup` of that share. The tip comes back by a fifth of how far it stood
    over the rim (5 to 8 mm on the cast), half way up the cloth goes out a little, and the rim stays: over the centimetre around the
    tip the cloth of a T-shirt stood 8 mm proud, and now 3. The skin under the cloth goes by the same rule, and stays under it.
    Unclothed, nothing changes.

    This file is numpy alone: `parle_polish.round_bust` reads the meshes.
"""
import numpy as np

# The window of the bust: heights as shares of the way from the hip joints to the shoulder joints, sides as shares of the way from
# the middle of the body to the shoulder joint.
LEVELS = (0.5, 0.95)
SIDES = (0.2, 0.95)
# The rim of the breast is a ring this far from its axis, as a share of the stature (5.7 cm on a woman of 1.62 m), and the plane of
# the rim is fitted to the points between these shares of it.
REACH = 0.035
RIM = (0.8, 1.15)
# The cup: the tip stands at CUP of where it stood; up to it the height follows a quarter of a sine wave, which is flat at the top,
# with STRAIGHT of the old height left in, so no two points on the way to the tip end at the same place. At 0.9 the tip of a T-shirt
# still read as a point from the side.
CUP = 0.8
STRAIGHT = 0.2
# A tip that stands less than this over its rim, in metres, is no breast to round: a flat chest, or a loose top.
TALL_LEAST_M = 0.01


def smooth(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def tip_of(points, hip, shoulder, middle, reach, side):
    """The tip of the breast of one side on the points of a top, or None when the top has no point there: the foremost of them (the
    face looks to -y) in the window of the bust. `hip` and `shoulder` are the heights of those joints, `middle` the x of the middle of
    the body, `reach` how far the shoulder joint is from it and `side` +1 for the side of +x, -1 for the other."""
    level = (points[:, 2] - hip) / (shoulder - hip)
    out = side * (points[:, 0] - middle) / reach
    window = (level > LEVELS[0]) & (level < LEVELS[1]) & (out > SIDES[0]) & (out < SIDES[1])
    if not window.any():
        return None
    inside = points[window]
    return inside[inside[:, 1].argmin()]


def rim_of(points, tip, reach):
    """The axis of a breast, pointing out of the body, and how far its tip stands over the plane of its rim, in metres; None when the
    points do not go all the way round the rim. `reach` is the distance from the axis to the rim, in metres."""
    axis = np.array([0.0, -1.0, 0.0])
    for _ in range(2):
        rel = points - tip
        along = rel @ axis
        across = np.linalg.norm(rel - np.outer(along, axis), axis=1)
        ring = points[(across > RIM[0] * reach) & (across < RIM[1] * reach) & (np.abs(along) < 2 * reach)]
        if len(ring) < 6:
            return None
        centre = ring.mean(axis=0)
        normal = np.linalg.svd(ring - centre)[2][2]
        axis = normal if normal @ axis > 0 else -normal
    return axis, float((tip - centre) @ axis)


def cup(share):
    """Where a point that stood `share` of the way from the rim (0) to the tip (1) stands in the cup."""
    share = np.clip(share, 0, 1)
    return CUP * ((1 - STRAIGHT) * np.sin(np.pi * share / 2) + STRAIGHT * share)


def cupped(points, tip, axis, tall, reach):
    """How far each point goes along the axis of the breast for it to stand as in a cup, in metres: back at the tip, out on the sides,
    and nowhere from the rim out. `tall` is how far the tip stands over the plane of the rim and `reach` how far the rim is from the axis."""
    rel = points - tip
    along = rel @ axis
    across = np.linalg.norm(rel - np.outer(along, axis), axis=1)
    share = np.clip(1 + along / tall, 0, 1)
    fade = smooth((reach - across) / (0.3 * reach))
    return (tall * (cup(share) - share) * fade)[:, None] * axis
