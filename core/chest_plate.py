"""The pectorals of a bare-chested muscled man as two plates, in the place of the two round mounds of the base mesh.

The base mesh's chest is a round mound under each nipple. With the sculpture's relief on top, on the muscled man of the gym it stood out
4.3 cm and hung over a hard straight ledge, and with the relief smoothed it was two soft mounds with a cone at each nipple: from the side
both read as breasts (owner, 2026-11-15: "o peitoral do homem negro está errado"). A pectoral is a plate from the breastbone to the armpit,
full at the bottom and thinner up to the collarbone, whose lower edge rises from the breastbone toward the armpit.

So, on the front of the chest: at each height the trunk's own curve across, without the mounds (`trunk_curve`), and a plate on it
(`plate_depth`). Numpy and scipy alone; `tools/chest_shape.py` runs it on the rigged body.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d

from core.body_shape import rigged_landmarks, smoothstep, welded

# The plate, in shares of crotch to neck pit (heights) and of the half width between the shoulder joints (sides). Its lower edge sits at
# EDGE_LOW at the breastbone and rises by EDGE_RISE to the armpit, and falls away over EDGE_FALL under it. It is THICKEST_M deep at its
# lower edge and loses TOP_LOSS of that up to the collarbone: kept whole up there, the upper chest sank under the collarbone as a groove.
# Judged by eye on the muscled man of the gym, from the front, at three quarters and from the side, against 1.5, 2.5 and 4.5 cm deep
# plates and lower edges at 0.60 and 0.64 (2026-11-15).
THICKEST_M = 0.035
EDGE_LOW = 0.62
EDGE_RISE = 0.12
EDGE_FALL = 0.05
TOP_LOSS = 0.25
# Where the new chest takes over from the body's own: whole between these, fading over the second number. Further up or further out it
# met the collarbone and the front of the armpit and carved a groove down from the collarbone and a notch at the armpit (2026-11-15).
TOP = (0.92, 0.12)
SIDE = (0.78, 0.35)
# The trunk's curve at each height: depth = c0 + c2·x² + c4·x⁴ across, fitted from under (what stands out of it is cut to it and the fit
# made again, ROUNDS times), in bands BAND tall, the bands smoothed along the height over BLUR of them.
BAND = 0.02
ROUNDS = 8
BLUR = 2


def trunk_curve(points, centre, level, chosen):
    """The depth (to the front, -y) of the trunk at each point without the mounds on it: per band of `level`, the curve across fitted
    from under to the `chosen` points."""
    depth = -points[:, 1]
    across = points[:, 0] - centre
    rows = np.arange(EDGE_LOW - 0.2, 1.1 + BAND, BAND)
    kept, fits = [], []
    for row in rows:
        pick = chosen & (np.abs(level - row) < BAND)
        if pick.sum() < 6:
            continue
        x, y = across[pick], depth[pick].copy()
        terms = np.stack([np.ones_like(x), x ** 2, x ** 4], axis=1)
        for _ in range(ROUNDS):
            fit = np.linalg.lstsq(terms, y, rcond=None)[0]
            y = np.minimum(y, terms @ fit + 0.002)
        kept.append(row)
        fits.append(fit)
    fits = gaussian_filter1d(np.array(fits), BLUR, axis=0, mode="nearest")
    at = np.stack([np.interp(level, kept, fits[:, term]) for term in range(3)], axis=1)
    return at[:, 0] + at[:, 1] * across ** 2 + at[:, 2] * across ** 4


def plate_depth(level, side):
    """How far the plate stands out on the trunk's curve at each point."""
    low = EDGE_LOW + EDGE_RISE * np.clip(side, 0, 1)
    up = np.clip((level - low) / np.maximum(1.0 - low, 1e-6), 0, 1)
    across = smoothstep((side - 0.03) / 0.12) * smoothstep((0.92 - side) / 0.40)
    return THICKEST_M * across * smoothstep((level - low + EDGE_FALL) / EDGE_FALL) * (1.0 - TOP_LOSS * up)


def plates(points, faces, trunk, arms, hips):
    """The shift (one per vertex, along y) that gives the front of the chest its two plates. `trunk` is the share of each vertex on the
    spine and hip bones, `arms` the two shoulder joints and `hips` the two hip joints of the body."""
    merged, twin, _, _ = welded(points, faces)
    shoulder, hip = (arms[0] + arms[1]) / 2, (hips[0] + hips[1]) / 2
    marks = rigged_landmarks(merged, float(shoulder[2]), float(hip[2]))
    centre, crotch, pit = marks[0], marks[2], marks[3]
    half = abs(arms[0][0] - centre)
    level = (merged[:, 2] - crotch) / (pit - crotch)
    side = np.abs(merged[:, 0] - centre) / half
    bone = np.zeros(len(merged))
    bone[twin] = trunk
    # The middle of the trunk front to back, centimetre by centimetre: the front is what stands ahead of it.
    inner = side < 1.0
    rows = np.round(merged[:, 2] * 100).astype(int)
    middle = {row: (merged[inner & (rows == row), 1].min() + merged[inner & (rows == row), 1].max()) / 2 for row in np.unique(rows[inner])}
    ahead = np.array([middle.get(row, y) - y for y, row in zip(merged[:, 1], rows)])
    curve = trunk_curve(merged, centre, level, (ahead > 0) & (bone > 0.6) & inner & (level > EDGE_LOW - 0.2) & (level < 1.1))
    weight = (smoothstep((level - (EDGE_LOW - 0.12)) / 0.06) * smoothstep((TOP[0] - level) / TOP[1]) * smoothstep((SIDE[0] - side) / SIDE[1])
              * smoothstep((ahead - 0.02) / 0.05) * smoothstep((bone - 0.3) / 0.5))
    depth = -merged[:, 1]
    new = depth + weight * (curve + plate_depth(level, side) - depth)
    shift = np.zeros_like(merged)
    shift[:, 1] = depth - new
    print("PLACA do peito: até", round(float(plate_depth(level, side).max()) * 1000, 1), "mm, monte tirado até",
          round(float(np.max(weight * np.maximum(depth - curve, 0))) * 1000, 1), "mm")
    return shift[twin]
