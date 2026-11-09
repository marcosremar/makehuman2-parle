#!/usr/bin/env python3
"""
    License information: data/licenses/makehuman_license.txt

    Paints the lips of a Parle cast skin, as a step of its own after `skin_style.py` (owner, 2026-11-07:
    the lip colour is set later, not only when the skin is made). The skin step leaves the skin without
    painted lips and a map of where the lip is; this step always starts from those two, so it can run
    again with another colour and nothing drifts.

    Natural lips are the person's own skin, 20% darker, rosier and a little less yellow in Lab, with a fifth
    of the lip's own light and shade kept as its drawing: the game's wine lipstick and grey lips read as
    make-up or as cold. A colour given in the look (`lipColor`) replaces the hue, and the drawing stays.

        parle_lips.py <skin-base.png> <lips-map.png> <out.png> [#rrggbb] [--man]
"""
import sys

import cv2
import numpy as np

from parle_beard import position_map

LIP_LIGHT = 0.8
# 11 and a half of the drawing left the crease and the corners dark and the colour weak next to the clean face (owner, 2026-11-08).
LIP_ROSE = 19.0
LIP_WARM = -7
LIP_DRAWING = 0.2
INSIDE = 0.5
# The same rose reads as lipstick on dark skin and on men (Seu Nelson): it fades with the skin's lightness, and a man's lip is less rosy.
ROSE_FULL_LIGHT = 170
MAN_ROSE = 0.55


# Where the lip is, on the mesh (owner, 2026-11-08: the women's lower lip looked too big). The old map was a soft patch centred 46 mm
# below the nose tip, 19 mm under the mouth, so the lower lip's colour ran down to the chin. The lip is now measured from the mouth's
# slit (the open edge of the mesh between the lips): the upper lip reaches 7 mm above it, the lower 8.5 mm below, both narrowing to
# nothing at the corners, and only on the face's front, not inside the mouth.
SLIT_BELOW_TIP_M = (0.012, 0.05)
SLIT_HALF_M = 0.035
LIP_UP_M = 0.007
LIP_LOW_M = 0.0085
CORNER_REACH = 1.08
FRONT_M = 0.006
SIZE = 1024
# The border of the lip is an edge, 1 mm wide. The upper lip's top has the two peaks of the bow, either side of a dip under the philtrum.
LIP_EDGE_M = 0.001
BOW_AT_M = 0.0065
BOW_WIDE_M = 0.006
BOW_PEAK = 0.14
BOW_DIP = 0.2
BOW_DIP_WIDE_M = 0.0035
# Where the lips meet there is a thin dark line in every photograph; it stops short of the corners.
# The very middle rolls into the mouth and is not seen: the line sits a little on the lower lip.
LINE_M = 0.0013
LINE_BELOW_M = 0.0007
SLIT_BIN_M = 0.003
SLIT_DEGREE = 4
LINE_ENDS = 4.0
LINE_DARK = 0.5
# Light comes from above: the upper lip faces down and is darker than the lower.
UPPER_SHADE = 10.0


def slit_points(data, points):
    """The open edge of the mesh between the lips, from one corner of the mouth to the other."""
    top = points[:, 2].max()
    head = points[points[:, 2] > top - 0.26]
    middle = float(np.median(head[:, 0]))
    tip = head[np.abs(head[:, 0] - middle) < 0.01]
    tip = tip[tip[:, 1].argmin()]
    triangles = data["tri"]
    edges = np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1)
    unique, count = np.unique(edges, axis=0, return_counts=True)
    edge = np.unique(unique[count == 1])
    below = tip[2] - points[edge, 2]
    slit = points[edge[(np.abs(points[edge, 0] - middle) < SLIT_HALF_M) & (below > SLIT_BELOW_TIP_M[0]) & (below < SLIT_BELOW_TIP_M[1]) & (points[edge, 1] < tip[1] + 0.04)]]
    return slit[np.argsort(slit[:, 0])]


def lip_points(data, points):
    """Which points of this person's mesh are lip."""
    top = points[:, 2].max()
    head = points[points[:, 2] > top - 0.26]
    middle = float(np.median(head[:, 0]))
    tip = head[np.abs(head[:, 0] - middle) < 0.01]
    tip = tip[tip[:, 1].argmin()]
    slit = slit_points(data, points)
    half = np.abs(slit[:, 0] - middle).max() * CORNER_REACH
    across = np.sqrt(np.clip(1 - ((points[:, 0] - middle) / half) ** 2, 0, 1))
    height = points[:, 2] - np.interp(points[:, 0], slit[:, 0], slit[:, 2])
    front = points[:, 1] < np.interp(points[:, 0], slit[:, 0], slit[:, 1]) + FRONT_M
    return front & (np.abs(points[:, 0] - middle) < half) & (height < LIP_UP_M * across) & (height > -LIP_LOW_M * across)


def lip_maps(data, points):
    """Texel by texel, from this person's mesh: how much is lip, the line where the lips meet, and which part is the upper lip.

    Next to photographs of closed lips the old map, whole triangles blurred, had no edge and no bow, and nothing marked where the lips meet (owner, 2026-11-09).
    """
    place = position_map({"uv": data["uv"], "tri": data["tri"], "pos": points}, SIZE)
    known = ~np.isnan(place[:, :, 2])
    place = np.nan_to_num(place)
    top = points[:, 2].max()
    head = points[points[:, 2] > top - 0.26]
    middle = float(np.median(head[:, 0]))
    slit = slit_points(data, points)
    half = np.abs(slit[:, 0] - middle).max() * CORNER_REACH
    x = (place[:, :, 0] - middle).astype(np.float32)
    across = np.sqrt(np.clip(1 - (x / half) ** 2, 0, 1))
    # The slit's edge zigzags between the two lips: read point by point, the line came out as dark blots along the mouth.
    bins = np.arange(slit[:, 0].min(), slit[:, 0].max() + SLIT_BIN_M, SLIT_BIN_M)
    centres = np.array([bin_start + SLIT_BIN_M / 2 for bin_start in bins if ((slit[:, 0] >= bin_start) & (slit[:, 0] < bin_start + SLIT_BIN_M)).any()])
    medians = np.array([np.median(slit[(slit[:, 0] >= bin_start) & (slit[:, 0] < bin_start + SLIT_BIN_M)], axis=0) for bin_start in bins if ((slit[:, 0] >= bin_start) & (slit[:, 0] < bin_start + SLIT_BIN_M)).any()])
    curve = np.polyfit(centres - middle, medians[:, 2], SLIT_DEGREE)
    height = (place[:, :, 2] - np.polyval(curve, np.clip(place[:, :, 0] - middle, centres[0] - middle, centres[-1] - middle))).astype(np.float32)
    front = place[:, :, 1] < np.interp(place[:, :, 0], centres, medians[:, 1]) + FRONT_M
    bow = 1 + BOW_PEAK * np.exp(-(((np.abs(x) - BOW_AT_M) / BOW_WIDE_M) ** 2)) - BOW_DIP * np.exp(-((x / BOW_DIP_WIDE_M) ** 2))
    inside = known & front & (np.abs(x) < half)
    share = np.clip(np.minimum(LIP_UP_M * across * bow - height, height + LIP_LOW_M * across) / LIP_EDGE_M, 0, 1) * inside
    line = np.exp(-(((height + LINE_BELOW_M) / LINE_M) ** 2)) * np.clip(across * LINE_ENDS, 0, 1) * inside
    upper = share * (height > 0)
    return cv2.GaussianBlur(share.astype(np.float32), (0, 0), 0.7), cv2.GaussianBlur(line.astype(np.float32), (0, 0), 0.5), cv2.GaussianBlur(upper.astype(np.float32), (0, 0), 1.0)


def lip_zone(data, points):
    """The lip on the UV map, from the cast mesh (`uvpos.npz`) and this person's own points."""
    return lip_maps(data, points)[0]


def lab_of(bgr):
    return cv2.cvtColor(np.array(bgr, np.uint8)[None, None, :], cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]


def painted(skin, share, colour=None, man=False, line=None, upper=None):
    lab = cv2.cvtColor(skin, cv2.COLOR_BGR2LAB).astype(np.float32)
    own = lab_of(np.median(skin.reshape(-1, 3), axis=0))
    inside = share > INSIDE
    light = float(np.median(lab[inside, 0])) if inside.any() else own[0]
    if colour:
        wanted = lab_of([int(colour[index:index + 2], 16) for index in (5, 3, 1)])
        base, hue = wanted[0], wanted[1:]
    else:
        rose = LIP_ROSE * min(1.0, (own[0] / ROSE_FULL_LIGHT) ** 2) * (MAN_ROSE if man else 1.0)
        base, hue = own[0] * LIP_LIGHT, own[1:] + np.array([rose, LIP_WARM])
    lip = lab.copy()
    lip[:, :, 0] = base + (lab[:, :, 0] - light) * LIP_DRAWING
    lip[:, :, 1:] = hue
    if upper is not None:
        lip[:, :, 0] -= UPPER_SHADE * upper
    lip = cv2.cvtColor(np.clip(lip, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
    mixed = skin * (1 - share[:, :, None]) + lip * share[:, :, None]
    if line is not None:
        mixed *= 1 - LINE_DARK * line[:, :, None]
    return mixed.round().astype(np.uint8)


def main():
    words = [word for word in sys.argv[1:] if word != "--man"]
    base, mask, out = words[:3]
    colour = words[3] if len(words) > 3 else None
    share = cv2.imread(mask, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
    drawn = cv2.imread(mask.replace(".lips.png", ".line.png"))
    line, upper = (drawn[:, :, 0].astype(np.float32) / 255, drawn[:, :, 1].astype(np.float32) / 255) if drawn is not None else (None, None)
    cv2.imwrite(out, painted(cv2.imread(base), share, colour, "--man" in sys.argv, line, upper))
    print("LABIOS", out.split("/")[-1], colour or "natural", "área", round(float((share > INSIDE).mean()), 4))


if __name__ == "__main__":
    main()
