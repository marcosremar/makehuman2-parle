#!/usr/bin/env python3
"""
    License information: data/licenses/makehuman_license.txt

    Paints the lips of a Parle cast skin, as a step of its own after `skin_style.py` (owner, 2026-11-07:
    the lip colour is set later, not only when the skin is made). The skin step leaves the skin without
    painted lips and a map of where the lip is; this step always starts from those two, so it can run
    again with another colour and nothing drifts.

    Natural lips are the person's own skin, 20% darker, rosier and a little less yellow in Lab, with half
    of the lip's own light and shade kept as its drawing: the game's wine lipstick and grey lips read as
    make-up or as cold. A colour given in the look (`lipColor`) replaces the hue, and the drawing stays.

        parle_lips.py <skin-base.png> <lips-map.png> <out.png> [#rrggbb] [--man]
"""
import sys

import cv2
import numpy as np

LIP_LIGHT = 0.8
LIP_ROSE = 11
LIP_WARM = -4
LIP_DRAWING = 0.5
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


def lip_points(data, points):
    """Which points of this person's mesh are lip."""
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
    slit = slit[np.argsort(slit[:, 0])]
    half = np.abs(slit[:, 0] - middle).max() * CORNER_REACH
    across = np.sqrt(np.clip(1 - ((points[:, 0] - middle) / half) ** 2, 0, 1))
    height = points[:, 2] - np.interp(points[:, 0], slit[:, 0], slit[:, 2])
    front = points[:, 1] < np.interp(points[:, 0], slit[:, 0], slit[:, 1]) + FRONT_M
    return front & (np.abs(points[:, 0] - middle) < half) & (height < LIP_UP_M * across) & (height > -LIP_LOW_M * across)


def lip_zone(data, points):
    """The lip on the UV map, from the cast mesh (`uvpos.npz`) and this person's own points."""
    lip = lip_points(data, points)
    triangles = data["tri"]
    corners = np.stack([data["uv"][:, :, 0] * SIZE, (1 - data["uv"][:, :, 1]) * SIZE], axis=2).round().astype(np.int32)
    canvas = np.zeros((SIZE, SIZE), np.float32)
    for corner, triangle in zip(corners, triangles):
        if lip[triangle].mean() > 0.5:
            cv2.fillConvexPoly(canvas, corner, 1.0)
    return cv2.GaussianBlur(canvas, (0, 0), 1.5)


def lab_of(bgr):
    return cv2.cvtColor(np.array(bgr, np.uint8)[None, None, :], cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]


def painted(skin, share, colour=None, man=False):
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
    lip = cv2.cvtColor(np.clip(lip, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
    return (skin * (1 - share[:, :, None]) + lip * share[:, :, None]).round().astype(np.uint8)


def main():
    words = [word for word in sys.argv[1:] if word != "--man"]
    base, mask, out = words[:3]
    colour = words[3] if len(words) > 3 else None
    share = cv2.imread(mask, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
    cv2.imwrite(out, painted(cv2.imread(base), share, colour, "--man" in sys.argv))
    print("LABIOS", out.split("/")[-1], colour or "natural", "área", round(float((share > INSIDE).mean()), 4))


if __name__ == "__main__":
    main()
