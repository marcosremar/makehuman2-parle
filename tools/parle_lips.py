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

import numpy as np

try:
    import cv2
    from parle_beard import position_map
except ImportError:
    # Blender has no OpenCV. The mesh side (`parle_lip_cut`) reads from here where the lip is; only the painting needs the pictures.
    cv2 = position_map = None

LIP_LIGHT = 0.8
# A woman's lips stand out more from her skin: the contrast in lightness between the lips and the skin round them is greater on
# women's faces, and a woman's face with more of it is found more feminine and more attractive, a man's with less (Russell 2003,
# Perception 32(9):1093-1107, DOI 10.1068/p5101; Russell 2009, Perception 38(8):1211-1219, DOI 10.1068/p6331; for the lips alone,
# Stephen & McKeegan 2010, Perception 39(8):1104-1110, DOI 10.1068/p6730: directions, read in the abstracts). Measured on the portrait
# (`parle_beauty.lip_contrast`, 2026-11-11) the cast's women had 0.06 where Sofia has 0.14 and the women of the game 0.10 to 0.18.
# Not as far as theirs: the game's wine lips read as make-up (above), and the lip stays lighter than 0.7 of the skin.
WOMAN_LIP_LIGHT = 0.71
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
# Towards the corners the lips narrow faster than an ellipse does: as one, their colour reached the corner a third as tall as at the
# middle and drew a band with blunt ends, which on a map with a millimetre and a half to the texel blurred into a smear past the
# corners of the mouth (2026-11-12, on the portrait). The height of each lip goes with this power of the ellipse.
LIP_TAPER = 1.5
# On a woman the colour of the lower lip goes further down. The jaw rests closed as far as the lips need to meet
# (`parle_face.CALM_LOOK`), which carries the lower lip some 3.5 mm up behind the upper one: of its 8.5 mm of colour 5 showed, a thin
# band over the pale roll of the lip, whose most forward point is 5 mm under the meeting and whose hollow 13 mm (2026-11-12, on the
# profile of the middle of the face). The heights above were set on 2026-11-08 with the jaw closed twice as far.
WOMAN_LIP_LOW_M = 0.011
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
LINE_M = 0.0024
LINE_BELOW_M = 0.0013
SLIT_BIN_M = 0.003
SLIT_DEGREE = 4
LINE_ENDS = 4.0
# Darkens less, and blue and green more than red (BGR): with the mouth open the line's texels stretch over the inner lip, and at 0.68 neutral
# they drew black wedges at the corners and a smeared-lipstick ring (owner, scoreboard of 2026-11-11). Calibrated by eye, no source: to pilot.
LINE_DARK = np.array((0.4, 0.4, 0.22), np.float32)
# Light comes from above: the upper lip faces down and is darker than the lower.
UPPER_SHADE = 10.0
# A woman's lips have their colour on a patch of the map of their own, and their border is an edge of the mesh (`parle_lip_cut`).
# Painted on the skin, the lip had some 12 by 45 texels of the map, a millimetre and a half to the texel: "an edgeless puffed pad
# under smeared colour", "red paint bleeds past ragged corners along a wavy lip line, like smeared lipstick" (the judge by sight,
# rounds 15 to 18, always the first fault of the face at rest). The patch, in texels of the map from its top left corner (x, y,
# wide, tall): a place no island of the cast's map comes within 8 texels of (measured on `uvpos.npz`, 2026-11-12). The places of
# the lip stay this many texels inside it, and the colour round it goes on as far again: smaller copies of the map (mips) and its
# blocks of compression mix what stands next to a patch into it.
LIP_PATCH_PX = (16, 16, 96, 40)
LIP_PATCH_EDGE_PX = 6
# On the patch the lips are drawn in their own measure: across from one corner of the mouth to the other, and up from the lower
# border (0) through the line where they meet (0.5) to the upper border (1). The colour is whole as far from the middle as the
# first of these shares of the half width and is the skin's own at the second: "colour edge hard in the centre, blurred at the
# corners", the corners "small compressed transitions, not drilled holes" (a painting guide and a sculpting guide read on
# 2026-11-12; no number was found, these two are chosen by eye). The lower lip catches the light in its middle by this much of
# lightness (chosen by eye), and the line where the lips meet is this share of a lip wide.
LIP_PATCH_CORNER = (0.8, 1.0)
LIP_PATCH_GLOW = 5.0
LIP_PATCH_LINE = 0.09
# On the patch the colour is whole up to the edge: at WOMAN_LIP_LIGHT, the lightness the painted lip was mixed to with the skin
# round it, it read as lipstick (2026-11-12, the first lips with an edge). Lips are darker than the skin round them by some 13 of
# lightness on a light skin (L* 43 against 56 measured in a colour study read on 2026-11-12), a fifth: this share of the skin's.
LIP_PATCH_LIGHT = 0.8


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


def lip_points(data, points, low=LIP_LOW_M):
    """Which points of this person's mesh are lip. `low` is how far under the slit the lower lip goes."""
    top = points[:, 2].max()
    head = points[points[:, 2] > top - 0.26]
    middle = float(np.median(head[:, 0]))
    tip = head[np.abs(head[:, 0] - middle) < 0.01]
    tip = tip[tip[:, 1].argmin()]
    slit = slit_points(data, points)
    half = np.abs(slit[:, 0] - middle).max() * CORNER_REACH
    across = np.sqrt(np.clip(1 - ((points[:, 0] - middle) / half) ** 2, 0, 1)) ** LIP_TAPER
    height = points[:, 2] - np.interp(points[:, 0], slit[:, 0], slit[:, 2])
    front = points[:, 1] < np.interp(points[:, 0], slit[:, 0], slit[:, 1]) + FRONT_M
    return front & (np.abs(points[:, 0] - middle) < half) & (height < LIP_UP_M * across) & (height > -low * across)


def mouth_line(data, points):
    """Where the mouth of a mesh is: the middle of the face, half the width of the lips, and the line where the lips meet, as the
    places it is read at across the face (`centres`), its point at each (`medians`) and its height as a curve of the way from the
    middle (`curve`)."""
    top = points[:, 2].max()
    head = points[points[:, 2] > top - 0.26]
    middle = float(np.median(head[:, 0]))
    slit = slit_points(data, points)
    half = np.abs(slit[:, 0] - middle).max() * CORNER_REACH
    # The slit's edge zigzags between the two lips: read point by point, the line came out as dark blots along the mouth.
    bins = np.arange(slit[:, 0].min(), slit[:, 0].max() + SLIT_BIN_M, SLIT_BIN_M)
    centres = np.array([bin_start + SLIT_BIN_M / 2 for bin_start in bins if ((slit[:, 0] >= bin_start) & (slit[:, 0] < bin_start + SLIT_BIN_M)).any()])
    medians = np.array([np.median(slit[(slit[:, 0] >= bin_start) & (slit[:, 0] < bin_start + SLIT_BIN_M)], axis=0) for bin_start in bins if ((slit[:, 0] >= bin_start) & (slit[:, 0] < bin_start + SLIT_BIN_M)).any()])
    curve = np.polyfit(centres - middle, medians[:, 2], SLIT_DEGREE)
    return {"middle": middle, "half": half, "centres": centres, "medians": medians, "curve": curve}


def lip_reach(x, half, low=LIP_LOW_M):
    """How far the upper lip goes above the line where the lips meet and the lower one under it, at `x` from the middle of the
    face: both narrow to nothing at the corners, and the upper one has the two peaks of the bow."""
    high = np.sqrt(np.clip(1 - (x / half) ** 2, 0, 1)) ** LIP_TAPER
    bow = 1 + BOW_PEAK * np.exp(-(((np.abs(x) - BOW_AT_M) / BOW_WIDE_M) ** 2)) - BOW_DIP * np.exp(-((x / BOW_DIP_WIDE_M) ** 2))
    return LIP_UP_M * high * bow, low * high


def lip_place(line, points, low=LIP_LOW_M):
    """Point by point of a mesh, with the line of its mouth (`mouth_line`): how far inside the lip the point is, in metres (nothing
    or less outside it: over the upper border, under the lower one, past a corner, or inside the mouth), and its place on the lips:
    across, from one corner of the mouth (0) to the other (1), and up, from the lower border (0) through the line where the lips
    meet (0.5) to the upper border (1)."""
    x = points[:, 0] - line["middle"]
    centres, medians = line["centres"], line["medians"]
    height = points[:, 2] - np.polyval(line["curve"], np.clip(x, centres[0] - line["middle"], centres[-1] - line["middle"]))
    in_front = np.interp(points[:, 0], centres, medians[:, 1]) + FRONT_M - points[:, 1]
    over, under = lip_reach(x, line["half"], low)
    inside = np.minimum(np.minimum(over - height, height + under), in_front)
    up = np.where(height >= 0, 0.5 + 0.5 * np.clip(height / np.maximum(over, 1e-9), 0, 1), 0.5 - 0.5 * np.clip(-height / np.maximum(under, 1e-9), 0, 1))
    return inside, np.clip(0.5 + x / (2 * line["half"]), 0, 1), up


def patch_uv(across, up):
    """Where a place of the lips (`lip_place`) is on the map: u and v from 0 to 1, v upwards, inside the patch of the lips."""
    left, top, wide, tall = LIP_PATCH_PX
    u = (left + LIP_PATCH_EDGE_PX + across * (wide - 2 * LIP_PATCH_EDGE_PX)) / SIZE
    v = 1 - (top + tall - LIP_PATCH_EDGE_PX - up * (tall - 2 * LIP_PATCH_EDGE_PX)) / SIZE
    return np.stack([u, v], axis=-1)


def lip_maps(data, points, low=LIP_LOW_M):
    """Texel by texel, from this person's mesh: how much is lip, the line where the lips meet, and which part is the upper lip.
    `low` is how far under the slit the lower lip goes.

    Next to photographs of closed lips the old map, whole triangles blurred, had no edge and no bow, and nothing marked where the lips meet (owner, 2026-11-09).
    """
    place = position_map({"uv": data["uv"], "tri": data["tri"], "pos": points}, SIZE)
    known = ~np.isnan(place[:, :, 2])
    place = np.nan_to_num(place)
    line = mouth_line(data, points)
    middle, half, centres, medians, curve = (line[name] for name in ("middle", "half", "centres", "medians", "curve"))
    x = (place[:, :, 0] - middle).astype(np.float32)
    across = np.sqrt(np.clip(1 - (x / half) ** 2, 0, 1))
    height = (place[:, :, 2] - np.polyval(curve, np.clip(place[:, :, 0] - middle, centres[0] - middle, centres[-1] - middle))).astype(np.float32)
    front = place[:, :, 1] < np.interp(place[:, :, 0], centres, medians[:, 1]) + FRONT_M
    inside = known & front & (np.abs(x) < half)
    over, under = lip_reach(x, half, low)
    share = np.clip(np.minimum(over - height, height + under) / LIP_EDGE_M, 0, 1) * inside
    line = np.exp(-(((height + LINE_BELOW_M) / LINE_M) ** 2)) * np.clip(across * LINE_ENDS, 0, 1) * inside
    upper = share * (height > 0)
    return cv2.GaussianBlur(share.astype(np.float32), (0, 0), 0.7), cv2.GaussianBlur(line.astype(np.float32), (0, 0), 0.5), cv2.GaussianBlur(upper.astype(np.float32), (0, 0), 1.0)


def lip_zone(data, points, low=LIP_LOW_M):
    """The lip on the UV map, from the cast mesh (`uvpos.npz`) and this person's own points."""
    return lip_maps(data, points, low)[0]


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
        base, hue = own[0] * (LIP_LIGHT if man else WOMAN_LIP_LIGHT), own[1:] + np.array([rose, LIP_WARM])
    lip = lab.copy()
    lip[:, :, 0] = base + (lab[:, :, 0] - light) * LIP_DRAWING
    lip[:, :, 1:] = hue
    if upper is not None:
        lip[:, :, 0] -= UPPER_SHADE * upper
    lip = cv2.cvtColor(np.clip(lip, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
    mixed = skin * (1 - share[:, :, None]) + lip * share[:, :, None]
    if line is not None:
        mixed *= 1 - LINE_DARK[None, None, :] * line[:, :, None]
    return mixed.round().astype(np.uint8)


def lip_colour(skin, share, colour=None, man=False):
    """The colour of the lips of this skin in Lab, as `painted` mixes it, and the colour of the skin round the mouth: lightness, then the two of hue."""
    lab = cv2.cvtColor(skin, cv2.COLOR_BGR2LAB).astype(np.float32)
    own = lab_of(np.median(skin.reshape(-1, 3), axis=0))
    around = (cv2.dilate((share > 0.1).astype(np.uint8), np.ones((25, 25), np.uint8)) > 0) & (share < 0.02)
    near = np.median(lab[around], axis=0) if around.any() else own
    if colour:
        wanted = lab_of([int(colour[index:index + 2], 16) for index in (5, 3, 1)])
        return wanted, near
    rose = LIP_ROSE * min(1.0, (own[0] / ROSE_FULL_LIGHT) ** 2) * (MAN_ROSE if man else 1.0)
    return np.array([own[0] * (LIP_LIGHT if man else WOMAN_LIP_LIGHT), own[1] + rose, own[2] + LIP_WARM], np.float32), near


def with_lip_patch(skin, share, colour=None):
    """The skin with no lips painted on the face and the lips of a woman drawn on their patch (LIP_PATCH_PX), which the lips of
    the mesh are mapped to (`parle_lip_cut`): the upper lip darker, the lower one lit in its middle, a dark line where they meet,
    and the colour going over to the skin's at the corners. Round the patch the same colours go on, for the smaller copies of the
    map."""
    lip, near = lip_colour(skin, share, colour)
    if not colour:
        lip[0] *= LIP_PATCH_LIGHT / WOMAN_LIP_LIGHT
    left, top, wide, tall = LIP_PATCH_PX
    edge = LIP_PATCH_EDGE_PX
    across = ((np.arange(wide + 2 * edge) - 2 * edge) / max(wide - 2 * edge - 1, 1))[None, :].clip(0, 1)
    up = (1 - (np.arange(tall + 2 * edge) - 2 * edge) / max(tall - 2 * edge - 1, 1))[:, None].clip(0, 1)
    from_middle = np.abs(across - 0.5) * 2
    lab = np.empty((tall + 2 * edge, wide + 2 * edge, 3), np.float32)
    lab[:, :, 0] = lip[0] - UPPER_SHADE * np.clip((up - 0.5) * 2, 0, 1) + LIP_PATCH_GLOW * np.exp(-(((up - 0.25) / 0.15) ** 2)) * (1 - from_middle ** 2)
    lab[:, :, 1:] = lip[1:]
    fade = np.clip((from_middle - LIP_PATCH_CORNER[0]) / (LIP_PATCH_CORNER[1] - LIP_PATCH_CORNER[0]), 0, 1)
    fade = (fade * fade * (3 - 2 * fade))[:, :, None]
    lab = lab * (1 - fade) + near[None, None, :] * fade
    drawn = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
    line = np.exp(-(((up - 0.5 + LIP_PATCH_LINE / 2) / LIP_PATCH_LINE) ** 2)) * np.clip((1 - from_middle) * LINE_ENDS, 0, 1)
    drawn *= 1 - LINE_DARK[None, None, :] * line[:, :, None] * (1 - fade)
    out = skin.copy()
    out[top - edge:top + tall + edge, left - edge:left + wide + edge] = drawn.round().astype(np.uint8)
    return out


def main():
    words = [word for word in sys.argv[1:] if word != "--man"]
    base, mask, out = words[:3]
    colour = words[3] if len(words) > 3 else None
    share = cv2.imread(mask, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
    if "--man" not in sys.argv:
        cv2.imwrite(out, with_lip_patch(cv2.imread(base), share, colour))
        print("LABIOS", out.split("/")[-1], colour or "natural", "no retalho", LIP_PATCH_PX)
        return
    drawn = cv2.imread(mask.replace(".lips.png", ".line.png"))
    line, upper = (drawn[:, :, 0].astype(np.float32) / 255, drawn[:, :, 1].astype(np.float32) / 255) if drawn is not None else (None, None)
    cv2.imwrite(out, painted(cv2.imread(base), share, colour, True, line, upper))
    print("LABIOS", out.split("/")[-1], colour or "natural", "área", round(float((share > INSIDE).mean()), 4))


if __name__ == "__main__":
    main()
