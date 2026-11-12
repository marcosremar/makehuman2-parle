#!/usr/bin/env python3
"""
    License information: data/licenses/makehuman_license.txt

    Paints the nostrils of a Parle cast skin, as a step of its own after `skin_style.py` and before the lips.

    The mesh step closes the notch under each wing of the nose and leaves a shallow dent in its place (`parle_nose`, 2026-11-12).
    A dent under a millimetre deep takes little shade of its own, and what makes it read as a nostril is paint: "sketched with
    a curve and never filled with black". So the texels of each dent get a soft shade, darker and redder than the person's own
    skin, strongest at the middle of the dent and gone a quarter of its size past its edge; both nostrils get the same, since
    the two dents are mirror images.

    Before that, whatever paint the skin has there is replaced by the skin round it. On the cast as it is, little is left to
    replace: `skin_style` has already cleared the dark nostril of the photograph, and the texels of the notch are 1 to 3 points
    of lightness, of 100, under the skin round them (five characters, 2026-11-12). But with the notch closed those texels lie open
    on the surface, and a darker patch with the outline of the old hole would show under the new shade.

    The dents are found here on the person's own points, the same way the mesh step finds them on the finished body
    (`parle_nose.closed_and_dented`), and the shade is laid by where each texel stands on the dented skin: the notch has some
    thirty points packed into 6 by 7 texels of the map, so on the map it has no shape to paint by. On the five the two meshes
    put the nostrils on the same texels (897 to 903 across, 515 to 525 and 533 to 543 down).

        parle_nostrils.py <skin.png> <uvpos.npz> <body.npz> [<out.png>]

    With no <out.png> the skin is painted in place: `skin_style.py` makes it anew each time, so the shade is never laid twice.
"""
import sys

import cv2
import numpy as np

import parle_nose
from parle_beard import position_map

SIZE = 1024
# The texels looked at are those of the triangles with a corner this near the middle of a dent.
NEAR_M = 0.012
# The old paint is cleared over the shape of the dent this many times its size, whole where the share of that shape is
# CLEAR_WHOLE or more (as far as the edge of the dent itself) and less out to its own edge, and what fills it is the skin
# round it carried inwards from CLEAR_FILL_PX away.
CLEAR_WIDER = 1.6
CLEAR_WHOLE = 0.4
CLEAR_FILL_PX = 3
# The shade, at the middle of the dent: this share darker, and this much redder and warmer (the a and b of Lab, of 255). It
# covers the shape of the dent SHADE_WIDER times its size. Chosen by eye on the five, in colour and with the paint alone: at
# 7 of red with 3 of warmth the shade read orange on the lightest skins (Dona Rosa, Lúcia). As it is, the middle of the shade
# is 10 to 13 points of lightness, of 100, under the skin (60 where Lúcia's skin has 71, 49 where Carlos's has 59), and its
# darkest texel is 46: no black. No source: a design choice to pilot.
SHADE_WIDER = 1.25
SHADE_DARK = 0.24
SHADE_RED = 6.0
SHADE_WARM = 1.0


def nostril_maps(data, points):
    """Texel by texel, from the cast mesh (`uvpos.npz`) and this person's own points: how much of the shade of a nostril each
    texel takes, how much of its old paint is cleared, and the dents (`parle_nose.closed_and_dented`): none, and two empty
    maps, when the nose has no notch."""
    moved, dents = parle_nose.closed_and_dented(points, data["tri"])
    shade, clear = np.zeros((SIZE, SIZE), np.float32), np.zeros((SIZE, SIZE), np.float32)
    for dent in dents.values():
        near = (np.linalg.norm(moved[data["tri"]] - dent["centre"], axis=2) < NEAR_M).any(axis=1)
        place = position_map({"uv": data["uv"][near], "tri": data["tri"][near], "pos": moved}, SIZE)
        known = ~np.isnan(place[:, :, 2])
        spots = np.nan_to_num(place).reshape(-1, 3)
        for wider, canvas in ((SHADE_WIDER, shade), (CLEAR_WIDER, clear)):
            share, over = parle_nose.dent_share(spots, dent, wider)
            canvas += (share * (np.abs(over) < parle_nose.DENT_NEAR_M)).reshape(SIZE, SIZE) * known
    return np.clip(shade, 0, 1), np.clip(clear / CLEAR_WHOLE, 0, 1), dents


def painted(skin, shade, clear):
    """The skin with the old paint of the nostrils replaced by the skin round them, and their shade put on."""
    hole = (clear > 0).astype(np.uint8)
    if not hole.any():
        return skin
    filled = cv2.inpaint(skin, hole, CLEAR_FILL_PX, cv2.INPAINT_TELEA).astype(np.float32)
    clean = skin * (1 - clear[:, :, None]) + filled * clear[:, :, None]
    lab = cv2.cvtColor(np.clip(clean, 0, 255).round().astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 0] *= 1 - SHADE_DARK * shade
    lab[:, :, 1] += SHADE_RED * shade
    lab[:, :, 2] += SHADE_WARM * shade
    shaded = cv2.cvtColor(np.clip(lab, 0, 255).round().astype(np.uint8), cv2.COLOR_LAB2BGR)
    return np.where((np.maximum(shade, clear) > 0)[:, :, None], shaded, skin)


def main():
    source, map_path, shape_path = sys.argv[1:4]
    out = sys.argv[4] if len(sys.argv) > 4 else source
    shade, clear, dents = nostril_maps(np.load(map_path), np.load(shape_path)["points"].astype(np.float64))
    cv2.imwrite(out, painted(cv2.imread(source), shade, clear))
    print("NARINAS", out.split("/")[-1], len(dents), "narinas; sombra em", int((shade > 0.05).sum()), "texels, pintura antiga trocada em", int((clear > 0.05).sum()))


if __name__ == "__main__":
    main()
