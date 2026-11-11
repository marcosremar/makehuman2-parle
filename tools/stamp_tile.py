#!/usr/bin/env python3
"""
    License information: data/licenses/makehuman_license.txt

    Joins the print laid on the cloth (`print_cloth.py`) to the picture of the garment baked from the game, in place:

        stamp_tile.py <baked.png> <print.png>

    The baked picture still says where there is print: a plain part, like Dona Rosa's black bodice, stays as it is, and what the bake
    did not reach is part of the same cloth (a skirt longer than the game's, a sleeve). What sits on the print without being of it
    stays too: the buttons of a shirt, the undershirt at its neck.
"""
import sys

import cv2
import numpy as np

BUSY_PX = 33
BUSY_SPREAD = 6.0
EDGE_PX = 25
# A colour is of the print when the print has it, or a mix of two of its neighbours, within this many steps of 8 in each of blue, green
# and red. The white of a button and the pale blue of an undershirt are not; the same green a little darker, in the shade the game
# painted on the cloth, is.
COLOUR_STEP = 8
COLOUR_NEAR = 2
MIX_PX = (1.0, 2.0, 4.0)
# What is kept is at least this wide: a button is 4 pixels, the edge of a check blurred by the bake is 1 or 2.
KEPT_LEAST_PX = 3
KEPT_SOFT = 1.0


def print_colours(laid, cloth):
    """The colours of the print, as a table of blue, green and red in steps: the ones on the picture, the mixes between neighbours, and
    the ones next to them."""
    cells = 256 // COLOUR_STEP
    table = np.zeros((cells, cells, cells), np.uint8)
    for soft in (0.0, *MIX_PX):
        seen = (cv2.GaussianBlur(laid, (0, 0), soft) if soft else laid)[cloth] // COLOUR_STEP
        table[seen[:, 0], seen[:, 1], seen[:, 2]] = 1
    for axis in range(3):
        padded = np.pad(table, [(COLOUR_NEAR, COLOUR_NEAR) if index == axis else (0, 0) for index in range(3)])
        table = np.maximum.reduce([np.take(padded, range(shift, shift + cells), axis=axis) for shift in range(2 * COLOUR_NEAR + 1)])
    return table


def kept(baked, laid, cloth, reached):
    """What sits on the print without being of it, from 0 to 1: reached by the bake, of a colour the print does not have, and wide enough
    to be a thing and not the edge of one."""
    table = print_colours(laid, cloth)
    steps = baked // COLOUR_STEP
    other = (table[steps[:, :, 0], steps[:, :, 1], steps[:, :, 2]] == 0) & cloth & reached
    wide = cv2.morphologyEx(other.astype(np.uint8), cv2.MORPH_OPEN, np.ones((KEPT_LEAST_PX, KEPT_LEAST_PX), np.uint8))
    return cv2.GaussianBlur(cv2.dilate(wide, np.ones((3, 3), np.uint8)).astype(np.float32), (0, 0), KEPT_SOFT)


def joined(baked, laid, triangles=None, unreached=None):
    """The baked picture with the laid print where the bake shows print or did not reach, less what is kept (`kept`). `triangles` are the
    UV triangles of the garment in pixels: past them each point takes the colour of the cloth nearest to it."""
    cloth = np.ones(baked.shape[:2], np.uint8)
    if triangles is not None:
        cloth[:] = 0
        for triangle in triangles:
            cv2.fillConvexPoly(cloth, triangle, 1)
    cloth = cloth > 0
    hole = np.zeros(baked.shape[:2], bool) if unreached is None else unreached > 0
    gray = cv2.cvtColor(baked, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.blur(gray, (BUSY_PX, BUSY_PX))
    spread = np.sqrt(np.maximum(cv2.blur(gray * gray, (BUSY_PX, BUSY_PX)) - mean * mean, 0))
    busy = cv2.morphologyEx((spread > BUSY_SPREAD).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((EDGE_PX, EDGE_PX), np.uint8))
    share = cv2.GaussianBlur(np.maximum(busy, hole.astype(np.uint8)).astype(np.float32), (EDGE_PX, EDGE_PX), 0)
    share = share * (1 - kept(baked, laid, cloth, ~hole))
    out = (baked.astype(np.float32) * (1 - share[:, :, None]) + laid.astype(np.float32) * share[:, :, None]).round().astype(np.uint8)
    if triangles is not None and cloth.any() and not cloth.all():
        _, labels = cv2.distanceTransformWithLabels((~cloth).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        rows, columns = np.nonzero(cloth)
        nearest = np.zeros((labels.max() + 1, 3), np.uint8)
        nearest[labels[rows, columns]] = out[rows, columns]
        out[~cloth] = nearest[labels[~cloth]]
    return out, float(share[cloth].mean())


def main():
    import os

    baked_path, laid_path = sys.argv[1], sys.argv[2]
    baked, laid = cv2.imread(baked_path), cv2.imread(laid_path)
    triangles = baked_path.replace(".png", ".uv.npy")
    unreached = cv2.imread(baked_path.replace(".png", ".hole.png"), cv2.IMREAD_GRAYSCALE)
    out, share = joined(baked, laid, np.load(triangles) if os.path.exists(triangles) else None, unreached)
    cv2.imwrite(baked_path, out)
    print("TILE", baked_path.split("/")[-1], "estampa em", round(share, 2), "do pano")


if __name__ == "__main__":
    main()
