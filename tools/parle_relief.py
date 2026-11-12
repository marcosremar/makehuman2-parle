#!/usr/bin/env python3
"""
    License information: data/licenses/makehuman_license.txt

    Flattens the face relief (normal map) on the mouth of a Parle cast member. The relief is baked from the game face, whose
    lips are full: in the player it drew a ring of light and shade around them, the outline of a big lip over the new, thinner
    one, and the mouth read as swollen (owner, 2026-11-08, pharmacy). Inside the lip map and FADE_PX around it the normal
    goes flat, fading back to the relief outside.

        parle_relief.py <relevo.png> <lips-map.png>   # rewrites the relief in place
"""
import sys

import cv2
import numpy as np

from parle_lips import LIP_PATCH_EDGE_PX, LIP_PATCH_PX

AROUND_PX = 10
FADE_PX = 8
FLAT = np.array((255, 128, 128), np.float32)  # BGR of a normal pointing straight out


def flattened(relief, lips):
    core = cv2.dilate((lips > 0.2).astype(np.uint8), np.ones((AROUND_PX * 2 + 1, AROUND_PX * 2 + 1), np.uint8)).astype(np.float32)
    share = cv2.GaussianBlur(core, (0, 0), FADE_PX)[:, :, None]
    share = np.maximum(share, core[:, :, None])
    return (relief.astype(np.float32) * (1 - share) + FLAT * share).round().astype(np.uint8)


def main():
    relief_path, lips_path = sys.argv[1:3]
    lips = cv2.imread(lips_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
    relief = flattened(cv2.imread(relief_path), lips)
    # The patch the lips of a woman are mapped to (`parle_lip_cut`) is flat too: whatever relief stood there would be drawn on her lips.
    left, top, wide, tall = LIP_PATCH_PX
    relief[top - LIP_PATCH_EDGE_PX:top + tall + LIP_PATCH_EDGE_PX, left - LIP_PATCH_EDGE_PX:left + wide + LIP_PATCH_EDGE_PX] = FLAT
    cv2.imwrite(relief_path, relief)
    print("RELEVO BOCA", relief_path.split("/")[-1])


if __name__ == "__main__":
    main()
