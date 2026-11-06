#!/usr/bin/env python3
"""
    License information: data/licenses/makehuman_license.txt

    Opaque teeth map for the Parle cast. The stock map is plain white and draws the teeth in its
    alpha channel, so with alpha blending whatever is behind leaks through them as dark blotches.
    The Rafa and the player have opaque teeth. Here the alpha becomes colour, on the dark of the
    inside of the mouth.

      python tools/teeth_texture.py <stock-teeth.png> <out.png>
"""
import sys

import cv2
import numpy as np

IVORY = np.array((0.84, 0.92, 0.96))  # BGR
MOUTH = np.array((0.10, 0.10, 0.20))
SIZE = 512


def main():
    source, out = sys.argv[1:3]
    alpha = cv2.imread(source, cv2.IMREAD_UNCHANGED)[:, :, 3].astype(np.float32) / 255
    alpha = cv2.resize(alpha, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)
    alpha = np.clip(alpha, 0, 1)[:, :, None]
    cv2.imwrite(out, (np.clip(IVORY * alpha + MOUTH * (1 - alpha), 0, 1) * 255 + 0.5).astype(np.uint8))
    print("TEETH", out)


if __name__ == "__main__":
    main()
