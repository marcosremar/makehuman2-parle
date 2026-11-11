#!/usr/bin/env python3
"""
    License information: data/licenses/makehuman_license.txt

    Teeth map of the Parle cast: every tooth a cell and every gum a strip of `parle_teeth.atlas`, painted opaque. The stock map drew them in its alpha over piled UV islands.

      python tools/teeth_texture.py <out.png>
"""
import sys

import cv2
import numpy as np

import parle_teeth


def main():
    out = sys.argv[1]
    cv2.imwrite(out, (np.clip(parle_teeth.tooth_map(), 0, 1) * 255 + 0.5).astype(np.uint8))
    print("TEETH", out)


if __name__ == "__main__":
    main()
