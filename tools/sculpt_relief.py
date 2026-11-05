#!/usr/bin/env python3
"""Turns a sculpture into the torso relief map core/body_shape reads. Run it again only for a new sculpture.

    python tools/sculpt_relief.py <sculpture.npz> data/sculpt/male-torso-relief.npz
"""
import sys

import numpy as np

sys.path.insert(0, ".")
from core.body_shape import relief_map  # noqa: E402


def main():
    source, target = sys.argv[1:3]
    relief = relief_map(np.load(source)["points"].astype(np.float64))
    np.savez_compressed(target, relief=relief.astype(np.float32))
    print("RELEVO", target, relief.shape, round(float(np.abs(relief).max()), 3))


main()
