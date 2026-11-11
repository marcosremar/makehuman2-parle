#!/usr/bin/env python3
"""
    License information: data/licenses/makehuman_license.txt

    Paints the eye map of the low-poly eye for the Parle cast, in the style of the two characters
    the owner likes (Rafa and the player): a light warm iris against a black pupil a bit over a third
    of its width, a thin dark ring and fine radial fibres. The stock map has a dark muddy iris, so at
    game distance pupil and iris merge and the eye reads as a hole. Same UV layout as stock: one
    iris per eye. The highlight comes from the renderer, like theirs.

      python tools/eye_texture.py <out-dir>
"""
import os
import sys

import cv2
import numpy as np

SIZE = 512
# Centres and radius of the iris in the stock map, as shares of the map.
CENTRES = ((0.703, 0.273), (0.296, 0.676))
IRIS_RADIUS = 0.115
# The cast has no eye bones and the globe rests looking slightly up: without this the pupil touches the upper lid.
GAZE_DOWN = 0.2
# Pupil as a share of the iris radius. Light-adapted pupils under interior light are about 0.25 to 0.33 of the iris (10 to 100 cd/m2,
# Watson & Yellott 2012, "A unified formula for light-adapted pupil size", Journal of Vision 12(10):12, DOI 10.1167/12.10.12). 0.36 sits
# just above that range and is chosen by eye in the player's bust shots (2026-11-09): 0.48 read as a dilated eye, 0.42 still did on a dark iris.
PUPIL_SHARE = 0.36
LIMBUS_SHARE = 0.90
# BGR, 0-1. Measured on the player's iris for the middle one, then 20% lighter with a darker limbus and a whiter sclera: next to her and Rafa the eye read as dull (owner, 2026-11-08).
# Honey was then too bright in the game (value 0.82: an orange disc with a dot, an owl's eye at game size, 2026-11-09). The owner chose
# the darker honey (2026-11-10): same hue at value 0.64, below the 0.68 it had before the 20%, with the same pupil as the brown irises.
# The two browns read orange in the player (34 of 42 bodies, 2026-11-11): value 0.50 and 0.67 at HSV hue 24 and saturation 0.59,
# CIELAB L* 40 and 52 with b* above a* (h_ab ~58). Measured brown irises are much darker and redder: in 87 brown irises photographed
# at high resolution, light / medium / dark brown average L* 26.0 / 20.5 / 15.3, a* 25.5 / 22.7 / 19.3, b* 19.5 / 15.7 / 12.1, so
# h_ab 32 to 37 and never b* over a* by much (Li, Hu, Yang, Xiao, Yamanari, Higashita & Liu 2026, "Quantitative measurement of iris
# melanin concentration by polarization-sensitive anterior segment optical coherence tomography", European Journal of Ophthalmology
# 36(4):872-880, DOI 10.1177/11206721251407025; numbers read in the Research Square preprint rs-5379311, Table 3).
# The L* below is anchored on that source and adjusted by eye: a photograph's L* depends on its exposure and is not the albedo of a
# texture under the game's light, so the numbers set the order of magnitude and the order of the classes, not the exact value.
# castanho sits at L* 31, inside the measured light-brown range (23.7 to 32.9, mean 26). castanho-escuro sits at L* 24.5, in the upper
# part of the medium-brown range (16.7 to 27.1, mean 20.5): at the medium mean it read as a black hole in the shop close-up, the pupil
# lost in the iris (2026-11-11). The two are CIELAB delta E 7.2 apart, so the names stay two colours (at L* 26 / 24.5 they were 1.6
# apart, the same brown to the eye). Hue h_ab 45 is a choice by eye, outside the measured class means (32 to 37) but with a* = b*,
# never b* over a* (the orange). Chroma 20 / 17 (a* = b* = 14.1 and 12.0) sits inside the dark-brown a* (12 to 23) and b* (7.7 to 15)
# ranges but below each class mean: the full measured chroma is HSV saturation 0.67, more than the orange we are removing. Chroma is
# also a choice by eye, to be judged in the player. Converted to sRGB.
IRISES = {
    "castanho-escuro": (0.16, 0.2, 0.316),
    "castanho": (0.202, 0.251, 0.393),
    "mel": (0.307, 0.47, 0.64),
}
LIMBUS = np.array((0.09, 0.07, 0.03))
PUPIL = np.array((0.0, 0.0, 0.0))
SCLERA = np.array((0.93, 0.95, 0.99))


def fibres(angle, seed):
    """Radial streaks: a sum of waves with fixed random phases, so the same map comes out every time."""
    random = np.random.default_rng(seed)
    wave = np.zeros_like(angle)
    for count in (23, 41, 67, 109):
        wave = wave + np.sin(angle * count + random.uniform(0, 6.283))
    return wave / 4


def smooth(low, high, value):
    t = np.clip((value - low) / (high - low), 0, 1)
    return t * t * (3 - 2 * t)


def paint(iris_colour):
    axis = (np.arange(SIZE) + 0.5) / SIZE
    x, y = np.meshgrid(axis, axis)
    image = np.ones((SIZE, SIZE, 3)) * SCLERA
    for index, (cx, cy) in enumerate(CENTRES):
        dx, dy = (x - cx) / IRIS_RADIUS, (y - cy) / IRIS_RADIUS - GAZE_DOWN
        radius = np.hypot(dx, dy)
        near = radius < 1.2
        # A little lighter near the pupil, as in both references.
        colour = np.array(iris_colour)[None, None] * (1.15 - 0.3 * np.clip((radius - PUPIL_SHARE) / (1 - PUPIL_SHARE), 0, 1))[..., None]
        colour = colour * (1 + 0.14 * fibres(np.arctan2(dy, dx), index)[..., None])
        ring = smooth(LIMBUS_SHARE - 0.03, LIMBUS_SHARE + 0.02, radius)[..., None]
        colour = colour * (1 - ring) + LIMBUS * ring
        iris = (1 - smooth(0.98, 1.02, radius))[..., None]
        pupil = (1 - smooth(PUPIL_SHARE - 0.02, PUPIL_SHARE + 0.02, radius))[..., None]
        eye = SCLERA * (1 - iris) + colour * iris
        eye = eye * (1 - pupil) + PUPIL * pupil
        image[near] = eye[near]
    return (np.clip(image, 0, 1) * 255 + 0.5).astype(np.uint8)


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    for name, colour in IRISES.items():
        cv2.imwrite(os.path.join(out, f"parle_eye_{name}.png"), paint(colour))
        print("EYE", name)


if __name__ == "__main__":
    main()
