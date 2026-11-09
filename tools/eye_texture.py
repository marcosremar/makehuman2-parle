#!/usr/bin/env python3
"""
    License information: data/licenses/makehuman_license.txt

    Paints the eye map of the low-poly eye for the Parle cast, in the style of the two characters
    the owner likes (Rafa and the player): a light warm iris against a black pupil about a third
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
# 0.48 read as a dilated eye in the player's interior close-ups (2026-11-09); 0.36 is the top of the light-adapted range below plus a small margin, chosen by eye.
PUPIL_SHARE = 0.36
# Bounds the regression test holds the painted pupil to, as shares of the iris radius. Light-adapted pupils under interior
# light are about 3 to 4 mm across on an iris of about 11.7 mm, so 0.26 to 0.34 (Watson & Yellott 2012, "A unified formula for
# light-adapted pupil size", Journal of Vision 12(10):12, DOI 10.1167/12.10.12). The top is 0.42, the share the game's eye
# already authored as a design choice between that and the larger pupil people read as attractive (map/copacabana/npc/npc-eyes.lua).
PUPIL_MIN_SHARE = 0.26
PUPIL_MAX_SHARE = 0.42
LIMBUS_SHARE = 0.90
# BGR, 0-1. Measured on the player's iris for the middle one, then 20% lighter with a darker limbus and a whiter sclera: next to her and Rafa the eye read as dull (owner, 2026-11-08).
IRISES = {
    "castanho-escuro": (0.204, 0.324, 0.504),
    "castanho": (0.276, 0.432, 0.672),
    "mel": (0.30, 0.552, 0.816),
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
