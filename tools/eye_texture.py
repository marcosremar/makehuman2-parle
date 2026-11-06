#!/usr/bin/env python3
"""
    License information: data/licenses/makehuman_license.txt

    Paints the eye map of the low-poly eye for the Parle cast, in the style of the two characters
    the owner likes (Rafa and the player): a light warm iris against a black pupil of about half
    its width, a thin dark ring and fine radial fibres. The stock map has a dark muddy iris, so at
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
PUPIL_SHARE = 0.48
LIMBUS_SHARE = 0.90
# BGR, 0-1, measured on the player's iris for the middle one.
IRISES = {
    "castanho-escuro": (0.17, 0.27, 0.42),
    "castanho": (0.23, 0.36, 0.56),
    "mel": (0.25, 0.46, 0.68),
}
LIMBUS = np.array((0.20, 0.16, 0.05))
PUPIL = np.array((0.0, 0.0, 0.0))
SCLERA = np.array((0.86, 0.89, 0.97))


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
