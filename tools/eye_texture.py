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
# The stock radius is 0.115, the iris of a real eye on its globe. A judge by sight read the cast's eyes as small next to Sofia's
# and asked for the iris bigger with the opening (2026-11-12, `parle_face.FAIR_LOOK`): on a drawn face the iris fills more of the eye.
# At 0.13 it filled the opening from lid to lid, with slivers of white: "two dark beads" (the same judge, five rounds on). Against
# the width of the face it measured 0.109, Sofia's 0.097 and the same woman's in the game 0.101: a little smaller.
# Round 21 (2026-11-16): with the lids opened rounder the white showed all about the iris, a scared look; the series' iris fills
# most of the eye.
# Round 23 (4.5): at 0.155 the iris filled the eye and no white showed: between the two.
# Round 25: with the lid lifted, an iris filling more of the width (about 55 to 60 % of it in the series).
# Round 31: the iris still "pequena, com branco dos dois lados" on all four.
# Round 40 (two judges): the iris fills the opening lid to lid and the eye reads as a dark disc; one judge asked it 30 % smaller.
# Round 41 (5.0): both judges asked the iris 10 to 15 % bigger within the same eye: between.
# Round 47 (5.4): both judges asked the iris a fifth bigger within the same eye, white about it "no limite do esbugalhado".
# Round 50: with the eyes back to the owner's sizes, both again "íris 20 % maior sem aumentar o olho".
# Round 66 (two judges, eight of the cast): with the lids back at the owner's opening (round 60), the iris filled the eye from
# corner to corner, about 95 % of its width where the series' fills 55 to 60, "olho só íris, sem branco"; one judge asked it 40 %
# smaller, the other 20: 0.14.
# Round 83 (measured on the portraits of round 81 against the series, a finder and a sceptic): the cast showed a quarter of the
# white the series' eye shows (0.11 of the opening against 0.46) and its iris was 1.4 times the series' (0.39 of the distance
# between the pupils against 0.28). 0.10 would be the series' size, as tall as the opening of the lids: a little over it, so
# the lids still lie across the top and the foot of the iris and no white shows about it (the stare of rounds 21 and 25).
IRIS_RADIUS = 0.115
# The cast has no eye bones and the globe rests looking slightly up: without this the pupil touches the upper lid.
# Measured on a woman of the cast (2026-11-12), the lids opened from 5 to 8 degrees above the middle of the iris to 21 to 27 below
# it: at 0.2 the upper lid still lay across the top of the pupil. With the lids as they are now (`parle_face.FAIR_LOOK`), at 0.3
# the upper lid covers the top of the iris and the lower one touches its foot; at 0.45 white showed over the iris, a look cast down.
GAZE_DOWN = 0.3
# Pupil as a share of the iris radius. Light-adapted pupils under interior light are about 0.25 to 0.33 of the iris (10 to 100 cd/m2,
# Watson & Yellott 2012, "A unified formula for light-adapted pupil size", Journal of Vision 12(10):12, DOI 10.1167/12.10.12). 0.36 sits
# just above that range and is chosen by eye in the player's bust shots (2026-11-09): 0.48 read as a dilated eye, 0.42 still did on a dark iris.
# Round 66: "pupila 30 % menor" (a judge by sight against the series): 0.30, inside that range.
# Round 83: with the iris smaller the pupil keeps the size it had on the ball (0.14 x 0.30); the series' is 0.45 of its iris, over
# the ceiling the test holds for the player's bust shots (0.38).
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
# The series the owner chose on 2026-11-15 has lighter, clearer brown irises: a judge by sight against it asked for them a quarter
# lighter (round 8 of the loop), so each colour is IRIS_LIGHTER times the one chosen above.
# Round 27: the dark brown iris read almost black; the series' are a mid brown.
# Round 33: at 1.6 the brown washed into the white, an iris "sumida".
# 2026-11-16: at 1.15 the light brown passed the value the owner's test holds (0.48 against 0.43, `tests/test_parle_eye_map.py`: a
# light brown reads orange in the player, 34 of 42 bodies on 2026-11-11): the browns as the owner chose them.
IRIS_LIGHTER = 1.0
# Honey read orange in the player as the browns had (2026-11-19, the baker Seu Jorge in the bakery close-up, loop of references
# cycle 3): at L* 54, a* 12, b* 29 (h_ab 68, chroma 32, HSV saturation 0.52) the game drew an orange disc, HSV hue 29 and saturation
# 0.71, where the brown of the pharmacist came out at 0.58. The series the owner chose has dark, clean eyes, and glasses never
# change their colour. Honey now keeps its place between the light brown and the old honey: L* 44, over the light brown's 31 and
# the measured light-brown range (23.7 to 32.9, Li et al. 2026 above), so it still reads lighter; a* 8, b* 16, hue h_ab 63, more
# yellow than the browns' 45 because honey is the yellow end of the browns (the intermediate colours sit between brown and green,
# Sturm & Larsson 2009, "Genetics of human iris colour and patterns", Pigment Cell & Melanoma Research 22(5):544-562,
# DOI 10.1111/j.1755-148X.2009.00606.x); chroma 18, under the browns' 20, because the game's light adds about half again to the
# painted saturation. Lightness and chroma are a choice by eye on the player's close-up, to be judged.
IRISES = {name: tuple(min(1.0, value * IRIS_LIGHTER) for value in colour) for name, colour in {
    "castanho-escuro": (0.16, 0.2, 0.316),
    "castanho": (0.202, 0.251, 0.393),
    "mel": (0.306, 0.387, 0.494),
}.items()}
LIMBUS = np.array((0.09, 0.07, 0.03))
PUPIL = np.array((0.0, 0.0, 0.0))
# Written by cv2 (blue first): the old bluish white came out pink on the eye (round 33: "branco rosado"). A neutral white.
SCLERA = np.array((0.96, 0.96, 0.96))
# One small clean catchlight up and to one side of the pupil, as every eye of the series the owner chose has (round 20 of the loop,
# 2026-11-16): a disc in iris radii, at (across, down) from the iris's middle. The iris drawn flat, with almost none of its fibres.
# Round 56: judge B asked it twice as big; bigger it cut the pupil's round edge, and further out its white took the brown iris's
# value past the owner's ceiling (`tests/test_parle_eye_map.py`): it stays as it was.
CATCHLIGHT = (0.36, -0.4)
CATCHLIGHT_RADIUS = 0.17
FIBRES = 0.04


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
        colour = colour * (1 + FIBRES * fibres(np.arctan2(dy, dx), index)[..., None])
        ring = smooth(LIMBUS_SHARE - 0.03, LIMBUS_SHARE + 0.02, radius)[..., None]
        colour = colour * (1 - ring) + LIMBUS * ring
        iris = (1 - smooth(0.98, 1.02, radius))[..., None]
        pupil = (1 - smooth(PUPIL_SHARE - 0.02, PUPIL_SHARE + 0.02, radius))[..., None]
        eye = SCLERA * (1 - iris) + colour * iris
        eye = eye * (1 - pupil) + PUPIL * pupil
        glint = (1 - smooth(CATCHLIGHT_RADIUS - 0.03, CATCHLIGHT_RADIUS + 0.03, np.hypot(dx - CATCHLIGHT[0], dy - CATCHLIGHT[1])))[..., None]
        eye = eye * (1 - glint) + glint
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
