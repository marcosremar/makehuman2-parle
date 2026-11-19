"""Regression for the eye map (2026-11-09): run with `.local/venv/bin/python tests/test_parle_eye_map.py`.

The critic read the cast's pupil as too big under interior light, and the honey iris as an owl's eye: a bright orange disc with a
dot. The map is painted, so the size and the roundness of the black disc and the brightness of the iris are measured on the painted
pixels of every iris colour, per eye.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import eye_texture as eye  # noqa: E402

# 0.42 on a dark iris read as dilated (0.48 on all of them before). 0.26 is the floor of the light-adapted range.
CEILING = 0.38
FLOOR = 0.26
# Mean value (the brightest channel) of the iris between pupil and limbus. Honey at 0.82 read as an orange disc and at 0.72 was still
# the lightest eye of the cast by far; the owner chose 0.64 (2026-11-10). The lightest iris is now honey, at about 0.65.
VALUE_CEILING = 0.70
# HSV saturation of the mean colour of a brown iris. At 0.59 both browns read orange in the player (2026-11-11); the new browns are
# about 0.50 (CIELAB from Li et al. 2026, Eur J Ophthalmol, DOI 10.1177/11206721251407025: see eye_texture.py). Honey is warm by name.
BROWN_SATURATION_CEILING = 0.55
# Mean value of a brown iris. The orange browns had 0.51 and 0.68; the new ones are about 0.32 and 0.39. Without this a light orange
# brown would pass on saturation alone.
BROWN_VALUE_CEILING = 0.43
# The two browns must be two colours: CIELAB delta E between them. At L* 26 / 24.5 they were 1.6 apart (2026-11-11), the same brown.
BROWN_DELTA_E_FLOOR = 5.0
# Honey read orange in the player's close-up too (2026-11-19, the baker Seu Jorge, loop of references cycle 3): painted at saturation
# 0.48 and value 0.66, the game drew it at HSV hue 29, saturation 0.71, an orange disc with a black dot, where the dark brown of the
# pharmacist (painted 0.42 / 0.35) came out at saturation 0.58. Under the game's light the painted saturation grows about half again:
# honey stays at or under 0.40 painted and under the old 0.55 of a light brown in value, still lighter than the light brown and a
# colour of its own beside it.
HONEY_SATURATION_CEILING = 0.40
HONEY_VALUE_CEILING = 0.55
HONEY_DELTA_E_FLOOR = 5.0


def lab(bgr):
    """CIELAB (D65) of a BGR colour in 0-1 sRGB."""
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in bgr[::-1]]
    r, g, b = linear
    x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883
    f = [t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116 for t in (x, y, z)]
    return np.array((116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])))


delta_e = np.linalg.norm(lab(eye.IRISES["castanho"]) - lab(eye.IRISES["castanho-escuro"]))
assert delta_e > BROWN_DELTA_E_FLOOR, f"castanho and castanho-escuro are delta E {delta_e:.1f} apart, not over {BROWN_DELTA_E_FLOOR}"
honey_e = np.linalg.norm(lab(eye.IRISES["mel"]) - lab(eye.IRISES["castanho"]))
assert honey_e > HONEY_DELTA_E_FLOOR, f"mel and castanho are delta E {honey_e:.1f} apart, not over {HONEY_DELTA_E_FLOOR}: one brown"
assert lab(eye.IRISES["mel"])[0] > lab(eye.IRISES["castanho"])[0], "mel is no longer lighter than castanho"

for name, colour in eye.IRISES.items():
    image = eye.paint(colour).astype(float) / 255
    for index, (cx, cy) in enumerate(eye.CENTRES):
        centre_x = cx * eye.SIZE
        centre_y = (cy + eye.GAZE_DOWN * eye.IRIS_RADIUS) * eye.SIZE
        radius_px = eye.IRIS_RADIUS * eye.SIZE
        # Walk out from the centre in 72 directions and stop where the map stops being black.
        reach = []
        for angle in np.arange(0, 2 * np.pi, 2 * np.pi / 72):
            for step in np.arange(0, radius_px, 0.25):
                x, y = int(centre_x + np.cos(angle) * step), int(centre_y + np.sin(angle) * step)
                if image[y, x].max() > 0.08:
                    reach.append(step)
                    break
        reach = np.array(reach)
        share = reach.mean() / radius_px
        assert share <= CEILING, f"{name} eye {index}: pupil is {share:.2f} of the iris, over {CEILING}"
        assert share >= FLOOR, f"{name} eye {index}: pupil is {share:.2f} of the iris, under {FLOOR}"
        assert reach.max() - reach.min() <= 2.0, f"{name} eye {index}: pupil outline varies {reach.max() - reach.min():.1f} px, not a circle"
        ys, xs = np.mgrid[0:eye.SIZE, 0:eye.SIZE]
        distance = np.hypot(xs + 0.5 - centre_x, ys + 0.5 - centre_y) / radius_px
        ring = (distance > share + 0.06) & (distance < eye.LIMBUS_SHARE - 0.06)
        value = image[ring].max(axis=1).mean()
        assert value <= VALUE_CEILING, f"{name} eye {index}: iris value is {value:.2f}, over {VALUE_CEILING}: it reads as a bright disc"
        if name.startswith("castanho"):
            mean = image[ring].mean(axis=0)
            saturation = 1 - mean.min() / mean.max()
            assert saturation <= BROWN_SATURATION_CEILING, f"{name} eye {index}: saturation {saturation:.2f}, over {BROWN_SATURATION_CEILING}: it reads orange"
            assert value <= BROWN_VALUE_CEILING, f"{name} eye {index}: brown value {value:.2f}, over {BROWN_VALUE_CEILING}: a light brown reads orange"
        if name == "mel":
            mean = image[ring].mean(axis=0)
            saturation = 1 - mean.min() / mean.max()
            assert saturation <= HONEY_SATURATION_CEILING, f"mel eye {index}: saturation {saturation:.2f}, over {HONEY_SATURATION_CEILING}: it reads orange"
            assert value <= HONEY_VALUE_CEILING, f"mel eye {index}: value {value:.2f}, over {HONEY_VALUE_CEILING}: an orange disc with a dot"

print("ok")


# Round 66: at the owner's opening of the lids the iris filled the eye from corner to corner (95 % of its width; the series' 55 to
# 60 %): it stays under 0.16 of the map.
assert eye.IRIS_RADIUS < 0.16, f"the iris fills the eye: {eye.IRIS_RADIUS}"
# Round 83 (measured): at 0.14 the cast showed a quarter of the white the series shows; under 0.10 the iris is shorter than the
# opening of the lids and white shows all about it (the stare the owner faults).
assert 0.10 < eye.IRIS_RADIUS <= 0.12, f"the iris hides the white, or leaves white all about it: {eye.IRIS_RADIUS}"
