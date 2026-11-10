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
# the lightest eye of the cast by far; the owner chose 0.64 (2026-11-10). The lightest iris is now the middle brown, at 0.67.
VALUE_CEILING = 0.70

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

print("ok")
