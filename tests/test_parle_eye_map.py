"""Regression for the eye map (2026-11-09): run with `.local/venv/bin/python tests/test_parle_eye_map.py`.

The critic read the cast's pupil as too big under interior light. The map is painted, so the size and the roundness of the
black disc are measured on the painted pixels of every iris colour, per eye.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import eye_texture as eye  # noqa: E402

# Ceilings by colour: 0.42 on a dark iris read as dilated, and 0.36 on honey read as an owl's eye. 0.26 is the floor of the light-adapted range.
CEILING = {"castanho-escuro": 0.38, "castanho": 0.38, "mel": 0.44}
FLOOR = 0.26

for name, colour in eye.IRISES.items():
    image = eye.paint(colour, eye.PUPIL_SHARES[name]).astype(float) / 255
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
        assert share <= CEILING[name], f"{name} eye {index}: pupil is {share:.2f} of the iris, over {CEILING[name]}"
        assert share >= FLOOR, f"{name} eye {index}: pupil is {share:.2f} of the iris, under {FLOOR}"
        assert reach.max() - reach.min() <= 2.0, f"{name} eye {index}: pupil outline varies {reach.max() - reach.min():.1f} px, not a circle"

print("ok")
