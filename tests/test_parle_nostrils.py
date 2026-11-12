"""Regression for the paint of the nostrils (2026-11-12): run with `.local/venv/bin/python tests/test_parle_nostrils.py`.

The judge by sight, rounds 13 to 18, on the nose of the cast: "torn, asymmetric dark notches", "a ragged dark blot from the front".
The mesh step closes the notch under each wing of the nose and leaves a shallow dent (`parle_nose`); `parle_nostrils` puts a soft
shade in each dent, darker and redder than the skin and never black, the same on both sides, and takes away whatever dark paint
the skin had where the hole was."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import parle_nose  # noqa: E402
from parle_nostrils import SHADE_DARK, SIZE, nostril_maps, painted  # noqa: E402

# The base of a nose that looks at -y: a plate that looks down, 4 cm wide and 3 cm deep, with a notch going up into the nose on
# each side of the middle, 5 mm across, 10 long and 3 deep. The tip of the nose is 4 mm ahead of the plate, at its level.
STEP = 0.001
XS = np.round(np.arange(-0.02, 0.0201, STEP), 6)
YS = np.round(np.arange(0.0, 0.0301, STEP), 6)
LEVEL, NOTCH_AT, NOTCH_HALF, NOTCH_DEEP = 1.5, (0.009, 0.014), (0.0025, 0.005), 0.003
TIP, TOP = (0.0, -0.004, LEVEL), (0.0, 0.03, LEVEL + 0.1)


def nose_base(deep=NOTCH_DEEP):
    """The points and the triangles (outer side down), the triangles of one half the mirror image of those of the other."""
    roof = lambda x, y: LEVEL + deep * max(0.0, 1 - ((abs(x) - NOTCH_AT[0]) / NOTCH_HALF[0]) ** 2 - ((y - NOTCH_AT[1]) / NOTCH_HALF[1]) ** 2) ** 2
    points = [(x, y, roof(x, y)) for y in YS for x in XS] + [TIP, TOP]
    wide, triangles = len(XS), []
    for row in range(len(YS) - 1):
        for col in range(wide - 1):
            a, b, c, d = row * wide + col, row * wide + col + 1, (row + 1) * wide + col + 1, (row + 1) * wide + col
            triangles.extend([(a, c, b), (a, d, c)] if XS[col] >= 0 else [(a, d, b), (b, d, c)])
    return np.array(points), np.array(triangles)


POINTS, TRIANGLES = nose_base()
# The map: 6 cm of the plate across the picture, so a texel is 0.06 mm.
UV = np.stack([(POINTS[TRIANGLES][:, :, 0] + 0.03) / 0.06, (POINTS[TRIANGLES][:, :, 1] + 0.01) / 0.06], axis=2)
DATA = {"uv": UV, "tri": TRIANGLES, "pos": POINTS}
texel = lambda x, y: (int(round((1 - (y + 0.01) / 0.06) * SIZE)), int(round((x + 0.03) / 0.06 * SIZE)))

# A skin of one colour, with the nostrils of a photograph painted dark where the notches are.
SKIN_COLOUR, DARK = (150, 175, 215), (55, 65, 85)
skin = np.full((SIZE, SIZE, 3), SKIN_COLOUR, np.uint8)
for side in (1, -1):
    cv2.ellipse(skin, texel(side * NOTCH_AT[0], NOTCH_AT[1])[::-1], (int(0.002 / 0.06 * SIZE), int(0.004 / 0.06 * SIZE)), 0, 0, 360, DARK, -1)
lab_of = lambda image: cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(float)
own = lab_of(np.full((1, 1, 3), SKIN_COLOUR, np.uint8))[0, 0]

shade, clear, dents = nostril_maps(DATA, POINTS)
assert sorted(dents) == [-1, 1], f"the nose of the test has {len(dents)} nostrils"
out = painted(skin, shade, clear)
lab = lab_of(out)
for side, dent in dents.items():
    middle = texel(dent["centre"][0], dent["centre"][1])
    assert abs(dent["centre"][0] - side * NOTCH_AT[0]) < 0.001 and abs(dent["centre"][1] - NOTCH_AT[1]) < 0.0015, f"the dent is at {dent['centre'] * 1000} mm"
    # The shade is whole at the middle of the dent...
    assert shade[middle] > 0.9, f"the shade at the middle of the dent is {shade[middle]:.2f}"
    # ... darker and redder than the skin there, by a modest amount, and never the black of a hole.
    light, red = lab[middle][0] / own[0], lab[middle][1] - own[1]
    assert 1 - SHADE_DARK - 0.03 < light < 1 - SHADE_DARK + 0.06 and 3 < red < 9, f"at the middle of the dent the skin has {light:.2f} of its lightness and {red:+.1f} of red"
    # It is gone a little way past the dent: 4 mm to the side of its middle and 7 mm along it the skin is the skin.
    for x, y in ((dent["centre"][0] + side * 0.004, dent["centre"][1]), (dent["centre"][0], dent["centre"][1] + 0.007), (dent["centre"][0], dent["centre"][1] - 0.007)):
        assert shade[texel(x, y)] == 0 and np.abs(out[texel(x, y)].astype(int) - SKIN_COLOUR).max() <= 2, f"the shade reaches {x * 1000:.1f}, {y * 1000:.1f} mm"
# The dark paint of the old hole is gone: nothing left on the skin is darker than the middle of the shade, where the paint had
# 0.35 of the lightness of the skin.
assert lab[:, :, 0].min() / own[0] > 1 - SHADE_DARK - 0.03, f"the darkest texel left has {lab[:, :, 0].min() / own[0]:.2f} of the lightness of the skin"
assert (lab_of(skin)[:, :, 0].min() / own[0]) < 0.45, "the test skin had no dark nostrils"
# Far from the nostrils nothing is touched: the middle of the nose base, and the whole skin 9 mm and more from a dent.
far = np.ones((SIZE, SIZE), bool)
for dent in dents.values():
    row, col = texel(dent["centre"][0], dent["centre"][1])
    far[max(row - 154, 0):row + 154, max(col - 154, 0):col + 154] = False
assert np.array_equal(out[far], skin[far]) and np.array_equal(out[texel(0.0, NOTCH_AT[1])], skin[texel(0.0, NOTCH_AT[1])]), "skin far from the nostrils was painted"
# The two nostrils are painted the same: one is the other in the mirror.
row, col = texel(dents[1]["centre"][0], dents[1]["centre"][1])
right = out[row - 120:row + 120, col - 80:col + 80].astype(int)
row, col = texel(dents[-1]["centre"][0], dents[-1]["centre"][1])
left = out[row - 120:row + 120, col - 80:col + 80][:, ::-1].astype(int)
assert np.abs(right - left).mean() < 1.5, f"the two nostrils differ by {np.abs(right - left).mean():.2f} levels a texel"

# A nose with no notch gets no paint.
flat, triangles = nose_base(0.0)
shade, clear, dents = nostril_maps({"uv": UV, "tri": triangles, "pos": flat}, flat)
assert not dents and not shade.any() and not clear.any() and np.array_equal(painted(skin, shade, clear), skin), "a nose with no notch was painted"
# And the shade is laid where the dent of the mesh step is: the same measure on the same points.
moved, same = parle_nose.closed_and_dented(POINTS, TRIANGLES)
assert np.allclose(same[1]["centre"], nostril_maps(DATA, POINTS)[2][1]["centre"]), "the paint and the mesh step put the dent at different places"
print("ok")
