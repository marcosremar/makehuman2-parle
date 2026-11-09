"""Regression for the mouth that looked lit from inside (2026-11-08): run with `.local/venv/bin/python tests/test_parle_mouth.py`."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import parle_mouth as mouth  # noqa: E402

# Front incisors at y=-0.1, molars 50 mm behind.
teeth = np.array([[0, -0.1, 0], [0, -0.05, 0]], float)
shade = mouth.teeth_shade(teeth)
assert shade[0] == 1.0, f"the incisor must stay lit: {shade[0]}"
assert shade[1] <= mouth.TEETH_SHADES[-1] + 1e-6, f"the molars must be in the dark: {shade[1]}"

tongue = np.array([[0, -0.08, 0], [0, -0.04, 0], [0, -0.03, 0]], float)
tongue_light = mouth.tongue_shade(tongue)
assert tongue_light[0] == mouth.TONGUE_FRONT < 0.8, "the tip of the tongue is already behind the teeth"
assert tongue_light[0] > tongue_light[1] >= tongue_light[2] and abs(tongue_light[2] - mouth.TONGUE_DEEP) < 1e-6, tongue_light

uv = np.array([[[0, 0], [0.5, 0], [0, 1]]], float)
canvas = mouth.shade_map(16, 16, uv, np.array([[0.2, 0.2, 0.2]], float))
overlap = mouth.shade_map(16, 16, np.concatenate([uv, uv]), np.array([[0.2, 0.2, 0.2], [0.9, 0.9, 0.9]], float))
assert abs(np.nanmax(overlap) - 0.2) < 1e-6, "where UV islands overlap, the darkest triangle must win"
assert np.nanmax(canvas) == np.nanmin(canvas) and abs(np.nanmin(canvas) - 0.2) < 1e-6
assert np.isnan(canvas).any(), "the map must say where no triangle falls"
filled = mouth.grow(canvas, 40)
assert not np.isnan(filled).any() and abs(filled.max() - 0.2) < 1e-6, "islands must bleed their own shade past their edge"

pixels = np.full((16, 16, 4), 0.8, np.float32)
dark = mouth.darken(pixels, np.full((16, 16), 0.5, np.float32))
assert abs(dark[0, 0, 0] - 0.4) < 1e-6 and dark[0, 0, 3] == pixels[0, 0, 3], "alpha must not change"

# Painted black between the teeth (luma 33 of 255) keeps its value: deep shade used to take it to luma 9, a hole at the player's size.
black = np.dstack([np.full((4, 4), v, np.float32) for v in (51 / 255, 25 / 255, 25 / 255, 1.0)])
held = mouth.darken(black, np.full((4, 4), 0.2, np.float32))
assert abs((held[:, :, :3] @ mouth.LUMA).min() - (black[:, :, :3] @ mouth.LUMA).min()) < 1e-3, "a texel already under the floor must stay as it was"
pale = np.dstack([np.full((4, 4), v, np.float32) for v in (0.5, 0.5, 0.5, 1.0)])
assert (mouth.darken(pale, np.full((4, 4), 0.1, np.float32))[:, :, :3] @ mouth.LUMA).min() >= mouth.LUMA_FLOOR - 1e-3, "a texel went under the floor"

# The pink of a gum goes down with the map, wherever the point is; ivory does not.
gum = np.dstack([np.full((4, 4), v, np.float32) for v in (196 / 255, 116 / 255, 112 / 255, 1.0)])
ivory = np.dstack([np.full((4, 4), v, np.float32) for v in (245 / 255, 235 / 255, 214 / 255, 1.0)])
lit = np.ones((4, 4), np.float32)
assert (mouth.darken(gum, lit, gums=True)[:, :, :3] @ mouth.LUMA).max() < (gum[:, :, :3] @ mouth.LUMA).max() * 0.45, "the gum must go down"
assert abs(mouth.darken(ivory, lit, gums=True)[0, 0, 0] - 245 / 255) < 1e-3, "a tooth is not a gum"

# Shaded wet parts bottom out on the wine colour, not on black; a lit texel is left alone.
wet = mouth.darken(black, np.full((4, 4), 0.2, np.float32), wine=True)
assert abs(wet[0, 0, 0] - mouth.WINE[0]) < 1e-3, "a shaded dark texel must rise to the colour of a wet mouth"
assert abs(mouth.darken(black, np.ones((4, 4), np.float32), wine=True)[0, 0, 0] - 51 / 255) < 1e-3, "an unshaded texel must not change"

deep_tooth = mouth.darken(ivory, np.full((4, 4), 0.3, np.float32), gums=True)
assert deep_tooth[0, 0, 0] > deep_tooth[0, 0, 2], "a shaded tooth must stay warm, or it reads teal"

# The body's inside of the mouth: in the island and behind the teeth, not the cheek beside them or the lip in front.
inner_teeth = np.array([[-0.03, -0.12, 1.36], [0.03, -0.06, 1.40]], float)
inner_body = np.array([[0.0, -0.1, 1.38], [0.0, -0.13, 1.38], [0.0, -0.1, 1.5], [0.05, -0.1, 1.38], [0.0, -0.1, 1.38]], float)
inner_uv = np.array([[0.8, 0.9], [0.8, 0.9], [0.8, 0.9], [0.8, 0.9], [0.5, 0.5]], float)
# Two pieces of one island: the mouth (a strip of points joined front to back, of any depth) and the nape, not joined to it.
inner_box = np.array([[-0.03, -0.12, 1.36], [0.03, -0.06, 1.40]], float)
depths = [0.01 * step for step in range(1, 11)]
mouth_strip = [[0.0, inner_box[0, 1] + d, 1.38] for d in depths] + [[0.01, inner_box[0, 1] + d, 1.38] for d in depths]
points = np.array(mouth_strip + [[0.0, inner_box[0, 1] + 0.125, 1.38], [0.01, inner_box[0, 1] + 0.125, 1.38], [0.0, inner_box[0, 1] + 0.14, 1.38]], float)
strip = [[i, i + 1, 10 + i] for i in range(9)] + [[i + 1, 10 + i, 11 + i] for i in range(9)]
nape_triangle = [[20, 21, 22]]
all_triangles = np.array(strip + nape_triangle)
piece_uv = np.full((len(points), 2), (0.8, 0.9), float)
inside_of = mouth.body_inside(points, piece_uv, inner_box, all_triangles)
assert inside_of[:20].all() and not inside_of[20:].any(), "the mouth is its own piece of the island: all of it, of any depth, and not the nape"
assert not mouth.body_inside(points, np.full((len(points), 2), (0.5, 0.5), float), inner_box, all_triangles).any(), "outside the island nothing is the inside of the mouth"
assert abs(mouth.grow(np.full((3, 3), np.nan, np.float32), 1, fill=1.0)[0, 0] - 1.0) < 1e-6, "an uncovered map must stay as it was"
tri_inside = np.array([True, False, False, False, False])
tri_uv = np.array([[0.8, 0.9], [0.8, 0.9], [0.8, 0.9], [0.8, 0.9], [0.5, 0.5]], float)
assert mouth.body_triangles(np.array([[0, 1, 2], [0, 1, 4], [1, 2, 3]]), tri_inside, tri_uv).tolist() == [True, False, False], "any corner inside takes the shade; a corner off the island does not"
assert abs(mouth.grow(np.full((3, 3), np.nan, np.float32), 1, fill=1.0)[0, 0] - 1.0) < 1e-6, "an uncovered map must stay as it was"

repo_map = Path(__file__).parent.parent / "data/eyes/hm08/materials/parle_teeth.png"
assert Path(mouth.clean_teeth(repo_map)).name == "parle_teeth_clean.png", "the shade must start from the clean map"

print("ok")
