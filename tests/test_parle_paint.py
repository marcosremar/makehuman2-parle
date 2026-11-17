"""Regression for the paint of a drawn face (2026-11-16): run with `python tests/test_parle_paint.py` (no Blender).

Round 20 of the loop judged by sight against the owner's chosen series: smudged grey eyebrows, grained skin, no blush. The eyebrow
becomes a clean dark band (light grey on the old), the skin map loses its grain, and the cheeks and the nose tip take a warm blush
that the forehead and the back of the head do not."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import parle_paint  # noqa: E402

brow = np.zeros((4, 4, 4), np.float32)
brow[:, :, :3] = 0.5
brow[:, :, 3] = np.linspace(0.0, 1.0, 16).reshape(4, 4)
young, old = parle_paint.drawn_brow(brow, False), parle_paint.drawn_brow(brow, True)
assert young[..., :3].max() < 0.7 * 0.5, "the young's eyebrow is not darker than its own colour"
# Grey on the old: from round 49 to 54 both judges asked it "cinza médio em vez de branco", neither dark nor white.
assert 0.4 < old[..., :3].mean() and old[..., :3].max() < 0.7, "the old's eyebrow is not a mid grey"
alpha = young[..., 3].ravel()
assert ((alpha < 0.02) | (alpha > 0.98)).mean() >= 0.6, "the eyebrow's edge is still a soft smudge"

# Round 66 (owner, 2026-11-17: "melhora a sobrancelha"): the kit's strands have a lumpy edge that the thickening of the old made a
# slab with a wavy top, and blunt ends. Drawn again, an eyebrow's upper edge runs smooth, its head is round, its tail a point, and
# it stays where the kit's was; the old's grey is darker at its lower edge than at its upper.
rows, columns = 128, 256
ys, xs = np.mgrid[0:rows, 0:columns]
along = (xs - 20) / 200.0
middle = 64 - 18 * np.sin(np.pi * np.clip(along, 0, 1) * 0.8)
half = (9 - 7 * np.clip(along, 0, 1)) + 2.5 * np.sin(xs / 4.0) * (ys < middle)
strands = (np.abs(ys - middle) <= half) & (along >= 0) & (along <= 1)
lumpy = np.zeros((rows, columns, 4), np.float32)
lumpy[..., :3], lumpy[..., 3] = 0.3, strands
drawn = parle_paint.drawn_brow(lumpy, True)
solid = drawn[..., 3] > 0.5
cols = np.where(solid.any(axis=0))[0]
top = solid[:, cols].argmax(axis=0)
kit_top = strands[:, cols].argmax(axis=0)
body = slice(0, int(len(cols) * 0.8))


def lumps(edge):
    return np.std(edge[body] - np.polyval(np.polyfit(cols[body], edge[body], 4), cols[body]))


assert lumps(top) < lumps(kit_top) / 3, f"the eyebrow's upper edge keeps the strands' lumps: {lumps(top):.2f} against {lumps(kit_top):.2f}"
thick = solid[:, cols].sum(axis=0)
assert thick[:3].max() < thick[8:20].max() and thick[-4:].max() <= 0.3 * thick[:30].max(), f"no round head or no pointed tail: {thick[:12]} {thick[-6:]}"
grown = parle_paint.box_blur(strands.astype(np.float32)[..., None], 3, passes=1)[..., 0] > 0
assert not (solid & ~grown).any(), "the eyebrow is drawn off where the kit's was"
across = parle_paint.brow_band(strands)[1]
upper, lower = solid & (across < -0.5), solid & (across > 0.5)
assert drawn[..., :3][lower].mean() < drawn[..., :3][upper].mean() * 0.85, "the old's eyebrow is one flat grey"

rng = np.random.default_rng(1)
skin = 0.6 + 0.1 * rng.standard_normal((64, 64, 3)).astype(np.float32)
smoothed = parle_paint.box_blur(skin, 3)
assert smoothed.shape == skin.shape and smoothed.std() < skin.std() / 3, "the skin keeps its grain"
assert abs(smoothed.mean() - skin.mean()) < 0.01, "the blur changes the skin's colour"

eyes = [(np.array([0.032, -0.06, 1.66]), 0.012), (np.array([-0.032, -0.06, 1.66]), 0.012)]
tip = np.array([0.0, -0.1, 1.62])
cheek = eyes[0][0] + np.array(parle_paint.CHEEK_SPOT) * 0.064
forehead, back = np.array([0.0, -0.08, 1.75]), np.array([0.0, 0.08, 1.66])
share = parle_paint.blush_share(np.array([cheek, tip, forehead, back]), tip, eyes, "male")
assert share[0] > 0.8 * parle_paint.BLUSH_SHARE["male"] and share[1] > 0.6 * parle_paint.BLUSH_SHARE["male"], "no blush on the cheek or the nose"
assert share[2] < 0.02 and share[3] < 0.001, "the forehead or the back of the head blush"
assert parle_paint.BLUSH_SHARE["male-old"] > parle_paint.BLUSH_SHARE["male"], "the old blush less than the young"
red = parle_paint.blushed(np.full((1, 1, 3), 0.6, np.float32), np.ones((1, 1)))
assert red[0, 0, 0] > red[0, 0, 2], "the blush is not warm"
# The blur of the skin keeps to the covered map (round 34): the empty map next to an island does not darken its edge.
island = np.zeros((32, 32, 3), np.float32)
island[:, 16:] = 0.8
covered = np.zeros((32, 32), bool)
covered[:, 16:] = True
assert np.allclose(parle_paint.masked_blur(island, covered, 4)[:, 16:], 0.8, atol=1e-4), "the empty map bleeds into the skin's edge"
# The shaved beard of the kit's photograph goes to the skin (round 36): the chin and the jaw clear, the forehead and the nape do not.
chin, jaw = tip + np.array([0.0, 0.0, -0.09]), tip + np.array([0.05, 0.02, -0.08])
nape = np.array([0.0, 0.1, 1.5])
share = parle_paint.jaw_share(np.array([chin, jaw, forehead, nape]), tip, eyes)
assert share[0] > 0.9 * parle_paint.JAW_TO_SKIN and share[1] > 0.5 * parle_paint.JAW_TO_SKIN, "the beard's shade stays on the chin or the jaw"
assert share[2] == 0 and share[3] < 0.01, "the forehead or the nape go to the face's skin"
# The lash line at the rim of the lids goes to the skin too (round 40: a dark ring about the eye).
rim = eyes[0][0] + np.array([0.0, -0.4, 0.95]) * eyes[0][1]
assert parle_paint.orbit_share(np.array([rim]), eyes)[0] > 0.5 * parle_paint.ORBIT_TO_SKIN, "the rim of the lids keeps its dark line"
# The see-through texels of a hair map take the strands' colour (round 44: a pale frayed band along the hairline).
hair = np.zeros((16, 16, 4), np.float32)
hair[:, :, :3] = 1.0
hair[:, :8, :3] = (0.3, 0.2, 0.1)
hair[:, :8, 3] = 1.0
out = parle_paint.bled(hair, 8)
assert np.allclose(out[:, 8:12, :3], (0.3, 0.2, 0.1), atol=1e-4), "the hair's edge keeps its white"
assert np.array_equal(out[:, :, 3], hair[:, :, 3]) and np.allclose(out[:, :8], hair[:, :8]), "the solid strands or the alpha changed"
# A scalp painted with skin goes to the strands' colour (round 45: a pale band of skin along the hairline).
assert np.allclose(parle_paint.solid_mean(hair), (0.3, 0.2, 0.1)), "the strands' colour is not their solid texels'"
scalp = np.full((2, 2, 4), 1.0, np.float32)
scalp[0, :, :3] = 0.8
scalp[1, :, :3] = (0.32, 0.2, 0.12)
dyed = parle_paint.toward(scalp[:, :, :3], parle_paint.scalp_share(scalp, (0.3, 0.2, 0.1)), (0.3, 0.2, 0.1))
assert dyed[0, 0, 0] < 0.4, "the scalp keeps the skin's light colour"
assert np.allclose(dyed[1], scalp[1, :, :3]), "the strands painted on the scalp lose their grain"
# The face's patch goes to the body's tone down the neck (round 46: a seam under the jaw on all four): none on the cheek, whole
# on the neck.
neck_point = tip + np.array([0.0, 0.03, -2.8 * 0.064])
shares = parle_paint.neck_share(np.array([cheek, neck_point]), tip, eyes)
assert shares[0] == 0 and shares[1] == 1, "the face goes to the body's tone, or the neck keeps the face's"
# The old blush wide and soft on the cheeks with the tip of the nose the reddest (round 51: "dois discos vermelhos" on the old man).
old = parle_paint.blush_share(np.array([cheek, tip]), tip, eyes, "male-old")
assert old[1] > old[0], "the old man's cheek is redder than the tip of his nose"
wide_point = cheek + np.array([0.0, 0.0, -0.4 * 0.064])
assert parle_paint.blush_share(np.array([wide_point]), tip, eyes, "male-old")[0] / parle_paint.BLUSH_SHARE["male-old"] > parle_paint.blush_share(np.array([wide_point]), tip, eyes, "male")[0] / parle_paint.BLUSH_SHARE["male"], "the old man's flush is no wider than the young man's"
# A woman's photographed lip goes almost all the way to the skin about her lip patch (round 59: a pink stain about the mouth).
mouth_point = tip + np.array(parle_paint.MOUTH_SPOT) * 0.064
assert parle_paint.mouth_share(np.array([mouth_point]), tip, eyes, parle_paint.LIP_TO_SKIN_WOMAN)[0] > parle_paint.LIP_TO_SKIN, "a woman's photographed lip stays about her lips"
print("ok")

# Round 66 (owner, 2026-11-17: "verifica o que mais está feio"): a beard painted on the map is kept through the paints of the face
# (the bar's attendant's came out a smudge behind the jaw), a lip's red is not; and the head goes to one tone (the florist's face was
# a pale mask on a yellow head), its detail and its dark texels kept.
face = np.array([0.95, 0.8, 0.7], np.float32)
texels = np.array([[[0.3, 0.25, 0.2], [0.85, 0.45, 0.45], face]], np.float32)
kept = parle_paint.kept_dark(texels, face)[0]
assert kept[0] > 0.99 and kept[1] < 0.01 and kept[2] < 0.01, f"what is kept through the face's paints: {kept}"
head_map = np.zeros((64, 64, 3), np.float32)
head_map[:, :32], head_map[:, 32:] = face, [1.0, 0.75, 0.4]
head_map[30:34, 40:44] -= 0.1
head_map[:8, :8] = [0.3, 0.25, 0.2]
dark_map = parle_paint.kept_dark(head_map, face)
even = parle_paint.evened(head_map, np.ones((64, 64)), np.ones((64, 64), bool), dark_map, face, 6)
assert np.abs(even[20:44, 50:60] - face).max() < 0.2 * np.abs(head_map[20:44, 50:60] - face).max() + 1e-3, "the head is not one tone"
assert abs(float((even[31, 41] - even[31, 47]).mean()) - float((head_map[31, 41] - head_map[31, 47]).mean())) < 0.02, "the detail went"
assert np.allclose(even[:4, :4], head_map[:4, :4]), "a dark texel (a beard) was moved to the skin's tone"
