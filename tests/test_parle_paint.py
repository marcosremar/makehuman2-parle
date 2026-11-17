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
assert drawn[..., :3][lower].mean() < drawn[..., :3][upper].mean() * 0.93, "the old's eyebrow is one flat grey"

rng = np.random.default_rng(1)
skin = 0.6 + 0.1 * rng.standard_normal((64, 64, 3)).astype(np.float32)
smoothed = parle_paint.box_blur(skin, 3)
assert smoothed.shape == skin.shape and smoothed.std() < skin.std() / 3, "the skin keeps its grain"
assert abs(smoothed.mean() - skin.mean()) < 0.01, "the blur changes the skin's colour"

eyes = [(np.array([0.032, -0.06, 1.66]), 0.012), (np.array([-0.032, -0.06, 1.66]), 0.012)]
tip = np.array([0.0, -0.1, 1.62])
cheek = eyes[0][0] + np.array(parle_paint.CHEEK_SPOT_BY_KIND["male"]) * 0.064
forehead, back = np.array([0.0, -0.08, 1.75]), np.array([0.0, 0.08, 1.66])
share = parle_paint.blush_share(np.array([cheek, tip, forehead, back]), tip, eyes, "male")
assert share[0] > 0.8 * parle_paint.BLUSH_SHARE["male"] and share[1] > 0.6 * parle_paint.BLUSH_SHARE["male"], "no blush on the cheek or the nose"
assert share[2] < 0.02 and share[3] < 0.001, "the forehead or the back of the head blush"
assert parle_paint.BLUSH_SHARE["male-old"] > parle_paint.BLUSH_SHARE["male"], "the old blush less than the young"
# Round 83 (measured against the series): the women's nose tip is no redder than their cheek, the old women's blush under the old man's,
# the young man's spot lower and further out than the others', and a young person's eyebrow darker than its hair.
for kind in ("female", "female-old"):
    woman = parle_paint.blush_share(np.array([eyes[0][0] + np.array(parle_paint.CHEEK_SPOT) * 0.064, tip]), tip, eyes, kind)
    assert woman[1] < woman[0], f"the tip of a woman's nose is redder than her cheek: {kind} {woman}"
assert parle_paint.BLUSH_SHARE["female-old"] < 0.6 < parle_paint.BLUSH_SHARE["male-old"], "the old women blush as the old man"
assert parle_paint.CHEEK_SPOT_BY_KIND["male"][0] > parle_paint.CHEEK_SPOT[0] and parle_paint.CHEEK_SPOT_BY_KIND["male"][2] < parle_paint.CHEEK_SPOT[2], "the young man's blush is not lower and further out"
assert parle_paint.BROW_YOUNG_DARKEN <= 0.45, "a young person's eyebrow is as light as the hair"
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


# Round 66: an old woman with dark brown hair had a light grey eyebrow (both judges: "não combina com o cabelo escuro"): with dark
# hair the old's eyebrow stays near its own colour; with grey hair it goes to the mid grey.
brown = np.zeros((4, 4, 4), np.float32)
brown[:, :, :3], brown[:, :, 3] = (0.24, 0.16, 0.1), 1.0
dark_old = parle_paint.drawn_brow(brown, True)[0, 0, :3]
grey_old = parle_paint.drawn_brow(np.concatenate([np.full((4, 4, 3), 0.55, np.float32), np.ones((4, 4, 1), np.float32)], axis=2), True)[0, 0, :3]
assert np.abs(dark_old - np.array(parle_paint.BROW_OLD_GREY)).sum() > 0.3 and dark_old.mean() < 0.3, f"a dark-haired old person's eyebrow is grey: {dark_old}"
assert np.abs(grey_old - np.array(parle_paint.BROW_OLD_GREY)).max() < 0.1, f"a grey-haired old person's eyebrow is not grey: {grey_old}"


# Round 68: the kept beard's edge in steps (both judges): with its outline rounded, a toothed edge runs smooth.
teeth = np.full((64, 64, 3), 0.95, np.float32)
for column in range(64):
    teeth[32 + (4 if (column // 4) % 2 else 0):, column] = 0.3
rounded = parle_paint.kept_dark(teeth, face, 4) > 0.5
edge_rows = rounded.argmax(axis=0)[8:56]
assert np.abs(np.diff(edge_rows)).sum() < 0.5 * np.abs(np.diff((parle_paint.kept_dark(teeth, face) > 0.5).argmax(axis=0)[8:56])).sum(), "the beard's edge keeps its teeth"

# Round 73 (both judges, the four men: "a boca é só um corte, sem lábio"): a man's patch has a lower lip darker and more rose than
# his skin, lit in its middle, a darker line where the lips meet that fades before the corners, and no light rim.
across = np.linspace(0, 1, 41)[None, :]
up = np.linspace(1, 0, 21)[:, None]
man = parle_paint.man_lip_patch(across, up, face)
lower_lip, meet = man[15, 20], man[10, 20]
assert 0.8 < lower_lip.mean() / face.mean() < 0.95 and lower_lip[0] / lower_lip[1] > face[0] / face[1] * 1.1, f"a man's lower lip: {lower_lip} on {face}"
assert meet.mean() < 0.7 * lower_lip.mean() and man[10, 1].mean() > 0.85 * man[15, 1].mean(), "the line where the lips meet: not dark in the middle, or dark at the corner"
assert man.max() <= face.max() + 1e-6, "a man's lip is lighter than his skin somewhere"
# A woman's lip gives up its colour towards the corners ("faixa rosa larga de canto a canto"): the middle keeps the common share.
ends = parle_paint.lip_ends_share(across, parle_paint.LIP_PATCH_TO_SKIN)[0]
assert abs(ends[20] - parle_paint.LIP_PATCH_TO_SKIN) < 1e-6 and ends[0] > 0.99 and ends[-1] > 0.99 and np.all(np.diff(ends[20:]) >= -1e-9), f"a woman's lip towards the corners: {ends}"

# Round 75 (both judges: seen from the side "não há ângulo entre o queixo e o pescoço"): the floor under the chin is shaded, a
# warm shade; the cheek, the under side of the nose, the hollow under the lips and the nape are not.
unit = float(np.linalg.norm(eyes[0][0] - eyes[1][0]))
spots = tip + unit * np.array([(0.0, 0.8, -1.4), (0.5, 1.0, -0.6), (0.0, 0.3, -0.1), (0.0, 0.4, -0.75), (0.0, 3.0, -1.6), (0.0, 0.8, -1.4)])
facing = np.array([(0, 0, -1.0), (0, -1.0, 0), (0, 0, -1.0), (0, -0.6, -0.8), (0, 0.6, -0.8), (0, -1.0, 0)])
shaded = parle_paint.jaw_shade_share(spots, facing, tip, eyes)
assert shaded[0] > 0.99 and np.all(shaded[1:] < 0.01), f"what is shaded under the jaw: {shaded}"
tone = np.array(parle_paint.JAW_SHADE_TONE)
assert tone[0] > tone[1] > tone[2] and 0.5 < tone.mean() < 0.85 and 0 < parle_paint.JAW_SHADE <= 0.8, "the shade under the jaw is not a warm, light one"

# Round 76: a woman's eyebrow had its round head at the temple and its point by the nose (the kit's thin stroke bends down at
# the temple, and is tall there column by column). The mesh says which end is by the nose: with the head at the last column, the
# point is at the first.
uv_corners = np.array([(0.1, 0.5), (0.5, 0.52), (0.9, 0.5), (0.1, 0.1), (0.5, 0.12), (0.9, 0.1)])
heads = parle_paint.brow_heads(uv_corners, np.array([0.06, 0.04, 0.02, 0.02, 0.04, 0.06]))
assert len(heads) == 2 and {head[2] for head in heads} == {True, False}, f"the head of each eyebrow, from the mesh: {heads}"
turned = parle_paint.brow_band(strands, [(0.0, 1.0, False)])[0] > 0.5
turned_cols = np.where(turned.any(axis=0))[0]
turned_thick = turned[:, turned_cols].sum(axis=0)
assert turned_thick[:4].max() <= 0.4 * turned_thick[-30:].max() and turned_thick[-3:].max() < turned_thick[-20:-8].max(), "the head is not at the end the mesh gives"


# Round 76 (both judges, the four old: crow's feet, the fold from the nose to the mouth and, on the men, two lines on the
# forehead, drawn): the old have their lines and the young none; a line darkens the texels it runs over, most at its middle, and
# leaves the map 5 texels away.
assert parle_paint.age_lines(tip, eyes, "female") == [] and parle_paint.age_lines(tip, eyes, "male") == []
old_woman, old_man = parle_paint.age_lines(tip, eyes, "female-old"), parle_paint.age_lines(tip, eyes, "male-old")
assert len(old_woman) == 8 and len(old_man) == 10, f"the lines of age: {len(old_woman)} on an old woman, {len(old_man)} on an old man"
assert all(len(line) == parle_paint.AGE_POINTS for line in old_man), "a line of age is not laid at its points"
wobbly = np.column_stack([np.arange(9.0), np.array([0, 1, -1, 1, -1, 1, -1, 1, 0.0])])
calm = parle_paint.evened_line(wobbly)
assert np.abs(calm[1:-1, 1]).max() < 0.5 and np.allclose(calm[[0, -1]], wobbly[[0, -1]]), "a line of age keeps its wobble, or its ends moved"
corner = np.linalg.norm(np.asarray(old_woman[0][0]) - eyes[0][0]) / eyes[0][1]
assert 1.0 < corner < 1.35, f"the crow's feet do not start at the corner of the eye: {corner:.2f} radii from its middle"
feet = [np.linalg.norm(np.asarray(line[-1]) - np.asarray(line[0])) for line in old_woman[:3]]
assert all(0.004 < length < 0.012 for length in feet), f"the crow's feet are not some 6 mm long: {feet}"
plain = np.full((40, 80, 3), 0.8, np.float32)
lined_map = parle_paint.stroked(plain, [[(10.0, 20.0), (70.0, 20.0)]], np.array([0.5, 0.4, 0.4], np.float32))
assert lined_map[20, 40, 0] < 0.7 and lined_map[20, 40, 0] < lined_map[20, 13, 0] and np.allclose(lined_map[27], 0.8) and np.allclose(lined_map[20, 75:], 0.8), "a line of age is not drawn, or runs over"
