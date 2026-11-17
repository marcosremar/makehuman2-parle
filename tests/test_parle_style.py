"""Regression for the style by kind of person (2026-11-15): run with `python tests/test_parle_style.py` (no Blender).

The owner asked for the faces of one animated series, by kind of person. The nose of each kind grows about a middle behind its tip,
across more than to the front, and the upper lip and the cheeks stay; every kind has its numbers; a sex with no style is refused."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import parle_style  # noqa: E402

assert set(parle_style.STYLE) == {"male", "male-old", "female", "female-old"}, "a kind of person has no style"
assert parle_style.kind_of("male", {"age": "old"}) == "male-old" and parle_style.kind_of("female", {}) == "female"
try:
    parle_style.kind_of("dog", {})
    raise AssertionError("a body that is neither a woman nor a man gets a style")
except ValueError:
    pass
men, old = parle_style.STYLE["male"], parle_style.STYLE["male-old"]
assert men["nose"] > parle_style.STYLE["female"]["nose"] and old["nose"] > men["nose"], "the series' men have the bigger nose, the old the biggest"
assert old["brow"] > men["brow"] > parle_style.STYLE["female"]["brow"], "the eyebrows of the kinds are not in the series' order"
# Round 66 (owner, 2026-11-17: "melhora a sobrancelha"): the old's eyebrows were a block (2.5 times the kit's, as thick at the
# temple as by the nose): under twice the kit's, and thinning to the temple.
for kind in ("male-old", "female-old"):
    assert parle_style.STYLE[kind]["brow"] < 2 and parle_style.STYLE[kind]["brow_outer"] < 0.8, f"the old's eyebrow is a block: {kind}"

tip = np.array([0.0, -0.1, 1.6])
field = parle_style.nose_field(tip, 1.6, 0.9)
side = np.array([[0.01, -0.09, 1.6]])
assert field(side)[0, 0] > side[0, 0] * 1.2, "the nose does not grow across"
assert abs(field(tip[None])[0, 1] - tip[1]) < abs(field(side)[0, 0] - side[0, 0]), "the nose comes out more than it widens"
lip, cheek = np.array([[0.0, -0.095, 1.6 - 0.02]]), np.array([[0.045, -0.07, 1.6]])
assert np.allclose(field(lip), lip, atol=2e-4) and np.allclose(field(cheek), cheek), "the upper lip or the cheek moves with the nose"
grey = np.array([[0.6, 0.5, 0.45]])
assert np.ptp(parle_style.warmer(grey, 1.3)) > np.ptp(grey), "the skin is not more saturated"
# The men's nose was pressed flat (owner, 2026-11-16, "o nariz dele está amassado"): it grows to the front more than across, and the
# ball of the sculpture only fills out, leaving the bridge above it where it was and bringing the tip forward.
for kind in ("male", "male-old"):
    assert parle_style.STYLE[kind]["forward"] > 1.0, f"the nose of {kind} is pushed back into the face"
    across, back, _ = parle_style.SCULPT[kind]["nose_ball"]
    assert back >= across, f"the nose ball of {kind} is wider than it is deep: a flat nose"
eyes = [(np.array([0.032, -0.06, 1.66]), 0.012), (np.array([-0.032, -0.06, 1.66]), 0.012)]
sculpt = parle_style.sculpt_field(tip, eyes, "male")
bridge = np.array([[0.0, -0.085, 1.628]])
assert sculpt(tip[None])[0, 1] < tip[1] - 0.001, "the tip of the nose does not come out into its ball"
assert np.allclose(sculpt(bridge), bridge), "the ball of the nose pulls the bridge in"
# Simple volumes (round 27): a bump on a smoothed patch goes, the volume stays, and the rims of the eyes and the lips are left.
grid = np.stack(np.meshgrid(np.arange(9.0), np.arange(9.0)), -1).reshape(-1, 2)
flat = np.c_[grid, np.zeros(len(grid))]
flat[40, 2] = 1.0
links = [(i, i + 1) for i in range(81) if i % 9 < 8] + [(i, i + 9) for i in range(72)]
done = parle_style.smoothed(flat, np.array(links), np.ones(81), rounds=10)
assert done[40, 2] < 0.4 and abs(done[:, 2].sum() - 1.0) < 0.5, "the bump is not smoothed out, or the volume went"
rim = eyes[0][0] + np.array([0.0, -0.012, 0.0])
lips = tip + np.array([0.0, 0.006, -0.04])
neck = tip + np.array([0.0, 0.06, -0.12])
weight = parle_style.simple_weight(np.array([rim, lips, neck]), tip, eyes)
assert weight[0] < 0.01 and weight[1] < 0.01 and weight[2] > 0.9, f"the smoothing takes the eye or the lips, or leaves the neck: {weight}"
# The lower face shorter (round 31): the chin comes up most, the mouth part of the way, and nothing above the nose tip nor far down
# the neck moves.
short = parle_style.lower_face_field(tip, eyes, "male")
probe = np.array([[0.0, -0.09, tip[2] - 0.083], [0.0, -0.09, tip[2] - 0.04], [0.0, -0.1, tip[2] + 0.01], [0.0, -0.05, 1.4]])
up = (short(probe) - probe)[:, 2]
assert up[0] > up[1] > 0.002 and abs(up[2]) < 1e-9 and abs(up[3]) < 1e-9, f"the lower face is not shortened as it should be: {up}"
# Worn meshes follow the skin (rounds 36 to 43: hairline frayed, earrings and glasses off the face, a torn collar): a point on the
# skin's moved patch goes with it, one far from it stays.
skin_before = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.5, 0.5, 0.5], [0.51, 0.5, 0.5]])
skin_after = skin_before + np.array([[0.0, -0.005, 0.0]] * 3 + [[0.0, 0.0, 0.0]] * 2)
follow = parle_style.follow_field(skin_before, skin_after)(np.array([[0.005, 0.005, 0.01], [0.3, 0.3, 0.3]]))
assert abs(follow[0, 1] - 0.0) < 1e-4, "the hair over the moved skin stays where it was"
assert np.allclose(follow[1], [0.3, 0.3, 0.3]), "a point far from the moved skin moves"
# The lid lifted by the blink run backwards no longer folds at the edge of the blink (rounds 24 to 46: a pale arc over each
# eyebrow): along a chain, the abrupt edge of a move becomes a slope, the pinned rim kept.
chain = np.zeros((12, 3))
chain[:4, 2] = 0.004
edges_chain = np.array([[i, i + 1] for i in range(11)])
pin = np.zeros(12, bool)
pin[0] = True
evened = parle_style.spread(chain, edges_chain, pin, np.arange(12) < 10, 200)
assert evened[0, 2] == 0.004, "the lid's rim moved"
assert np.abs(np.diff(evened[:, 2])).max() < 0.5 * np.abs(np.diff(chain[:, 2])).max(), "the edge of the lift stays a fold"
assert evened[11, 2] == 0.0, "a point past the reach moved"
# The skin that stayed counts too (round 47: an eyebrow went down with the lid under it): a point next to skin that did not move
# stays, even with moved skin a centimetre off.
lid_before = np.array([[0.0, 0.0, 0.0], [0.002, 0.0, 0.0], [0.0, 0.0, 0.012], [0.002, 0.0, 0.012]])
lid_after = lid_before + np.array([[0.0, 0.0, -0.003]] * 2 + [[0.0, 0.0, 0.0]] * 2)
brow = parle_style.follow_field(lid_before, lid_after)(np.array([[0.001, -0.002, 0.013]]))
assert abs(brow[0, 2] - 0.013) < 0.0005, "the eyebrow goes down with the lid under the skin it sits on"
# The old open their eyes less than the young (round 47: white above the iris of the old woman and the old man).
assert parle_style.EYE_TALL_LEAST_BY_KIND["female-old"] < parle_style.EYE_TALL_LEAST_BY_KIND["female"], "the old open as wide as the young"
assert parle_style.EYE_TALL_LEAST_BY_KIND["male-old"] < parle_style.EYE_TALL_LEAST_BY_KIND["male"], "the old open as wide as the young"
# The hollow between the lid and the eyebrow fills (rounds 38 to 48: "cavidade 40 a 50 % mais rasa"): along a chain, a point
# behind its neighbours comes forward (-y), a point in front of them stays.
row = np.zeros((9, 3))
row[:, 0] = np.arange(9) * 0.002
row[4, 1] = 0.003
row[7, 1] = -0.002
edges_row = np.array([[i, i + 1] for i in range(8)])
filled_row = parle_style.hollows_filled(row, edges_row, np.ones(9))
assert filled_row[4, 1] < 0.5 * row[4, 1], "the hollow keeps its depth"
assert filled_row[7, 1] == row[7, 1], "a ridge went back"
# The smoothing comes in up to the lips (rounds 48 to 50: an unsmoothed oval 80 mm wide about the mouth stood out as a muzzle with a
# crease round it): the cheek beside the corner of the mouth is smoothed, the lips are not.
eyes_m = [(np.array([0.032, -0.06, 1.66]), 0.012), (np.array([-0.032, -0.06, 1.66]), 0.012)]
tip_m = np.array([0.0, -0.1, 1.62])
mouth_m = tip_m + np.array(parle_style.SIMPLE_MOUTH) * 0.064
beside = mouth_m + np.array([0.7 * 0.064, 0.0, 0.0])
weights = parle_style.simple_weight(np.array([mouth_m, beside]), tip_m, eyes_m)
assert weights[0] == 0 and weights[1] > 0.18, f"the lips are smoothed, or the cheek beside the mouth is not: {weights}"
# A frame on the eyes comes in to the face at the temples (round 51: the old man's glasses stood out past his face in 3/4).
face_pts = np.array([[x, y, 1.60] for x in np.linspace(-0.065, 0.065, 27) for y in (-0.09, -0.07, -0.05)])
frame_pts = np.array([[x, -0.1, 1.6] for x in np.linspace(-0.08, 0.08, 17)] + [[0.08, -0.1 + d, 1.6] for d in (0.02, 0.05)])
scale_f = parle_style.frame_scale(frame_pts, face_pts, 1.6)
assert abs(scale_f - (0.065 + parle_style.FRAME_CLEAR_M) / 0.08) < 1e-6, f"the frame is not narrowed to the face: {scale_f}"
assert parle_style.frame_scale(frame_pts * np.array([0.5, 1, 1]), face_pts, 1.6) == 1.0, "a frame narrower than the face was narrowed"
# Glasses on top of the head rest on the hair (round 66: the young woman's stood 1.4 to 4 cm over it): a ring over a dome of hair
# comes down to it, its nearest point FRAME_ON_HAIR_M off, none nearer; one already resting stays.
theta, phi = np.meshgrid(np.linspace(0, np.pi / 2, 30), np.linspace(0, 2 * np.pi, 60))
dome = 0.1 * np.column_stack([(np.sin(theta) * np.cos(phi)).ravel(), (np.sin(theta) * np.sin(phi)).ravel(), np.cos(theta).ravel()])
angle = np.linspace(0, 2 * np.pi, 40)
ring = np.column_stack([0.07 * np.cos(angle), 0.07 * np.sin(angle), np.full(40, 0.11)])
lifted = ring + parle_style.settle_on(ring, dome)
gap = parle_style.nearest_on(lifted, dome)[0]
assert abs(gap.min() - parle_style.FRAME_ON_HAIR_M) < 0.001 and gap.min() >= parle_style.FRAME_ON_HAIR_M - 1e-6, f"the glasses float or sink: {gap.min()}"
assert lifted[:, 2].mean() < ring[:, 2].mean() - 0.02, "the glasses did not come down onto the hair"
assert np.allclose(parle_style.settle_on(lifted, dome), 0, atol=1e-3), "glasses already resting on the hair moved"
# The lower edge of short hair lies on the skin (round 66: a cap 7 mm off the young man's neck, a helmet's rim): a strip of hair
# whose lower row goes onto the skin moves the row over it less, and the rows further than HAIR_TUCK_FADE_M not at all.
rows_z = np.arange(0.0, 0.06, 0.005)
strip = np.array([(x, 0.0, z) for z in rows_z for x in np.arange(0.0, 0.05, 0.005)])
across = 10
links = [(r * across + c, r * across + c + 1) for r in range(len(rows_z)) for c in range(across - 1)]
links += [(r * across + c, (r + 1) * across + c) for r in range(len(rows_z) - 1) for c in range(across)]
lower = np.arange(across)
tuck = parle_style.tucked_edge(strip, np.array(links), lower, strip[lower] + np.array([0.0, 0.007, 0.0]))
by_row = tuck[:, 1].reshape(len(rows_z), across).mean(axis=1)
assert np.allclose(by_row[0], 0.007) and np.all(np.diff(by_row) <= 1e-9), f"the hair does not follow its edge evenly: {by_row}"
assert np.allclose(by_row[rows_z > parle_style.HAIR_TUCK_FADE_M + 1e-9], 0), f"the top of the hair moved with its edge: {by_row}"
assert 0 < by_row[2] < 0.007, f"the hair next to the edge did not follow it: {by_row}"
# An eyebrow follows the skin the same across it (round 66: the old man's lower edge went down with his lowered lids and its upper
# edge stayed, a tuft 3 cm tall by the nose): its thickness is kept, its bend along it followed.
brow_pts = np.array([(sign * x, 0.0, 1.6 + 0.004 * edge) for sign in (-1, 1) for x in np.linspace(0.01, 0.05, 9) for edge in (-1, 1)])
stretch = np.where(brow_pts[:, 2:] < 1.6, [0.0, 0.0, -0.003], 0.0) + np.column_stack([np.zeros(len(brow_pts)), np.zeros(len(brow_pts)), 0.02 * (np.abs(brow_pts[:, 0]) - 0.03) ** 2])
kept = brow_pts + parle_style.along_only(brow_pts, stretch)
tall = [np.ptp(kept[np.isclose(brow_pts[:, 0], x), 2]) for x in np.unique(brow_pts[:, 0])]
assert np.allclose(tall, 0.008, atol=1e-6), f"the eyebrow's thickness went with the skin: {tall}"
bent = 0.02 * (np.abs(brow_pts[:, 0]) - 0.03) ** 2
assert np.allclose(parle_style.along_only(brow_pts, np.column_stack([0 * bent, 0 * bent, bent]))[:, 2], bent, atol=1e-9), "the bend is not followed"
# The folds about the mouth fill (rounds 48 to 54): the skin in a ring about the lips takes part, the lips and the cheekbone do not.
eyes_f = [(np.array([0.032, -0.06, 1.66]), 0.012), (np.array([-0.032, -0.06, 1.66]), 0.012)]
tip_f = np.array([0.0, -0.1, 1.62])
mouth_f = tip_f + np.array(parle_style.SIMPLE_MOUTH) * 0.064
fold_f = mouth_f + np.array([0.6 * 0.064, 0.0, 0.1 * 0.064])
cheekbone_f = eyes_f[0][0] + np.array([0.0, -0.01, -0.02])
weights_f = parle_style.fold_weight(np.array([mouth_f, fold_f, cheekbone_f]), tip_f, eyes_f)
assert weights_f[0] == 0 and weights_f[1] > 0.5 and weights_f[2] == 0, f"the folds about the mouth: {weights_f}"
# The lower part of the lower lip is smoothed with the chin (round 63: "um vinco duro e reto logo abaixo do lábio inferior").
under_lip = mouth_m + np.array([0.0, 0.0, -0.15 * 0.064])
assert parle_style.simple_weight(np.array([under_lip]), tip_m, eyes_m)[0] > 0.02, "the smoothing stops at the lower lip's edge"
print("ok")
# A cap sits on the head (round 66: the ice-cream girl's stood 13 to 43 mm off her forehead): a rim 2 cm out on a head of 8 cm comes
# in to HAT_CLEAR_M off it; a rim already on the head stays.
around = np.linspace(0, 2 * np.pi, 40, endpoint=False)
rim_pts = np.column_stack([0.1 * np.cos(around), 0.1 * np.sin(around), np.full(40, 1.5)])
narrowed = parle_style.hat_scale(rim_pts, np.full(40, 0.02), (0.0, 0.0))
assert abs(narrowed - (0.08 + parle_style.HAT_CLEAR_M) / 0.1) < 1e-6, f"the cap is not brought onto the head: {narrowed}"
assert parle_style.hat_scale(rim_pts, np.full(40, 0.002), (0.0, 0.0)) == 1.0, "a cap on the head was narrowed"

assert parle_style.HAIR_TUCK_ABOVE_M < 0, "the hair's edge at the temples goes onto the skin (round 66: a hole at each temple)"
assert parle_style.STYLE["male-old"]["brow_inner"] == 0.0, "the old man's eyebrows go up at the inner end (round 66: a sad roof)"

# The arms of a frame lie along the head (round 68: the market owner's stood 11 to 24 mm off it): an arm running straight back
# 15 mm off the side of a head comes to FRAME_ON_HAIR_M of it, slice by slice and whole; the front of the frame stays.
side_head = np.array([(0.07, y, z) for y in np.linspace(-0.1, 0.05, 60) for z in np.linspace(-0.02, 0.02, 9)])
lens_pts = np.array([(x, -0.11, z) for x in np.linspace(-0.06, 0.06, 13) for z in (-0.01, 0.0, 0.01)])
arm_pts = np.array([(0.085, y, z) for y in np.linspace(-0.105, 0.03, 28) for z in (-0.001, 0.001)])
glasses_pts = np.concatenate([lens_pts, arm_pts])
laid = glasses_pts + parle_style.arms_on_head(glasses_pts, side_head)
assert np.allclose(laid[:len(lens_pts)], lens_pts), "the front of the frame moved"
far_back = laid[len(lens_pts):][arm_pts[:, 1] > -0.07]
gaps = parle_style.nearest_on(far_back, side_head)[0]
assert np.all(gaps >= parle_style.FRAME_ON_HAIR_M - 1e-4) and gaps.max() < parle_style.FRAME_ON_HAIR_M + 0.002, f"the arm is not on the head: {gaps.max()}"
assert np.allclose(np.ptp(laid[len(lens_pts):].reshape(-1, 2, 3)[:, :, 2], axis=1), 0.002), "an arm's slice lost its thickness"

# Round 69: the arms came out in a zigzag (judge A): on a head whose side bulges and hollows, the laid arm runs smooth (no slice
# turns back) and no slice goes nearer than FRAME_ON_HAIR_M.
bumpy_head = np.array([(0.07 + 0.006 * np.sin(y * 300), y, z) for y in np.linspace(-0.1, 0.05, 120) for z in np.linspace(-0.02, 0.02, 9)])
bent = glasses_pts + parle_style.arms_on_head(glasses_pts, bumpy_head)
arm_x = bent[len(lens_pts):].reshape(-1, 2, 3)[:, 0, 0]
steps_x = np.diff(arm_x[arm_pts[::2, 1] > -0.085])
turns = np.diff(np.sign(steps_x[np.abs(steps_x) > 0.0002]))
assert np.count_nonzero(turns) <= 2, f"the arm zigzags: {np.round(arm_x * 1000, 1)}"
assert parle_style.nearest_on(bent[len(lens_pts):], bumpy_head)[0].min() >= parle_style.FRAME_ON_HAIR_M - 1e-4, "an arm went into the head"
assert np.all(parle_style.under_smooth(np.array([5.0, 1.0, 5.0, 5.0])) <= np.array([5.0, 1.0, 5.0, 5.0]) + 1e-12), "the curve passes a gap"
# Round 71: the smile and the "o" creased the skin at the corners of the mouth (both judges, all eight): along a chain, a move
# that stops abruptly becomes a slope (its steepest step under half of what it was), and most of the move stays.
chain = np.array([(index, index + 1) for index in range(19)])
abrupt = np.zeros((20, 3))
abrupt[:10, 2] = 0.004
opened = parle_style.creases_out(abrupt, chain)
assert np.abs(np.diff(opened[:, 2])).max() < 0.5 * 0.004, f"the crease is still a step: {np.abs(np.diff(opened[:, 2])).max()}"
assert opened[:5, 2].min() > 0.9 * 0.004 and np.allclose(opened[15:], 0, atol=2e-4), f"the shape went with its crease: {opened[:, 2]}"
# Round 74: the folds under the jaw (both judges, all eight: "quina dura da mandíbula até o pescoço", "vinco sob o queixo"). The
# skin under the chin and down the front of the throat is smoothed again; the lips, the cheeks, the collar and the nape are left.
unit = 0.064
spots = tip + unit * np.array([(0.0, 0.5, -1.5), (0.5, 0.9, -1.8), (0.0, 0.1, -0.62), (0.6, 0.6, -0.3), (0.0, 0.8, -3.0), (0.0, 2.6, -1.6)])
under = parle_style.jaw_neck_weight(spots, tip, eyes)
assert under[0] > 0.99 and under[1] > 0.9, f"under the chin and the side of the throat are not smoothed: {under}"
assert np.all(under[2:] < 0.01), f"the lips, the cheek, the collar or the nape are smoothed with the jaw: {under}"
fold = np.array([(x, 0.0, abs(x) * 0.6) for x in np.linspace(-0.02, 0.02, 21)])
eased = parle_style.smoothed(fold, np.array([(index, index + 1) for index in range(20)]), np.r_[0.0, np.ones(19), 0.0], parle_style.JAW_NECK_ROUNDS)
assert eased[10, 2] > 0.0005 and np.allclose(eased[[0, 20]], fold[[0, 20]]), f"a fold is not opened, or its ends moved: {eased[10]}"

# Round 78 (owner, 2026-11-17: "deve melhorar o maxilar"): the women's faces measured wider than tall, the jaw narrow against the
# cheekbones. Their cheeks are narrower across than the men's, and their face is shortened less under the nose.
for old_or_young in ("", "-old"):
    assert parle_style.SCULPT["female" + old_or_young]["cheek_wide"] < 0.9 and "cheek_wide" not in parle_style.SCULPT["male" + old_or_young], "the women's cheeks are as wide as the men's"
    assert parle_style.LOWER_FACE["female" + old_or_young] > parle_style.LOWER_FACE["male" + old_or_young] + 0.05, "the women's face is as short under the nose as the men's"

# The owner, 2026-11-17: "o maxilar do homem e da mulher ainda está com falhas". (1) A ball's rim: a flat sheet through a ball is
# pushed out inside it; with the ease, the sheet leaves the ball with no crease (the push next to the rim is a small share of
# what it was). (2) The cheeks' and the chin's balls are no longer cut at a plane behind the nose: two points 2 mm apart across
# that plane, both inside the cheek's ball, go out by nearly the same.
sheet = np.array([(x, 0.03, 0.0) for x in np.linspace(-0.06, 0.06, 121)])
ball = (np.zeros(3), (0.05, 0.05, 0.05))
hard = parle_style.onto_ball(sheet, ball[0], ball[1], 0.9, True) - sheet
eased = parle_style.onto_ball(sheet, ball[0], ball[1], 0.9, True, parle_style.BALL_EASE) - sheet
rim = np.abs(np.abs(sheet[:, 0]) - 0.04) < 0.0021
inside = np.abs(sheet[:, 0]) < 0.01
assert np.linalg.norm(eased[rim], axis=1).max() < 0.15 * np.linalg.norm(hard[rim], axis=1).max(), "the skin still turns at a crease at the rim of a ball"
assert np.allclose(eased[inside], hard[inside]), "the ease changed the middle of the ball"
sculpt = parle_style.sculpt_field(tip, eyes, "male")
unit_s = float(np.linalg.norm(eyes[0][0] - eyes[1][0]))
cheek_at = eyes[0][0] + np.array([parle_style.CHEEK_AT[0], parle_style.CHEEK_AT[1], parle_style.CHEEK_AT[2]]) * unit_s
across = np.array([(cheek_at[0] + 0.3 * unit_s, tip[1] + 0.8 * unit_s + side * 0.001, cheek_at[2]) for side in (-1, 1)])
steps = np.linalg.norm(sculpt(across) - across, axis=1)
assert steps.min() > 0.001 and steps.min() > 0.5 * steps.max(), f"a step across a plane behind the nose: {steps}"
# Round 80: the balls leave the angle of the jaw and the floor under the jaw alone (a jowl on the women, a notch in the line of
# the men's jaw where the chin's ball ended): a point inside the cheek's ball 1.3 distances between the eyes behind the nose tip,
# and one inside the chin's ball 0.9 behind it, stay.
jaw_angle = np.array([(cheek_at[0] + 0.2 * unit_s, tip[1] + 1.3 * unit_s, cheek_at[2] - 0.2 * unit_s)])
chin_at = tip + np.array(parle_style.CHIN_AT) * unit_s
under_jaw = np.array([(0.2 * unit_s, tip[1] + 0.9 * unit_s, chin_at[2] - 0.1 * unit_s)])
assert np.allclose(sculpt(jaw_angle), jaw_angle, atol=1e-6) and np.allclose(sculpt(under_jaw), under_jaw, atol=1e-6), "a ball pushes the angle of the jaw or the floor under it"
