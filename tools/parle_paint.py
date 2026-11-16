"""
    License information: data/licenses/makehuman_license.txt

    The paint of a drawn face (owner, 2026-11-15 and 16: the faces "idênticos" to those of one animated series, "iluminação e cores e
    tudo", and the loop of marks run until 9 in 10). Round 20 of the loop, judged by sight against the series (3.9 in 10), named as
    common to the four people: eyebrows a smudge of grey with soft edges where the series' are one dark band with a clean edge; the
    skin grained where the series' is smooth modelling clay; no blush, where the series' cheeks and nose tip are warm red-orange.

    The numbers here are pure (no bpy) and tested; the steps that write a map take the map's pixels as an array.
"""
import numpy as np

# An eyebrow's edge: the alpha of its strands drawn as a clean band, from soft to whole between these.
BROW_EDGE = (0.18, 0.5)
# How dark a young person's eyebrow is drawn, of its own colour; an old person's goes this share to a light grey instead (the series'
# old man has white-grey brows with a clean outline).
# Round 21 (4.0): the young's still read as a grey smudge at 0.45 of their own colour, the old's dark grey: both now go most of the
# way to a colour of their own.
BROW_DARK = (0.09, 0.06, 0.045)
BROW_DARK_SHARE = 0.8
BROW_OLD_GREY = (0.9, 0.88, 0.84)
BROW_OLD_SHARE = 0.85
# Box blur of the skin map, in pixels of a 1024 map (scaled to the map), three passes: the pores and grain of the kit's photograph go.
SKIN_BLUR_PX = 5
# The blush: a warm red-orange (sRGB) laid over the skin on each cheek and on the nose tip, most at the middle, fading with a
# gaussian of the given width in distances between the eyes. The old take more, as the series' old man.
BLUSH = (0.86, 0.38, 0.28)
BLUSH_SHARE = {"female": 0.4, "female-old": 0.45, "male": 0.36, "male-old": 0.6}
CHEEK_SPOT = (0.12, -0.25, -0.72)
CHEEK_WIDE = 0.34
NOSE_SPOT_WIDE = 0.22
NOSE_SHARE = 0.8
# The lips: the series draws no lip, the mouth is a cut with curled corners (round 21: "a série não desenha lábio"). The skin about
# the mouth goes this share of the way to the face's own skin colour, so the painted lip of the kit's photograph goes.
MOUTH_SPOT = (0.0, -0.05, -0.62)
MOUTH_WIDE = (0.42, 0.16)
LIP_TO_SKIN = 0.8
# A woman's lips are drawn on a patch of their own (`parle_lips`): it goes this share of the way to the skin at their corners.
LIP_PATCH_TO_SKIN = 0.7


def smooth(low, high, value):
    t = np.clip((np.asarray(value, float) - low) / (high - low), 0, 1)
    return t * t * (3 - 2 * t)


def drawn_brow(pixels, old):
    """An eyebrow map (rows x columns x RGBA, 0..1, sRGB) drawn as one clean band: dark on the young, light grey on the old."""
    out = np.array(pixels, np.float32, copy=True)
    out[:, :, 3] = smooth(BROW_EDGE[0], BROW_EDGE[1], out[:, :, 3])
    if old:
        out[:, :, :3] = out[:, :, :3] * (1 - BROW_OLD_SHARE) + np.asarray(BROW_OLD_GREY, np.float32) * BROW_OLD_SHARE
    else:
        out[:, :, :3] = out[:, :, :3] * (1 - BROW_DARK_SHARE) + np.asarray(BROW_DARK, np.float32) * BROW_DARK_SHARE
    return out


def box_blur(image, radius, passes=3):
    """The image (rows x columns x channels) blurred by `passes` box filters of `radius` pixels, edges held."""
    out = np.asarray(image, np.float32)
    if radius < 1:
        return out
    for _ in range(passes):
        for axis in (0, 1):
            padded = np.concatenate([np.repeat(out.take([0], axis), radius + 1, axis), out, np.repeat(out.take([-1], axis), radius, axis)], axis)
            total = np.cumsum(padded, axis=axis)
            size = out.shape[axis]
            out = (total.take(range(2 * radius + 1, 2 * radius + 1 + size), axis) - total.take(range(0, size), axis)) / (2 * radius + 1)
    return out


def blush_share(points, tip, eyes, kind):
    """How much blush each point takes (0..1 of BLUSH_SHARE): a spot on each cheek under its eye and one on the nose tip. Blender's
    axes, the face looking at -y."""
    points = np.asarray(points, float)
    centres = [np.asarray(centre, float) for centre, _ in eyes]
    unit = float(np.linalg.norm(centres[0] - centres[1]))
    middle_x = (centres[0][0] + centres[1][0]) / 2
    share = np.zeros(len(points))
    for centre in centres:
        side = np.sign(centre[0] - middle_x)
        at = centre + np.array([side * CHEEK_SPOT[0], CHEEK_SPOT[1], CHEEK_SPOT[2]]) * unit
        far = np.linalg.norm(points - at, axis=1) / (CHEEK_WIDE * unit)
        share = np.maximum(share, np.exp(-far * far))
    far = np.linalg.norm(points - np.asarray(tip, float), axis=1) / (NOSE_SPOT_WIDE * unit)
    share = np.maximum(share, NOSE_SHARE * np.exp(-far * far))
    # Only the front of the face.
    share *= smooth(-0.2, 0.2, (np.asarray(tip, float)[1] + 1.2 * unit - points[:, 1]) / unit)
    return share * BLUSH_SHARE[kind]


def mouth_share(points, tip, eyes):
    """How much of its colour each point about the mouth gives up to the skin's (0..LIP_TO_SKIN): an oval across the mouth."""
    points = np.asarray(points, float)
    centres = [np.asarray(centre, float) for centre, _ in eyes]
    unit = float(np.linalg.norm(centres[0] - centres[1]))
    at = np.asarray(tip, float) + np.array(MOUTH_SPOT) * unit
    off = (points - at) / unit
    far = np.hypot(off[:, 0] / MOUTH_WIDE[0], off[:, 2] / MOUTH_WIDE[1])
    front = points[:, 1] < np.asarray(tip, float)[1] + 0.6 * unit
    return np.where(front, LIP_TO_SKIN * smooth(1.3, 0.8, far), 0.0)


def toward(pixels, canvas, colour):
    """The map's RGB with `colour` laid over by `canvas` (0..1, NaN = none)."""
    weight = np.nan_to_num(np.asarray(canvas, np.float32))[:, :, None]
    return np.asarray(pixels, np.float32) * (1 - weight) + np.asarray(colour, np.float32) * weight


def blushed(pixels, canvas):
    """The skin map's RGB (rows x columns x 3) with BLUSH laid over by `canvas` (rows x columns, 0..1, NaN = none)."""
    weight = np.nan_to_num(np.asarray(canvas, np.float32))[:, :, None]
    return np.asarray(pixels, np.float32) * (1 - weight) + np.asarray(BLUSH, np.float32) * weight
