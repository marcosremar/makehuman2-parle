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
# Round 40: at 0.18 the faint strands over the brow came out whole, a second arc above it; at 0.3 still (round 45).
BROW_EDGE = (0.42, 0.62)
# How dark a young person's eyebrow is drawn, of its own colour; an old person's goes this share to a light grey instead (the series'
# old man has white-grey brows with a clean outline).
# Round 21 (4.0): the young's still read as a grey smudge at 0.45 of their own colour, the old's dark grey: both now go most of the
# way to a colour of their own.
BROW_DARK = (0.09, 0.06, 0.045)
BROW_DARK_SHARE = 0.8
BROW_OLD_GREY = (0.8, 0.79, 0.76)
BROW_OLD_SHARE = 0.85
# Box blur of the skin map, in pixels of a 1024 map (scaled to the map), three passes: the pores and grain of the kit's photograph go.
SKIN_BLUR_PX = 9
# The blush: a warm red-orange (sRGB) laid over the skin on each cheek and on the nose tip, most at the middle, fading with a
# gaussian of the given width in distances between the eyes. The old take more, as the series' old man.
BLUSH = (0.86, 0.38, 0.28)
# Round 30: at a half the blush read as red patches and the skin orange: back to round 29's.
BLUSH_SHARE = {"female": 0.4, "female-old": 0.45, "male": 0.45, "male-old": 0.55}
CHEEK_SPOT = (0.12, -0.25, -0.72)
CHEEK_WIDE = 0.34
NOSE_SPOT_WIDE = 0.22
NOSE_SHARE = 0.8
# The lips: the series draws no lip, the mouth is a cut with curled corners (round 21: "a série não desenha lábio"). The skin about
# the mouth goes this share of the way to the face's own skin colour, so the painted lip of the kit's photograph goes.
MOUTH_SPOT = (0.0, -0.05, -0.62)
MOUTH_WIDE = (0.42, 0.16)
# Round 25 (4.75): gone almost all the way, the mouth was "um risco apagado": the series' lips have a little colour.
LIP_TO_SKIN = 0.45
# A woman's lips are drawn on a patch of their own (`parle_lips`): it goes this share of the way to the skin at their corners.
# Round 38: at 0.35 a woman's lip read "mancha rosa borrada fora da boca" to both judges.
LIP_PATCH_TO_SKIN = 0.6
# The kit's photograph darkens the skin about each eye (round 29: "anel cinza-escuro da órbita, que lê como olheira"): from
# the rim out to ORBIT_OUT radii of the globe it goes ORBIT_TO_SKIN of the way to the face's skin.
# Round 40 (two judges): "anel escuro em volta do olho", the eye sunk with dark circles: the dark line the photograph paints at
# the rim of the lids (the lash line of a face with lashes) was left, inside 1.05 radii. From inside the rim, and most of the way.
ORBIT_IN = 0.9
ORBIT_OUT = 2.1
ORBIT_TO_SKIN = 0.85
# The kit's photograph shades the jaw and the chin of a man (a shaved beard) and the skin under the chin: the mouth's oval cleared to
# the face's skin then read as a light band under the lip (round 36, Carlos). The lower face, from under the nose down past the jaw,
# goes this share of the way to the face's skin, in distances between the eyes below the nose tip and in front of it.
JAW_TOP = 0.35
JAW_BOTTOM = 2.3
JAW_BACK = 0.9
JAW_TO_SKIN = 0.7


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


# A hair map's see-through texels keep whatever colour they were saved with, often white: blended at the strands' soft edge they
# showed as a pale frayed band along the hairline (rounds 36 to 44). They take the colour of the strands about them, out to this
# many pixels of a 1024 map.
HAIR_BLEED_PX = 24
# How far a scalp cap painted with a skin map goes to the colour of the strands over it.
SCALP_TO_HAIR = 0.9


def scalp_share(pixels, strands):
    """How far each texel of a scalp map goes to the strands' colour: only the skin it shows (lighter than the strands), so the
    strands painted on it keep their grain (round 45: dyed whole, the cap read as a flat brown helmet)."""
    light = np.asarray(pixels, np.float32)[:, :, :3].mean(axis=2)
    above = light - float(np.mean(strands))
    t = np.clip((above - 0.04) / 0.14, 0, 1)
    return SCALP_TO_HAIR * t * t * (3 - 2 * t)


def solid_mean(pixels):
    """The mean colour of a map's solid texels (alpha at least a half)."""
    pixels = np.asarray(pixels, np.float32)
    solid = pixels[:, :, 3] >= 0.5
    return pixels[:, :, :3][solid].mean(axis=0) if solid.any() else pixels[:, :, :3].reshape(-1, 3).mean(axis=0)


def bled(pixels, radius):
    """A map (rows x columns x RGBA, 0..1) whose see-through texels (alpha under a half) take the mean colour of the solid ones
    about them, growing out to `radius` pixels; the solid ones and every alpha kept."""
    out = np.array(pixels, np.float32, copy=True)
    solid = (out[:, :, 3] >= 0.5).astype(np.float32)[:, :, None]
    if solid.all() or not solid.any():
        return out
    reach = 1
    filled = solid.copy()
    colour = out[:, :, :3] * solid
    while reach <= radius:
        weight = box_blur(filled, reach, passes=1)
        mean = box_blur(colour, reach, passes=1) / np.maximum(weight, 1e-6)
        new = (filled == 0) & (weight > 1e-6)
        colour = np.where(new, mean, colour)
        filled = np.where(new, 1.0, filled)
        reach *= 2
    out[:, :, :3] = np.where(solid > 0, out[:, :, :3], np.where(filled > 0, colour, out[:, :, :3]))
    return out


def masked_blur(image, covered, radius):
    """`box_blur` of the texels of the map that a face of the mesh covers (`covered`, rows x columns of bool), weighted by the
    cover so the empty map between islands does not bleed in (round 34: a band with a straight edge on the chin), the rest kept."""
    weight = covered.astype(np.float32)[:, :, None]
    blurred = box_blur(np.asarray(image, np.float32) * weight, radius) / np.maximum(box_blur(weight, radius), 1e-6)
    return np.where(weight > 0, blurred, image)


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


def jaw_share(points, tip, eyes):
    """How much of its colour each point of the lower face gives up to the skin's (0..JAW_TO_SKIN): below the nose, in front."""
    points = np.asarray(points, float)
    centres = [np.asarray(centre, float) for centre, _ in eyes]
    unit = float(np.linalg.norm(centres[0] - centres[1]))
    tip = np.asarray(tip, float)
    below = (tip[2] - points[:, 2]) / unit
    behind = (points[:, 1] - tip[1]) / unit
    share = smooth(JAW_TOP - 0.2, JAW_TOP + 0.1, below) * smooth(JAW_BOTTOM, JAW_BOTTOM - 0.4, below) * smooth(JAW_BACK + 0.3, JAW_BACK - 0.2, behind)
    return JAW_TO_SKIN * share


def orbit_share(points, eyes):
    """How much each point about an eye gives up its colour to the skin's (0..ORBIT_TO_SKIN), in front of the globe."""
    points = np.asarray(points, float)
    share = np.zeros(len(points))
    for centre, radius in eyes:
        off = points - np.asarray(centre, float)
        far = np.linalg.norm(off, axis=1) / radius
        ring = smooth(ORBIT_IN, ORBIT_IN + 0.15, far) * smooth(ORBIT_OUT, ORBIT_OUT - 0.4, far) * (off[:, 1] < 0.4 * radius)
        share = np.maximum(share, ring)
    return share * ORBIT_TO_SKIN


def toward(pixels, canvas, colour):
    """The map's RGB with `colour` laid over by `canvas` (0..1, NaN = none)."""
    weight = np.nan_to_num(np.asarray(canvas, np.float32))[:, :, None]
    return np.asarray(pixels, np.float32) * (1 - weight) + np.asarray(colour, np.float32) * weight


def blushed(pixels, canvas):
    """The skin map's RGB (rows x columns x 3) with BLUSH laid over by `canvas` (rows x columns, 0..1, NaN = none)."""
    weight = np.nan_to_num(np.asarray(canvas, np.float32))[:, :, None]
    return np.asarray(pixels, np.float32) * (1 - weight) + np.asarray(BLUSH, np.float32) * weight
