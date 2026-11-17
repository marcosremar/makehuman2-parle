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
# Round 57, both judges: the young woman's eyebrows "castanha como o cabelo (hoje quase preta)": a young person's eyebrow is its
# own colour (the cast dyes it in the hair's) made darker, no longer one near-black for all.
# Round 63: "castanho-médio como o cabelo (hoje quase preta)" (judge B): 0.6.
BROW_YOUNG_DARKEN = 0.6
# Rounds 49 to 52: both judges asked the old's eyebrows "cinza médio em vez de branco".
# Round 54: "sem o tom azulado, 20 % mais escura" (judge B): a warm mid grey. Round 55: still "cinza-claro, some na pele": darker.
BROW_OLD_GREY = (0.46, 0.43, 0.4)
BROW_OLD_SHARE = 0.85
# Round 66 (both judges): on the old women with dark brown hair the grey eyebrow "não combina com o cabelo escuro", "castanho-escuro
# com fios cinza". An old person's eyebrow goes BROW_OLD_SHARE to the grey when its own colour (the hair's) is grey, light and with
# little colour, and only this share, darkened as a young person's, when it is dark or coloured.
BROW_OLD_SHARE_DARK_HAIR = 0.3
BROW_GREY_CHROMA = (0.1, 0.04)
BROW_GREY_VALUE = (0.25, 0.4)
# Box blur of the skin map, in pixels of a 1024 map (scaled to the map), three passes: the pores and grain of the kit's photograph go.
SKIN_BLUR_PX = 9
# The blush: a warm red-orange (sRGB) laid over the skin on each cheek and on the nose tip, most at the middle, fading with a
# gaussian of the given width in distances between the eyes. The old take more, as the series' old man.
BLUSH = (0.86, 0.38, 0.28)
# Round 30: at a half the blush read as red patches and the skin orange: back to round 29's.
# Round 48: both judges asked the old woman's and the old man's nose tip and cheeks a quarter redder, as the series' old man.
# Round 52: both judges asked the nose tip and the cheeks of all four "25 a 30 % mais vermelhos": a seventh more.
# Round 56: the old's nose tip and cheeks "+25 a 30 % de saturação, como o idoso da série" (judge B).
BLUSH_SHARE = {"female": 0.4, "female-old": 0.74, "male": 0.52, "male-old": 0.82}
# Round 56: the young woman's blush reached the mouth and the chin ("tirar as manchas rosadas em volta da boca", judge B): the spot
# a little higher on the cheek.
# Round 57: the young woman's blush "na lateral, até a borda do rosto" (judge B), "40 % mais fraco, sem as duas manchas" (judge A):
# on the round of the cheek under the eye, and softer and narrower on her.
# Round 61, both judges: at -0.56 the young woman's blush reached the chin and the low cheek, "o rosa que borra em volta da boca";
# the series' sits "no alto das maçãs", just under the eye.
CHEEK_SPOT = (0.1, -0.25, -0.42)
CHEEK_WIDE = 0.34
NOSE_SPOT_WIDE = 0.22
# Round 54: both judges, the tip of the nose "15 a 25 % mais vermelha" on all four.
NOSE_SHARE = 1.0
# The old: on the old man "dois discos vermelhos" on the cheeks, where the series' old man has a wide soft flush and the tip of his
# nose is the reddest of his face (both judges, round 51). By kind: the width of the cheek's spot, how much of the blush the cheek
# takes and how much the nose's tip takes.
# Round 61: the old man's blush still "em círculo de borda dura" (judge B): wider.
CHEEK_WIDE_BY_KIND = {"male-old": 0.64, "female-old": 0.5, "female": 0.3}
# Round 52: with the cheek's part weaker one judge read the old paler; round 53, both judges: "trocar os dois discos vermelhos por um
# avermelhado difuso 40 % mais fraco" on the old man, "blush 30 % mais fraco e espalhado" on the old woman.
CHEEK_PART_BY_KIND = {"male-old": 0.6, "female-old": 0.7}
NOSE_SHARE_BY_KIND = {"male-old": 1.3, "female-old": 1.25}
# The lips: the series draws no lip, the mouth is a cut with curled corners (round 21: "a série não desenha lábio"). The skin about
# the mouth goes this share of the way to the face's own skin colour, so the painted lip of the kit's photograph goes.
MOUTH_SPOT = (0.0, -0.05, -0.62)
MOUTH_WIDE = (0.42, 0.16)
# Round 25 (4.75): gone almost all the way, the mouth was "um risco apagado": the series' lips have a little colour.
LIP_TO_SKIN = 0.45
# A woman's lips take their colour from a patch of their own (`parle_lips.with_lip_patch`): the lip the kit's photograph paints on
# her face showed about the patch as "uma mancha rosada de textura em volta da boca e no queixo" (round 59, the young woman).
# Round 62: at 0.9 the cheeks' tone, lighter than the skin about the mouth, drew "um halo claro em volta da boca" (judge A): 0.7.
LIP_TO_SKIN_WOMAN = 0.7
# A woman's lips are drawn on a patch of their own (`parle_lips`): it goes this share of the way to the skin at their corners.
# Round 38: at 0.35 a woman's lip read "mancha rosa borrada fora da boca" to both judges.
# Round 52: the women's lips "com cor mais perto da pele" (judge B, the young and the old woman): 0.72. Round 55, both judges: "a
# boca é um risco fino", "15 % mais rosada": 0.55. Round 59: "25 % mais saturados que a pele": 0.45. Round 61: "lábio só 1 tom
# acima da pele" (judge B): 0.55.
# Rounds 66 to 69, both judges on every woman of eight: "placa pálida sem contorno", "cor 15 % mais rosada e escura que a pele": 0.45.
LIP_PATCH_TO_SKIN = 0.45
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


# Round 66 (owner, 2026-11-17: "melhora a sobrancelha"): the kit's strands cut at BROW_EDGE and then stretched up to 2.5 times
# about their middle line on the old (`parle_style.STYLE`) came out a grey slab with a lumpy top on the old woman, a block on the old
# man and a band with blunt ends on the young. Each eyebrow of the map is drawn again as the series draws one: its middle line and
# its thickness read off the kit's shape column by column and smoothed (polynomials of these degrees along its length), a round head
# by the nose and a point at the temple over this last share of its length, the edge this many pixels soft.
BROW_LINE_DEGREE = 5
BROW_THICK_DEGREE = 3
BROW_TAIL = 0.3
BROW_SOFT_PX = 1.0
# The old's grey goes this much darker towards its lower edge: one flat light grey read as a smudge on the skin (rounds 49 to 65).
# Round 66: at 0.3 "sombra escura embaixo", "parece adesivo de pelúcia" (judge B), where judge A asked a darker base: less.
BROW_OLD_UNDER = 0.15


def brow_islands(solid):
    """The rows (start, end) of each eyebrow of a map: runs of rows with a solid texel, split by empty rows."""
    rows = np.where(solid.any(axis=1))[0]
    if not len(rows):
        return []
    cuts = np.where(np.diff(rows) > 1)[0]
    return list(zip(np.concatenate([[rows[0]], rows[cuts + 1]]), np.concatenate([rows[cuts], [rows[-1]]]) + 1))


def brow_band(solid):
    """Each eyebrow of a map (rows x columns of bool, the kit's strands) drawn again as one smooth band (BROW_LINE_DEGREE): its
    alpha (0..1) and, across it, where each texel lies from its upper edge (-1) to its lower one (1). An eyebrow too small to
    smooth keeps its own shape."""
    rows = solid.shape[0]
    alpha, across = np.zeros(solid.shape, np.float32), np.zeros(solid.shape, np.float32)
    for top_row, end_row in brow_islands(solid):
        part = solid[top_row:end_row]
        cols = np.where(part.any(axis=0))[0]
        if len(cols) <= BROW_LINE_DEGREE + 2:
            alpha[top_row:end_row] = np.maximum(alpha[top_row:end_row], part)
            continue
        top = part[:, cols].argmax(axis=0) + top_row
        bottom = end_row - part[::-1, cols].argmax(axis=0)
        first, last = cols[0], cols[-1] + 1
        length = last - first
        line = np.polyfit((cols + 0.5 - first) / length, (top + bottom) / 2, BROW_LINE_DEGREE)
        thick = np.polyfit((cols + 0.5 - first) / length, bottom - top, BROW_THICK_DEGREE)
        head_first = np.polyval(thick, 0.1) >= np.polyval(thick, 0.9)
        reach = 2 * (end_row - top_row)
        ys, xs = np.mgrid[max(top_row - reach, 0):min(end_row + reach, rows), first:last]
        at = (xs + 0.5 - first) / length
        along = at if head_first else 1 - at
        half = np.maximum(np.polyval(thick, at), 0) / 2
        cap = max(np.polyval(thick, 0.0 if head_first else 1.0) / 2 / length, 1e-9)
        half = half * np.sqrt(1 - np.clip((cap - along) / cap, 0, 1) ** 2)
        half = half * np.sin(np.pi / 2 * np.clip((1 - along) / BROW_TAIL, 0, 1))
        off = ys + 0.5 - np.polyval(line, at)
        drawn = np.clip(0.5 - (np.abs(off) - half) / BROW_SOFT_PX, 0, 1).astype(np.float32)
        region = (slice(ys[0, 0], ys[-1, 0] + 1), slice(first, last))
        across[region] = np.where(drawn > alpha[region], np.clip(off / np.maximum(half, 1e-6), -1, 1), across[region])
        alpha[region] = np.maximum(alpha[region], drawn)
    return alpha, across


def drawn_brow(pixels, old):
    """An eyebrow map (rows x columns x RGBA, 0..1, sRGB) drawn as the series draws one (`brow_band`), in one colour: its own made
    darker on the young, a mid grey on the old that darkens towards its lower edge (BROW_OLD_UNDER)."""
    pixels = np.asarray(pixels, np.float32)
    alpha, across = brow_band(smooth(BROW_EDGE[0], BROW_EDGE[1], pixels[:, :, 3]) >= 0.5)
    colour = solid_mean(pixels)
    out = np.empty_like(pixels)
    if old:
        grey_hair = float(smooth(*BROW_GREY_CHROMA, float(colour.max() - colour.min())) * smooth(*BROW_GREY_VALUE, float(colour.max())))
        share = BROW_OLD_SHARE_DARK_HAIR + (BROW_OLD_SHARE - BROW_OLD_SHARE_DARK_HAIR) * grey_hair
        own = colour * (BROW_YOUNG_DARKEN + (1 - BROW_YOUNG_DARKEN) * grey_hair)
        tone = own * (1 - share) + np.asarray(BROW_OLD_GREY, np.float32) * share
        shade = 1 - BROW_OLD_UNDER * smooth(-0.2, 1.0, across)
    else:
        tone, shade = colour * BROW_YOUNG_DARKEN, np.ones(alpha.shape, np.float32)
    out[:, :, :3] = tone[None, None, :] * shade[:, :, None]
    out[:, :, 3] = alpha
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
    wide, part = CHEEK_WIDE_BY_KIND.get(kind, CHEEK_WIDE), CHEEK_PART_BY_KIND.get(kind, 1.0)
    for centre in centres:
        side = np.sign(centre[0] - middle_x)
        at = centre + np.array([side * CHEEK_SPOT[0], CHEEK_SPOT[1], CHEEK_SPOT[2]]) * unit
        far = np.linalg.norm(points - at, axis=1) / (wide * unit)
        share = np.maximum(share, part * np.exp(-far * far))
    far = np.linalg.norm(points - np.asarray(tip, float), axis=1) / (NOSE_SPOT_WIDE * unit)
    share = np.maximum(share, NOSE_SHARE_BY_KIND.get(kind, NOSE_SHARE) * np.exp(-far * far))
    # Only the front of the face.
    share *= smooth(-0.2, 0.2, (np.asarray(tip, float)[1] + 1.2 * unit - points[:, 1]) / unit)
    return share * BLUSH_SHARE[kind]


def mouth_share(points, tip, eyes, most=LIP_TO_SKIN):
    """How much of its colour each point about the mouth gives up to the skin's (0..`most`): an oval across the mouth."""
    points = np.asarray(points, float)
    centres = [np.asarray(centre, float) for centre, _ in eyes]
    unit = float(np.linalg.norm(centres[0] - centres[1]))
    at = np.asarray(tip, float) + np.array(MOUTH_SPOT) * unit
    off = (points - at) / unit
    far = np.hypot(off[:, 0] / MOUTH_WIDE[0], off[:, 2] / MOUTH_WIDE[1])
    front = points[:, 1] < np.asarray(tip, float)[1] + 0.6 * unit
    return np.where(front, most * smooth(1.3, 0.8, far), 0.0)


# The face is painted on a patch of its own, the neck and the body on one flat skin tone about 20 % lighter: the step at the
# patch's edge under the jaw read as a seam between head and neck on all four (round 46, both judges). From NECK_FROM to
# NECK_TO distances between the eyes below the nose tip, all round the head, the face's patch goes to the body's tone.
NECK_FROM = 1.7
NECK_TO = 2.6
# The body's tone is read on the skin this far below the nose tip and lower (the chest and the shoulders).
BODY_TONE_BELOW = 4.0


def neck_share(points, tip, eyes):
    """How far each point's colour goes to the body's skin tone (0..1): none on the face, whole down the neck."""
    points = np.asarray(points, float)
    centres = [np.asarray(centre, float) for centre, _ in eyes]
    unit = float(np.linalg.norm(centres[0] - centres[1]))
    below = (np.asarray(tip, float)[2] - points[:, 2]) / unit
    return smooth(NECK_FROM, NECK_TO, below)


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



# Round 66 (owner, 2026-11-17: "verifica o que mais está feio"): on the bar's attendant the beard painted on the map (the cast's own,
# `skin_style`, since 2026-11-09) came out a brown smudge behind the jaw, its front washed to the skin by the paints of the lower
# face. A texel darker than the face's skin past these shares of it, by its brightest channel (a lip's red does not count), is kept
# through those paints as it was after the smoothing: the beard, and the hair painted on a scalp.
KEPT_DARK = (0.7, 0.55)
# Round 68 (both judges): the beard kept so had "borda em degraus", the teeth of the strands along its upper edge: its outline is
# rounded, the kept share blurred over this many pixels of a 1024 map and cut again at its half.
KEPT_DARK_ROUND_PX = 8


def kept_dark(pixels, face_skin, radius=0):
    """How much of each texel of the map is kept through the paints of the face (0..1): the ones much darker than `face_skin`,
    the outline rounded over `radius` pixels."""
    ratio = np.asarray(pixels, np.float32)[:, :, :3].max(axis=2) / max(float(np.max(face_skin)), 1e-6)
    kept = smooth(KEPT_DARK[0], KEPT_DARK[1], ratio)
    return smooth(0.35, 0.65, box_blur(kept[:, :, None], radius)[:, :, 0]) if radius >= 1 else kept


# And the florist's face was a pale mask on a yellow head: the game's face is painted on the map lighter than the kit's skin about
# it, and the series' warmer skin turns that yellow-orange. The skin of the head goes this share of the way to the cheeks' tone,
# its own light and shade kept: its tone is the map blurred over this many pixels of a 1024 map, the dark (`kept_dark`) and the
# empty map left out. From NECK_FROM down, the neck goes on to the body's tone (`neck_share`).
EVEN_TONE = 0.85
EVEN_TONE_PX = 40


def evened(pixels, canvas, covered, dark, colour, radius):
    """The map's RGB with its tone (blurred over `radius` pixels, read off the `covered` texels that are not `dark`) moved to
    `colour` by `canvas` (0..1, NaN = none) times EVEN_TONE, the detail on it kept; the dark texels keep theirs."""
    pixels = np.asarray(pixels, np.float32)
    dark = np.asarray(dark, np.float32)
    tone = masked_blur(pixels, np.asarray(covered, bool) & (dark < 0.5), radius)
    weight = (np.nan_to_num(np.asarray(canvas, np.float32)) * EVEN_TONE * (1 - dark))[:, :, None]
    return pixels + weight * (np.asarray(colour, np.float32) - tone)


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
