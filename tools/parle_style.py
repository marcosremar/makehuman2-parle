"""
    License information: data/licenses/makehuman_license.txt

    The style of the cast's faces, by kind of person (owner, 2026-11-15). He sent the opening of an animated series and asked for the
    faces to be "idêntico aos rostos dessa série", then "categoriza cada personagem e modifica o makehuman2-parle para isso". Its
    characters were sorted by kind (24 portraits measured with the ruler of `parle_beauty`; the catalogue stays with the game's
    repository, next to the registry of the faces), and what sets each kind apart that a chart seen from the front (`parle_head`)
    cannot give is volume: the young and grown men have a big, long nose with a round tip, the old man a bigger one still and heavy
    eyebrows, the women a small nose. That is sculpted here, as a field of points with no bpy, by kind: STYLE.

    The kind is the sex and, for the old, "-old" (`kind_of`). A job says it is old with "age": "old"; with nothing said a person is a
    grown one. No mesh, picture or point of any character is kept: the numbers are read off the middle of each group by sight.
"""
import os

import numpy as np

# How much the nose grows about a point behind its tip, and how thick the eyebrows are left (`parle_face.BROW_THIN` is the kind's
# own default, which this replaces). The series' young men have noses 0.31 to 0.37 of the face long against 0.29 to 0.31 on the cast,
# and much further out; its old man's is the biggest thing on his face.
STYLE = {
    # The eyes (2026-11-16, round 49): grown 1.25 on the men and 1.12 on the women for the series, the skin round them kept under a
    # third of its room (0.36 on the men, 0.22 on the women: the bags under the eyes the judges asked away, the women past the 1.3
    # the owner's test holds, `tests/test_parle_eyes.py`). The women's eyes back to 1.3 of the kit's, the men's to 1.25.
    "male": {"forward": 1.25, "nose": 1.6, "tip": 1.5, "brow": 1.4, "eye": 1.25 / 1.1, "skin": 1.55},
    # Round 41: the old man's eyebrow a flat white patch over the glasses (2.8) and his eye white all about the iris (`lid` in SCULPT lower).
    # Round 66 (owner, 2026-11-17: "melhora a sobrancelha"): at 2.5 his eyebrow stood 3 cm tall by the nose, a grey block over the
    # glasses, as thick at the temple as by the nose: thinner, and to `brow_outer` of that at its outer end (the series' old man's
    # are bushy by the nose and thin out to the temple). The young's keep `parle_face.BROW_OUTER`.
    # Round 66 (both judges): his eyebrows a roof, "a ponta de dentro alta deixa a cara triste": the inner end does not go up
    # (`parle_face.BROW_INNER_UP_M`, asked for the young man's angry slope).
    "male-old": {"forward": 1.3, "nose": 2.0, "tip": 1.65, "brow": 1.9, "brow_outer": 0.65, "brow_inner": 0.0, "eye": 1.25 / 1.1, "skin": 1.55},
    # Round 48 (5.7): both judges, three rounds running, asked the young woman's nose a third smaller, a button ("hoje é nariz de
    # homem"), and the old woman's tip shorter.
    # Round 53: both judges asked the women's and the old man's eyebrows a half to twice as thick; the young woman's stays under
    # the young man's (the series' order, `tests/test_parle_style.py`).
    # Round 63, both judges asked the young woman's eyebrows thicker; at 1.55 (round 64) both "30 a 40 % mais finas": back to 1.35.
    # Round 76 (both judges, seven of eight: the eyebrow "em vírgula, mais grossa na cabeça e afinando até a ponta"): a woman's
    # thins to 0.6 of its head at the temple (`parle_face.BROW_OUTER` 0.75 until then).
    "female": {"forward": 0.8, "nose": 1.15, "tip": 1.1, "brow": 1.35, "brow_outer": 0.6, "eye": 1.0, "skin": 1.55},
    # Rounds 54 and 58: the old woman's nose "20 % menor, ... hoje parece nariz de homem" (both judges).
    # Round 66: at 2.4 the old woman's eyebrow was a flat grey slab over the eye: thinner, and thinning to the temple.
    "female-old": {"forward": 0.85, "nose": 1.3, "tip": 1.2, "brow": 1.7, "brow_outer": 0.6, "eye": 1.0, "skin": 1.55},
}
# Round 3 of the loop judged by sight against the series (2026-11-15, 3.1 in 10) asked first for eyes about a third bigger and round,
# with white showing and a highlight; then warmer, more saturated skin; a bulb at the tip of the nose; eyebrows half as thick again.
# Round 7 (4.0): the nose too long and pointed from the side; it is to grow across and round, and come out less ("forward" is how
# much it grows to the front, against "nose" across and up).
# The owner, 2026-11-16, on the sculpted Carlos: "corrige o nariz dele que está amassado". Grown across 1.9 and to the front only
# 0.95, the men's nose came out wide and pressed flat, and the ball of the sculpture then pulled its bridge in: it now grows to the
# front more than across, and the ball only fills out (SCULPT, `sculpt_field`).
# Round 21 (2026-11-16, 4.0): on all four the nose small and flat for the style; the young man's eyebrow too wide. The nose balls
# grew by a fifth to a quarter (SCULPT), the young man's eyebrow down from 1.8.
# Round 6 (3.4) asked for skin smooth and matte, with no relief of pores; a wider mouth; fuller cheeks; thicker, higher eyebrows.
# Round 4 (3.25) asked first for eyes 40% more open with the white showing, then a bigger nose and fuller cheeks.
# "eye" multiplies the size the eyes of the kit's sex already take (`parle_polish.eye_scale`); "tip" grows the tip of the nose alone
# within NOSE_TIP_REACH_M; "skin" is how much more saturated the skin's colour is, turned a little towards orange (SKIN_WARM).
NOSE_TIP_REACH_M = 0.016
# How round an eye opens at least, its height over its width: the series' eyes are 0.6 to 0.7 (read on its portraits); the cast's
# lids gave 0.42, with the iris filling the opening (round 5).
# Round 20 (2026-11-16): at 0.58 the upper lid still covered the top third of the iris, a sleepy look, on all four.
# Round 29 (5.0): at 0.66 the women and the old man stared (white all about the iris): the series' lid covers 15 to 20 % of it.
EYE_TALL_LEAST = 0.6
# Round 47: with the lid's lift evened out (no fold left over the lid) the old woman and the old man opened white above the iris,
# "no limite do esbugalhado"; both judges asked their upper lid over a fifth of the top of the iris. By kind.
# Round 48: the old man still "esbugalhado", both judges asking his upper lid over a fifth to a quarter of the iris: his lids
# open 0.45 of the width as the kit has them, and are no longer lifted.
# Round 53 (with the iris a tenth bigger since round 51): both judges asked the upper lid over 10 to 20 % of the top of the iris,
# most on the old, "tira o ar arregalado".
# 2026-11-16, after round 59: under the owner's most (`parle_polish.EYE_TALL_MOST_BY_SEX`, 0.50 for a woman and 0.43 for a man), a
# little more open than the lids the owner approved on 2026-11-12 (0.41 to 0.425), so the lid lies over the top of the iris.
EYE_TALL_LEAST_BY_KIND = {"male": 0.40, "female": 0.46, "male-old": 0.30, "female-old": 0.43}
# How much the eye's own map lights it (emission): the white of the series' eyes is white even in the shade of the brow.
# Round 30 (2026-11-16): the white of the eye read grey-pink in the orbit's shade; the series' is clean white.
# Round 38 (two judges, 4.1): the white still "sombra funda em volta" on all four.
# Round 56, judge B on all four: "esclera branca e limpa, cerca de 70 % menos da sombra cinza da pálpebra sobre o branco". Round 61:
# still "de cinza para branco-marfim, só a cor" (judge B): 0.65.
EYE_GLOW = 0.65
SKIN_WARM = 0.012
# The nose's reach about its tip, in metres; below the tip it is shorter (NOSE_BELOW), so the upper lip stays where it was, and up
# the bridge longer (NOSE_ABOVE). The middle it grows about stands NOSE_BACK behind the tip, a little above it.
NOSE_REACH_M = 0.032
NOSE_BELOW = 2.2
NOSE_ABOVE = 0.75
NOSE_BACK_M = 0.024
NOSE_UP_M = 0.004


def kind_of(sex, job):
    """The kind of a person: "female" or "male", with "-old" when the job says the person is old."""
    if sex not in ("female", "male"):
        raise ValueError(f"no style for {sex!r}")
    return sex + ("-old" if str((job or {}).get("age", "")) == "old" else "")


def smooth(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def nose_tip(points, eyes):
    """The tip of the nose: the point furthest to the front (-y) on the middle line, between a little and some way below the eyes."""
    centres = np.array([centre for centre, _ in eyes], float)
    middle = centres.mean(axis=0)
    unit = float(np.linalg.norm(centres[0] - centres[1]))
    near = (np.abs(points[:, 0] - middle[0]) < 0.1 * unit) & (points[:, 2] < middle[2] - 0.25 * unit) & (points[:, 2] > middle[2] - 1.1 * unit)
    if not near.any():
        raise ValueError("no nose on the middle line under the eyes")
    found = points[near]
    return found[np.argmin(found[:, 1])]


def nose_field(tip, grow, forward=None):
    """`field(points)`: each point near the nose moved away from a middle behind the tip by (grow - 1) of its distance, fading out."""
    middle = np.asarray(tip, float) + np.array([0.0, NOSE_BACK_M, NOSE_UP_M])
    forward = grow if forward is None else forward

    def field(points):
        points = np.asarray(points, float)
        off = points - tip
        tall = np.where(off[:, 2] < 0, off[:, 2] * NOSE_BELOW, off[:, 2] * NOSE_ABOVE)
        away = np.sqrt(off[:, 0] ** 2 + off[:, 1] ** 2 + tall ** 2) / NOSE_REACH_M
        # Only the face: nothing behind the middle (the cheeks and the inside of the head stay).
        front = smooth((middle[1] + 0.006 - points[:, 1]) / 0.012)
        share = smooth(1 - away) * front
        return points + share[:, None] * (points - middle) * (np.array([grow, forward, grow]) - 1)

    return field


def eye_grow(sex):
    """How much bigger than its sex's own size an eye of the series' style is; the same for young and old."""
    return STYLE[kind_of(sex, {})]["eye"]


def tip_field(tip, grow, forward=1.0):
    """The bulb at the tip of the nose: points within NOSE_TIP_REACH_M of the tip grown about a middle a little behind it."""
    middle = np.asarray(tip, float) + np.array([0.0, 0.6 * NOSE_TIP_REACH_M, 0.0])

    def field(points):
        points = np.asarray(points, float)
        share = smooth(1 - np.linalg.norm(points - tip, axis=1) / NOSE_TIP_REACH_M)
        return points + share[:, None] * (points - middle) * (np.array([grow, forward, grow]) - 1)

    return field


def warmer(rgb, amount):
    """Colours (0-1, n x 3) with their saturation times `amount` about their grey and turned SKIN_WARM towards orange."""
    rgb = np.asarray(rgb, float)
    grey = rgb.mean(axis=1, keepdims=True)
    out = grey + (rgb - grey) * amount
    out = out + np.array([1.0, 0.25, -0.6]) * SKIN_WARM
    return np.clip(out, 0, 1)


# The sculpted head (owner, 2026-11-15, "cabeça base esculpida"): the trait morphs and the fields above stopped near 4.5 in 10 against
# the series, the judge by sight naming what a real face keeps under them: deep orbits with the lids of an anatomy, the folds from
# the nose to the mouth, hollow cheeks, a nose with grooved nostrils. The series' heads are simple volumes. So each place is laid
# onto a simple form here: the nose onto a ball, each cheek out onto a ball under the eye, the skin about each eye out onto a thick
# shell over its globe, the chin onto a ball. Every form is placed from the person's own eyes and nose tip, in distances between
# the eyes (`unit`), so one set of numbers serves every person of a kind. SCULPT has them by kind; a share is how much of the way
# to its form a point goes.
SCULPT = {
    "male": {"nose_ball": (0.41, 0.42, 0.33), "nose_share": 0.95, "cheek": 0.66, "cheek_share": 0.76, "chin": 0.5, "lid": 0.1, "ridge": 0.0},
    "male-old": {"nose_ball": (0.44, 0.46, 0.38), "nose_share": 0.95, "cheek": 0.735, "cheek_share": 0.92, "chin": 0.45, "lid": -0.32, "ridge": 0.0},
    "female": {"nose_ball": (0.22, 0.22, 0.19), "nose_share": 0.9, "cheek": 0.62, "cheek_wide": 0.86, "cheek_share": 0.82, "chin": 0.36, "lid": -0.02, "ridge": 0.0},
    "female-old": {"nose_ball": (0.29, 0.29, 0.25), "nose_share": 0.9, "cheek": 0.7, "cheek_wide": 0.86, "cheek_share": 0.92, "chin": 0.38, "lid": -0.08, "ridge": 0.0},
}
# Where the forms stand, in distances between the eyes from the middle of the eye globe of that side (cheeks) or from the nose tip
# (the nose ball behind it, the chin under it): across, back (+y), up.
# Round 35: the cheek a high round mass ("maçã alta"), its ball higher under the eye (0.14, 0.58, -0.48), and a brow ridge forward
# (`ridge` in SCULPT) and lighter smoothing: round 36 fell from 5.5 to 4.5, so back to round 34's.
# Round 57, both judges on all four: the cheeks "15 a 20 % mais cheias e redondas, com o volume subido para a maçã": the balls a
# twelfth bigger (SCULPT) and a little higher.
# Round 76 (both judges, all eight: "bochechas 10 % mais cheias no terço de baixo", "maçã 5 a 8 % mais larga na altura da asa do
# nariz"): the balls lower and a little further out, and a twentieth bigger (SCULPT). Lower more than out: the owner faults a
# wide face over a narrow jaw (2026-11-12, `parle_face.FAIR_LOOK`).
CHEEK_AT = (0.14, 0.62, -0.64)
# Round 78 (owner, 2026-11-17: "deve melhorar o maxilar"): on the women the cheeks' balls made the face a wide round over a
# narrow jaw, the jaw 0.77 to 0.79 of the width at the cheekbones where the series' women have 0.82 and the faces he approved
# 0.80 (he faults "queixo fino e rosto largo", 2026-11-12). Their balls are narrower across (`cheek_wide` in SCULPT), and back
# to the size they had before round 77.
# Round 55: "boca e queixo muito atrás da ponta do nariz" in profile (judge A): 3 mm forward in round 57; the crease under the young
# man's lower lip came back where the chin's ball ends ("uma segunda boca", round 60), as in round 26: back.
CHIN_AT = (0.0, 0.55, -1.25)
# How far about each form its pull reaches, in its own radii, and the zone of the nose about the tip, in distances between the eyes.
REACH = 1.4
# Where the cheeks' and the chin's balls stop taking the head, in distances between the eyes behind the nose tip: whole to the
# first, none past the second (the ears stand some 1.9 behind it).
FRONT_FADE = (0.9, 1.6)
# How much of the way to the chin's ball a point goes (round 25: the young man's chin "recua muito" from the side).
# Round 26: at 0.8 a crease stood under the lower lip of both men, where the chin's ball ended.
CHIN_SHARE = 0.6
NOSE_BALL_UP = 0.1
# The hollow between each upper lid and its eyebrow filled (`hollows_filled`): judges by sight asked the orbit "40 a 50 % mais
# rasa, sem a sombra escura entre o olho e a sobrancelha" from round 38 to 48. The shell over the eye that was to do it moved no
# point (its radius was never past the skin's: found on 2026-11-16). The skin from ORBIT_IN radii of the globe (the rim of the lid
# stays) out to ORBIT_OUT, above ORBIT_LOW radii under the globe's middle and within ORBIT_ACROSS radii across, comes forward
# towards its neighbours where it stands behind them, never back, ORBIT_ROUNDS times by ORBIT_STEP.
ORBIT_IN = 1.2
ORBIT_OUT = 2.6
ORBIT_LOW = -0.2
ORBIT_ACROSS = 1.9
# Round 50: at 30 the eye still "fundo na órbita, com sombra em volta" on all four (judge B). Round 55: at 60 "cerca de 1,5 mm
# menos fundo" (judge B, the young two).
ORBIT_ROUNDS = 90
ORBIT_STEP = 0.5
# The brow ridge: where it is most, in radii of the globe over its middle, how tall, and how far forward by kind (SCULPT).
BROW_RIDGE_UP = 1.75
BROW_RIDGE_TALL = 0.55
# How far the upper lid is lifted, in radii of the globe, and how far from the globe's middle the lift reaches.
# Round 25 (4.75): at 0.22 white showed all about the iris on three of four, the stare the owner calls "esbugalhado"; the
# series' upper lid always covers a little of the top of the iris.
# Round 26: the men's lids right at 0.1; the women's eyes, larger and rounder in their kind, still stared: theirs come a
# little down. By kind, in SCULPT ("lid").
LID_LIFT = 0.1  # the men's, as SCULPT["male"]["lid"]
LID_REACH = 1.9
# The ball stands this share of its depth behind the tip, so the tip comes out a little into a round bulb and nothing is pulled in.
NOSE_BALL_BACK = 0.78
NOSE_ZONE = 0.55
# The lower face (round 31, 2026-11-16, 5.0): "a estrutura ainda é a de uma cabeça humana realista … rosto comprido e estreito
# embaixo"; the series' faces are short under the nose. From the nose tip down to the chin (LOWER_CHIN distances between the eyes
# under the tip) the face is this share as tall; the neck takes the move back over LOWER_NECK, and the whole width of the jaw
# (LOWER_WIDE either side) goes with it.
# Round 37 (5.5): with the crease gone, the chin still "quadrado" and long on all four: a little shorter.
# Round 78 made each 0.05 shorter for the judges by sight ("nariz–queixo 6 % mais curto"): measured with the ruler on the portraits
# of round 77 (`parle_beauty`, the owner, 2026-11-17: "deve melhorar o maxilar"), the women's faces were already wider than tall
# (width over height 1.01 to 1.09; the series' women 0.91, the faces the owner approved 0.90) and the men's as wide as his (0.83
# to 0.89 against 0.83; the series' men 0.77). Back for the men; the women's face is left longer under the nose.
LOWER_FACE = {"male": 0.83, "male-old": 0.87, "female": 0.93, "female-old": 0.93}
# A trial of another share for the lower face, for every kind, without touching this file: a QA door, not a mode of the cast.
LOWER_FACE_TRIAL = "PARLE_LOWER_FACE"
LOWER_CHIN = -1.3
# Round 38: at 1.6 the lift reached a woman's collar and the throat came through it.
LOWER_NECK = 1.1
LOWER_BACK = 2.4
LOWER_BACK_FADE = 1.8
LOWER_WIDE = 2.4
# Simple volumes (round 27, 2026-11-16, 4.1: "pele cerosa e realista", the folds of the neck and the chin, the lips of a real mouth,
# "volumes mais simples e redondos"). The face and the neck are smoothed where a real face keeps its small forms: about a middle
# behind and under the nose (in distances between the eyes), whole within SIMPLE_IN and none past SIMPLE_OUT; the rims of the eyes
# (in radii of the globe) and the lips (an oval about the mouth) are left.
SIMPLE_MIDDLE = (0.0, 1.0, -0.9)
SIMPLE_IN = 2.0
SIMPLE_OUT = 2.9
# Round 40: a bag line under the eye (Carlos, Dona Marta) between 1.25 and 1.9 radii, half smoothed: closer to the rim.
SIMPLE_EYE = (1.18, 1.55)
SIMPLE_MOUTH = (0.0, 0.1, -0.62)
# The lips keep their own shape; from rounds 24 to 50 an oval some 80 mm wide about the mouth did, the cheeks round it smoothed and
# not it: the block stood out as a muzzle with a crease round it, from the wing of the nose to under the corner of the mouth ("um
# vinco que vai da asa do nariz até abaixo do canto da boca", both judges, rounds 48 to 50). The lips only, and the smoothing
# coming in over SIMPLE_MOUTH_RAMP times their size.
# Round 63, both judges: "um vinco duro e reto logo abaixo do lábio inferior", "uma linha dupla de boneco", where the smoothing
# stopped at the edge of the lips' oval: the lower part of the lower lip smoothed too (a longer way in, 2.2, smoothed the cheek
# beside the mouth less than round 51 did).
SIMPLE_MOUTH_WIDE = (0.48, 0.12)
SIMPLE_MOUTH_RAMP = 1.5
# Round 35 (5.4) read "um balão liso" at 25; 12 in round 36 did not help (above).
SIMPLE_ROUNDS = 25
SIMPLE_IN_STEP = 0.5
SIMPLE_OUT_STEP = -0.53


# The rim of a ball (owner, 2026-11-17: "o maxilar do homem e da mulher ainda está com falhas"). A point inside a ball went the
# whole share of the way out to it right up to the ball's rim, and a point outside not at all: where the skin crossed the ball it
# turned at a crease, a notch on each side of the chin at the rim of the chin's ball (0.5 distances between the eyes from the
# middle, the men) and a corner at the angle of the jaw at the rim of the cheek's. With `ease`, the pull of a point inside falls to
# nothing over the last `ease` of the radius, with no slope at the rim: a fillet where there was a crease.
BALL_EASE = 0.3


def onto_ball(points, centre, radii, share, outward_only, ease=0.0):
    """Each point moved `share` of the way to the surface of the ellipsoid about `centre` (radii across, back, up), along the line
    from the centre; with `outward_only`, only the points inside it, which fills hollows and leaves the rest, and with `ease` the
    points within that share of the radius of its rim go less, down to nothing at the rim."""
    off = points - centre
    scaled = np.linalg.norm(off / np.asarray(radii, float), axis=1)
    scaled = np.maximum(scaled, 1e-9)
    onto = centre + off / scaled[:, None]
    pull = share * smooth((REACH - scaled) / (REACH - 1.0))
    if outward_only:
        pull = np.where(scaled < 1.0, pull, 0.0)
        if ease > 0:
            pull = pull * smooth((1.0 - scaled) / ease)
    return points + pull[:, None] * (onto - points)


def simple_weight(points, tip, eyes):
    """How much each point is smoothed into simple volumes (`smoothed`): the face and the neck under it, but not the rims of the
    eyes nor the lips, which the lids, the blink and speech need as they are. Blender's axes, the face looking at -y."""
    points = np.asarray(points, float)
    centres = [np.asarray(centre, float) for centre, _ in eyes]
    unit = float(np.linalg.norm(centres[0] - centres[1]))
    tip = np.asarray(tip, float)
    middle = tip + np.array(SIMPLE_MIDDLE) * unit
    weight = smooth((SIMPLE_OUT - np.linalg.norm(points - middle, axis=1) / unit) / (SIMPLE_OUT - SIMPLE_IN))
    for (centre, radius) in eyes:
        far = np.linalg.norm(points - np.asarray(centre, float), axis=1) / radius
        weight *= smooth((far - SIMPLE_EYE[0]) / (SIMPLE_EYE[1] - SIMPLE_EYE[0]))
    mouth = tip + np.array(SIMPLE_MOUTH) * unit
    off = (points - mouth) / unit
    weight *= smooth((np.hypot(off[:, 0] / SIMPLE_MOUTH_WIDE[0], off[:, 2] / SIMPLE_MOUTH_WIDE[1]) - 1.0) / SIMPLE_MOUTH_RAMP)
    return weight



# Under the jaw (round 74, both judges on all eight, since round 66: "quina dura da mandíbula até o pescoço", "vinco sob o queixo",
# "costura serrilhada sob o queixo"): the face is made shorter after it is smoothed (`lower_face_field`), and the floor under the
# chin, lifted more at the front than at the back, met the jaw and the throat in folds of 20 to 35 degrees. The skin from just
# over the chin (JAW_NECK_FROM distances between the eyes under the nose tip) down the throat (to JAW_NECK_TO), at the front and
# the sides of the neck (no further back than JAW_NECK_BACK behind the tip), is smoothed again for JAW_NECK_ROUNDS; the lips are
# left as in `simple_weight`.
JAW_NECK_FROM = (0.8, 1.05)
JAW_NECK_TO = (2.3, 2.7)
JAW_NECK_BACK = (1.5, 2.3)
# Round 76: at 20 rounds a small step stayed under the men's chin in profile ("uma ondulação mole no meio", "dobra dupla").
JAW_NECK_ROUNDS = 50


def jaw_neck_weight(points, tip, eyes):
    """How much each point of the skin is smoothed again under the jaw (0..1), the face looking at -y."""
    points = np.asarray(points, float)
    centres = [np.asarray(centre, float) for centre, _ in eyes]
    unit = float(np.linalg.norm(centres[0] - centres[1]))
    tip = np.asarray(tip, float)
    below = (tip[2] - points[:, 2]) / unit
    behind = (points[:, 1] - tip[1]) / unit
    weight = smooth((below - JAW_NECK_FROM[0]) / (JAW_NECK_FROM[1] - JAW_NECK_FROM[0])) * smooth((JAW_NECK_TO[1] - below) / (JAW_NECK_TO[1] - JAW_NECK_TO[0]))
    weight *= smooth((JAW_NECK_BACK[1] - behind) / (JAW_NECK_BACK[1] - JAW_NECK_BACK[0]))
    off = (points - (tip + np.array(SIMPLE_MOUTH) * unit)) / unit
    return weight * smooth((np.hypot(off[:, 0] / SIMPLE_MOUTH_WIDE[0], off[:, 2] / SIMPLE_MOUTH_WIDE[1]) - 1.0) / SIMPLE_MOUTH_RAMP)


def orbit_weight(points, tip, eyes):
    """How much each point of the skin takes part in filling the hollow between the upper lid and the eyebrow (0..1), the face
    looking at -y."""
    points = np.asarray(points, float)
    centres = [np.asarray(centre, float) for centre, _ in eyes]
    unit = float(np.linalg.norm(centres[0] - centres[1]))
    front = points[:, 1] < np.asarray(tip, float)[1] + 0.9 * unit
    weight = np.zeros(len(points))
    for centre, radius in eyes:
        off = points - np.asarray(centre, float)
        far = np.linalg.norm(off, axis=1) / radius
        share = smooth((far - ORBIT_IN) / 0.25) * smooth((ORBIT_OUT - far) / 0.6) * smooth((off[:, 2] / radius - ORBIT_LOW) / 0.3)
        share *= smooth((ORBIT_ACROSS - np.abs(off[:, 0]) / radius) / 0.5) * front * (off[:, 1] < 0.3 * radius)
        weight = np.maximum(weight, share)
    return weight


# The folds about the mouth filled as the orbit is (`hollows_filled`): the one from the wing of the nose to the corner of the mouth
# and the one under the lower lip, which both judges asked 50 to 70 % shallower from round 48 to 54 ("pele lisa como a da série").
# A ring about the lips (SIMPLE_MOUTH_WIDE), from FOLD_IN to FOLD_OUT times their size, in front.
FOLD_IN = 1.05
FOLD_OUT = 3.2
# Round 56: at 40 the folds still asked "50 % mais suaves" (judge A, the old woman and the young man).
# Round 57: the young man's folds still "fundos" (judge B): 110.
FOLD_ROUNDS = 110


def fold_weight(points, tip, eyes):
    """How much each point of the skin takes part in filling the folds about the mouth (0..1), the face looking at -y."""
    points = np.asarray(points, float)
    centres = [np.asarray(centre, float) for centre, _ in eyes]
    unit = float(np.linalg.norm(centres[0] - centres[1]))
    tip = np.asarray(tip, float)
    off = (points - (tip + np.array(SIMPLE_MOUTH) * unit)) / unit
    far = np.hypot(off[:, 0] / SIMPLE_MOUTH_WIDE[0], off[:, 2] / SIMPLE_MOUTH_WIDE[1])
    front = points[:, 1] < tip[1] + 0.9 * unit
    return smooth((far - FOLD_IN) / 0.3) * smooth((FOLD_OUT - far) / 0.6) * front * (off[:, 2] < 0.45)


def hollows_filled(points, edges, weight, rounds=ORBIT_ROUNDS):
    """The points of a mesh by its `edges` (m x 2) brought forward (-y) towards the mean of their neighbours where they stand
    behind it, by `weight` of ORBIT_STEP each round: a hollow fills, a ridge stays."""
    points = np.asarray(points, float).copy()
    edges = np.asarray(edges, int)
    count = np.zeros(len(points))
    np.add.at(count, edges[:, 0], 1)
    np.add.at(count, edges[:, 1], 1)
    weight = np.asarray(weight, float)
    for _ in range(rounds):
        total = np.zeros(len(points))
        np.add.at(total, edges[:, 0], points[edges[:, 1], 1])
        np.add.at(total, edges[:, 1], points[edges[:, 0], 1])
        ahead = total / np.maximum(count, 1) - points[:, 1]
        points[:, 1] += ORBIT_STEP * weight * np.minimum(ahead, 0.0)
    return points


def smoothed(points, edges, weight, rounds=None):
    """Taubin smoothing (Taubin 1995, SIGGRAPH '95, DOI 10.1145/218380.218473: a step in and a step out, so the volume is kept)
    of the points by their `edges` (n x 2), each point taking `weight` of each step. Returns the new points."""
    points = np.asarray(points, float).copy()
    edges = np.asarray(edges, int)
    count = np.zeros(len(points))
    np.add.at(count, edges[:, 0], 1)
    np.add.at(count, edges[:, 1], 1)
    count = np.maximum(count, 1)[:, None]
    weight = np.asarray(weight, float)[:, None]
    for _ in range(SIMPLE_ROUNDS if rounds is None else rounds):
        for step in (SIMPLE_IN_STEP, SIMPLE_OUT_STEP):
            total = np.zeros_like(points)
            np.add.at(total, edges[:, 0], points[edges[:, 1]])
            np.add.at(total, edges[:, 1], points[edges[:, 0]])
            points = points + step * weight * (total / count - points)
    return points


# How a worn mesh follows the skin (`parle_polish.follow_skin`): the mean move of the FOLLOW_NEAREST nearest points of the skin
# that moved, whole within FOLLOW_NEAR metres of them and none past FOLLOW_FAR.
FOLLOW_NEAREST = 4
FOLLOW_NEAR = 0.025
FOLLOW_FAR = 0.06


def spread(step, edges, pinned, free, rounds):
    """The move `step` (n x 3) of a mesh by its `edges` (m x 2) evened out over the points `free` (n, bool) by `rounds` of taking
    the mean of their neighbours' move, the `pinned` ones and the rest kept: an abrupt edge of a move becomes a slope."""
    step = np.asarray(step, float).copy()
    edges = np.asarray(edges, int)
    loose = np.asarray(free, bool) & ~np.asarray(pinned, bool)
    count = np.zeros(len(step))
    np.add.at(count, edges[:, 0], 1)
    np.add.at(count, edges[:, 1], 1)
    for _ in range(rounds):
        total = np.zeros_like(step)
        np.add.at(total, edges[:, 0], step[edges[:, 1]])
        np.add.at(total, edges[:, 1], step[edges[:, 0]])
        mean = total / np.maximum(count, 1)[:, None]
        step[loose] = mean[loose]
    return step



# The lower edge of short hair lies on the skin (round 66: the cap under the young man's curls stood 7 mm off his neck, up to 15,
# and the old man's 6.6 mm, a helmet's rim behind the ear; the series paints a hairline flush on the head). An edge point of a hair
# that ends above the jaw, at the nape, round the ears or at the sideburns and no further than HAIR_TUCK_REACH_M from the skin goes
# to HAIR_ON_SKIN_M off it; the rest of the hair follows, less and less, out to HAIR_TUCK_FADE_M from the edge.
HAIR_ON_SKIN_M = 0.001
HAIR_TUCK_REACH_M = 0.02
HAIR_TUCK_FADE_M = 0.03
HAIR_TUCK_ROUNDS = 60
# A hair that ends above the eyes less this much is short; below it (long hair to the shoulders) its edge is left. Only the edge
# below the eyes plus HAIR_TUCK_ABOVE_M and this far behind them goes (not a fringe on the forehead).
# Round 66: up to 3 cm over the eyes, the edge at the temples went in too and the young man showed a light hole at each temple
# (both judges): only the edge 1.5 cm or more below the eyes, the nape and behind the ears.
HAIR_SHORT_M = 0.1
HAIR_TUCK_ABOVE_M = -0.015
HAIR_TUCK_BEHIND_M = 0.04


def tucked_edge(points, edges, edge, targets):
    """The move (n x 3) of a hair whose `edge` points (indices) go to `targets` (their places on the skin): the points further than
    HAIR_TUCK_FADE_M from every one of them stay, the ones between follow evenly (`spread`)."""
    points = np.asarray(points, float)
    step = np.zeros_like(points)
    edge = np.asarray(edge, int)
    if not len(edge):
        return step
    step[edge] = np.asarray(targets, float) - points[edge]
    far = nearest_on(points, points[edge])[0] > HAIR_TUCK_FADE_M
    pinned = far.copy()
    pinned[edge] = True
    return spread(step, edges, pinned, np.ones(len(points), bool), HAIR_TUCK_ROUNDS)



# A shape of the face (a smile, a rounded mouth) whose move falls off abruptly creases the skin there (round 71, both judges on
# all eight: the smile's "vincos fundos nos cantos", the "o"'s "vincos saindo dos cantos"). Its move goes this share of the way to
# the same move evened over the mesh for this many rounds (`spread`): the crease opens into a slope, the shape stays.
CREASE_ROUNDS = 3
CREASE_SHARE = 0.6


def creases_out(move, edges, rounds=CREASE_ROUNDS, share=CREASE_SHARE):
    """The move (n x 3) of a shape with its creases opened: `share` of the way to itself evened along the mesh's `edges`."""
    move = np.asarray(move, float)
    none = np.zeros(len(move), bool)
    return move * (1 - share) + spread(move, edges, none, ~none, rounds) * share


# A frame on the eyes is no wider than the face at the temples (round 51: the old man's glasses stood out past his face in 3/4,
# fitted to the kit's head before the drawn one narrowed it): the front of the frame comes in to the face's width there and this
# much more, the frame narrowed across as a whole.
FRAME_CLEAR_M = 0.003
# The face at the temples: the skin within this height of the eyes and this far behind the front of the frame.
FRAME_TEMPLE_TALL_M = 0.01
FRAME_TEMPLE_BEHIND_M = 0.03


def frame_scale(frame, skin, eye_z):
    """How much a frame of glasses on the eyes (`frame`, its points) is narrowed across to sit on the face (`skin`, its points): 1
    when it is no wider than the face at the temples with FRAME_CLEAR_M to spare. The face looks at -y."""
    frame, skin = np.asarray(frame, float), np.asarray(skin, float)
    middle = float(np.median(skin[:, 0]))
    front_y = float(frame[:, 1].min())
    front = frame[frame[:, 1] < front_y + 0.01]
    half = float(np.abs(front[:, 0] - middle).max())
    side = skin[(np.abs(skin[:, 2] - eye_z) < FRAME_TEMPLE_TALL_M) & (skin[:, 1] > front_y) & (skin[:, 1] < front_y + FRAME_TEMPLE_BEHIND_M)]
    if not len(side) or half <= 0:
        return 1.0
    face = float(np.abs(side[:, 0] - middle).max())
    return min(1.0, (face + FRAME_CLEAR_M) / half)

# Glasses worn on top of the head rest on the hair (round 66: the young woman's sunglasses stood 1.4 to 4 cm over her hair, the hair
# having followed the drawn head and the glasses not): moved as a whole until their nearest point is this far from the hair's
# nearest point (points of the hair, so a little more than the distance to its surface), over at most this many steps.
FRAME_ON_HAIR_M = 0.003
FRAME_SETTLE_ROUNDS = 8
# Each step goes towards the hair along the mean way from this share of the frame's points nearest it.
FRAME_SETTLE_SHARE = 0.1

# A cap or a hat sits on the head (round 66: the ice-cream girl's cap, fitted to the game's head, stood 13 to 43 mm off the drawn one
# at the forehead, a helmet): it is narrowed across about the head's upright axis until this share of the points of its rim that
# lie this high over the eyes stand HAT_CLEAR_M off the head or the hair under it.
HAT_CLEAR_M = 0.003
HAT_RIM_BAND_M = (0.02, 0.09)
# Round 68: at 0.9 a tenth of the rim went into the hair, which then stood out under the cap at the temples ("pecinhas escuras
# saem das têmporas", both judges): the whole rim stays clear.
HAT_RIM_SHARE = 1.0


def hat_scale(rim, gaps, centre, clear=HAT_CLEAR_M):
    """How much a hat is narrowed across (x and y) about the head's upright axis through `centre` for its `rim` points, each `gaps`
    off the head, to stand `clear` off it: the narrowing that leaves HAT_RIM_SHARE of them clear, and 1 when it is not wider."""
    rim, gaps = np.asarray(rim, float), np.asarray(gaps, float)
    if not len(rim):
        return 1.0
    reach = np.hypot(rim[:, 0] - centre[0], rim[:, 1] - centre[1])
    need = (reach - gaps + clear) / np.maximum(reach, 1e-9)
    return float(min(1.0, np.quantile(need, HAT_RIM_SHARE)))

# The arms of a frame lie along the head (round 68: the market owner's stood 11 to 24 mm off it and ended in hooks in the air, the
# young woman's sunglasses' ran straight back off her hair, both judges): from FRAME_ARM_FROM_M behind the front of the frame, each
# slice of an arm FRAME_ARM_SLICE_M long goes whole towards the head, fully from FRAME_ARM_RAMP_M further back; the front stays.
# Round 69: each slice taken to its own gap, the arms came out "em zigue-zague, como arame quebrado" (judge A): an arm goes one
# way, and by a smooth curve that stays under every slice's gap (so no slice goes nearer than FRAME_ON_HAIR_M), evened over
# FRAME_ARM_EVEN slices for FRAME_ARM_EVEN_ROUNDS rounds.
FRAME_ARM_FROM_M = 0.01
FRAME_ARM_RAMP_M = 0.015
FRAME_ARM_SLICE_M = 0.005
FRAME_ARM_EVEN = 7
FRAME_ARM_EVEN_ROUNDS = 30


def under_smooth(gaps, width=FRAME_ARM_EVEN, rounds=FRAME_ARM_EVEN_ROUNDS):
    """A smooth curve under `gaps` (one per slice, in order): evened over `width` slices and kept under them, round after round."""
    gaps = np.asarray(gaps, float)
    curve = gaps.copy()
    half = width // 2
    for _ in range(rounds):
        padded = np.concatenate([np.repeat(curve[:1], half), curve, np.repeat(curve[-1:], half)])
        curve = np.minimum(gaps, np.convolve(padded, np.ones(width) / width, mode="valid"))
    return curve


def arms_on_head(frame, surface, clear=FRAME_ON_HAIR_M):
    """The move (n x 3) of each point of a frame of glasses (`frame`, the face looking at -y) that lays its arms on the head
    (`surface`, its points): slice by slice along each arm, as a whole, one way for the arm, by a smooth curve (`under_smooth`)."""
    frame = np.asarray(frame, float)
    move = np.zeros_like(frame)
    behind = frame[:, 1] - frame[:, 1].min()
    middle = float(np.median(frame[:, 0]))
    for side in (frame[:, 0] < middle, frame[:, 0] >= middle):
        arm = side & (behind > FRAME_ARM_FROM_M)
        if not arm.any():
            continue
        slices = np.floor((behind[arm] - FRAME_ARM_FROM_M) / FRAME_ARM_SLICE_M).astype(int)
        distance, nearest = nearest_on(frame[arm], surface)
        count = slices.max() + 1
        gaps, ways, filled = np.zeros(count), np.zeros((count, 3)), np.zeros(count, bool)
        for index in np.unique(slices):
            here = slices == index
            filled[index] = True
            gaps[index] = max(float(distance[here].min()) - clear, 0.0)
            ways[index] = (nearest[here] - frame[arm][here]).mean(axis=0)
        if not gaps.any():
            continue
        # An empty slice takes the gap of the nearest filled one before it.
        for index in range(count):
            if not filled[index]:
                gaps[index] = gaps[index - 1] if index else gaps[filled.argmax()]
        way = (ways * gaps[:, None]).sum(axis=0)
        way /= max(np.linalg.norm(way), 1e-9)
        curve = under_smooth(gaps)
        move[arm] = way * (curve[slices] * np.clip((behind[arm] - FRAME_ARM_FROM_M) / FRAME_ARM_RAMP_M, 0, 1))[:, None]
    return move


# Round 69 (judge A): the old men's glasses "flutuam ~1 cm à frente dos olhos": a frame on the eyes goes back as a whole till the
# point of its front nearest the face (the bridge on the nose) is this far off it.
FRAME_ON_NOSE_M = 0.002


def nearest_on(points, surface, chunk=256):
    """For each of `points`, its distance to the nearest of `surface`'s points, and that point."""
    points, surface = np.asarray(points, float), np.asarray(surface, float)
    distance, nearest = np.empty(len(points)), np.empty_like(points)
    for start in range(0, len(points), chunk):
        part = points[start:start + chunk]
        gaps = np.linalg.norm(part[:, None, :] - surface[None, :, :], axis=2)
        index = gaps.argmin(axis=1)
        distance[start:start + chunk], nearest[start:start + chunk] = gaps[np.arange(len(part)), index], surface[index]
    return distance, nearest


def settle_on(frame, surface, clear=FRAME_ON_HAIR_M):
    """The move (x, y, z) that takes a frame of glasses (`frame`, its points) as a whole onto `surface` (its points), its nearest
    point `clear` from it: no point of the frame comes nearer than `clear`, since no point moves further than the gap it closes."""
    frame = np.asarray(frame, float)
    move = np.zeros(3)
    for _ in range(FRAME_SETTLE_ROUNDS):
        distance, nearest = nearest_on(frame + move, surface)
        gap = float(distance.min()) - clear
        if gap <= 1e-4:
            break
        close = distance <= np.quantile(distance, FRAME_SETTLE_SHARE)
        way = (nearest[close] - (frame + move)[close]).mean(axis=0)
        move += way / max(np.linalg.norm(way), 1e-9) * gap
    return move



# An eyebrow follows the skin under it along its length, the same across it (round 66: the old man's lids lowered, the skin under
# his eyebrows went down with them and took their lower edge and not their upper one, a tuft by the nose 3 cm tall). Each side's
# move is fitted along the eyebrow by a polynomial of this degree.
BROW_FOLLOW_DEGREE = 3


def along_only(points, move, degree=BROW_FOLLOW_DEGREE):
    """The move (n x 3) of an eyebrow's `points` (both eyebrows, side by side across x) made the same for every point at the same
    place along it: each side's, fitted along x, so the eyebrow bends with the skin and keeps its thickness."""
    points, move = np.asarray(points, float), np.asarray(move, float)
    out = move.copy()
    middle = points[:, 0].mean()
    for side in (points[:, 0] < middle, points[:, 0] >= middle):
        if side.sum() <= degree + 1:
            continue
        for axis in range(3):
            out[side, axis] = np.polyval(np.polyfit(points[side, 0], move[side, axis], degree), points[side, 0])
    return out


def follow_field(before, after):
    """`field(points)`: each point moved as the skin nearest it moved from `before` to `after` (n x 3, the same points), the skin
    that stayed included: an eyebrow over a lid that came down stays with the skin under it (round 47: Dona Marta's eyebrow went
    down onto her lid, the nearest skin that had moved)."""
    before, after = np.asarray(before, float), np.asarray(after, float)
    shift = after - before
    moved = np.linalg.norm(shift, axis=1) > 1e-5
    if moved.any():
        low, high = before[moved].min(axis=0) - FOLLOW_FAR, before[moved].max(axis=0) + FOLLOW_FAR
        about = np.all((before > low) & (before < high), axis=1)
    else:
        low = high = None
        about = moved
    source, step = before[about].astype(np.float32), shift[about]

    def field(points):
        points = np.asarray(points, float)
        out = points.copy()
        if low is None:
            return out
        near = np.flatnonzero(np.all((points > low) & (points < high), axis=1))
        for start in range(0, len(near), 128):
            chunk = near[start:start + 128]
            far = np.linalg.norm(points[chunk, None, :].astype(np.float32) - source[None, :, :], axis=2)
            count = min(FOLLOW_NEAREST, len(source))
            nearest = np.argpartition(far, count - 1, axis=1)[:, :count]
            distance = np.take_along_axis(far, nearest, axis=1)
            weight = 1.0 / (distance + 1e-4)
            mean = (step[nearest] * weight[..., None]).sum(axis=1) / weight.sum(axis=1)[:, None]
            out[chunk] += mean * smooth((FOLLOW_FAR - distance.min(axis=1)) / (FOLLOW_FAR - FOLLOW_NEAR))[:, None]
        return out

    return field


def lower_face_field(tip, eyes, kind):
    """`field(points)`: the face under the nose made shorter by its kind's LOWER_FACE (the chin comes up, the mouth part of the
    way), the neck under the chin following it back to where it was. For the skin and the teeth and tongue alike."""
    centres = [np.asarray(centre, float) for centre, _ in eyes]
    unit = float(np.linalg.norm(centres[0] - centres[1]))
    tip = np.asarray(tip, float)
    chin = tip[2] + LOWER_CHIN * unit
    lift = (1 - float(os.environ.get(LOWER_FACE_TRIAL) or LOWER_FACE[kind])) * (tip[2] - chin)

    def field(points):
        points = np.asarray(points, float).copy()
        z = points[:, 2]
        share = np.where(z > chin, (tip[2] - z) / (tip[2] - chin), 1.0 - (chin - z) / (LOWER_NECK * unit))
        # Eased at both ends and wide across (round 36: a crease along the jaw and under the chin where the lift turned at a corner).
        share = smooth(np.clip(share, 0, 1)) * (z < tip[2])
        share *= smooth((LOWER_WIDE - np.abs(points[:, 0] - tip[0]) / unit) / 0.9)
        # The front of the face and the throat only (round 32: moved all round, the back of the neck came out through the collar).
        # Faded back over LOWER_BACK_FADE (round 37: over half a distance the floor under the chin tilted up and creased along the jaw).
        share *= smooth((tip[1] + LOWER_BACK * unit - points[:, 1]) / (LOWER_BACK_FADE * unit))
        points[:, 2] += lift * share
        return points

    return field


def sculpt_field(tip, eyes, kind):
    """`field(points)`: the head laid onto the forms of its kind (SCULPT), the face looking at -y."""
    spec = SCULPT[kind]
    centres = [np.asarray(centre, float) for centre, _ in eyes]
    unit = float(np.linalg.norm(centres[0] - centres[1]))
    tip = np.asarray(tip, float)

    def field(points):
        points = np.asarray(points, float).copy()
        front = points[:, 1] < tip[1] + 0.9 * unit
        # The cheeks' and the chin's balls take the front of the head whole and fade out behind it (FRONT_FADE): cut at a plane
        # 0.9 distances between the eyes behind the nose tip, the skin in front of it went out to the cheek's ball by millimetres
        # and the skin behind it not at all, a step down the side of the face to the line of the jaw (owner, 2026-11-17).
        forward = smooth((tip[1] + FRONT_FADE[1] * unit - points[:, 1]) / ((FRONT_FADE[1] - FRONT_FADE[0]) * unit))[:, None]
        # The upper lid lifted off the iris (round 24, 2026-11-16: "olhos pequenos e fundos, sem branco em volta"; the blink run
        # backwards stops at half the width): the skin over each globe goes up by its kind's "lid" of its radius at the rim, fading out.
        for (centre, radius) in eyes:
            centre = np.asarray(centre, float)
            off = points - centre
            far = np.linalg.norm(off, axis=1) / radius
            share = smooth((LID_REACH - far) / (LID_REACH - 1.15)) * smooth((off[:, 2] / radius - 0.1) / 0.4)
            share *= smooth((1.3 - np.abs(off[:, 0]) / radius) / 0.5) * front * (off[:, 1] < 0.3 * radius)
            points[:, 2] += spec["lid"] * radius * share
        # The brow ridge over each eye comes forward (round 35: "arco da sobrancelha que avança", the eye set in an orbit and not
        # laid on the surface): the skin from the top of the globe to above the eyebrow, most at BROW_RIDGE_UP radii over it.
        for (centre, radius) in eyes:
            off = points - np.asarray(centre, float)
            up = off[:, 2] / radius
            share = np.exp(-((up - BROW_RIDGE_UP) / BROW_RIDGE_TALL) ** 2) * smooth((1.6 - np.abs(off[:, 0]) / radius) / 0.6)
            share *= front * (off[:, 1] < 0.5 * radius) * (up > 0.9)
            points[:, 1] -= spec["ridge"] * radius * share
        # The cheeks out onto a ball under each eye.
        for centre, _ in eyes:
            side = np.sign(centre[0] - (centres[0][0] + centres[1][0]) / 2)
            at = np.asarray(centre, float) + np.array([side * CHEEK_AT[0], CHEEK_AT[1], CHEEK_AT[2]]) * unit
            radius = spec["cheek"] * unit
            points = points + forward * (onto_ball(points, at, (radius * spec.get("cheek_wide", 1.0), radius, radius * 0.9), spec["cheek_share"], True, BALL_EASE) - points)
        # The chin onto a ball.
        chin = tip + np.array(CHIN_AT) * unit
        radius = spec["chin"] * unit
        points = points + forward * (onto_ball(points, chin, (radius * 1.1, radius, radius * 0.85), CHIN_SHARE, True, BALL_EASE) - points)
        # The nose onto a ball behind its tip, in the zone of the nose only.
        radii = np.array(spec["nose_ball"]) * unit
        # The ball stands a little above the tip, so the tip comes up into it: a nose shorter and round, not hanging (round 13).
        ball = tip + np.array([0.0, NOSE_BALL_BACK * radii[1], NOSE_BALL_UP * unit])
        zone = np.linalg.norm((points - tip) * np.array([1.0, 1.0, 1.4]), axis=1) < NOSE_ZONE * unit
        zone &= points[:, 1] < tip[1] + 2 * radii[1]
        # Only outwards (2026-11-16): pulled onto the ball from outside too, the bridge and the wings were pressed into it.
        points[zone] = onto_ball(points[zone], ball, radii, spec["nose_share"], True)
        return points

    return field
