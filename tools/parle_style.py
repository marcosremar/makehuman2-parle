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
import numpy as np

# How much the nose grows about a point behind its tip, and how thick the eyebrows are left (`parle_face.BROW_THIN` is the kind's
# own default, which this replaces). The series' young men have noses 0.31 to 0.37 of the face long against 0.29 to 0.31 on the cast,
# and much further out; its old man's is the biggest thing on his face.
STYLE = {
    "male": {"forward": 1.25, "nose": 1.6, "tip": 1.5, "brow": 1.4, "eye": 1.25, "skin": 1.55},
    # Round 41: the old man's eyebrow a flat white patch over the glasses (2.8) and his eye white all about the iris (`lid` in SCULPT lower).
    "male-old": {"forward": 1.3, "nose": 2.0, "tip": 1.65, "brow": 2.0, "eye": 1.25, "skin": 1.55},
    # Round 48 (5.7): both judges, three rounds running, asked the young woman's nose a third smaller, a button ("hoje é nariz de
    # homem"), and the old woman's tip shorter.
    "female": {"forward": 0.8, "nose": 1.15, "tip": 1.1, "brow": 1.2, "eye": 1.12, "skin": 1.55},
    "female-old": {"forward": 0.85, "nose": 1.45, "tip": 1.3, "brow": 1.9, "eye": 1.12, "skin": 1.55},
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
EYE_TALL_LEAST_BY_KIND = {"male": EYE_TALL_LEAST, "female": EYE_TALL_LEAST, "male-old": 0.45, "female-old": 0.52}
# How much the eye's own map lights it (emission): the white of the series' eyes is white even in the shade of the brow.
# Round 30 (2026-11-16): the white of the eye read grey-pink in the orbit's shade; the series' is clean white.
# Round 38 (two judges, 4.1): the white still "sombra funda em volta" on all four.
EYE_GLOW = 0.3
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
    "male": {"nose_ball": (0.41, 0.42, 0.33), "nose_share": 0.95, "cheek": 0.55, "cheek_share": 0.55, "chin": 0.5, "lid": 0.1, "ridge": 0.0},
    "male-old": {"nose_ball": (0.44, 0.46, 0.38), "nose_share": 0.95, "cheek": 0.62, "cheek_share": 0.85, "chin": 0.45, "lid": -0.16, "ridge": 0.0},
    "female": {"nose_ball": (0.22, 0.22, 0.19), "nose_share": 0.9, "cheek": 0.54, "cheek_share": 0.6, "chin": 0.36, "lid": -0.02, "ridge": 0.0},
    "female-old": {"nose_ball": (0.32, 0.32, 0.28), "nose_share": 0.9, "cheek": 0.62, "cheek_share": 0.85, "chin": 0.38, "lid": -0.08, "ridge": 0.0},
}
# Where the forms stand, in distances between the eyes from the middle of the eye globe of that side (cheeks) or from the nose tip
# (the nose ball behind it, the chin under it): across, back (+y), up.
# Round 35: the cheek a high round mass ("maçã alta"), its ball higher under the eye (0.14, 0.58, -0.48), and a brow ridge forward
# (`ridge` in SCULPT) and lighter smoothing: round 36 fell from 5.5 to 4.5, so back to round 34's.
CHEEK_AT = (0.12, 0.62, -0.62)
CHIN_AT = (0.0, 0.55, -1.25)
# How far about each form its pull reaches, in its own radii, and the zone of the nose about the tip, in distances between the eyes.
REACH = 1.4
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
ORBIT_ROUNDS = 30
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
LOWER_FACE = {"male": 0.83, "male-old": 0.87, "female": 0.85, "female-old": 0.87}
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
SIMPLE_MOUTH_WIDE = (0.62, 0.3)
# Round 35 (5.4) read "um balão liso" at 25; 12 in round 36 did not help (above).
SIMPLE_ROUNDS = 25
SIMPLE_IN_STEP = 0.5
SIMPLE_OUT_STEP = -0.53


def onto_ball(points, centre, radii, share, outward_only):
    """Each point moved `share` of the way to the surface of the ellipsoid about `centre` (radii across, back, up), along the line
    from the centre; with `outward_only`, only the points inside it, which fills hollows and leaves the rest."""
    off = points - centre
    scaled = np.linalg.norm(off / np.asarray(radii, float), axis=1)
    scaled = np.maximum(scaled, 1e-9)
    onto = centre + off / scaled[:, None]
    pull = share * smooth((REACH - scaled) / (REACH - 1.0))
    if outward_only:
        pull = np.where(scaled < 1.0, pull, 0.0)
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
    weight *= smooth((np.hypot(off[:, 0] / SIMPLE_MOUTH_WIDE[0], off[:, 2] / SIMPLE_MOUTH_WIDE[1]) - 1.0) / 0.6)
    return weight


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
    lift = (1 - LOWER_FACE[kind]) * (tip[2] - chin)

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
            points[front] = onto_ball(points[front], at, (radius, radius, radius * 0.9), spec["cheek_share"], True)
        # The chin onto a ball.
        chin = tip + np.array(CHIN_AT) * unit
        radius = spec["chin"] * unit
        points[front] = onto_ball(points[front], chin, (radius * 1.1, radius, radius * 0.85), CHIN_SHARE, True)
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
