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
    "male": {"forward": 0.95, "nose": 1.9, "tip": 1.6, "brow": 1.8, "eye": 1.25, "skin": 1.3},
    "male-old": {"forward": 1.0, "nose": 2.4, "tip": 1.75, "brow": 2.2, "eye": 1.25, "skin": 1.3},
    "female": {"forward": 0.8, "nose": 1.45, "tip": 1.3, "brow": 1.2, "eye": 1.12, "skin": 1.3},
    "female-old": {"forward": 0.85, "nose": 1.65, "tip": 1.4, "brow": 1.3, "eye": 1.12, "skin": 1.3},
}
# Round 3 of the loop judged by sight against the series (2026-11-15, 3.1 in 10) asked first for eyes about a third bigger and round,
# with white showing and a highlight; then warmer, more saturated skin; a bulb at the tip of the nose; eyebrows half as thick again.
# Round 7 (4.0): the nose too long and pointed from the side; it is to grow across and round, and come out less ("forward" is how
# much it grows to the front, against "nose" across and up).
# Round 6 (3.4) asked for skin smooth and matte, with no relief of pores; a wider mouth; fuller cheeks; thicker, higher eyebrows.
# Round 4 (3.25) asked first for eyes 40% more open with the white showing, then a bigger nose and fuller cheeks.
# "eye" multiplies the size the eyes of the kit's sex already take (`parle_polish.eye_scale`); "tip" grows the tip of the nose alone
# within NOSE_TIP_REACH_M; "skin" is how much more saturated the skin's colour is, turned a little towards orange (SKIN_WARM).
NOSE_TIP_REACH_M = 0.016
# How round an eye opens at least, its height over its width: the series' eyes are 0.6 to 0.7 (read on its portraits); the cast's
# lids gave 0.42, with the iris filling the opening (round 5).
EYE_TALL_LEAST = 0.58
# How much the eye's own map lights it (emission): the white of the series' eyes is white even in the shade of the brow.
EYE_GLOW = 0.0
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
    "male": {"nose_ball": (0.32, 0.27, 0.25), "nose_share": 0.95, "cheek": 0.52, "cheek_share": 0.45, "shell": 0.28, "chin": 0.42},
    "male-old": {"nose_ball": (0.38, 0.31, 0.3), "nose_share": 0.95, "cheek": 0.56, "cheek_share": 0.5, "shell": 0.3, "chin": 0.45},
    "female": {"nose_ball": (0.21, 0.18, 0.17), "nose_share": 0.9, "cheek": 0.5, "cheek_share": 0.45, "shell": 0.22, "chin": 0.36},
    "female-old": {"nose_ball": (0.25, 0.21, 0.2), "nose_share": 0.9, "cheek": 0.53, "cheek_share": 0.5, "shell": 0.24, "chin": 0.38},
}
# Where the forms stand, in distances between the eyes from the middle of the eye globe of that side (cheeks) or from the nose tip
# (the nose ball behind it, the chin under it): across, back (+y), up.
CHEEK_AT = (0.12, 0.62, -0.62)
CHIN_AT = (0.0, 0.55, -1.25)
# How far about each form its pull reaches, in its own radii, and the zone of the nose about the tip, in distances between the eyes.
REACH = 1.4
# The shell over an eye leaves the rim of the opening alone: from this many radii of the globe out (round 12: thickened over the rim,
# it closed the eye).
SHELL_FROM = 1.18
NOSE_BALL_UP = 0.1
NOSE_ZONE = 0.55


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


def sculpt_field(tip, eyes, kind):
    """`field(points)`: the head laid onto the forms of its kind (SCULPT), the face looking at -y."""
    spec = SCULPT[kind]
    centres = [np.asarray(centre, float) for centre, _ in eyes]
    unit = float(np.linalg.norm(centres[0] - centres[1]))
    tip = np.asarray(tip, float)

    def field(points):
        points = np.asarray(points, float).copy()
        front = points[:, 1] < tip[1] + 0.9 * unit
        # The skin about each eye out onto a thick shell over its globe: a lid with no fold of an anatomy, the orbit shallow.
        for (centre, radius) in eyes:
            centre = np.asarray(centre, float)
            off = points - centre
            far = np.maximum(np.linalg.norm(off, axis=1), 1e-9)
            ahead = (-off[:, 1] / far > 0.0) & front
            # The rim of the opening stays where the lids put it (SHELL_FROM): the shell thickens from there outwards.
            want = radius * (1.0 + spec["shell"] * smooth((far / radius - SHELL_FROM) / 0.6))
            near = ahead & (far < want) & (far < 2.4 * radius) & (far > SHELL_FROM * radius)
            fade = smooth((2.4 * radius - far) / (0.8 * radius))
            points[near] = centre + off[near] / far[near, None] * (far[near] + (want[near] - far[near]) * fade[near])[:, None]
        # The cheeks out onto a ball under each eye.
        for centre, _ in eyes:
            side = np.sign(centre[0] - (centres[0][0] + centres[1][0]) / 2)
            at = np.asarray(centre, float) + np.array([side * CHEEK_AT[0], CHEEK_AT[1], CHEEK_AT[2]]) * unit
            radius = spec["cheek"] * unit
            points[front] = onto_ball(points[front], at, (radius, radius, radius * 0.9), spec["cheek_share"], True)
        # The chin onto a ball.
        chin = tip + np.array(CHIN_AT) * unit
        radius = spec["chin"] * unit
        points[front] = onto_ball(points[front], chin, (radius * 1.1, radius, radius * 0.85), 0.5, True)
        # The nose onto a ball behind its tip, in the zone of the nose only.
        radii = np.array(spec["nose_ball"]) * unit
        # The ball stands a little above the tip, so the tip comes up into it: a nose shorter and round, not hanging (round 13).
        ball = tip + np.array([0.0, radii[1], NOSE_BALL_UP * unit])
        zone = np.linalg.norm((points - tip) * np.array([1.0, 1.0, 1.4]), axis=1) < NOSE_ZONE * unit
        zone &= points[:, 1] < tip[1] + 2 * radii[1]
        points[zone] = onto_ball(points[zone], ball, radii, spec["nose_share"], False)
        return points

    return field
