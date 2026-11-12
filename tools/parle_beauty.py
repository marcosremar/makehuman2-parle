"""
    License information: data/licenses/makehuman_license.txt

    How good-looking a face of the cast is, by measure.

    Owner, 2026-11-11: "estou achando o rosto do avatar ainda muito feio. busca na internet forma de medir a beleza e usa isso como
    critério", and the same day, on what the cast is for: "o objetivo é ter avatares tipo o Rafa e Sofia, para serem mais estilo Disney".

    A face is measured on a portrait from the front, at rest, by the points of the Face Mesh (MediaPipe Face Landmarker, 478 points) and,
    for the lips against the skin, by the picture itself. Each measure is a share of the face, so the size of the picture does not
    count; the whole face has to be on it (`whole`), and a face is read on several portraits and given the middle value of each
    measure (`settled`). What the literature says counts, and which way (every DOI checked against Crossref, 2026-11-11):
      - A woman's face is found more attractive with bigger eyes (their height over the length of the face, r = +.50), eyes further
        apart, a smaller nose, a shorter chin, a face narrower at the mouth against the cheekbones and higher eyebrows: Cunningham
        (1986), "Measuring the physical in physical attractiveness: quasi-experiments on the sociobiology of female facial beauty",
        Journal of Personality and Social Psychology 50(5):925-935, DOI 10.1037/0022-3514.50.5.925. Correlations on photographs of
        women, with no value to aim at.
      - The eyes of an attractive face stand about 46% of the width of the face apart, pupil to pupil: Pallett, Link & Lee (2010), "New
        'golden' ratios for facial beauty", Vision Research 50(2):149-154, DOI 10.1016/j.visres.2009.11.003. Their width is taken
        between the ears and ours between the cheekbones; their other ratio, eyes to mouth over the length of the face from the
        hairline, is not measured here: the Face Mesh has no hairline.
      - The two halves of an attractive face are alike (r = .25 over the studies gathered): Rhodes (2006), "The evolutionary psychology
        of facial beauty", Annual Review of Psychology 57:199-226, DOI 10.1146/annurev.psych.57.102904.190208.
      - A woman's lips and eyes are darker against her skin than a man's, and her face is found more attractive with more of that
        contrast, his with less: Russell (2003), "Sex, beauty, and the relative luminance of facial features", Perception
        32(9):1093-1107, DOI 10.1068/p5101; Russell (2009), "A sex difference in facial contrast and its exaggeration by cosmetics",
        Perception 38(8):1211-1219, DOI 10.1068/p6331; for the lips, Stephen & McKeegan (2010), "Lip colour affects perceived sex
        typicality and attractiveness of human faces", Perception 39(8):1104-1110, DOI 10.1068/p6730. Directions, from the abstracts.
      - A face at rest that looks like an expression is judged as that expression, so a mouth whose corners turn down reads as sad or
        cross: Said, Sebe & Todorov (2009), "Structural resemblance to emotional expressions predicts evaluation of emotionally neutral
        faces", Emotion 9(2):260-264, DOI 10.1037/a0014681. A smiling face is found more attractive: Reis et al. (1990), "What is
        smiling is beautiful and good", European Journal of Social Psychology 20(3):259-267, DOI 10.1002/ejsp.2420200307.
      - How far a feature may go depends on the head it is on. On photographs, eyes are judged too large from 116% of their size on
        (SD 9), and eyes of 150% are unpleasant on a realistic face and hardly matter on a drawn one: Seyama & Nagayama (2007), "The
        uncanny valley: effect of realism on the impression of artificial human faces", Presence 16(4):337-351, DOI
        10.1162/pres.16.4.337, and (2009), "Probing the uncanny valley with the eye size aftereffect", Presence 18(5):321-339, DOI
        10.1162/pres.18.5.321. Shape and surface are best drawn to the same degree: Zell et al. (2015), "To stylize or not to
        stylize? The effect of shape and material stylization on the perception of computer-generated faces", ACM Transactions on
        Graphics 34(6):184, DOI 10.1145/2816795.2818126.
    None of them gives the size a drawn eye should have. How far each measure should go is read on the two faces the owner gave as the
    aim, Sofia and Rafa, measured the same way (`AIM`): a choice of design, to be judged by eye in the game, not a finding.

    What it does not see is the surface: a fold or a bag in the skin, the light on it, how thick the eyebrows are drawn. On 2026-11-11
    a face whose eyes had been grown 1.4 times went from 6.3 to 8.8 with a bag under each eye. The mark is a criterion, and every change
    is looked at on the portrait as well.

    The ruler itself was wrong in three places, found on 2026-11-12 with the measures drawn over the portraits (a ruler that accuses
    what the eye does not see is mended in the ruler):
      - The portrait cut the chin of a big head, Sofia's among them, and the aims had been read on it (`whole`).
      - The eye as it shows was whatever stood apart from the skin round the eye, by colour. On the lit portrait the shade at the temple
        joined the eye and counted (Sofia 0.099 with it, 0.077 without). Without the light, three quarters of Sofia's was the line
        drawn on her lids, and the cast has none: owner, 2026-11-06, 09 and 11, "sem cílios". And the white of the eye of a fair face
        fell in or out of it with the threshold. It is now the opening between the lids, by the Face Mesh (`eye_shown`): what
        Cunningham measured was the eye. How dark the eye is against the skin (Russell 2009) is then not marked: without a drawn line
        it would mark the iris against the skin, and a darker skin would lose by its colour.
      - The width of the mouth was not marked. No study read here gives it a direction: it is a choice of design, the mouth at rest
        of the two faces the owner gave as the aim, small against the wide flat mouth the women of the cast had (seen on the portrait).

    From the side the mouth has a measure of its own, read on the silhouette of the head (`parle_profile`): the upper lip leads the
    lower one. It is marked here with the rest when the caller has it.

    Numpy alone, and OpenCV for the measure read on the picture. The portraits and their points come from the caller.
"""
import numpy as np

import parle_profile
import parle_symmetry

# Face Mesh points. Left and right are the picture's.
TOP, CHIN = 10, 152
CHEEKS, JAW = (234, 454), (172, 397)
EYES = ((33, 133, 159, 145), (263, 362, 386, 374))  # outer corner, inner corner, upper lid, lower lid
BROWS = (105, 334)
NASION, SUBNASALE, NOSE = 168, 2, (48, 278)
MOUTH, LIP_TOP, LIP_BOTTOM, LIP_MEET = (61, 291), 0, 17, (13, 14)
LIPS = (61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146)
PAIRS = ((33, 263), (133, 362), (159, 386), (145, 374), (105, 334), (48, 278), (61, 291), (234, 454), (172, 397), (70, 300), (50, 280))
# The ring of points round the opening of each eye, lower lid from the outer corner to the inner one and back along the upper lid.
LIDS = ((33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246),
        (263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466))
# The skin round the lips is a ring from 2 to 7% of the width of the face away from them.
LIP_RING = (0.02, 0.07)
# A face is whole on its portrait with this share of the picture between it and the border (`whole`).
EDGE = 0.02

# The aim, by sex: each measure on Sofia's face (frontend/game/assets/npcs/models/mulher-jeans-disney.glb) and on Rafa's
# (home-jeans-disney.glb), read on 2026-11-12 on three portraits with the whole face on them (the first reading, of 2026-11-11, was on
# one portrait that cut Sofia's chin), and how far from it a face scores nothing. A measure with `more` scores full from the aim on, up
# to MOST times the aim: past that a feature is no longer bigger, it is another face. The corners of the mouth score full from the aim
# up to a reach above it (`up`): Sofia's sit level with the middle of the lips and those of the women of the game up to 4% of the
# width of the mouth above it; Rafa's 2% below. A man's lips score nearest to Rafa's, not darker (Russell 2003). The width of the
# mouth is a choice of design with no study behind it: nearest to the aim, with a longer reach than the rest. The two halves of a face
# cannot be too much alike (`least`): with `less`, a face more even than Sofia's by a third lost marks for it.
AIM = {
    "female": {
        "eye_shown": (0.024, 0.015, "more"),
        "eye_open": (0.084, 0.035, "more"),
        "eyes_apart": (0.46, 0.06, "near"),
        "eye_line": (0.328, 0.060, "near"),
        "brow_high": (0.172, 0.045, "more"),
        "nose_wide": (0.226, 0.070, "less"),
        "chin_long": (0.203, 0.060, "less"),
        "jaw_wide": (0.775, 0.100, "less"),
        "lips_full": (0.340, 0.150, "near"),
        "mouth_wide": (0.341, 0.120, "near"),
        "mouth_curve": (0.003, 0.040, "up"),
        "lip_contrast": (0.145, 0.120, "more"),
        "uneven": (0.015, 0.035, "least"),
    },
    "male": {
        "eye_shown": (0.016, 0.010, "more"),
        "eye_open": (0.059, 0.030, "more"),
        "eyes_apart": (0.46, 0.06, "near"),
        "eye_line": (0.308, 0.060, "near"),
        "brow_high": (0.152, 0.045, "more"),
        "nose_wide": (0.263, 0.070, "less"),
        "chin_long": (0.251, 0.070, "near"),
        "jaw_wide": (0.822, 0.100, "near"),
        "lips_full": (0.241, 0.150, "near"),
        "mouth_wide": (0.408, 0.120, "near"),
        "mouth_curve": (-0.019, 0.040, "up"),
        "lip_contrast": (0.032, 0.120, "near"),
        "uneven": (0.015, 0.035, "least"),
    },
}
# And from the side, read on the silhouette (`parle_profile`): the upper lip ahead of the lower one.
AIM = {sex: {**rules, **parle_profile.AIM[sex]} for sex, rules in AIM.items()}
MOST = 1.6
# The faces the owner likes. On 2026-11-12 he looked at 52 portraits from the front, the faces of the game and five of the new
# cast, and said of each whether it is beautiful ("guardar métrica de avatares que eu acho bonito para usar eles como referência"):
# 15 women and 17 men are. Each measure on those faces, read as the cast's are (three portraits, the middle value): its mean and
# how far they spread about it (standard deviation, n - 1). The verdicts and every reading are in babylon-cinema,
# docs/reports/2026-11-11-beleza-rosto/gabarito/. A face is nearer to what he likes the nearer it stands to their middle: a face
# made of many is found more attractive than nearly all the faces it is made of (Langlois & Roggman 1990, "Attractive faces are
# only average", Psychological Science 1(2):115-121, DOI 10.1111/j.1467-9280.1990.tb00079.x), and the average of the faces found
# attractive more than the average of all (Perrett, May & Yoshikawa 1994, "Facial shape and judgements of female attractiveness",
# Nature 368(6468):239-242, DOI 10.1038/368239a0); an attractive face is not always an average one (DeBruine, Jones, Unger, Little
# & Feinberg 2007, "Dissociating averageness and attractiveness: attractive faces are not always average", Journal of
# Experimental Psychology: Human Perception and Performance 33(6):1420-1430, DOI 10.1037/0096-1523.33.6.1420). The three read in
# their abstracts, the DOIs checked against their records, 2026-11-12. The studies lay the whole shape of each face on the others
# (Procrustes) and take the distance; here it is taken on the measures of this ruler, which say which feature stands off.
# What told the faces he faulted from these, in his words and on these measures: a woman's jaw of 0.75 to 0.765 ("queixo fino e
# rosto largo, parece um triângulo invertido"; theirs 0.798), a man's face of 0.864 to 0.900 wide ("maxilar muito largo"; theirs
# 0.832, none past 0.860), eyebrows 2.5 to 3.7 deviations low ("sobrancelha feia"). The measures of how alike the two halves are
# did not: the faces he faulted are no less even (`parle_symmetry`, `parle_mirror`). Each of the faces he likes, measured against
# the others of its sex, stands 0.6 to 2.4 off (1.0 the middle one); the faces he faulted, 0.75 to 1.7 (1.2). Of the women, 12 of
# the 15 he likes stand within 1.1 and the three he faulted at 1.4 to 1.7; the men he faulted do not stand further than the others.
# Fifteen and seventeen faces, most of them drawn on one head: on a measure they all share (the upper lip ahead of the lower one
# in a woman's profile, 0.108 with a deviation of 0.007) any face built another way stands ten deviations off. So how far a face
# stands counts each measure for no more than LIKED_MOST deviations, and what is to be looked at is which measures stand off.
LIKED = {
    "female": {
        "eye_shown": (0.0250, 0.0027), "eye_open": (0.0838, 0.0070), "eye_wide": (0.2205, 0.0072), "eyes_apart": (0.4662, 0.0111),
        "eye_line": (0.3263, 0.0102), "brow_high": (0.1768, 0.0054), "nose_wide": (0.2549, 0.0137), "nose_long": (0.3016, 0.0052),
        "mouth_wide": (0.3886, 0.0483), "lips_full": (0.3167, 0.0483), "mouth_curve": (0.0275, 0.0416), "chin_long": (0.2040, 0.0130),
        "jaw_wide": (0.7984, 0.0226), "face_wide": (0.9008, 0.0215), "lip_contrast": (0.1487, 0.1106), "upper_leads": (0.1083, 0.0066),
        "uneven": (0.0129, 0.0043), "sym_sides": (0.3304, 0.1783), "sym_level": (0.0035, 0.0012), "sym_midline": (0.0073, 0.0041),
        "sym_size": (0.0586, 0.0277),
    },
    "male": {
        "eye_shown": (0.0158, 0.0013), "eye_open": (0.0588, 0.0038), "eye_wide": (0.2015, 0.0058), "eyes_apart": (0.4421, 0.0137),
        "eye_line": (0.3109, 0.0118), "brow_high": (0.1605, 0.0105), "nose_wide": (0.2741, 0.0170), "nose_long": (0.2895, 0.0091),
        "mouth_wide": (0.4344, 0.0317), "lips_full": (0.2390, 0.0371), "mouth_curve": (-0.0123, 0.0309), "chin_long": (0.2551, 0.0174),
        "jaw_wide": (0.8385, 0.0176), "face_wide": (0.8316, 0.0183), "lip_contrast": (-0.0308, 0.1087), "upper_leads": (0.0652, 0.0305),
        "uneven": (0.0140, 0.0036), "sym_sides": (0.1394, 0.0499), "sym_level": (0.0034, 0.0009), "sym_midline": (0.0082, 0.0035),
        "sym_size": (0.0496, 0.0138),
    },
}
LIKED_MOST = 4.0
NAMES = {
    **parle_profile.NAMES,
    "eye_shown": "olho que aparece (abertura/rosto)", "eye_open": "olho aberto (altura/rosto)", "eyes_apart": "olhos afastados (entre centros/largura)",
    "eye_line": "altura dos olhos no rosto", "brow_high": "sobrancelha acima do olho", "nose_wide": "largura do nariz", "chin_long": "queixo (lábio ao queixo)",
    "jaw_wide": "mandíbula contra as maçãs", "lips_full": "lábios (altura/largura da boca)", "mouth_wide": "largura da boca",
    "mouth_curve": "cantos da boca (acima do meio)",
    "lip_contrast": "lábio contra a pele (contraste)", "uneven": "assimetria",
}


def measures(points):
    """The measures of a face, from its Face Mesh points in the pixels of a portrait from the front: each a share of the face."""
    spot = lambda index: np.asarray(points[index][:2], float)
    apart = lambda a, b: float(np.linalg.norm(spot(a) - spot(b)))
    tall, wide = apart(TOP, CHIN), apart(*CHEEKS)
    centres = [(spot(outer) + spot(inner)) / 2 for outer, inner, _, _ in EYES]
    eyes_y = float(np.mean([centre[1] for centre in centres]))
    eye_wide = float(np.mean([apart(outer, inner) for outer, inner, _, _ in EYES]))
    middle = float(np.mean([spot(index)[0] for index in (TOP, CHIN, NASION, SUBNASALE)]))
    uneven = np.mean([abs((spot(a)[0] - middle) + (spot(b)[0] - middle)) + abs(spot(a)[1] - spot(b)[1]) for a, b in PAIRS]) / wide
    return {
        "eye_shown": eye_shown(points),
        "eye_open": float(np.mean([apart(up, low) for _, _, up, low in EYES])) / tall,
        "eye_wide": eye_wide / wide,
        "eyes_apart": float(np.linalg.norm(centres[0] - centres[1])) / wide,
        "eye_line": (eyes_y - spot(TOP)[1]) / tall,
        "brow_high": (eyes_y - float(np.mean([spot(index)[1] for index in BROWS]))) / tall,
        "nose_wide": apart(*NOSE) / wide,
        "nose_long": apart(NASION, SUBNASALE) / tall,
        "mouth_wide": apart(*MOUTH) / wide,
        "lips_full": apart(LIP_TOP, LIP_BOTTOM) / apart(*MOUTH),
        "mouth_curve": float(np.mean([spot(index)[1] for index in LIP_MEET]) - np.mean([spot(index)[1] for index in MOUTH])) / apart(*MOUTH),
        "chin_long": (spot(CHIN)[1] - spot(LIP_BOTTOM)[1]) / tall,
        "jaw_wide": apart(*JAW) / wide,
        "face_wide": wide / tall,
        "uneven": float(uneven),
        **parle_symmetry.measures(points),
    }


def whole(points, size):
    """Whether a face is all on its portrait: the top of the forehead, the chin and the cheekbones at least EDGE of the picture from its
    border. `size` is the picture's width and height. The Face Mesh fits the face as a whole, so with the chin cut every point moves:
    on 2026-11-12 Sofia's eyes read 0.089 open and her chin 0.181 long on a portrait that cut it, and 0.083 and 0.202 with the whole
    face on the picture. The aims had been read on the cut one."""
    spots = np.asarray([points[index][:2] for index in (TOP, CHIN, *CHEEKS)], float)
    low, high = EDGE * np.asarray(size, float), (1 - EDGE) * np.asarray(size, float)
    return bool((spots >= low).all() and (spots <= high).all())


def settled(reads):
    """The measures of one face read on several portraits: the middle value of each. From one framing of the same face to another the
    Face Mesh moves a measure by up to 0.01, a sixth of what a mark has to fall from full to nothing."""
    names = {name for found in reads for name, value in found.items() if value is not None}
    return {name: float(np.median([found[name] for found in reads if found.get(name) is not None])) for name in names}


def eye_shown(points):
    """The eye as it shows, as a share of the box of the face: the opening between the lids of the two eyes, by their rings of points.
    What is drawn round the eye, a lash or a line on the lid, is not eye."""
    spot = lambda index: np.asarray(points[index][:2], float)
    tall, wide = np.linalg.norm(spot(TOP) - spot(CHIN)), np.linalg.norm(spot(CHEEKS[0]) - spot(CHEEKS[1]))
    shown = 0.0
    for ring in LIDS:
        x, y = np.array([spot(index) for index in ring]).T
        shown += 0.5 * abs(float(x @ np.roll(y, 1) - y @ np.roll(x, 1)))
    return shown / float(wide * tall)


def lip_contrast(image, points):
    """How much darker the lips are than the skin round them: (skin - lips) / (skin + lips) in lightness, the contrast between a
    feature and the skin about it. `image` is the portrait without light and shade, the colours of the face alone, in blue, green and
    red; with the light on it the shadow under the lower lip would count as lip. None when the lips have no room on the picture."""
    import cv2

    light = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    spot = lambda index: np.asarray(points[index][:2], float)
    wide = float(np.linalg.norm(spot(CHEEKS[0]) - spot(CHEEKS[1])))
    lips = np.zeros(light.shape, np.uint8)
    cv2.fillPoly(lips, [np.round([spot(index) for index in LIPS]).astype(np.int32)], 1)
    grown = lambda share: cv2.dilate(lips, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(share * wide) + 1,) * 2))
    inside, around = cv2.erode(lips, np.ones((3, 3), np.uint8)), grown(LIP_RING[1]) & ~grown(LIP_RING[0]) & 1
    if not inside.any() or not around.any():
        return None
    lip, skin = float(light[inside > 0].mean()), float(light[around > 0].mean())
    return (skin - lip) / max(skin + lip, 1e-6)


def mark(value, aim, reach, way):
    """How well one measure does, from 0 to 1: full at the aim, nothing `reach` away from it. With `more` or `less` the side that makes
    a face more attractive scores full too, up to MOST times the aim (or down to the aim over MOST). With `up`, a measure that may be
    nothing or less, full is from the aim to a reach above it. With `least`, full is from the aim down, without end."""
    if way == "up":
        return float(max(0.0, 1 - max(aim - value, value - aim - reach, 0.0) / reach))
    if way == "least":
        return float(max(0.0, 1 - max(value - aim, 0.0) / reach))
    if way == "more" and aim <= value <= aim * MOST or way == "less" and aim / MOST <= value <= aim:
        return 1.0
    edge = aim * MOST if way == "more" and value > aim else aim / MOST if way == "less" and value < aim else aim
    return float(max(0.0, 1 - abs(value - edge) / reach))


def likeness(found, sex):
    """How far a face stands from the faces the owner likes (LIKED): for each measure both have, the difference from their mean
    in their deviations, and over all of them the root of the mean square, each counted for no more than LIKED_MOST. None when
    no measure is shared."""
    far = {name: (found[name] - mean) / spread for name, (mean, spread) in LIKED[sex].items() if isinstance(found.get(name), float)}
    if not far:
        return None, far
    return float(np.sqrt(np.mean(np.square(np.clip(list(far.values()), -LIKED_MOST, LIKED_MOST))))), far


def score(found, sex):
    """The mark of a face, 0 to 10, and the mark of each measure, 0 to 1. `found` holds the measures (`measures`, `lip_contrast` when
    the portrait without light was read, `upper_leads` when the profile was); a measure that is missing counts for nothing, neither
    way."""
    marks = {name: mark(found[name], *rule) for name, rule in AIM[sex].items() if found.get(name) is not None}
    return (10 * float(np.mean(list(marks.values()))) if marks else 0.0), marks
