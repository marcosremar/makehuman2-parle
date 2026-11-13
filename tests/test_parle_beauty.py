"""Regression for the looks of a face of the cast (2026-11-11): run with `.local/venv/bin/python tests/test_parle_beauty.py`.

The owner found the faces "muito feio" and asked for a measure of beauty as the criterion. `parle_beauty` measures a face by the points of the
Face Mesh on a portrait and marks it against the two faces the cast aims at, Sofia and Rafa, in the directions the literature gives.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import parle_beauty  # noqa: E402
import parle_symmetry  # noqa: E402


def face(eye_open=10.0, eyes_apart=92.0, chin=36.0, nose=44.0, brow=36.0, lean=0.0, corners=0.0):
    """Face Mesh points of a face 200 px wide and 200 px tall, the top at y = 0: only the points the measures read are set."""
    points = np.zeros((478, 2))
    put = lambda index, x, y: points.__setitem__(index, (x + lean * y, y))
    put(parle_beauty.TOP, 0, 0), put(parle_beauty.CHIN, 0, 200)
    put(parle_beauty.CHEEKS[0], -100, 80), put(parle_beauty.CHEEKS[1], 100, 80)
    put(parle_beauty.JAW[0], -78, 150), put(parle_beauty.JAW[1], 78, 150)
    for sign, (outer, inner, up, low), ring in zip((-1, 1), parle_beauty.EYES, parle_beauty.LIDS):
        middle = sign * eyes_apart / 2
        # The opening of the eye as an ellipse 44 px wide and `eye_open` high: the ring runs from the outer corner along the lower lid.
        for turn, index in zip(np.linspace(0, 2 * np.pi, len(ring), endpoint=False), ring):
            put(index, middle + sign * 22 * np.cos(turn), 70 + eye_open / 2 * np.sin(turn))
        put(outer, middle + sign * 22, 70), put(inner, middle - sign * 22, 70), put(up, middle, 70 - eye_open / 2), put(low, middle, 70 + eye_open / 2)
    put(parle_beauty.BROWS[0], -46, 70 - brow), put(parle_beauty.BROWS[1], 46, 70 - brow)
    put(parle_beauty.NASION, 0, 62), put(parle_beauty.SUBNASALE, 0, 122)
    put(parle_beauty.NOSE[0], -nose / 2, 116), put(parle_beauty.NOSE[1], nose / 2, 116)
    put(parle_beauty.MOUTH[0], -35, 142 - corners), put(parle_beauty.MOUTH[1], 35, 142 - corners)
    put(parle_beauty.LIP_MEET[0], 0, 141), put(parle_beauty.LIP_MEET[1], 0, 143)
    put(parle_beauty.LIP_TOP, 0, 132), put(parle_beauty.LIP_BOTTOM, 0, 200 - chin)
    for a, b in ((70, 300), (50, 280)):
        put(a, -60, 50), put(b, 60, 50)
    return points


found = parle_beauty.measures(face())
assert abs(found["eye_open"] - 0.05) < 1e-9 and abs(found["eyes_apart"] - 0.46) < 1e-9, f"the eyes are not read as shares of the face: {found}"
assert abs(found["chin_long"] - 0.18) < 1e-9 and abs(found["nose_wide"] - 0.22) < 1e-9 and abs(found["brow_high"] - 0.18) < 1e-9, f"chin, nose or brows: {found}"
assert abs(found["eye_line"] - 0.35) < 1e-9 and abs(found["jaw_wide"] - 0.78) < 1e-9 and found["uneven"] < 1e-9, f"where the eyes stand, the jaw or the two halves: {found}"
assert abs(parle_beauty.measures(face() * 3.7 + 50)["chin_long"] - found["chin_long"]) < 1e-9, "the size of the portrait, or where the face is on it, counts"
# The corners of the mouth against the middle of the lips, over the width of the mouth: up is a smile, down a sad or cross mouth.
assert abs(found["mouth_curve"]) < 1e-9, f"a level mouth is read as curved: {found['mouth_curve']}"
up, down = (parle_beauty.measures(face(corners=lift))["mouth_curve"] for lift in (1.4, -1.4))
assert abs(up - 0.02) < 1e-3 and abs(down + 0.02) < 1e-3, f"corners 1.4 px above or below the lips on a mouth of 70 are not 2% of it: {up}, {down}"
assert parle_beauty.measures(face(lean=0.1))["uneven"] > 0.02, "a face with one half higher than the other is as even as any"
assert abs(found["mouth_wide"] - 0.35) < 1e-9, f"a mouth of 70 px on a face of 200 is not 35% of it: {found['mouth_wide']}"

# The eye as it shows is the opening between the lids, as a share of the box of the face: two ellipses 44 px wide and 10 high on a face
# of 200 by 200 (the ring of 16 points holds 97% of the ellipse). A more open eye shows more; what is drawn round it does not count,
# since nothing but the points is read. Until 2026-11-12 it was what stood apart from the skin by colour, and three quarters of
# Sofia's was the line drawn on her lids.
ellipse = lambda high: 2 * np.pi * 22 * high / 2 / (200 * 200)
assert 0.95 * ellipse(10) < found["eye_shown"] < ellipse(10), f"the eye as it shows is not the opening between the lids: {found['eye_shown']}"
wider = parle_beauty.measures(face(eye_open=15.0))["eye_shown"]
assert abs(wider / found["eye_shown"] - 1.5) < 1e-6, f"an eye half as open again does not show half as much again: {wider}"

# A face is whole on its portrait when the forehead, the chin and the cheekbones stand clear of the border: on a portrait that cut her
# chin Sofia's eyes read 0.089 open, and 0.084 on a whole one. And a face read on several portraits has the middle value of each measure.
shifted = face() + (120, 20)
assert parle_beauty.whole(shifted, (240, 240)), "a face with room all round it is not whole on its portrait"
assert not parle_beauty.whole(shifted, (240, 215)), "a face whose chin is off the picture is whole"
assert not parle_beauty.whole(face() + (95, 20), (240, 240)), "a face whose cheek is off the picture is whole"
middle = parle_beauty.settled([{"a": 1.0, "b": 5.0}, {"a": 3.0, "b": None}, {"a": 2.0}])
assert middle == {"a": 2.0, "b": 5.0}, f"the middle value of each measure, leaving out what was not read: {middle}"

# A mark is full at the aim and nothing a reach away. The side the literature says is better is full too, but not without end: past MOST
# times the aim a bigger eye is another face (big eyes on a realistic face are eerie, Seyama & Nagayama 2007).
assert parle_beauty.mark(0.10, 0.10, 0.05, "near") == 1 and parle_beauty.mark(0.15, 0.10, 0.05, "near") < 1e-9 and abs(parle_beauty.mark(0.125, 0.10, 0.05, "near") - 0.5) < 1e-9
assert parle_beauty.mark(0.12, 0.10, 0.05, "more") == 1 and abs(parle_beauty.mark(0.075, 0.10, 0.05, "more") - 0.5) < 1e-9, "more of a good feature loses marks"
assert parle_beauty.mark(0.10 * parle_beauty.MOST + 0.05, 0.10, 0.05, "more") < 1e-9, "there is no end to how much bigger scores full"
assert parle_beauty.mark(0.08, 0.10, 0.05, "less") == 1 and abs(parle_beauty.mark(0.125, 0.10, 0.05, "less") - 0.5) < 1e-9, "less of a feature that is better small loses marks"

# The two halves of a face cannot be too much alike: full from the aim down to nothing, less above it.
assert parle_beauty.mark(0.0, 0.015, 0.035, "least") == parle_beauty.mark(0.015, 0.015, 0.035, "least") == 1, "a face with its two halves alike loses marks for it"
assert abs(parle_beauty.mark(0.0325, 0.015, 0.035, "least") - 0.5) < 1e-9 and parle_beauty.AIM["female"]["uneven"][2] == "least", "an uneven face scores as an even one"
# A measure that may be nothing or less (the corners of the mouth): full from the aim to a reach above it, less on either side.
assert parle_beauty.mark(0.011, 0.011, 0.04, "up") == 1 and parle_beauty.mark(0.05, 0.011, 0.04, "up") == 1, "corners a little up lose marks"
assert abs(parle_beauty.mark(-0.009, 0.011, 0.04, "up") - 0.5) < 1e-9 and parle_beauty.mark(-0.03, 0.011, 0.04, "up") < 1e-9, "corners turned down score as level ones"
assert abs(parle_beauty.mark(0.071, 0.011, 0.04, "up") - 0.5) < 1e-9, "a wide smile at rest scores as a calm mouth"
assert parle_beauty.mark(-0.017, -0.017, 0.04, "up") == 1 and parle_beauty.mark(-0.041, -0.017, 0.04, "up") < 0.5, "an aim below nothing (Rafa's mouth) is not held"

# The face the women aim at scores 10; smaller eyes, a longer chin, a wider nose and lower brows each take marks off, and all of them more.
aim = {name: rule[0] for name, rule in parle_beauty.AIM["female"].items()}
whole, marks = parle_beauty.score(aim, "female")
assert whole == 10 and set(marks) == set(parle_beauty.AIM["female"]), f"the aim does not score full on every measure: {whole}"
worse = [parle_beauty.score(dict(aim, **change), "female")[0] for change in (dict(eye_shown=0.018), dict(chin_long=0.23), dict(nose_wide=0.25), dict(brow_high=0.16), dict(mouth_wide=0.41))]
assert all(8.5 < value < 10 for value in worse), f"one feature off the aim does not cost its share: {worse}"
# Lúcia before any work on her face, read on 2026-11-12 on whole portraits: her lower lip led the upper one and the corners of her mouth turned down.
plain = dict(aim, eye_open=0.077, eyes_apart=0.437, brow_high=0.161, nose_wide=0.253, jaw_wide=0.801, mouth_wide=0.383, mouth_curve=-0.018, lip_contrast=0.058, upper_leads=-0.168)
assert 6 < parle_beauty.score(plain, "female")[0] < 8, f"a woman of the cast as she was on 2026-11-11 does not score about seven: {parle_beauty.score(plain, 'female')[0]}"
assert parle_beauty.score(dict(aim, lip_contrast=0.06), "female")[0] < 10, "a woman's lips as pale as her skin score as Sofia's"
his = {name: rule[0] for name, rule in parle_beauty.AIM["male"].items()}
assert parle_beauty.score(his, "male")[0] == 10 and parle_beauty.score(dict(his, lip_contrast=0.142), "male")[0] < 10, "a man's lips as dark against the skin as a woman's score as his own"
assert parle_beauty.score(dict(aim, eye_shown=None), "female")[0] == 10, "a measure that was not taken counts against the face"
assert set(parle_beauty.AIM["male"]) == set(parle_beauty.AIM["female"]) == set(parle_beauty.NAMES), "a measure has no aim for one sex, or no name"

# How far a face stands from the faces the owner likes: nothing at their middle, one deviation with every measure one deviation off,
# and a measure they all share counts for no more than LIKED_MOST however far a face is on it.
for sex in ("female", "male"):
    middle = {name: mean for name, (mean, _) in parle_beauty.LIKED[sex].items()}
    assert parle_beauty.likeness(middle, sex)[0] == 0, "the middle of the faces the owner likes stands off them"
    off = {name: mean + spread for name, (mean, spread) in parle_beauty.LIKED[sex].items()}
    assert abs(parle_beauty.likeness(off, sex)[0] - 1) < 1e-9, "a face one deviation off on every measure does not stand one deviation off"
    mean, spread = parle_beauty.LIKED[sex]["upper_leads"]
    far, by = parle_beauty.likeness(dict(middle, upper_leads=mean - 10 * spread, lip_contrast=None), sex)
    assert "lip_contrast" not in by and abs(far - parle_beauty.LIKED_MOST / np.sqrt(len(by))) < 1e-9 and abs(by["upper_leads"] + 10) < 1e-9, f"one measure ten deviations off: {far}, {by['upper_leads']}"
    assert set(parle_beauty.LIKED[sex]) >= set(parle_beauty.AIM[sex]), "a measure that is marked is not among the ones read on the faces the owner likes"
assert parle_beauty.likeness({}, "female") == (None, {}), "a face with no measure stands somewhere"
assert set(parle_symmetry.NAMES) <= set(parle_beauty.measures(face())), "the measures of how alike the two halves are do not come with the rest"

# The details of a face (2026-11-13): on a face drawn with the eyes 44 px wide and 48 px between their inner corners, the chin 36 px
# under the lower lip and the brows 36 px over the middle of the eyes, each detail reads what was drawn; a taller opening reads rounder
# and nothing else moves; a longer chin reads on the chin and on the lower third.
def whole_face(**given):
    """`face` with the points only the details read: the pupils at the middle of each eye, an iris 20 px across, the eyebrows as
    lines 50 px long ending 20 px from the middle, the corners of the jaw, the sides of the chin and the tip of the nose."""
    points = face(**given)
    put = lambda index, x, y: points.__setitem__(index, (x, y))
    for sign, pupil, (near, far), (outer, top, inner) in zip((-1, 1), parle_beauty.PUPILS, parle_beauty.IRIS_SIDES, parle_beauty.BROW_ENDS):
        put(pupil, sign * 46, 70), put(near, sign * 46 - 10, 70), put(far, sign * 46 + 10, 70)
        put(outer, sign * 70, 36), put(top, sign * 46, 34), put(inner, sign * 20, 36)
    put(parle_beauty.GONION[0], -70, 160), put(parle_beauty.GONION[1], 70, 160)
    put(parle_beauty.CHIN_SIDES[0], -18, 196), put(parle_beauty.CHIN_SIDES[1], 18, 196)
    put(parle_beauty.NOSE_TIP, 0, 112)
    return points


drawn = parle_beauty.details(whole_face())
for name, wanted in (("face_wide_of_tall", 1.0), ("eye_round", 10 / 44), ("eye_tall", 10 / 200), ("eyes_gap_in_eyes", 48 / 44), ("pupils_apart", 92 / 200),
                     ("iris_of_eye", 20 / 44), ("eye_level", 130 / 200), ("brow_over_lid", (36 - 5) / 200), ("brows_apart", 40 / 200), ("forehead_tall", 34 / 200),
                     ("nose_wide_of_eye", 1.0), ("nose_tip_over_base", 10 / 200), ("mouth_wide_of_nose", 70 / 44), ("lips_tall", 32 / 200), ("nose_to_lip", 10 / 200),
                     ("chin_tall", 36 / 200), ("chin_wide", 36 / 200), ("jaw_corner_wide", 140 / 200), ("third_lower", 78 / 200)):
    assert abs(drawn[name] - wanted) < 1e-9, f"{name} reads {drawn[name]}, drawn {wanted}"
assert abs(drawn["third_upper"] + drawn["third_middle"] + drawn["third_lower"] - 1) < 1e-9, "the three thirds of the face do not make the face"
rounder = parle_beauty.details(whole_face(eye_open=20.0))
assert abs(rounder["eye_round"] - 20 / 44) < 1e-9 and rounder["chin_tall"] == drawn["chin_tall"], "a taller opening does not read rounder, or moves the chin"
longer = parle_beauty.details(whole_face(chin=50.0))
assert longer["chin_tall"] > drawn["chin_tall"] and longer["lips_tall"] < drawn["lips_tall"] and longer["third_lower"] == drawn["third_lower"], "a longer chin under the same face"
assert not set(drawn) & set(parle_beauty.measures(face())), "a detail has the name of a measure"

# The pretty faces: every measure of the table is one this ruler reads off a portrait (or the profile), their middle stands nowhere off
# them, and the file of the faces says what the table says. A measure they all share counts with a spread of PRETTY_SPREAD of its mean.
import json  # noqa: E402

kept = json.loads((Path(__file__).parent.parent / "data" / "parle_beauty" / "pretty_faces.json").read_text())
read = set(parle_beauty.measures(face())) | set(drawn) | {"lip_contrast", "upper_leads"}
for sex in ("female", "male"):
    table = parle_beauty.PRETTY[sex]
    assert set(table) <= read, f"the table of the pretty faces has measures this ruler does not read: {sorted(set(table) - read)}"
    assert all(spread > 0 for _, spread in table.values()), "a measure of the pretty faces with no spread"
    middle = {name: mean for name, (mean, _) in table.items()}
    assert parle_beauty.likeness(middle, sex, parle_beauty.PRETTY)[0] == 0, "the middle of the pretty faces stands off them"
    for name, (mean, spread) in table.items():
        assert abs(kept["standard"][sex][name]["mean"] - mean) < 1e-9 and abs(kept["standard"][sex][name]["sd"] - spread) < 1e-9, f"{sex} {name}: the table and the file differ"
    name = min(table, key=lambda key: table[key][1] / abs(table[key][0]))
    mean, spread = table[name]
    step = parle_beauty.PRETTY_SPREAD * abs(mean)
    assert spread < step, "no measure of the pretty faces is shared closely enough to need the least spread"
    far, by = parle_beauty.likeness(dict(middle, **{name: mean + step}), sex, parle_beauty.PRETTY)
    assert abs(by[name] - 1) < 1e-9, f"a measure the pretty faces share, {name}, counts by its own spread: {by[name]}"
    faces = [one for one in kept["faces"] if one["group"] == "standard" and one["sex"] == sex]
    assert len(faces) == {"female": 6, "male": 4}[sex] and all("marcos" not in one and "points" not in one for one in faces), "the faces of the standard"
assert parle_beauty.likeness(dict({name: mean for name, (mean, _) in parle_beauty.LIKED["female"].items()}), "female")[0] == 0, "the faces the owner likes are still the table when none is given"

try:
    import cv2
except ImportError:
    cv2 = None
if cv2 is not None:
    # The lips against the skin round them, on the portrait without light: lips of the skin's own lightness have no contrast, darker ones more.
    def mouth(lightness):
        image = np.full((400, 400, 3), 170, np.uint8)
        points = np.zeros((478, 2))
        points[list(parle_beauty.CHEEKS)] = ((100, 180), (300, 180))
        ring = np.linspace(0, 2 * np.pi, len(parle_beauty.LIPS), endpoint=False)
        points[list(parle_beauty.LIPS)] = np.stack([200 + 36 * np.cos(ring), 250 + 12 * np.sin(ring)], axis=1)
        cv2.ellipse(image, (200, 250), (37, 13), 0, 0, 360, (lightness,) * 3, -1)
        return image, points
    same, dark, darker = (parle_beauty.lip_contrast(*mouth(lightness)) for lightness in (170, 130, 100))
    assert abs(same) < 0.005 and 0.1 < dark < darker < 0.3, f"the lips against the skin: {same}, {dark}, {darker}"
    assert parle_beauty.lip_contrast(np.full((40, 40, 3), 170, np.uint8), np.zeros((478, 2))) is None, "lips with no room on the picture have a contrast"
print("ok")
