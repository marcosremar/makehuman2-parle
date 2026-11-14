"""Regression for the body measured by its joints (2026-11-14): run with `python tests/test_body_joints.py` (no Blender).

The owner picked a man for his body, dressed in a T-shirt and shorts: "o homem tem bom corpo, copia toda a estrutura e formato".
`body_measure` found his armpit and his crotch above the joints of his arms and of his hips, so no row it read of him was where
it should be. `body_joints` reads a body at levels set by its skeleton and matches a kit to such a
measure, leaving the kit as it is where the reference has no row. This checks the reading on a body made of boxes, dressed and not;
that a pull stays between the reference's rows; that the trunk may go wider than `body_match` lets it and a limb may not; and the
reference and the profile that came of it.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import body_joints  # noqa: E402
import body_match  # noqa: E402
import parle_build  # noqa: E402


def box(low, high, cuts=6):
    """A closed box as points and triangles, its sides cut in `cuts` along the height so that level slices have points to show."""
    xs, ys, zs = (low[0], high[0]), (low[1], high[1]), np.linspace(low[2], high[2], cuts + 1)
    points = np.array([[x, y, z] for z in zs for x, y in ((xs[0], ys[0]), (xs[1], ys[0]), (xs[1], ys[1]), (xs[0], ys[1]))], float)
    faces = []
    for layer in range(cuts):
        for side in range(4):
            a, b = layer * 4 + side, layer * 4 + (side + 1) % 4
            faces += [(a, b, b + 4), (a, b + 4, a + 4)]
    top = cuts * 4
    faces += [(0, 2, 1), (0, 3, 2), (top, top + 1, top + 2), (top, top + 2, top + 3)]
    return points, np.array(faces)


def body(pieces):
    points, faces, start = [], [], 0
    for low, high in pieces:
        mine, mesh = box(low, high)
        points.append(mine)
        faces.append(mesh + start)
        start += len(mine)
    return np.concatenate(points), np.concatenate(faces)


# A man of boxes, 1 tall: a trunk 0.30 wide and 0.18 deep from 0.50 to 0.82, a head, two legs, and two arms straight down at the sides.
JOINTS = dict(hip=(0.08, 0, 0.50), knee=(0.08, 0, 0.27), ankle=(0.08, 0, 0.04), shoulder=(0.205, 0, 0.80), elbow=(0.205, 0, 0.60), wrist=(0.205, 0, 0.42),
              neck=(0, 0, 0.82), head=(0, 0, 0.88), shoulder_r=(-0.205, 0, 0.80), elbow_r=(-0.205, 0, 0.60), wrist_r=(-0.205, 0, 0.42), ankle_r=(-0.08, 0, 0.04))
PIECES = [((-0.15, -0.09, 0.50), (0.15, 0.09, 0.82)), ((-0.04, -0.05, 0.82), (0.04, 0.05, 0.88)), ((-0.07, -0.09, 0.88), (0.07, 0.09, 1.0)),
          ((0.03, -0.05, 0.0), (0.13, 0.05, 0.50)), ((-0.13, -0.05, 0.0), (-0.03, 0.05, 0.50)),
          ((0.18, -0.03, 0.40), (0.23, 0.03, 0.80)), ((-0.23, -0.03, 0.40), (-0.18, 0.03, 0.80))]


def measured(pieces, folder, name):
    points, faces = body(pieces)
    path = f"{folder}/{name}.npz"
    np.savez(path, points=points, faces=faces, flip=False, **{f"j_{key}": np.array(value, float) for key, value in JOINTS.items()})
    return body_joints.measure(path, path)


import tempfile  # noqa: E402

folder = tempfile.mkdtemp()
bare = measured(PIECES, folder, "bare")
rows = [row for row in bare["trunk"] if row and row["width"]]
assert rows and all(abs(row["width"] - 0.30) < 2e-3 and abs(row["depth"] - 0.18) < 2e-3 for row in rows), f"the trunk is not read between the arms: {bare['trunk']}"
assert abs(bare["bideltoid"] - 0.46) < 2e-3, f"the breadth across the shoulders is the whole width at the joints of the arms: {bare['bideltoid']}"
assert abs(bare["neck_width"] - 0.08) < 2e-3 and abs(bare["neck_depth"] - 0.10) < 2e-3, f"the neck is its narrowest level: {bare['neck_width']}, {bare['neck_depth']}"
assert all(abs(row["girth"] - np.sqrt(0.05 * 0.06)) < 2e-3 for row in bare["arm"] if row), f"an arm is cut square to its bone: {bare['arm']}"
assert all(abs(row["width"] - 0.10) < 2e-3 for row in bare["leg"] if row), f"a leg is one leg and not the two: {bare['leg']}"

# Dressed: a shirt as an open tube round the trunk, 1 cm off it, hanging to below the hips. It is read as the trunk, layers and open
# hem and all, where `body_measure` wants closed loops of skin.
shirt = [((-0.16, -0.10, 0.46), (0.16, 0.10, 0.83))]
dressed = measured(PIECES + shirt, folder, "dressed")
rows = [row for row in dressed["trunk"] if row and row["width"]]
assert rows and all(abs(row["width"] - 0.32) < 2e-3 and abs(row["depth"] - 0.20) < 2e-3 for row in rows), f"a dressed trunk is its cloth: {dressed['trunk']}"

# Arms that touch the trunk leave no width there: a level where the arms are not loose is not the trunk's.
close = [piece for piece in PIECES[:5]] + [((0.15, -0.03, 0.40), (0.20, 0.03, 0.80)), ((-0.20, -0.03, 0.40), (-0.15, 0.03, 0.80))]
stuck = measured(close, folder, "stuck")
assert all(row is None or row["width"] is None for row in stuck["trunk"] if row is None or row["z"] > 0.45), "arms that touch the trunk were read as trunk"

# `about`: loops make one stretch while no hole between them is wider than the gap, and what is past a hole stands apart.
loop = lambda low, high: dict(low=np.array([low, 0, 0.0]), high=np.array([high, 0, 0.0]), points=np.array([[low, 0, 0.0], [high, 0, 0.0]]))
mine, left, right = body_joints.about([loop(-0.1, 0.02), loop(0.021, 0.1), loop(0.2, 0.25), loop(-0.3, -0.2)], 0.0, 0.006)
assert len(mine) == 2 and left and right, "two loops that touch are one trunk, and an arm on each side stands apart"
assert body_joints.about([loop(0.2, 0.25)], 0.0, 0.006)[0] == [], "with nothing about the middle there is no trunk"

# A pull stays between the reference's rows and dies away past them; the curve keeps its own shape where nothing pulls it.
curve = list(np.linspace(0.9, 1.1, len(parle_build.LEG_Q)))
pulled = body_joints.pull("leg_x", curve, [1.2, 1.5], [0.9, 0.9], body_joints.LIMB_FADE)
thigh, calf = parle_build.LEG_Q < 1.2 - body_joints.LIMB_FADE - 0.2, (parle_build.LEG_Q >= 1.2) & (parle_build.LEG_Q <= 1.5)
assert np.allclose(np.array(pulled)[thigh], np.array(curve)[thigh], atol=1e-3), "the ratio of the calf thinned the thigh, which the reference does not show"
assert (np.array(pulled)[calf] < np.array(curve)[calf] - 0.03).all(), "the calf did not take its ratio"
# No curve goes past the limits of the loop on skin (`body_match.limits`).
wide = body_joints.pull("trunk_x", [1.2] * len(parle_build.TRUNK_W), [0.2, 0.4], [1.5, 1.5], body_joints.TRUNK_FADE)
assert max(wide) <= body_match.LIMITS[1] + 1e-9, f"the trunk went past its limit: {max(wide)}"
assert max(body_joints.pull("leg_x", [1.1] * len(parle_build.LEG_Q), [0.5, 1.0], [1.5, 1.5], body_joints.LIMB_FADE)) <= body_match.LIMB_LIMITS[1] + 1e-9, "a leg went past its limit"

# One round: a wider reference widens the trunk, rows the reference lacks pull nothing, and the shoulders follow its breadth.
profile = parle_build.identity("male-test")
reference = json.loads(json.dumps(bare))
for row in reference["trunk"]:
    if row and row["width"]:
        row["width"] = round(row["width"] * 1.1, 4)
reference["leg"] = [None] * len(reference["leg"])
reference["bideltoid"] = bare["bideltoid"] + 0.02
body_joints.update(profile, bare, reference)
assert max(profile["trunk_x"]) > 1.05 and profile["trunk_y"] == [1.0] * len(parle_build.TRUNK_W), f"the trunk does not follow the reference's width alone: {profile['trunk_x']}"
assert profile["leg_x"] == [1.0] * len(parle_build.LEG_Q) and profile["girdle"] > 1.02, "a leg moved with no row to follow, or the shoulders did not"
percent, parts = body_joints.likeness(bare, reference)
assert percent < 100 and "leg.width" not in parts and body_joints.likeness(bare, bare)[0] == 100, "the likeness counts rows the reference lacks, or a body is not like itself"

# The reference: measures only, and nothing where the clothes hide the body.
dressed_man = json.load(open(ROOT / "data" / "parle_build" / "reference-dressed-man.json"))
assert set(dressed_man) == {"stature_m", "shoulders_apart", "bideltoid", "neck_width", "neck_depth", "trunk", "arm", "leg", "arm_deg", "about"}, f"the reference keeps more than measures: {sorted(dressed_man)}"
assert all(row is None for row in dressed_man["leg"] if row is None or row["t"] < 1.05), "the thigh under the shorts was kept"
assert all(row is None for row in dressed_man["trunk"] if row is None or row["u"] < 0.55), "the waist and the abdomen under the hanging shirt were kept"
assert 0.28 < dressed_man["bideltoid"] < 0.31, f"the breadth across his shoulders: {dressed_man['bideltoid']}"

# The normal man's profile came of it, started from the one he had: shoulders on the reference's breadth, a chest as deep, and the
# abdomen and the thigh as they were.
normal = json.load(open(ROOT / "data" / "parle_build" / "male-normal.json"))
assert normal["reference"] == "reference-dressed-man.json" and normal["by"] == "joints" and normal["started_from"]["reference"] == "reference-men.json"
assert abs(normal["bideltoid"] - dressed_man["bideltoid"]) <= body_match.GIRDLE_NEAR and 1.05 < normal["girdle"] <= body_match.GIRDLE_LIMITS[1]
assert max(normal["trunk_x"]) <= body_match.LIMITS[1] + 1e-9 and normal["similarity"] > 97, f"the normal man is not like the reference: {normal['similarity']}"
# The waist as the reference men had left it (the curve of the trunk's width at the levels of the waist), since the shirt hides it.
assert np.allclose(normal["trunk_x"][8:11], [1.1502, 1.1918, 1.2162], atol=0.02), f"the waist moved under the shirt: {normal['trunk_x'][8:11]}"
# The thigh as the reference men had left it (the first six rings of the leg, hip to mid-thigh).
assert np.allclose(normal["leg_x"][:6], [0.9968, 0.9829, 0.9545, 0.9142, 0.8716, 0.85], atol=1e-4), f"the thigh moved under the shorts: {normal['leg_x'][:6]}"
print("ok")
