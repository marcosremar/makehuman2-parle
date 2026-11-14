"""Regression for the loop of the build (2026-11-10): run with `.local/venv/bin/python tests/test_parle_body_match.py`.

The body the loop measures has to be the body the cast builds. Twice it was not. Measured with a shoulder polish that fitted the hanging arm, the
loop pushed the arms out to undo it (male-slim: from 1.6 cm in to 1.5 cm out on each side) and the dressed arms stood away from the trunk. Measured
without it while the cast still ran it, the dressed shoulders came out 6 to 8 cm narrower on each side than the silhouette shown as matched (owner,
2026-11-10). Now one call, `parle_polish.natural_body`, is the whole step, in the cast and in the dump of the loop, and the polish is gone.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import body_match  # noqa: E402

options = body_match.dump_options(dict(height="1.74", sex="male"))
assert options == ["height=1.74", "sex=male"], f"the loop runs a step of its own, not the cast's: {options}"
assert "chest=shift.npy" in body_match.dump_options(dict(height="1.74", sex="male", chest="shift.npy"))

# Blender is not here: the sources are read as text.
polish = (ROOT / "tools" / "parle_polish.py").read_text()
whole = re.search(r"def natural_body\(meshes, rig, height, sex, kit\):.*?(?=\n\n\ndef )", polish, re.S)
assert whole, "the cast and the loop have no step in common"
steps = re.findall(r"\b(natural_\w+)\(", whole.group(0).split('"""')[-1])
assert steps == ["natural_arms", "natural_build"], f"the body is the arms, then the build: {steps}"
dump = (ROOT / "tools" / "body_dump.py").read_text()
assert re.findall(r"parle_polish\.(natural_\w+)\(", dump) == ["natural_body"], "the dump measures a body the cast does not build"
assert "body=" in (ROOT / "tools" / "body_match.py").read_text(), "the loop does not ask the dump for the cast's step"
for path in sorted((ROOT / "tools").glob("*.py")):
    assert "def natural_shoulders" not in path.read_text() and "natural_shoulders(" not in path.read_text(), f"{path.name} still polishes the shoulders"

profiles = sorted(path for path in (ROOT / "data" / "parle_build").glob("*.json") if not path.name.startswith("reference"))
assert len(profiles) == 6, f"a profile per kit: {[path.name for path in profiles]}"
for path in profiles:
    arm_out = json.load(open(path))["arm_out"]
    assert abs(arm_out) <= body_match.ARM_OUT_LIMIT + 1e-9, f"{path.name} moves the arms further than the limit: {arm_out}"
assert body_match.ARM_OUT_LIMIT <= 0.005, "the arm moves on its own again: 3 cm into the trunk pinched the skin between the arm and the shoulder (2026-11-11)"

# The shoulder line (2026-11-11): no ratio moved it, and the muscled man's stood 16 to 40 mm under the reference body's. The loop now lifts it
# where the reference's line stands higher, by a share of the height, and stops at the limit; a body without the rows is left as it is.
import numpy as np  # noqa: E402

import parle_build  # noqa: E402

frame = dict(joints=dict(shoulder=[0.15, 0.0, 1.3], shoulder_r=[-0.15, 0.0, 1.3]), tall=1.75, move=[0.0, 0.0, 0.0])
found = dict(line=[dict(share=0.5, x=0.04, rise=0.08), dict(share=1.0, x=0.09, rise=0.07), None])
reference = dict(line=[dict(share=0.5, x=0.05, rise=0.10), dict(share=1.0, x=0.10, rise=0.07), dict(share=1.2, x=0.12, rise=0.06)])
profile = parle_build.identity("test")
body_match.lift(profile, found, frame, reference)
rise = np.array(profile["shoulder_rise"])
assert 0.01 < rise[0] < 0.02 and abs(rise[-1]) < 1e-3, f"the line goes up next to the neck, where the reference's is higher, and only there: {rise}"
for _ in range(4):
    body_match.lift(profile, found, frame, reference)
assert profile["shoulder_rise"][0] == body_match.RISE_LIMITS[1], "the lift went past its limit"
assert body_match.at_limit(profile).get("shoulder_rise"), "a line held at the limit is not told"
kept = parle_build.identity("test")
body_match.lift(kept, dict(line=[None, None, None]), frame, reference)
assert kept["shoulder_rise"] == [0.0] * len(parle_build.RISE_S), "a body without the rows of the line was moved"
# The breadth across the shoulders (2026-11-11). The loop set the width of the trunk above the armpit by rows cut across the arms held out,
# and at 27 degrees below the horizontal such a row is where the top of the arm crosses its level: the height the arm leaves the trunk
# at, more than the breadth. Matched on them, the three kits of women stood 25.7 to 26.5% of their stature across the shoulders with the
# arms hanging, the two reference women 23.2 and 23.9%, and the men of the cast 25.8 to 26.4% (owner: "os ombros da mulher estão largos
# demais"). The reference women are now measured with their arms hanging too, the cast is dumped the same way, and the ratio of the
# shoulders of each kit of women brings that breadth to the reference's: 23.6%.
import contextlib  # noqa: E402
import io  # noqa: E402
import tempfile  # noqa: E402

import body_measure  # noqa: E402


def box(low, high):
    """A closed box as points and triangles."""
    corners = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])], float)
    faces = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    return corners, np.array(faces)


def standing(arm_from, arm_to):
    """A trunk 1.5 m tall and 30 cm wide with an arm hanging on each side, from `arm_from` to `arm_to` of the middle, dumped as `body_dump` does."""
    parts = [box((-0.15, -0.1, 0.0), (0.15, 0.1, 1.5)), box((arm_from, -0.05, 0.7), (arm_to, 0.05, 1.3)), box((-arm_to, -0.05, 0.7), (-arm_from, 0.05, 1.3))]
    points = np.concatenate([part[0] for part in parts])
    faces = np.concatenate([part[1] + 8 * index for index, part in enumerate(parts)])
    path = tempfile.mktemp(suffix=".npz")
    np.savez(path, points=points, faces=faces, flip=False)
    return path


broad, narrow = body_measure.bideltoid(standing(0.17, 0.27), 0.8), body_measure.bideltoid(standing(0.15, 0.23), 0.8)
assert abs(broad - 0.54 / 1.5) < 1e-3 and abs(narrow - 0.46 / 1.5) < 1e-3, f"the breadth is the whole width at the armpit, arms and all: {broad}, {narrow}"
assert abs(body_measure.bideltoid(standing(0.17, 0.27), 0.2) - 0.3 / 1.5) < 1e-3, "below the arms the breadth is the trunk's"


def body(bideltoid=None, across=0.30, pit=0.17):
    """A measure with the rows `update` reads: `across` is the width at the armpit of the rows cut across the arms held out, `pit` the
    width of the trunk at the armpit."""
    rows = dict(crotch=0.48, armpit=0.77, neck=0.88, neck_width=0.055, neck_depth=0.07, arm_angle=27.0, line=[None] * len(body_measure.LINE_SHARES),
                torso=[dict(u=round(float(u), 2), z=round(0.484 + float(u) * 0.278, 4), width=pit if u == 1 else 0.17, depth=0.12) for u in body_measure.TORSO_SHARES],
                shoulder=[dict(k=k, z=round(0.77 + k * 0.11, 4), width=round((across if k <= body_measure.SHOULDER_TIP else 0.30) - 0.25 * k, 4))
                          for k in body_measure.SHOULDER_SHARES],
                leg=[dict(v=round(float(v), 2), z=round(float(v) * 0.48, 4), width=0.09, depth=0.1) for v in body_measure.LEG_SHARES],
                arm=[dict(t=round(float(t), 2), x=0.12 + 0.2 * float(t), y=0.0, z=0.8 - 0.1 * float(t), depth=0.05, thick=0.045) for t in body_measure.ARM_SHARES])
    if bideltoid is not None:
        rows["bideltoid"] = bideltoid
    return rows


flat = body_measure.flatten(body(0.26))
gone = [index for index, row in enumerate(body()["shoulder"]) if flat[f"shoulder.width.{index:02d}"] is None]
assert gone == [index for index, k in enumerate(body_measure.SHOULDER_SHARES) if k <= body_measure.SHOULDER_TIP], f"the rows across the arms still count: {gone}"
last = len(body_measure.TORSO_SHARES) - 1
assert flat[f"torso.width.{last:02d}"] is None and flat[f"torso.depth.{last:02d}"] == 0.12 and flat[f"torso.width.{last - 1:02d}"] == 0.17, "the trunk at the armpit"
whole = body_measure.flatten(body())
assert flat["bideltoid"] == 0.26 and all(value is not None for key, value in whole.items() if key.startswith(("shoulder", "torso"))), "a body without the breadth lost a row"
alike, parts = body_measure.similarity(body(0.2356, across=0.36, pit=0.19), body(0.2356))
assert alike == 100 and parts["bideltoid"] == 100, f"two bodies as broad with the arms hanging are told apart by the widths cut across their arms: {alike}"
wider, parts = body_measure.similarity(body(0.26), body(0.2356))
assert wider < 100 and abs(parts["bideltoid"] - 100 * (1 - (0.26 - 0.2356) / 0.2356)) < 0.1, f"a body broader across the shoulders is not told: {parts}"
assert body_match.SHOULDER_REACH == body_measure.SHOULDER_TIP

joints = dict(shoulder=[0.16, 0.0, 1.30], shoulder_r=[-0.16, 0.0, 1.30], elbow=[0.38, 0.0, 1.19], wrist=[0.58, 0.0, 1.09], hip=[0.09, 0.0, 0.82],
              knee=[0.09, 0.0, 0.45], ankle=[0.09, 0.0, 0.08], neck=[0.0, 0.0, 1.34])
frame = dict(joints={key: np.array(value) for key, value in joints.items()}, tall=1.6, move=[0.0, 0.0, 0.0])
# The joints stand 20% of the stature apart, and the cast is 2.44% broader than the reference: the joints come 12.2% closer.
ratio = body_match.closer(body(0.26), body(0.2356), frame)
assert abs(ratio - (1 - 0.0244 / 0.2)) < 1e-9, f"the shoulders do not come in by the gap of the breadth: {ratio}"
assert body_match.hanging(body(0.26), body(0.2356)) and not body_match.hanging(body(0.26), body()) and not body_match.hanging(body(), body(0.2356))

# With the arms hanging read on both, the ratio of the shoulders alone moves, from the kit's own: the widths cut across the arms held out,
# 6% of the stature wider on this reference above the armpit and 2% at it, pull nothing, and the arm stays where the trunk carries it.
stations = parle_build.CURVES["trunk_x"]
profile = parle_build.identity("test")
body_match.update(profile, body(0.26), frame, body(0.2356, across=0.36, pit=0.19))
assert abs(profile["girdle"] - ratio ** body_match.STEP) < 1e-9, f"one round does not close its share of the gap: {profile['girdle']}"
assert profile["arm_out"] == 0.0, "the arm moved on its own: the shoulders bring the breadth to the reference's"
still = ("trunk_x", "trunk_y", "leg_x", "leg_y", "arm_depth", "arm_thick")
assert all(np.abs(np.array(profile[name]) - 1).max() < 1e-6 for name in still), "a curve moved with nothing to follow"
# A neck that stays thinner than the reference's pulls every round and goes to its limit. The levels of the shoulders, with no pull of
# their own, pass from the chest to the neck: once they took the pull of the neck too, and went 25% wider than the kit, past both.
for _ in range(20):
    body_match.update(profile, dict(body(0.30), neck_width=0.04), frame, body(0.2356))
trunk = np.array(profile["trunk_x"])
between = trunk[(stations > 0.76) & (stations < 0.86)]
assert trunk[stations > 1.0].min() == body_match.NECK_LIMITS[1] and np.all(np.diff(trunk) >= -1e-9), f"the neck did not go to its limit, or the curve does not pass evenly: {trunk}"
assert 1.0 < between.min() and between.max() < body_match.NECK_LIMITS[1] - 0.02, f"the neck drove the levels of the shoulders past its own width: {trunk}"
assert profile["girdle"] == body_match.GIRDLE_LIMITS[0] and body_match.at_limit(profile).get("girdle"), "shoulders held at their limit are not told"
# A reference that was not measured with its arms hanging is followed by its rows, as before: the trunk above the armpit and the arm.
for cast in (body(), body(0.26)):
    profile = parle_build.identity("test")
    body_match.update(profile, cast, frame, body(across=0.36))
    assert np.array(profile["trunk_x"])[(stations > 0.8) & (stations < 1.0)].min() > 1.05 and profile["arm_out"] > 0, "a reference with its rows alone is no longer followed"
    assert profile["girdle"] is None, "the shoulders moved by a breadth the reference does not have"

# `shoulders` finds that ratio on a profile that is already there and leaves its curves as they are. The kits of women were matched before
# their reference had the breadth, and matched again from nothing they came back with a neck 15% deeper, by a chance of the measure. The dump
# here is a stand-in: a body whose breadth is an arm on each side plus the joints, which stand 20% of the stature apart on the kit.
def stand_in(kit_glb, out, options):
    held = json.load(open(next(item for item in options if item.startswith("body=")).split("=", 1)[1]))
    assert f"hang={body_match.HANG_DEG}" in options and "arm=27.0" in options, f"the body is not dumped with its arms hanging, or in another pose: {options}"
    for path in (out, body_match.hang_path(out)):
        open(path, "w").close()
    apart = dict(joints, shoulder=[0.16 * held["girdle"], 0.0, 1.30], shoulder_r=[-0.16 * held["girdle"], 0.0, 1.30])
    return body(round(0.05 + 0.2 * held["girdle"], 4)), dict(frame, joints={key: np.array(value) for key, value in apart.items()})


folder = tempfile.mkdtemp()
settled = dict(parle_build.identity("test"), trunk_x=[1.1] * len(stations), leg_x=[1.05] * len(parle_build.LEG_Q), arm_out=-0.004, arm_deg=27.0)
parle_build.save_profile(settled, f"{folder}/kit.json")
json.dump(body(0.2356), open(f"{folder}/reference.json", "w"))
real, body_match.dump = body_match.dump, stand_in
with contextlib.redirect_stdout(io.StringIO()):
    body_match.shoulders("kit.glb", f"{folder}/reference.json", f"{folder}/kit.json", dict(height="1.6", sex="female", work=f"{folder}/work"))
body_match.dump = real
found = json.load(open(f"{folder}/kit.json"))
assert abs(found["bideltoid"] - 0.2356) <= body_match.GIRDLE_NEAR / 3 and abs(0.05 + 0.2 * found["girdle"] - found["bideltoid"]) < 1e-4, f"the shoulders did not come to the reference's: {found['bideltoid']}"
assert found["arm_out"] == 0.0 and found["reference"] == "reference.json", "the arm kept the way out its old rows gave it"
assert all(found[name] == settled[name] for name in parle_build.CURVES) and found["shoulder_rise"] == settled["shoulder_rise"], "a curve of the profile moved with the shoulders"

assert body_match.hang_path("work/round.npz") == "work/round.hang.npz"
loop = (ROOT / "tools" / "body_match.py").read_text()
assert 'f"hang={HANG_DEG}"] if reference.get("bideltoid")' in loop, "the loop does not dump the cast with its arms hanging when the reference was"
assert 'if "hang" in opts' in dump and ".hang.npz" in dump, "the dump no longer writes the body with its arms hanging"

women = json.load(open(ROOT / "data" / "parle_build" / "reference-women.json"))
assert 0.22 < women["bideltoid"] < 0.25, f"the reference women have no breadth across the shoulders, or another one: {women.get('bideltoid')}"
# A profile has a ratio of the shoulders only where its reference has a breadth for it to follow, and then it stands on that breadth: the
# reference women since 2026-11-11, and since 2026-11-14 the dressed man the normal man is matched to by his joints (`body_joints`).
for path in profiles:
    profile = json.load(open(path))
    wanted = json.load(open(ROOT / "data" / "parle_build" / profile["reference"])).get("bideltoid")
    if not wanted:
        assert profile.get("girdle") is None, f"{path.name} has the shoulders moved with no breadth to follow"
        continue
    gap = profile["bideltoid"] - wanted
    assert abs(gap) <= body_match.GIRDLE_NEAR + 1e-9, f"{path.name} stands {gap * 100:.1f}% of the stature off its reference across the shoulders"
    assert not body_match.at_limit(profile).get("girdle"), f"{path.name} holds the shoulders at the limit"
    assert profile["reference"] != "reference-women.json" or profile["arm_out"] == 0.0, f"{path.name} moves the arm on its own"
print("ok")
