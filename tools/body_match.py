"""
    License information: data/licenses/makehuman_license.txt

    Finds the build profile of a kit (`data/parle_build/<kit>.json`) by a loop: dump the unclothed kit body with the profile, in the arm pose of
    the reference, measure it (`body_measure`), pull every ratio towards the reference's measure, and stop once the likeness has reached
    `want` and a round no longer brings it closer (owner, 10/11/2026: the loop only stops when the body is that close to the reference, and
    the body has to be as close as it can), when it stalls short of it, or after `rounds`.
      python tools/body_match.py match <kit.glb> <reference.json> <profile.json> kit=<name> height=<m> sex=<male|female> [chest=<npy>] [want=95] [rounds=12] [work=<folder>]
      python tools/body_match.py shoulders <kit.glb> <reference.json> <profile.json> height=<m> sex=<male|female> [chest=<npy>] [rounds=8] [work=<folder>]
      python tools/body_match.py reference <out.json> <measure.json> [<measure.json> ...]
    `reference` keeps the mean of the measures of one or more bodies: a kit between two reference bodies is matched to their mean. Only measures
    of a reference are kept, never its mesh or its texture. A reference measured with its arms hanging too (`body_measure.py hanging=`)
    has the breadth across its shoulders, and the loop then reads the cast's the same way, a second dump each round.
    `shoulders` takes a profile that is already there and finds the ratio of its shoulders alone, every curve left as it is: for a kit
    whose reference gained that breadth after the kit was matched.
"""
import copy
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import body_measure  # noqa: E402
import parle_build  # noqa: E402

BLENDER = os.environ.get("BLENDER", "/Applications/Blender-4.5-LTS.app/Contents/MacOS/Blender")
# How far a ratio may go from the kit's own shape, and how much of each gap one round closes. The neck and the top of the chest stay nearer: a
# reference's neck is read up to the jaw, and a neck 15% thicker is already another person. An arm and a leg come back to their own girth over
# the last `TAPER` before the wrist and the ankle (stations run 0 to 2): the hand and the foot keep their shape, and a ratio held up to them
# left a step there. The trunk may go down to 0.72: at 0.85 the muscled man's waist stayed 20 to 40 mm deeper than the reference body from
# the front to the back, a belly where the reference has a flat abdomen (2026-11-11). A ratio only scales the kit's own shape, corners and
# all: a calf a quarter wider turned into a wedge and a forearm into a
# club, so arms and legs stay within 15%. A ratio that follows the reference ring by ring leaves steps, so each round smooths `SOFT` times.
LIMITS, LIMB_LIMITS, NECK_LIMITS, NECK_FROM_W, TAPER = (0.72, 1.25), (0.85, 1.15), (0.9, 1.15), 0.85, 0.5
STEP, SOFT = 0.8, 3
# How far the arm moves out or in on its own, as a share of the height. It was 3%: the muscled man had the arms drawn 3 cm into the trunk
# to match the width across the arms, and the slim woman 1.7 cm. Only the arm's skin moves, so the skin between the arm and the shoulder
# was pinched into a fold, and the muscled man ended with shoulders narrower than the slim one's (owner, 2026-11-11: "a dobra entre o
# braço e o ombro está estranha"). The width of the shoulders is now the trunk's, level by level, and the arm keeps to a few millimetres.
ARM_OUT_LIMIT = 0.004
# The shoulder rows that set the width of the trunk above the armpit, as shares of armpit to neck: the row at the armpit itself is the
# chest's, which the torso rows already give, and above 0.45 the row is cut over the tip of the shoulder.
SHOULDER_TRUNK = (0.15, 0.45)
# The shoulder line goes up by a share of the height, not by a ratio: 2% is 35 mm on a man of 1.75 m, a trapezius, and more is a hump. It
# never goes down: the measure asked the slim man's line 7 to 18 mm lower, more at the tip, and next to his reference, whose shoulders are
# square, he stood with the shoulders of a bottle. The rise is read from the armpit, and his armpit sits lower than the reference's.
RISE_LIMITS = (0.0, 0.02)
# The shoulder rows up to this share of armpit to neck are the arm's reach; above it they are the slope. It was 0.6, and the row at 0.6 is
# cut above the tip of the shoulder on a body with a longer neck: read as 7 to 10% of the height too narrow there, the women had their
# arms pushed 1 to 3 cm out on each side, and stood 2 to 4 cm wider across the deltoids than the reference women, with the shoulders of a
# man over their hips (owner, 2026-11-11: "o ombro da mulher está muito largo").
SHOULDER_REACH = body_measure.SHOULDER_TIP
# That was not enough (owner, the same day: "os ombros da mulher estão largos demais"): the rows with the arms out say how high the arm
# leaves the trunk as much as how broad the shoulders are (`body_measure.SHOULDER_TIP`). When the reference was also measured with its arms
# hanging, the loop dumps the cast that way too, HANG_DEG below the horizontal, and the breadth read there (`body_measure.bideltoid`)
# sets the ratio of the shoulders (`parle_build`'s `girdle`), as far from the kit's own as any ratio of the trunk may go. The curve of
# the trunk then pulls nothing above the armpit: its rows there hold it where it is, and it passes from the chest to the neck. Without
# them the levels between the armpit and the neck took the pull of the neck every round, and a neck held at its limit, never as thick
# as the reference's, drove them to their own limit, 25% wider than the kit. The row of the chest at the armpit itself leaves too: it
# is cut through the roots of the arms, and with the arms closer it reads narrower whatever the chest (the slim woman's fell 4% while
# the curve there rose 7%). 78 degrees is the arm of the cast standing in its idle clip, 10 to 13 degrees from the vertical; at the
# vertical the arm of a broad body goes into its hip. A round counts as near once that breadth is within GIRDLE_NEAR of the
# reference's, 5 mm on a woman of 1.60 m: the loop does not stop before, and keeps a near round over one that is not.
HANG_DEG = 78
GIRDLE_NEAR = 0.003
GIRDLE_LIMITS = LIMITS


def hang_path(out):
    """Where `body_dump.py hang=` writes the body with its arms hanging, beside the dump `out`."""
    return out[:-len(".npz")] + ".hang.npz"


def dump(kit_glb, out, options):
    for path in (out, hang_path(out)):
        if os.path.exists(path):
            os.remove(path)
    done = subprocess.run([BLENDER, "-b", "--factory-startup", "-noaudio", "-P", os.path.join(HERE, "body_dump.py"), "--", out, kit_glb, "kind=kit"] + options, capture_output=True, text=True)
    if not os.path.exists(out) or (any(item.startswith("hang=") for item in options) and not os.path.exists(hang_path(out))):
        raise SystemExit("the dump failed:\n" + (done.stdout + done.stderr)[-3000:])
    return body_measure.measure(out, hang_path(out) if os.path.exists(hang_path(out)) else None)


def rows(found, reference, group, field):
    """(row of the cast, its value, the reference's value) for the rows both bodies have."""
    return [(a, a[field], b[field]) for a, b in zip(found[group], reference[group]) if a and b and a[field] and b[field]]


def limits(name):
    """The lowest and the highest ratio of each station of that curve."""
    stations = parle_build.CURVES[name]
    if name.startswith("trunk"):
        tight = stations > NECK_FROM_W
        return np.where(tight, NECK_LIMITS[0], LIMITS[0]), np.where(tight, NECK_LIMITS[1], LIMITS[1])
    free = parle_build.smooth((2 - stations) / TAPER)
    return 1 - (1 - LIMB_LIMITS[0]) * free, 1 + (LIMB_LIMITS[1] - 1) * free


def soft(values):
    """Each station takes a quarter of each neighbour: a ratio that jumps from one ring to the next leaves a step on the body."""
    logs = np.log(values)
    for _ in range(SOFT):
        padded = np.concatenate([logs[:1], logs, logs[-1:]])
        logs = 0.25 * padded[:-2] + 0.5 * logs + 0.25 * padded[2:]
    return np.exp(logs)


def pull(name, curve, params, ratios):
    order = np.argsort(params)
    gain = np.interp(parle_build.CURVES[name], np.array(params)[order], np.array(ratios)[order] ** STEP)
    return list(np.clip(soft(np.array(curve) * gain), *limits(name)))


def hanging(found, reference):
    """Whether both bodies had the breadth across their shoulders read with the arms hanging."""
    return bool(found.get("bideltoid") and reference.get("bideltoid"))


def closer(found, reference, frame):
    """The ratio that brings the breadth across the shoulders of the cast, arms hanging, to the reference's: the breadth is the joints
    of the arms apart plus an arm, so the joints come closer, or go apart, by the whole gap."""
    joints = frame["joints"]
    apart = abs(joints["shoulder"][0] - joints["shoulder_r"][0]) / frame["tall"]
    return 1 + (reference["bideltoid"] - found["bideltoid"]) / apart


def update(profile, found, frame, reference):
    """One round: every curve is pulled by the ratio of the reference's measure to the cast's, read where the cast's rows are."""
    joints, tall, move = frame["joints"], frame["tall"], np.asarray(frame["move"], float)
    height = lambda z: z * tall + move[2]
    hip, knee, ankle, neck = (joints[key][2] for key in ("hip", "knee", "ankle", "neck"))
    level = lambda z: (height(z) - hip) / (neck - hip)
    for curve, field, neck_field in (("trunk_x", "width", "neck_width"), ("trunk_y", "depth", "neck_depth")):
        pairs = rows(found, reference, "torso", field)
        # Above the armpit the width of the trunk is the width of the shoulders, read across the arms: the arms ride with their joints.
        girdle = [item for item in rows(found, reference, "shoulder", "width") if SHOULDER_TRUNK[0] <= item[0]["k"] <= SHOULDER_TRUNK[1]] if curve == "trunk_x" else []
        if curve == "trunk_x" and hanging(found, reference):
            # Two bodies measured with their arms hanging have the ratio of the shoulders for that: the rows above the armpit hold the
            # curve where it is, and the row at the armpit, cut through the roots of the arms, leaves.
            pairs, girdle = [item for item in pairs if item[0]["u"] < 1], [(row, 1.0, 1.0) for row, _, _ in girdle]
        params = [level(row["z"]) for row, _, _ in pairs + girdle] + [level(found["neck"])]
        ratios = [want / have for _, have, want in pairs + girdle] + [reference[neck_field] / found[neck_field]]
        profile[curve] = pull(curve, profile[curve], params, ratios)
    for curve, field in (("leg_x", "width"), ("leg_y", "depth")):
        pairs = rows(found, reference, "leg", field)
        params = [float(parle_build.leg_param(height(row["z"]), hip, knee, ankle)) for row, _, _ in pairs]
        profile[curve] = pull(curve, profile[curve], params, [want / have for _, have, want in pairs])
    # The arm takes one ratio for both ways across, the girth's: how far a forearm is turned changes from body to body, and a ratio per axis
    # made the cast's forearm, turned palm down, a flat tube.
    chain = [np.asarray(joints[key], float) for key in ("shoulder", "elbow", "wrist")]
    pairs = [(a, np.sqrt(b["depth"] * b["thick"] / (a["depth"] * a["thick"]))) for a, b in zip(found["arm"], reference["arm"])
             if a and b and a["depth"] and a["thick"] and b["depth"] and b["thick"]]
    spots = np.array([[row["x"], row["y"], row["z"]] for row, _ in pairs]) * tall + move
    params, ratios = list(parle_build.arm_param(spots, *chain)[0]), [ratio for _, ratio in pairs]
    girth = pull("arm_depth", np.sqrt(np.array(profile["arm_depth"]) * np.array(profile["arm_thick"])), params, ratios)
    profile["arm_depth"], profile["arm_thick"] = list(girth), list(girth)
    # With the breadth across the shoulders read with the arms hanging, the ratio of the shoulders alone brings it to the reference's,
    # and the arm stays where the trunk carries it.
    if hanging(found, reference):
        profile["girdle"] = float(np.clip((profile.get("girdle") or 1.0) * closer(found, reference, frame) ** STEP, *GIRDLE_LIMITS))
    else:
        reach = [(want - have) / 2 for row, have, want in rows(found, reference, "shoulder", "width") if row["k"] <= SHOULDER_REACH]
        profile["arm_out"] = float(np.clip(profile["arm_out"] + STEP * np.mean(reach), -ARM_OUT_LIMIT, ARM_OUT_LIMIT))
    lift(profile, found, frame, reference)


def lift(profile, found, frame, reference):
    """The shoulder line: each station goes up by the gap between the reference's line and the cast's, read where the cast's rows are."""
    pairs = rows(found, reference, "line", "rise")
    if not pairs:
        return
    joints, tall, move = frame["joints"], frame["tall"], np.asarray(frame["move"], float)
    mid = (joints["shoulder"][0] + joints["shoulder_r"][0]) / 2 if "shoulder_r" in joints else 0.0
    params = np.array([abs(row["x"] * tall + move[0] - mid) for row, _, _ in pairs]) / abs(joints["shoulder"][0] - mid)
    gaps, order = np.array([want - have for _, have, want in pairs]), np.argsort(params)
    curve = np.array(profile.get("shoulder_rise") or [0.0] * len(parle_build.RISE_S)) + STEP * np.interp(parle_build.RISE_S, params[order], gaps[order])
    for _ in range(SOFT):
        padded = np.concatenate([curve[:1], curve, curve[-1:]])
        curve = 0.25 * padded[:-2] + 0.5 * curve + 0.25 * padded[2:]
    profile["shoulder_rise"] = list(np.clip(curve, *RISE_LIMITS))


def at_limit(profile):
    """The stations that sit on a limit: there the kit could not go as far as the reference."""
    held = {}
    for name in parle_build.CURVES:
        values, (low, high) = np.array(profile[name]), limits(name)
        count = int(np.sum(((values <= low + 1e-6) | (values >= high - 1e-6)) & (high - low > 0.02)))
        if count:
            held[name] = count
    if abs(profile["arm_out"]) >= ARM_OUT_LIMIT - 1e-6:
        held["arm_out"] = 1
    if not GIRDLE_LIMITS[0] + 1e-6 < (profile.get("girdle") or 1.0) < GIRDLE_LIMITS[1] - 1e-6:
        held["girdle"] = 1
    rise = np.array(profile.get("shoulder_rise") or [0.0])
    stuck = int(np.sum(rise >= RISE_LIMITS[1] - 1e-6))
    if stuck:
        held["shoulder_rise"] = stuck
    return held


def crotch_share(found, frame):
    hip, knee = frame["joints"]["hip"][2], frame["joints"]["knee"][2]
    return round(float((hip - (found["crotch"] * frame["tall"] + frame["move"][2])) / (hip - knee)), 4)


def dump_options(opts):
    """What every dump of the loop gets besides the profile of the round. No step of its own: the dump runs `parle_polish.natural_body`, the
    same call as the cast, so the loop measures the body that goes under the clothes. Twice it did not (2026-11-10): measured with a shoulder
    polish that fitted the hanging arm, the loop pushed the arms up to 3 cm out on each side to undo it; measured without it while the cast
    still ran it, the dressed shoulders came out 6 to 8 cm narrower on each side than the body shown as matched."""
    return [f"height={opts['height']}", f"sex={opts['sex']}"] + ([f"chest={opts['chest']}"] if "chest" in opts else [])


def match(kit_glb, reference_path, out, opts):
    reference = json.load(open(reference_path))
    work = opts.get("work") or tempfile.mkdtemp()
    os.makedirs(work, exist_ok=True)
    base = dump_options(opts) + ([f"hang={HANG_DEG}"] if reference.get("bideltoid") else [])
    profile, arm_deg = parle_build.identity(opts["kit"]), float(reference["arm_angle"])
    best, history, round_npz, best_npz = None, [], os.path.join(work, "round.npz"), os.path.join(work, "best.npz")
    for turn in range(int(opts.get("rounds", 12)) + 1):
        parle_build.save_profile(profile, os.path.join(work, "profile.json"))
        found, frame = dump(kit_glb, round_npz, base + [f"body={os.path.join(work, 'profile.json')}", f"arm={arm_deg:.1f}"])
        percent, parts = body_measure.similarity(found, reference)
        history.append(percent)
        near = not hanging(found, reference) or abs(found["bideltoid"] - reference["bideltoid"]) <= GIRDLE_NEAR
        across = f" shoulders {found['bideltoid']} of {reference['bideltoid']}" if hanging(found, reference) else ""
        print(f"round {turn}: {percent:.2f}% arm {arm_deg:.1f}{across} {parts}", flush=True)
        if turn == 0:
            profile["crotch"] = crotch_share(found, frame)
        if best is None or (near, percent) > (best["near"], best["similarity"]):
            best = dict(copy.deepcopy(profile), similarity=round(percent, 2), parts=parts, arm_deg=round(arm_deg, 1), near=near)
            best["bideltoid"] = found.get("bideltoid")
            for path, kept in ((round_npz, best_npz), (hang_path(round_npz), hang_path(best_npz))):
                if os.path.exists(path):
                    os.replace(path, kept)
        # Past `want` it goes on while a round still brings the body closer; short of it, it gives up when three rounds bring nothing.
        enough = percent >= float(opts.get("want", 95)) and turn and percent - history[turn - 1] < 0.2
        if near and (enough or (turn >= 3 and percent - history[turn - 3] < 0.05)):
            break
        update(profile, found, frame, reference)
        arm_deg = float(np.clip(arm_deg + reference["arm_angle"] - found["arm_angle"], 10, 60))
    best.pop("near")
    best.update(reference=os.path.basename(reference_path), before=round(history[0], 2), rounds=len(history) - 1, at_limit=at_limit(best))
    parle_build.save_profile(best, out)
    print(f"done: {best['before']}% -> {best['similarity']}% in {best['rounds']} rounds, at the limit: {best['at_limit']}", flush=True)


def shoulders(kit_glb, reference_path, out, opts):
    """Brings the breadth across the shoulders of the profile at `out` to the reference's, arms hanging, by the ratio of the shoulders
    alone. Its curves stay as they are: matched again from nothing, the kits of women came back with other things moved by the chance
    of a measure (the neck of one was read 2 mm lower, under the jaw, and came out 15% deeper than the kit's, where it had been 1 to
    7%), and only their shoulders were to change (2026-11-11). The arm's own way out goes back to none: it was set by the rows this
    breadth replaces. Stops within a third of GIRDLE_NEAR, or keeps the nearest of `rounds`."""
    reference, profile = json.load(open(reference_path)), json.load(open(out))
    work = opts.get("work") or tempfile.mkdtemp()
    os.makedirs(work, exist_ok=True)
    base = dump_options(opts) + [f"hang={HANG_DEG}", f"body={os.path.join(work, 'profile.json')}", f"arm={profile['arm_deg']:.1f}"]
    profile.update(arm_out=0.0, girdle=profile.get("girdle") or 1.0)
    best, round_npz, best_npz = None, os.path.join(work, "round.npz"), os.path.join(work, "best.npz")
    for turn in range(int(opts.get("rounds", 8)) + 1):
        parle_build.save_profile(profile, os.path.join(work, "profile.json"))
        found, frame = dump(kit_glb, round_npz, base)
        gap = found["bideltoid"] - reference["bideltoid"]
        percent, parts = body_measure.similarity(found, reference)
        print(f"round {turn}: shoulders {found['bideltoid']} of {reference['bideltoid']} with the ratio {profile['girdle']:.4f}, {percent:.2f}% {parts}", flush=True)
        if best is None or abs(gap) < abs(best["bideltoid"] - reference["bideltoid"]):
            best = dict(copy.deepcopy(profile), similarity=round(percent, 2), parts=parts, bideltoid=found["bideltoid"])
            for path, kept in ((round_npz, best_npz), (hang_path(round_npz), hang_path(best_npz))):
                os.replace(path, kept)
        if abs(gap) <= GIRDLE_NEAR / 3:
            break
        profile["girdle"] = float(np.clip(profile["girdle"] * closer(found, reference, frame) ** STEP, *GIRDLE_LIMITS))
    best.update(reference=os.path.basename(reference_path), at_limit=at_limit(best))
    parle_build.save_profile(best, out)
    print(f"done: shoulders {best['bideltoid']} of {reference['bideltoid']}, ratio {best['girdle']:.4f}, at the limit: {best['at_limit']}", flush=True)


def mean_measure(out, paths):
    """The mean of the measures of several bodies, row by row; a row one of them lacks is left out."""
    found = [json.load(open(path)) for path in paths]

    def mean(items):
        if any(item is None for item in items):
            return None
        if isinstance(items[0], dict):
            return {key: mean([item.get(key) for item in items]) for key in items[0]}
        if isinstance(items[0], list):
            return [mean(list(row)) for row in zip(*items)]
        return round(float(np.mean(items)), 4)

    json.dump(mean(found), open(out, "w"), indent=1)


if __name__ == "__main__":
    paths = [item for item in sys.argv[2:] if "=" not in item]
    options = dict(item.split("=", 1) for item in sys.argv[2:] if "=" in item)
    if sys.argv[1] in ("match", "shoulders"):
        (match if sys.argv[1] == "match" else shoulders)(paths[0], paths[1], paths[2], options)
    else:
        mean_measure(paths[0], paths[1:])
