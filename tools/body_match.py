"""
    License information: data/licenses/makehuman_license.txt

    Finds the build profile of a kit (`data/parle_build/<kit>.json`) by a loop: dump the unclothed kit body with the profile, in the arm pose of
    the reference, measure it (`body_measure`), pull every ratio towards the reference's measure, and stop once the likeness has reached
    `want` and a round no longer brings it closer (owner, 10/11/2026: the loop only stops when the body is that close to the reference, and
    the body has to be as close as it can), when it stalls short of it, or after `rounds`.
      python tools/body_match.py match <kit.glb> <reference.json> <profile.json> kit=<name> height=<m> sex=<male|female> [chest=<npy>] [want=95] [rounds=12] [work=<folder>]
      python tools/body_match.py reference <out.json> <measure.json> [<measure.json> ...]
    `reference` keeps the mean of the measures of one or more bodies: a kit between two reference bodies is matched to their mean. Only measures
    of a reference are kept, never its mesh or its texture.
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
# left a step there. A ratio only scales the kit's own shape, corners and all: a calf a quarter wider turned into a wedge and a forearm into a
# club, so arms and legs stay within 15%. A ratio that follows the reference ring by ring leaves steps, so each round smooths `SOFT` times.
LIMITS, LIMB_LIMITS, NECK_LIMITS, NECK_FROM_W, TAPER = (0.85, 1.25), (0.85, 1.15), (0.9, 1.15), 0.85, 0.5
STEP, SOFT = 0.8, 3
ARM_OUT_LIMIT = 0.03
# The shoulder rows below this share of armpit to neck are the arm's reach; above it they are the slope, which no ratio here moves.
SHOULDER_REACH = 0.6


def dump(kit_glb, out, options):
    if os.path.exists(out):
        os.remove(out)
    done = subprocess.run([BLENDER, "-b", "--factory-startup", "-noaudio", "-P", os.path.join(HERE, "body_dump.py"), "--", out, kit_glb, "kind=kit"] + options, capture_output=True, text=True)
    if not os.path.exists(out):
        raise SystemExit("the dump failed:\n" + (done.stdout + done.stderr)[-3000:])
    return body_measure.measure(out)


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


def update(profile, found, frame, reference):
    """One round: every curve is pulled by the ratio of the reference's measure to the cast's, read where the cast's rows are."""
    joints, tall, move = frame["joints"], frame["tall"], np.asarray(frame["move"], float)
    height = lambda z: z * tall + move[2]
    hip, knee, ankle, neck = (joints[key][2] for key in ("hip", "knee", "ankle", "neck"))
    level = lambda z: (height(z) - hip) / (neck - hip)
    for curve, field, neck_field in (("trunk_x", "width", "neck_width"), ("trunk_y", "depth", "neck_depth")):
        pairs = rows(found, reference, "torso", field)
        params = [level(row["z"]) for row, _, _ in pairs] + [level(found["neck"])]
        ratios = [want / have for _, have, want in pairs] + [reference[neck_field] / found[neck_field]]
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
    reach = [(want - have) / 2 for row, have, want in rows(found, reference, "shoulder", "width") if row["k"] <= SHOULDER_REACH]
    profile["arm_out"] = float(np.clip(profile["arm_out"] + STEP * np.mean(reach), -ARM_OUT_LIMIT, ARM_OUT_LIMIT))


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
    base = dump_options(opts)
    profile, arm_deg = parle_build.identity(opts["kit"]), float(reference["arm_angle"])
    best, history = None, []
    for turn in range(int(opts.get("rounds", 12)) + 1):
        parle_build.save_profile(profile, os.path.join(work, "profile.json"))
        found, frame = dump(kit_glb, os.path.join(work, "round.npz"), base + [f"body={os.path.join(work, 'profile.json')}", f"arm={arm_deg:.1f}"])
        percent, parts = body_measure.similarity(found, reference)
        history.append(percent)
        print(f"round {turn}: {percent:.2f}% arm {arm_deg:.1f} {parts}", flush=True)
        if turn == 0:
            profile["crotch"] = crotch_share(found, frame)
        if best is None or percent > best["similarity"]:
            best = dict(copy.deepcopy(profile), similarity=round(percent, 2), parts=parts, arm_deg=round(arm_deg, 1))
            os.replace(os.path.join(work, "round.npz"), os.path.join(work, "best.npz"))
        # Past `want` it goes on while a round still brings the body closer; short of it, it gives up when three rounds bring nothing.
        if (percent >= float(opts.get("want", 95)) and turn and percent - history[turn - 1] < 0.2) or (turn >= 3 and percent - history[turn - 3] < 0.05):
            break
        update(profile, found, frame, reference)
        arm_deg = float(np.clip(arm_deg + reference["arm_angle"] - found["arm_angle"], 10, 60))
    best.update(reference=os.path.basename(reference_path), before=round(history[0], 2), rounds=len(history) - 1, at_limit=at_limit(best))
    parle_build.save_profile(best, out)
    print(f"done: {best['before']}% -> {best['similarity']}% in {best['rounds']} rounds, at the limit: {best['at_limit']}", flush=True)


def mean_measure(out, paths):
    """The mean of the measures of several bodies, row by row; a row one of them lacks is left out."""
    found = [json.load(open(path)) for path in paths]

    def mean(items):
        if any(item is None for item in items):
            return None
        if isinstance(items[0], dict):
            return {key: mean([item[key] for item in items]) for key in items[0]}
        if isinstance(items[0], list):
            return [mean(list(row)) for row in zip(*items)]
        return round(float(np.mean(items)), 4)

    json.dump(mean(found), open(out, "w"), indent=1)


if __name__ == "__main__":
    paths = [item for item in sys.argv[2:] if "=" not in item]
    options = dict(item.split("=", 1) for item in sys.argv[2:] if "=" in item)
    if sys.argv[1] == "match":
        match(paths[0], paths[1], paths[2], options)
    else:
        mean_measure(paths[0], paths[1:])
