"""
    License information: data/licenses/makehuman_license.txt

    A body measured at levels set by its joints, and the build of a kit matched to such a measure (owner, 2026-11-13 and 14: of a man
    he had picked, dressed in a T-shirt and shorts, "consegue usar o corpo desse avatar padrão para homem?" and "o homem tem bom corpo,
    copia toda a estrutura e formato").
      python tools/body_joints.py measure <dump.npz> <out.json> [hanging=<dump.hang.npz>]
      python tools/body_joints.py match <kit.glb> <reference.json> <profile.json> kit=<name> height=<m> sex=<male|female> [chest=<npy>] [rounds=4] [work=<folder>]
    `tools/body_measure.py` finds the crotch, the armpit and the neck on the skin and reads its rows between them: on this dressed
    body it found the armpit above the joints of the arms (0.84 of the stature, the joints at 0.81) and the crotch at 0.545, above
    the joints of the hips, so no row of its was read where it should be. Here the levels come from the skeleton, which a dressed model has too: the trunk from the joints of the hips (0)
    to the joints of the arms (1), each arm and each leg along its own bones. A level slice is read as a cloud of points and not as
    loops (cloth is open at the hem and lies in layers): the trunk is what lies about the middle with nothing for TRUNK_GAP on each
    side, and a level where the arms are not loose from it has no width.
    A reference measured this way may lack rows: what its clothes hide is left out by whoever prepares it (and what they only
    thicken is taken off there), and `match` moves nothing where the reference says nothing. `match` starts from the profile that is
    there, so what an earlier reference gave the kit stays where this one is silent. Only measures of a reference are kept.
    A reference may also say how high its joints stand barefoot (`joints_up`: the knees and the hips, as shares of the stature): the
    kit then takes the lengths that put its own there (`parle_build.LENGTHS`), its foot as it is.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import body_match  # noqa: E402
import body_measure  # noqa: E402
import parle_build  # noqa: E402

# Trunk levels: 0 at the joints of the hips, 1 at the joints of the arms.
TRUNK_U = np.round(np.arange(0.0, 1.0001, 0.05), 3)
# Arm rings: 0 shoulder, 1 elbow, 2 wrist. Leg rings: 0 hip, 1 knee, 2 ankle.
ARM_T = np.round(np.arange(0.3, 1.9001, 0.1), 3)
LEG_Q = np.round(np.arange(0.2, 1.9001, 0.1), 3)
# As shares of the stature: the nothing that parts an arm from the trunk, how far from its bone a ring of an arm or a leg may reach, and
# the band about the joints of the arms where the breadth across the shoulders is read, arms hanging.
TRUNK_GAP, LIMB_REACH, DELTOID_BAND = 0.006, 0.075, (-0.07, 0.0)
ROUNDS = 4
# How far past the reference's last row a curve still takes some of its pull: of hips to neck for the trunk, of a bone for a limb.
TRUNK_FADE, LIMB_FADE = 0.15, 0.3


def frame_of(path):
    """Points with the floor at 0 and the stature at 1 (`body_measure.load`), and the joints in the same frame."""
    points, faces, frame = body_measure.load(path)
    joints = {name: (np.asarray(joint, float) - frame["move"]) / frame["tall"] for name, joint in frame["joints"].items()}
    return points, faces, joints, frame


def cloud(loops):
    return np.concatenate([loop["points"] for loop in loops]) if loops else np.zeros((0, 3))


def about(loops, centre, gap, axis=0):
    """The loops of a slice that make one stretch about `centre` along `axis`, with no hole wider than `gap` between them, and whether
    there is something else on each side of it. Each loop counts from end to end: a coarse cloth has few points on a long side."""
    spans = sorted(((loop["low"][axis], loop["high"][axis], number) for number, loop in enumerate(loops)), key=lambda span: abs((span[0] + span[1]) / 2 - centre))
    first = next((span for span in spans if span[0] - gap <= centre <= span[1] + gap), None)
    if first is None:
        return [], False, False
    low, high, mine = first[0], first[1], {first[2]}
    grown = True
    while grown:
        grown = False
        for start, end, number in spans:
            if number not in mine and start <= high + gap and end >= low - gap:
                low, high, grown = min(low, start), max(high, end), True
                mine.add(number)
    rest = [span for span in spans if span[2] not in mine]
    return [loops[number] for number in sorted(mine)], any(span[1] < low for span in rest), any(span[0] > high for span in rest)


def trunk_rows(points, faces, joints):
    hip, shoulder = joints["hip"][2], joints["shoulder"][2]
    rows = []
    for u in TRUNK_U:
        z = hip + u * (shoulder - hip)
        mine, left, right = about(body_measure.level(points, faces, z), 0.0, TRUNK_GAP)
        if not mine:
            rows.append(None)
            continue
        # With an arm on each side the stretch is the trunk alone; below the hands there is no arm to be taken for it.
        spots, loose = cloud(mine), (left and right) or z < min(joints["wrist"][2], joints["wrist_r"][2]) - TRUNK_GAP
        rows.append(dict(u=float(u), z=round(float(z), 4), width=round(float(np.ptp(spots[:, 0])), 4) if loose else None, depth=round(float(np.ptp(spots[:, 1])), 4) if loose else None))
    return rows


def limb_rows(points, faces, chain, stations, level):
    """Rings along two bones, `chain` being the three joints. `level` cuts by level slices (a leg); otherwise square to the bone (an arm)."""
    rows = []
    for t in stations:
        start, end = (chain[0], chain[1]) if t <= 1 else (chain[1], chain[2])
        spot = start + (t if t <= 1 else t - 1) * (end - start)
        way = np.array([0.0, 0.0, 1.0]) if level else (end - start) / np.linalg.norm(end - start)
        loops = [loop for loop in body_measure.cut(points, faces, spot, way) if np.linalg.norm(loop["mid"] - spot) < LIMB_REACH]
        if level:
            loops = about(loops, spot[0], TRUNK_GAP)[0]
        spots = cloud(loops)
        if len(spots) < 6:
            rows.append(None)
            continue
        if level:
            rows.append(dict(t=float(t), width=round(float(np.ptp(spots[:, 0])), 4), depth=round(float(np.ptp(spots[:, 1])), 4)))
            continue
        across = np.array([0.0, 1.0, 0.0]) - way[1] * way
        across /= np.linalg.norm(across)
        depth, thick = float(np.ptp(spots @ across)), float(np.ptp(spots @ np.cross(way, across)))
        rows.append(dict(t=float(t), girth=round(float(np.sqrt(depth * thick)), 4)))
    return rows


def neck_of(points, faces, joints):
    """The narrowest level between the joints of the arms and the joint of the head: how wide and how deep."""
    found = []
    for z in np.arange(joints["shoulder"][2] + 0.01, joints["head"][2] + 0.02, 0.003):
        spots = cloud(about(body_measure.level(points, faces, z), 0.0, TRUNK_GAP)[0])
        if len(spots):
            found.append((float(np.ptp(spots[:, 0])), float(np.ptp(spots[:, 1]))))
    return min(found) if found else (None, None)


def deltoids(path):
    """The breadth across the shoulders of a body dumped with its arms hanging: the widest level about the joints of the arms."""
    points, faces, joints, _ = frame_of(path)
    wide = [float(np.ptp(spots[:, 0])) for spots in (cloud(body_measure.level(points, faces, joints["shoulder"][2] + lift)) for lift in np.arange(*DELTOID_BAND, 0.005)) if len(spots)]
    return round(max(wide), 4) if wide else None


def measure(path, hanging=None):
    points, faces, joints, frame = frame_of(path)
    arm = [joints[name] for name in ("shoulder", "elbow", "wrist")]
    leg = [joints[name] for name in ("hip", "knee", "ankle")]
    neck_width, neck_depth = neck_of(points, faces, joints)
    return dict(stature_m=round(frame["tall"], 4), joints={name: [round(float(value), 4) for value in joint] for name, joint in joints.items()},
                shoulders_apart=round(float(abs(joints["shoulder"][0] - joints["shoulder_r"][0])), 4), bideltoid=deltoids(hanging) if hanging else None,
                neck_width=None if neck_width is None else round(neck_width, 4), neck_depth=None if neck_depth is None else round(neck_depth, 4),
                trunk=trunk_rows(points, faces, joints), arm=limb_rows(points, faces, arm, ARM_T, False), leg=limb_rows(points, faces, leg, LEG_Q, True))


def pairs(found, reference, group, field):
    """(row of the kit, the ratio the reference asks of it) where both bodies have that row."""
    return [(a, b[field] / a[field]) for a, b in zip(found[group], reference[group]) if a and b and a.get(field) and b.get(field)]


def pull(name, curve, params, ratios, fade):
    """`body_match.pull`, but only where the reference has rows: between the first and the last of `params` the curve takes the
    ratios, and outside them it is left as it is, the pull dying away over `fade` of the curve's own measure. Held to the end as
    `body_match.pull` holds it, the ratio of the calf thinned the thigh, which the shorts hide, and the ratio of the waist widened
    the hips (2026-11-14, the first run). And it is the pull that is made even from station to station, not the curve: evened
    whole every round, the curve lost its own shape where nothing pulled it (the thigh again). Past the fade no station moves at
    all: the evening out let a little of the pull through to the next station or two, and matched again, with the legs their
    new length, the waist under the shirt went on widening, half a percent a run."""
    stations = parle_build.CURVES[name]
    order = np.argsort(params)
    places, gains = np.array(params, float)[order], np.array(ratios, float)[order] ** body_match.STEP
    inside = np.clip(1 - np.maximum(places[0] - stations, stations - places[-1]) / fade, 0, 1)
    low, high = body_match.limits(name)
    pulled = np.where(inside > 0, body_match.soft(np.interp(stations, places, gains) ** inside), 1.0)
    return list(np.clip(np.array(curve) * pulled, low, high))


def update(profile, found, reference):
    """One round: each curve is pulled by the ratio of the reference's measure to the kit's (`body_match.pull`), where the reference has one."""
    joints = {name: np.asarray(joint, float) for name, joint in found["joints"].items()}
    hip, shoulder, neck = joints["hip"][2], joints["shoulder"][2], joints["neck"][2]
    level = lambda u: (u * (shoulder - hip)) / (neck - hip)
    for curve, field, neck_field in (("trunk_x", "width", "neck_width"), ("trunk_y", "depth", "neck_depth")):
        rows = pairs(found, reference, "trunk", field)
        if not rows:
            continue
        params, ratios = [level(row["u"]) for row, _ in rows], [ratio for _, ratio in rows]
        if found.get(neck_field) and reference.get(neck_field):
            # The neck is read above the base of the neck: its ratio stands at the top of the curve.
            params, ratios = params + [float(parle_build.TRUNK_W[-1])], ratios + [reference[neck_field] / found[neck_field]]
        else:
            # A reference with no neck leaves the kit's: the curve is held where it is from the base of the neck up.
            params, ratios = params + [1.0], ratios + [1.0]
        profile[curve] = pull(curve, profile[curve], params, ratios, TRUNK_FADE)
    rows = pairs(found, reference, "arm", "girth")
    if rows:
        girth = pull("arm_depth", np.sqrt(np.array(profile["arm_depth"]) * np.array(profile["arm_thick"])), [row["t"] for row, _ in rows], [ratio for _, ratio in rows], LIMB_FADE)
        profile["arm_depth"], profile["arm_thick"] = list(girth), list(girth)
    for curve, field in (("leg_x", "width"), ("leg_y", "depth")):
        rows = pairs(found, reference, "leg", field)
        if rows:
            profile[curve] = pull(curve, profile[curve], [row["t"] for row, _ in rows], [ratio for _, ratio in rows], LIMB_FADE)
    if found.get("bideltoid") and reference.get("bideltoid"):
        closer = 1 + (reference["bideltoid"] - found["bideltoid"]) / found["shoulders_apart"]
        profile["girdle"] = float(np.clip((profile.get("girdle") or 1.0) * closer ** body_match.STEP, *body_match.GIRDLE_LIMITS))
    if reference.get("joints_up"):
        profile["lengths"] = lengths(profile.get("lengths"), {name: joint[2] for name, joint in joints.items()}, reference["joints_up"])


def lengths(now, mine, theirs):
    """The lengths of a profile (`parle_build.LENGTHS`) that put the knees and the hips of the kit as high as the reference's:
    `now` are the lengths the kit was measured with, `mine` how high its joints stand with them and `theirs` how high the
    reference's do, all in shares of the stature. The ankle stays where the kit has it: the foot keeps its shape."""
    now = now or {}
    thigh = (theirs["hip"] - theirs["knee"]) / (mine["hip"] - mine["knee"])
    shank = (theirs["knee"] - mine["ankle"]) / (mine["knee"] - mine["ankle"])
    return dict(thigh=round(float(now.get("thigh", 1.0) * thigh), 4), shank=round(float(now.get("shank", 1.0) * shank), 4))


def likeness(found, reference):
    """How like the reference the kit is where the reference has a measure: 100 less the mean gap, in percent, and the gaps by part."""
    parts = {}
    for group, field in (("trunk", "width"), ("trunk", "depth"), ("arm", "girth"), ("leg", "width"), ("leg", "depth")):
        gaps = [abs(ratio - 1) for _, ratio in pairs(found, reference, group, field)]
        if gaps:
            parts[f"{group}.{field}"] = round(100 * (1 - float(np.mean(gaps))), 1)
    for name in ("bideltoid", "neck_width", "neck_depth"):
        if found.get(name) and reference.get(name):
            parts[name] = round(100 * (1 - abs(reference[name] / found[name] - 1)), 1)
    if reference.get("joints_up"):
        parts["joints_up"] = round(100 * (1 - float(np.mean([abs(reference["joints_up"][name] / found["joints"][name][2] - 1) for name in ("knee", "hip")]))), 1)
    return round(float(np.mean(list(parts.values()))), 2) if parts else 0.0, parts


def match(kit_glb, reference_path, out, opts):
    reference = json.load(open(reference_path))
    work = opts.get("work") or os.path.join(os.path.dirname(os.path.abspath(out)), "work")
    os.makedirs(work, exist_ok=True)
    profile = parle_build.load_profile(out) or parle_build.identity(opts["kit"])
    # What the profile was matched to before, for the record: this match starts from it and keeps it where the reference is silent.
    # Matched again to the reference it has (which may have learnt to say more), the record is the one it carries.
    again = profile.get("reference") == os.path.basename(reference_path)
    started = profile.get("started_from") if again else {key: profile.get(key) for key in ("reference", "similarity") if profile.get(key)}
    for key in ("similarity", "parts", "before", "rounds", "arm_deg", "bideltoid"):
        profile.pop(key, None)
    base = body_match.dump_options(opts) + [f"hang={body_match.HANG_DEG}", f"arm={float(reference.get('arm_deg', 40)):.1f}"]
    best, round_npz = None, os.path.join(work, "round.npz")
    history = []
    for turn in range(int(opts.get("rounds", ROUNDS)) + 1):
        parle_build.save_profile(profile, os.path.join(work, "profile.json"))
        body_match.dump(kit_glb, round_npz, base + [f"body={os.path.join(work, 'profile.json')}"])
        found = measure(round_npz, body_match.hang_path(round_npz))
        percent, parts = likeness(found, reference)
        history.append(percent)
        print(f"round {turn}: {percent:.2f}% {parts}", flush=True)
        if best is None or percent > best["similarity"]:
            best = dict(json.loads(json.dumps(profile)), similarity=percent, parts=parts, bideltoid=found.get("bideltoid"))
            json.dump(found, open(os.path.join(work, "best.json"), "w"), indent=1)
        if turn and percent - history[turn - 1] < 0.1:
            break
        update(profile, found, reference)
    best.update(reference=os.path.basename(reference_path), by="joints", before=history[0], rounds=len(history) - 1, arm_deg=float(reference.get("arm_deg", 40)),
                started_from=started or None, at_limit=body_match.at_limit(best))
    parle_build.save_profile(best, out)
    print(f"done: {history[0]}% -> {best['similarity']}% in {len(history) - 1} rounds, at the limit: {best['at_limit']}", flush=True)


if __name__ == "__main__":
    paths = [item for item in sys.argv[2:] if "=" not in item]
    options = dict(item.split("=", 1) for item in sys.argv[2:] if "=" in item)
    if sys.argv[1] == "match":
        match(paths[0], paths[1], paths[2], options)
    else:
        json.dump(measure(paths[0], options.get("hanging")), open(paths[1], "w"), indent=1)
        print("measured", paths[1])
