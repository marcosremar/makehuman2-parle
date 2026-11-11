"""
    License information: data/licenses/makehuman_license.txt

    Silhouette of an unclothed body in the A pose, by slices, as shares of the stature:
    `python tools/body_measure.py <dump.npz> [<out.json>] [hanging=<dump.hang.npz>]`.
    The dump comes from `tools/body_dump.py`. Runs outside Blender (numpy and scipy). Trunk: level slices from the crotch to the armpit. Shoulder: whole
    width with the arms between the armpit and the neck, and the height of the shoulder line at shares of the half chest. Leg: level slices of one
    leg. Arm: upright slices from the side of the chest to the wrist. With `hanging=`, the same body dumped with its arms hanging
    (`body_dump.py hang=`), the breadth across the shoulders too (`bideltoid`). `similarity` compares two bodies measure by measure.
"""
import json
import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

TORSO_SHARES = np.linspace(0, 1, 21)
SHOULDER_SHARES = (0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9)
LINE_SHARES = (0.5, 0.65, 0.8, 0.95, 1.1, 1.25)
LEG_SHARES = np.linspace(0.16, 0.96, 17)
ARM_SHARES = np.linspace(0.1, 1.0, 10)
WELD = 2e-4
ARM_WIDEST = 0.11
MIDDLE = 0.012
LANDMARKS = ("crotch", "armpit", "neck", "neck_width", "neck_depth")
# The shoulder rows up to this share of armpit to neck are cut across the arms; above it they are cut over the tip of the shoulder. With the
# arms out, 27 degrees below the horizontal, the width of such a row is where the top of the arm crosses its level: a body whose arm leaves
# the trunk 1% of the stature higher reads 4% of the stature wider, as broad or as narrow as its shoulders may be. Matched to the reference
# women on these rows, the three kits of women stood 2.1 to 2.9% of the stature broader across the shoulders than them once the arms hung,
# 3.4 to 4.6 cm: 25.7 to 26.5%, as broad as the men of the cast, 25.8 to 26.4% (owner, 2026-11-11: "os ombros da mulher estão largos
# demais"). A body that was also dumped with its arms hanging says how broad its shoulders are by `bideltoid`, and those rows count no
# more; nor does the width of the trunk at the armpit itself, a row cut through the roots of the arms.
SHOULDER_TIP = 0.45


def load(path):
    """Points with the floor at 0, the stature at 1, the middle of the head at x = 0 and the face to -y; `back` returns a point to the dump's metres."""
    data = np.load(path)
    turn = np.array([-1.0, -1.0, 1.0]) if bool(data["flip"]) else np.ones(3)
    points, faces = data["points"].astype(float) * turn, data["faces"].astype(np.int64)
    _, first, back = np.unique(np.round(points / WELD).astype(np.int64), axis=0, return_index=True, return_inverse=True)
    points, faces = points[first], back.ravel()[faces]
    faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])]
    floor, tall = points[:, 2].min(), float(np.ptp(points[:, 2]))
    move = np.array([np.median(points[points[:, 2] > floor + 0.93 * tall][:, 0]), 0.0, floor])
    joints = {key[2:]: data[key] * turn for key in data.files if key.startswith("j_")}
    return (points - move) / tall, faces, dict(tall=tall, move=move, turn=turn, joints=joints)


def cut(points, faces, origin, normal):
    """Slices the mesh by a plane: one entry per closed loop, with its points and box."""
    normal = np.asarray(normal, float)
    dist = (points - origin) @ normal
    df = dist[faces]
    side = df >= 0
    hit = side.any(1) & ~side.all(1)
    tri, df, side = faces[hit], df[hit], side[hit]
    if not len(tri):
        return []
    found, keys, cross = [], [], []
    with np.errstate(divide="ignore", invalid="ignore"):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            t = df[:, a] / (df[:, a] - df[:, b])
            found.append(points[tri[:, a]] + t[:, None] * (points[tri[:, b]] - points[tri[:, a]]))
            keys.append(np.minimum(tri[:, a], tri[:, b]) * len(points) + np.maximum(tri[:, a], tri[:, b]))
            cross.append(side[:, a] != side[:, b])
    found, keys, cross = np.stack(found, 1), np.stack(keys, 1), np.stack(cross, 1)
    pick = np.argsort(~cross, axis=1, kind="stable")[:, :2]
    rows = np.arange(len(tri))
    key_all = np.concatenate([keys[rows, pick[:, 0]], keys[rows, pick[:, 1]]])
    point_all = np.concatenate([found[rows, pick[:, 0]], found[rows, pick[:, 1]]])
    _, first, node = np.unique(key_all, return_index=True, return_inverse=True)
    node, spots = node.ravel(), point_all[first]
    a, b = node[:len(tri)], node[len(tri):]
    count, label = connected_components(coo_matrix((np.ones(len(a)), (a, b)), shape=(len(spots), len(spots))), directed=False)
    loops = []
    for index in range(count):
        mine = spots[label == index]
        if len(mine) >= 6:
            loops.append(dict(points=mine, low=mine.min(0), high=mine.max(0), mid=(mine.min(0) + mine.max(0)) / 2, size=np.ptp(mine, axis=0)))
    return loops


def level(points, faces, z):
    return cut(points, faces, np.array([0, 0, z]), [0, 0, 1])


def upright(points, faces, x):
    return cut(points, faces, np.array([x, 0, 0]), [1, 0, 0])


def central(loops):
    """The loop of the trunk: the widest one that crosses the middle of the body."""
    mine = [loop for loop in loops if loop["low"][0] < -MIDDLE and loop["high"][0] > MIDDLE and loop["size"][0] > 0.05]
    return max(mine, key=lambda loop: loop["size"][0]) if mine else None


def landmarks(points, faces):
    """Crotch: the lowest skin on the middle plane. Armpit: the highest level where both arms are loose from the trunk. Neck: its narrowest level."""
    middle = [loop for loop in upright(points, faces, 0.001) if loop["high"][2] > 0.6]
    crotch = float(min(loop["low"][2] for loop in middle))

    def arms_apart(z):
        loops = level(points, faces, z)
        trunk = central(loops)
        sides = [loop for loop in loops if loop is not trunk and abs(loop["mid"][0]) > 0.07 and loop["size"][0] > 0.02]
        return trunk is not None and any(loop["mid"][0] > 0 for loop in sides) and any(loop["mid"][0] < 0 for loop in sides)

    armpit = float(next(z for z in np.arange(0.86, 0.55, -0.002) if arms_apart(z)))
    necks = []
    for z in np.arange(armpit + 0.04, 0.93, 0.002):
        mine = [loop for loop in level(points, faces, z) if abs(loop["mid"][0]) < 0.05]
        if mine:
            cloud = np.concatenate([loop["points"] for loop in mine])
            necks.append((float(np.ptp(cloud[:, 0])), float(z), float(np.ptp(cloud[:, 1]))))
    width, neck, depth = min(necks)
    return crotch, armpit, neck, width, depth


def arm_rows(points, faces, start):
    """The arm's path is the centre of the highest loop of each upright slice, from the side of the chest outward; the arm is then cut square to it."""
    tip = float(points[:, 0].max())
    path = []
    for x in np.arange(start + 0.004, tip - 0.004, 0.004):
        loops = [loop for loop in upright(points, faces, x) if loop["size"][1] > 0.01]
        if loops:
            path.append(max(loops, key=lambda item: item["mid"][2])["mid"])
    path = np.array(path)
    path = np.stack([np.convolve(np.pad(path[:, axis], 3, mode="edge"), np.ones(7) / 7, mode="valid") for axis in range(3)], 1)
    ways = np.gradient(path, axis=0)
    ways /= np.linalg.norm(ways, axis=1)[:, None]
    cuts = []
    for spot, way in zip(path, ways):
        loops = cut(points, faces, spot, way)
        if not loops:
            continue
        loop = min(loops, key=lambda item: np.linalg.norm(item["mid"] - spot))
        across = np.array([0.0, 1.0, 0.0]) - way[1] * way
        across /= np.linalg.norm(across)
        depth, thick = float(np.ptp(loop["points"] @ across)), float(np.ptp(loop["points"] @ np.cross(way, across)))
        if np.linalg.norm(loop["mid"] - spot) < 0.03 and max(depth, thick) < ARM_WIDEST:
            cuts.append((*spot, depth, thick))
    cuts = np.array(cuts)
    along = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(cuts[:, :3], axis=0), axis=1))])
    span = along / along[-1]
    straight = (span > 0.2) & (span < 0.5)
    slope = np.polyfit(cuts[straight, 0], cuts[straight, 2], 1)[0]
    hand = (span > 0.5) & (span < 0.9)
    wrist = float(along[hand][np.argmin((cuts[:, 3] * cuts[:, 4])[hand])])
    rows = []
    for share in ARM_SHARES:
        at = [float(np.interp(share * wrist, along, cuts[:, column])) for column in range(5)]
        rows.append(dict(t=round(float(share), 2), x=round(at[0], 4), y=round(at[1], 4), z=round(at[2], 4), depth=round(at[3], 4), thick=round(at[4], 4)))
    return rows, round(float(np.degrees(np.arctan(-slope))), 1)


def bideltoid(path, armpit):
    """The breadth across the shoulders of a body dumped with its arms hanging, as a share of the stature: the whole width, arms and all,
    at the level of the armpit, which is found on the same body with its arms out. With the arms down that width hardly changes from one
    level to the next (0.1 to 0.3% of the stature for each 1% of level), so where exactly the armpit was found matters little."""
    points, faces, _ = load(path)
    cloud = np.concatenate([loop["points"] for loop in level(points, faces, armpit)])
    return round(float(np.ptp(cloud[:, 0])), 4)


def measure(path, hanging=None):
    points, faces, frame = load(path)
    crotch, armpit, neck, neck_width, neck_depth = landmarks(points, faces)
    out = dict(stature_m=round(frame["tall"], 4), crotch=round(crotch, 4), armpit=round(armpit, 4), neck=round(neck, 4), neck_width=round(neck_width, 4), neck_depth=round(neck_depth, 4))
    if hanging:
        out["bideltoid"] = bideltoid(hanging, armpit)
    torso = []
    for share in TORSO_SHARES:
        z = crotch + 0.004 + share * (armpit - 0.004 - crotch - 0.004)
        loop = central(level(points, faces, z))
        torso.append(None if loop is None else dict(u=round(float(share), 2), z=round(float(z), 4), width=round(float(loop["size"][0]), 4), depth=round(float(loop["size"][1]), 4)))
    out["torso"] = torso
    shoulder = []
    for share in SHOULDER_SHARES:
        z = armpit + share * (neck - armpit)
        cloud = np.concatenate([loop["points"] for loop in level(points, faces, z)])
        shoulder.append(dict(k=share, z=round(float(z), 4), width=round(float(np.ptp(cloud[:, 0])), 4)))
    out["shoulder"] = shoulder
    # The shoulder line is read at shares of the body's own half chest, so a narrow body is not read on the arm.
    chest = next((row["width"] / 2 for row in reversed(torso) if row), 0.09)
    line = []
    for share in LINE_SHARES:
        x = share * chest
        tops = [loop["high"][2] for loop in upright(points, faces, x) if loop["high"][2] < neck + 0.005 and loop["high"][2] > armpit - 0.08]
        line.append(dict(share=share, x=round(float(x), 4), rise=round(float(max(tops) - armpit), 4)) if tops else None)
    out["line"] = line
    legs = []
    for share in LEG_SHARES:
        loops = [loop for loop in level(points, faces, share * crotch) if loop["mid"][0] > 0.01 and loop["low"][0] > -MIDDLE]
        if not loops:
            legs.append(None)
            continue
        loop = max(loops, key=lambda item: item["size"][0] * item["size"][1])
        legs.append(dict(v=round(float(share), 2), z=round(float(share * crotch), 4), width=round(float(loop["size"][0]), 4), depth=round(float(loop["size"][1]), 4)))
    out["leg"] = legs
    try:
        chest = next(row for row in reversed(torso) if row)
        out["arm"], out["arm_angle"] = arm_rows(points, faces, chest["width"] / 2)
    except Exception:
        out["arm"], out["arm_angle"] = [None] * len(ARM_SHARES), None
    return out, frame


def flatten(found):
    """Every measure that counts for the likeness, by name. A body that has the breadth across its shoulders with the arms hanging counts
    that, and not the widths cut across its arms: the shoulder rows up to the tip of the shoulder (SHOULDER_TIP) and the trunk at the armpit."""
    flat = {key: found[key] for key in LANDMARKS}
    flat["armpit"], flat["neck"] = found["armpit"] - found["crotch"], found["neck"] - found["armpit"]
    flat["bideltoid"] = found.get("bideltoid")
    for group, fields in (("torso", ("width", "depth")), ("shoulder", ("width",)), ("line", ("rise",)), ("leg", ("width", "depth")), ("arm", ("depth", "thick"))):
        for index, row in enumerate(found[group]):
            arms = row is not None and ((group == "shoulder" and row["k"] <= SHOULDER_TIP) or (group == "torso" and row["u"] >= 1))
            for field in fields:
                left_out = arms and field == "width" and flat["bideltoid"] is not None
                flat[f"{group}.{field}.{index:02d}"] = None if row is None or left_out else row[field]
    return flat


def similarity(cast, reference):
    """100 x (1 - mean relative gap) over the measures both bodies have, whole and by part."""
    a, b = flatten(cast), flatten(reference)
    gaps = {key: min(1.0, abs(a[key] - b[key]) / abs(b[key])) for key in b if a.get(key) is not None and b[key] is not None and abs(b[key]) > 1e-6}
    parts = {}
    for key, gap in gaps.items():
        parts.setdefault("landmarks" if key in LANDMARKS else ".".join(key.split(".")[:2]), []).append(gap)
    return 100 * (1 - float(np.mean(list(gaps.values())))), {part: round(100 * (1 - float(np.mean(values))), 1) for part, values in parts.items()}


if __name__ == "__main__":
    paths = [item for item in sys.argv[1:] if "=" not in item]
    options = dict(item.split("=", 1) for item in sys.argv[1:] if "=" in item)
    result, _ = measure(paths[0], options.get("hanging"))
    if len(paths) > 1:
        json.dump(result, open(paths[1], "w"), indent=1)
    print(json.dumps({key: value for key, value in result.items() if not isinstance(value, list)}))
