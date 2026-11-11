"""
    License information: data/licenses/makehuman_license.txt

    A defined male torso for makehuman2-parle: the muscles of a sculpture on the proportions of the body.

    The shoulders grow with the artist-made MakeHuman deltoid targets; the lat and V targets made wings
    6 to 10 cm wider than the reference body, and a MakeHuman pectoral in the place of a band of the
    trunk left a crease across the back and a sagging chest (2026-11-10). The muscles of the trunk come
    from the sculpture, the way a displacement map works: both torsos go to the same chart, the trunk
    seen from the front and from the back, as wide as the trunk at each height, with the armpits at the
    same height and without the arms. What each has between two sizes, less what a height has in common
    across the trunk, is its muscle relief, and the body trades its own for the sculpture's, as depth. On the
    chest the step under the pectorals stays in, although it is the same from side to side: it is what
    makes a pectoral a plate.
    The proportions and the posture stay the body's: the profile of the kit owns them.

    Works in metres with Z up.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from core.garment_fit import vertex_normals

COLUMNS = 240
HEIGHTS = 420
# Height of the chart in shares of crotch to neck pit: the top of the hips to the collarbones.
CHART_LOW = -0.15
CHART_HIGH = 1.08
# Width of the chart in half widths of the trunk at each height: a little past the flank on each side.
CHART_SIDE = 1.1
# Half the width of a trunk in shares of crotch to neck pit: it turns a size on the body into columns of the chart.
TRUNK_SHARE = 0.27
# Where the armpit of every torso sits on the chart, and the least gap between the arm and the trunk under it, in the same shares.
ARMPIT_LEVEL = 0.72
GAP_SHARE = 0.02
# The muscles are what is smaller than this, in the same shares, front and back; the proportions of the trunk, which the profile of the
# kit owns, are not. On the back a lat and the groove of the spine fit in 0.14. On the front a pectoral is a plate a quarter of the trunk
# tall, and at 0.14 only its middle came through: with 0.26 the whole plate does, the upper chest fills up to the collarbone and the
# cleft between the two goes shallow (2026-11-11). At 0.07 only the fine detail came through (2026-11-10).
RELIEF_SHARE = (0.26, 0.14)
# The trunk has a vertex every 2 to 3 cm: detail finer than this lands on one vertex and not on the next, and reads as dents. At
# 0.03 the edge under the pectorals and the rows of the abdomen were blurred away and the front stayed as it came (2026-11-10).
FINEST_SHARE = 0.012
# The most a vertex moves, front and back: the plates of the chest stand further out than anything on the back.
MOST_RELIEF_M = (0.035, 0.018)
# How much of the body's own relief gives way to the sculpture's, front and back: all of it. The base mesh's pectorals are two round
# mounds in the middle of the chest. Kept, with the sculpture's on top, they stood out as cones with a deep cleft between them, where
# the reference bodies have a wide flat plate from the breastbone to the armpit with a straight lower edge (owner, 2026-11-11: "o
# peitoral do homem está feio").
OWN_TAKEN = (1.0, 1.0)
# How strong the sculpture's relief comes, front and back: the trunk has a vertex every 2 to 3 cm and rounds the edge of a plate away.
GAIN = (1.3, 1.0)
# What a height has in common from flank to flank is posture, not muscle, and goes out. On the back all of it does: kept, it came out as
# bands across the back. On the front only what is smooth over this share of the height goes: the step under the pectorals is the same
# from one side to the other, and with all of it out the plates never came through, only the mounds in the middle (2026-11-11).
POSTURE_SHARE = (0.15, 0.0)
# The pit of the neck over the shoulder joints and the crotch under the hip joints, in shares of the height between the two.
PIT_OVER_SHOULDER = 0.147
CROTCH_UNDER_HIP = 0.371
# A man's nipples. They sit between these heights, in shares of crotch to neck pit, and these distances from the midline, in half widths
# of the trunk. The mesh around one has edges shorter than NIPPLE_EDGE of that height (8 mm; the chest around it, 2 to 3 cm). The patch
# laid again around it is NIPPLE_PATCH wide to each side (3 cm), and NIPPLE_KEPT of what stood out comes back. Twice the wrong thing was
# laid flat: looked for as the point furthest in front of its neighbours, the nipple lost to the top of the pectoral and to a corner of
# the ribs, whose edges are ten times longer (2026-11-11).
NIPPLE_LEVELS = (0.62, 0.8)
NIPPLE_SIDES = (0.2, 0.7)
NIPPLE_EDGE = 0.012
NIPPLE_PATCH = 0.045
NIPPLE_KEPT = 0.2
# The flanks belong to the profile of the kit, which sets the width of the trunk level by level: the relief is whole up to the first
# of these and gone at the second, in half widths of the trunk from the midline.
EDGE = (0.8, 1.0)
# The relief is a depth, front or back: skin that faces up or sideways less than the first of these gets none, and all of it from the second.
FACING = (0.15, 0.5)
# The relief starts above the hips: below them the sculpture is a nude and the body wears shorts.
AREA_LOW = 0.12
AREA_RISE = 0.18
# Distance between the points taken on the faces of a torso to fill its chart, in shares of crotch to neck pit: a cell is 0.003 tall.
SAMPLE_SHARE = 0.002
MUSCLES = (
    "data/target/hm08/armslegs/l-upperarm-shoulder-muscle-incr.target",
    "data/target/hm08/armslegs/r-upperarm-shoulder-muscle-incr.target",
)
DECIMETRES = 10.0


def smoothstep(value):
    value = np.clip(value, 0, 1)
    return value * value * (3 - 2 * value)


def welded(points, faces):
    _, twin = np.unique(np.round(points, 4), axis=0, return_inverse=True)
    twin = twin.ravel()
    merged = np.zeros((twin.max() + 1, 3))
    merged[twin] = points
    pairs = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.unique(np.sort(twin[pairs], axis=1), axis=0)
    return merged, twin, edges[edges[:, 0] != edges[:, 1]], twin[faces]


def landmarks(points):
    """Midline, crotch and the pit of the neck, read from the surface: the sculpture has no skeleton."""
    centre = float(np.median(points[:, 0]))
    height = np.ptp(points[:, 2])
    bottom = points[:, 2].min()
    line = points[np.abs(points[:, 0] - centre) < height * 0.006]
    crotch = float(line[:, 2].min())
    # In profile the front of the neck pit is the deepest point between the chest and the chin.
    zones = np.linspace(bottom + height * 0.76, bottom + height * 0.9, 40)
    front = []
    for low, high in zip(zones[:-1], zones[1:]):
        band = line[(line[:, 2] >= low) & (line[:, 2] < high)]
        front.append(band[:, 1].min() if len(band) else -np.inf)
    pit = float(zones[int(np.argmax(front))])
    torso = points[(points[:, 2] > crotch) & (points[:, 2] < pit) & (np.abs(points[:, 0] - centre) < height * 0.1)]
    axis = float((torso[:, 1].min() + torso[:, 1].max()) / 2)
    return centre, axis, crotch, pit


def rigged_landmarks(points, shoulder, hip):
    """Midline, crotch and the pit of the neck of a body that has a skeleton, from the heights of its shoulder and hip joints.

    Read from the surface (`landmarks`), the dip between the pectorals of a slim man passed for the pit of his neck, 12 cm low, and his
    crotch came out 6 cm higher than the muscled man's: the chart of his trunk was a quarter short, the plates of the sculpture landed
    on his ribs and his nipples fell outside the patch that lays them flat (2026-11-11). The shares are those the surface gives on the
    muscled man, where it reads right and where every size of the chart was judged."""
    reach = shoulder - hip
    centre = float(np.median(points[:, 0]))
    crotch, pit = hip - CROTCH_UNDER_HIP * reach, shoulder + PIT_OVER_SHOULDER * reach
    torso = points[(points[:, 2] > crotch) & (points[:, 2] < pit) & (np.abs(points[:, 0] - centre) < np.ptp(points[:, 2]) * 0.1)]
    return centre, float((torso[:, 1].min() + torso[:, 1].max()) / 2), float(crotch), float(pit)


def surface(points, faces, step):
    """Points on the faces, `step` apart at most: the body has too few vertices to fill a chart."""
    corners = points[faces]
    cuts = np.ceil(np.linalg.norm(corners - corners[:, [1, 2, 0]], axis=2).max(axis=1) / step).astype(int)
    found = [points]
    for cut in np.unique(cuts[cuts > 1]):
        first, second = np.meshgrid(np.arange(cut + 1), np.arange(cut + 1), indexing="ij")
        inside = first + second <= cut
        first, second = first[inside] / cut, second[inside] / cut
        shares = np.stack([1 - first - second, first, second], axis=1)
        found.append(np.einsum("sk,fkd->fsd", shares, corners[cuts == cut]).reshape(-1, 3))
    return np.concatenate(found)


def armpit(points, marks):
    """Level of the armpits and half the width of the trunk there, in shares of crotch to neck pit: where the gap to the arm closes."""
    centre, _, crotch, pit = marks
    span = pit - crotch
    side = np.abs(points[:, 0] - centre) / span
    level = (points[:, 2] - crotch) / span
    found = None
    for low in np.arange(0.5, 1.0, 0.01):
        band = np.sort(side[(level >= low) & (level < low + 0.02)])
        gaps = np.where(np.diff(band) > GAP_SHARE)[0]
        if not len(gaps):
            break
        found = float(low + 0.01), float(band[gaps[0]])
    if found is None:
        raise ValueError("no gap between the arms and the trunk: the torso has to come with its arms away from it")
    return found


def aligned(level, own):
    """Levels with the armpit of every torso at the same height of the chart: the pectorals and the lats hang from it."""
    return np.where(level < own, level * ARMPIT_LEVEL / own, ARMPIT_LEVEL + (level - own) * (1 - ARMPIT_LEVEL) / (1 - own))


def half_widths(points, marks, pits):
    """Half the width of the trunk on each row of the chart, in shares of crotch to neck pit and without the arms."""
    centre, _, crotch, pit = marks
    span = pit - crotch
    side = np.abs(points[:, 0] - centre) / span
    row = rows_of((points[:, 2] - crotch) / span, pits).astype(int)
    inside = (row >= 0) & (row < HEIGHTS) & (side < pits[1])
    widths = np.zeros(HEIGHTS)
    np.maximum.at(widths, row[inside], side[inside])
    filled = np.flatnonzero(widths > 0)
    return gaussian_filter(np.interp(np.arange(HEIGHTS), filled, widths[filled]), 3, mode="nearest")


def rows_of(level, pits):
    return (aligned(level, pits[0]) - CHART_LOW) / (CHART_HIGH - CHART_LOW) * HEIGHTS


def chart(points, marks, pits, widths):
    """Where each point falls on the chart: row, column, side in half widths of the trunk and how far to the front it is."""
    centre, _, crotch, pit = marks
    span = pit - crotch
    row = rows_of((points[:, 2] - crotch) / span, pits)
    side = (points[:, 0] - centre) / span / np.interp(row, np.arange(HEIGHTS) + 0.5, widths)
    return row, (side / CHART_SIDE + 1) / 2 * COLUMNS, side, -points[:, 1] / span


def relief_map(points, marks, pits, widths):
    """Muscle relief of a torso, front and back, and how much of each is known: what is left of its depth between FINEST_SHARE and RELIEF_SHARE.

    `points` cover the surface closer than a cell of the chart. The chart is the trunk seen from the front and from the back, as wide
    as the trunk at each height and without the arms. A chart of the radius around the trunk kept the arm of each torso in its own
    pose and four tenths of the oval of the trunk, so the difference between two torsos was pose and proportion, not muscle (2026-11-10).
    Each height loses what it has in common across the trunk: the hollow of the back of the sculpture sits at another height than
    the body's, and with it in the difference came out as bands across the back.
    """
    row, column, side, front = chart(points, marks, pits, widths)
    inside = (row >= 0) & (row < HEIGHTS) & (np.abs(side) < CHART_SIDE) & (np.abs(points[:, 0] - marks[0]) < pits[1] * (marks[3] - marks[2]))
    cells = row[inside].astype(int), column[inside].astype(int)
    across = ((np.arange(COLUMNS) + 0.5) / COLUMNS * 2 - 1) * CHART_SIDE

    def blurred(grid, known, share):
        sigmas = (share / (CHART_HIGH - CHART_LOW) * HEIGHTS, share / (2 * CHART_SIDE * TRUNK_SHARE) * COLUMNS)
        weight = gaussian_filter(known, sigmas, mode="nearest")
        return gaussian_filter(grid, sigmas, mode="nearest") / np.maximum(weight, 1e-6), weight

    reliefs, knowns = [], []
    for face, out in enumerate((front[inside], -front[inside])):
        grid = np.full((HEIGHTS, COLUMNS), -np.inf)
        np.maximum.at(grid, cells, out)
        known = np.isfinite(grid).astype(np.float64)
        grid = np.where(known > 0, grid, 0.0)
        fine, weight = blurred(grid, known, FINEST_SHARE)
        relief = fine - blurred(grid, known, RELIEF_SHARE[face])[0]
        # What a height has in common from flank to flank is posture, not muscle: the curve of the spine and the line of the belly (POSTURE_SHARE).
        middle = weight * (np.abs(across) < EDGE[0])
        line = (relief * middle).sum(axis=1) / np.maximum(middle.sum(axis=1), 1e-6)
        if POSTURE_SHARE[face] > 0:
            line = gaussian_filter1d(line, POSTURE_SHARE[face] / (CHART_HIGH - CHART_LOW) * HEIGHTS, mode="nearest")
        reliefs.append(relief - line[:, None])
        knowns.append(weight)
    return np.array(reliefs), np.array(knowns)


def torso_chart(points, faces, marks=None):
    """Landmarks, armpits, widths and relief of a torso, with its surface on the chart closer than a cell. `marks` are the landmarks of a
    body that has a skeleton (`rigged_landmarks`); without them they are read from the surface."""
    marks = landmarks(points) if marks is None else marks
    span = marks[3] - marks[2]
    level = (points[:, 2] - marks[2]) / span
    near = (level > CHART_LOW - 0.1) & (level < CHART_HIGH + 0.1)
    skin = surface(points, faces[near[faces].all(axis=1)], span * SAMPLE_SHARE)
    pits = armpit(skin, marks)
    widths = half_widths(skin, marks, pits)
    return marks, pits, widths, relief_map(skin, marks, pits, widths)


def sample(grid, row, column):
    row = np.clip(row - 0.5, 0, HEIGHTS - 1.001)
    column = np.clip(column - 0.5, 0, COLUMNS - 1.001)
    r0, c0 = row.astype(int), column.astype(int)
    fr, fc = row - r0, column - c0
    return grid[r0, c0] * (1 - fr) * (1 - fc) + grid[r0 + 1, c0] * fr * (1 - fc) + grid[r0, c0 + 1] * (1 - fr) * fc + grid[r0 + 1, c0 + 1] * fr * fc


def read_target(path, count):
    shift = np.zeros((count, 3))
    with open(path) as lines:
        for line in lines:
            if line.startswith("#") or not line.strip():
                continue
            index, x, y, z = line.split()[:4]
            shift[int(index)] = (float(x), -float(z), float(y))
    return shift / DECIMETRES


def muscles(count):
    """Shoulders of the muscled man, one shift per vertex of the base mesh: the sum of the artist-made targets, and nothing taken away."""
    return sum(read_target(path, count) for path in MUSCLES)


def without_nipples(points, depth, edges, where, span, centre=0.0, patch=None):
    """`depth` (y, the front of the body at the smaller values) with the nipple of each side of `where` laid flat.

    The base mesh models a nipple as a small cone with a blunt tip, drawn with triangles a few millimetres wide in the middle of a chest
    whose triangles are 2 to 3 cm wide: it is found as the point of `where` with the shortest edges, on each side of `centre`, when they
    are shorter than NIPPLE_EDGE of `span`. The patch around it, NIPPLE_PATCH of `span` to each side (or `patch`), is laid again as the
    skin that joins its rim, each point the mean of its neighbours, and a share of what stood out comes back so the chest is not ironed
    flat. With a wider patch the mound under the nipple goes with it."""
    depth = np.asarray(depth, float).copy()
    near = [[] for _ in range(len(depth))]
    for a, b in edges:
        near[a].append(b)
        near[b].append(a)
    inside = np.array([index for index in np.flatnonzero(where) if near[index]], dtype=int)
    if not len(inside):
        return depth
    reach = np.array([np.mean(np.linalg.norm(points[near[index]] - points[index], axis=1)) for index in inside])
    for side in (points[inside, 0] >= centre, points[inside, 0] < centre):
        if not side.any() or reach[side].min() > NIPPLE_EDGE * span:
            continue
        tip = inside[side][np.argmin(reach[side])]
        flat = np.hypot(points[inside, 0] - points[tip, 0], points[inside, 2] - points[tip, 2])
        ring = inside[flat < (NIPPLE_PATCH if patch is None else patch) * span]
        place = {int(index): row for row, index in enumerate(ring)}
        # Each point of the patch is the mean of its neighbours, and the rim keeps its depth: one linear system, small enough to solve whole.
        system, known = np.zeros((len(ring), len(ring))), np.zeros(len(ring))
        for row, index in enumerate(ring):
            system[row, row] = len(near[index])
            for other in near[index]:
                if int(other) in place:
                    system[row, place[int(other)]] -= 1
                else:
                    known[row] += depth[other]
        laid = np.linalg.solve(system, known)
        depth[ring] = laid + NIPPLE_KEPT * np.clip(depth[ring] - laid, None, 0)
    return depth


def moulded(points, faces, relief, trunk, strength=1.0, soft=0, patch=None, joints=None):
    """The muscles of the sculpture in the place of the body's own, as depth on the front and on the back.

    `relief` is the pair relief_map gave for the sculpture, and `trunk` is the share of each vertex that belongs to the torso bones.
    `strength` is how much of the sculpture's relief a body takes: all of it on a muscled man, less on a man who is not. `soft` is how
    many times each vertex then takes half of what its neighbours got: on a man who does not train the edge of a plate is not a ledge.
    `patch` is how wide the skin is laid again around each nipple, in shares of crotch to neck pit (`without_nipples`). `joints` are the
    heights of the shoulder and the hip joints of the body (`rigged_landmarks`)."""
    merged, twin, edges, welded_faces = welded(points, faces)
    marks, pits, widths, (own, known) = torso_chart(merged, welded_faces, None if joints is None else rigged_landmarks(merged, *joints))
    row, column, side, _ = chart(merged, marks, pits, widths)
    span = marks[3] - marks[2]
    level = row / HEIGHTS * (CHART_HIGH - CHART_LOW) + CHART_LOW
    facing = vertex_normals(merged, welded_faces)[:, 1]
    back = facing > 0

    def read(grids):
        return np.where(back, sample(grids[1], row, column), sample(grids[0], row, column))

    bones = np.zeros(len(merged))
    bones[twin] = trunk
    area = smoothstep((bones - 0.6) / 0.35) * smoothstep((level - AREA_LOW) / AREA_RISE) * smoothstep((1.0 - level) / 0.12)
    area = area * smoothstep((EDGE[1] - np.abs(side)) / (EDGE[1] - EDGE[0])) * smoothstep((np.abs(facing) - FACING[0]) / (FACING[1] - FACING[0]))
    area = area * smoothstep((np.minimum(read(relief[1]), read(known)) - 0.5) / 0.4)
    # A soft ceiling: a hard one cuts the largest muscles flat and leaves a ledge around them.
    taken = np.where(back, OWN_TAKEN[1], OWN_TAKEN[0])
    wanted = (read(relief[0]) * np.where(back, GAIN[1], GAIN[0]) * strength - read(own) * taken) * span
    most = np.where(back, MOST_RELIEF_M[1], MOST_RELIEF_M[0])
    detail = np.tanh(wanted / most) * most * area
    for _ in range(soft):
        around, count = np.zeros(len(merged)), np.zeros(len(merged))
        for here, there in ((edges[:, 0], edges[:, 1]), (edges[:, 1], edges[:, 0])):
            np.add.at(around, here, detail[there])
            np.add.at(count, here, 1)
        detail = 0.5 * detail + 0.5 * around / np.maximum(count, 1)
    print("MOLDE marcos", np.round(marks, 3), "axila", np.round(pits, 3), "relevo", round(float(np.abs(detail).max()) * 1000, 1), "mm")
    shift = np.zeros((len(merged), 3))
    shift[:, 1] = np.where(back, detail, -detail)
    # The base mesh models a man's nipples as cones 6 to 9 mm tall: bare-chested they stood out as points, and under a T-shirt they marked the cloth.
    height = (merged[:, 2] - marks[2]) / span
    chest = (~back) & (np.abs(side) > NIPPLE_SIDES[0]) & (np.abs(side) < NIPPLE_SIDES[1]) & (height > NIPPLE_LEVELS[0]) & (height < NIPPLE_LEVELS[1]) & (bones > 0.3)
    shift[:, 1] = without_nipples(merged, merged[:, 1] + shift[:, 1], edges, chest, span, marks[0], patch) - merged[:, 1]
    return shift[twin]
