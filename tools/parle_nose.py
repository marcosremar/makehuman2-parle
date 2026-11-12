"""
    License information: data/licenses/makehuman_license.txt

    The nostrils of the Parle cast: the notch under each wing of the nose closed, and a shallow dent left in its place.

    The head of the kit is a real adult's, and under each wing of its nose the skin goes up into the nose. Cut down to the points
    of the game and evened out with the rest of the face (`parle_polish.fair_face` holds what has skin ahead of it, so the inside
    of the nose stays as it came), what is left is a notch of 28 to 39 points a side, 6 mm across and 8 or 9 long, whose roof
    stands 1.7 to 2.9 mm from a ball of 4.5 mm laid across it (five characters measured on 2026-11-12: Lúcia, Dona Marta, Dona
    Rosa, Juliana and Carlos, whose notch is a slit under 4 mm across). The light does not reach its walls, and with so few
    points its edge is a row of teeth: "torn, asymmetric dark notches with hanging spikes; the right one is a ripped hole",
    "a ragged dark blot from the front" (the judge by sight, rounds 13 to 18). The paint was not the cause: on the skin of the cast
    the texels of the notch are 1 to 3 points of lightness, of 100, under the skin round them (`skin_style` has already taken
    the dark nostril of the photograph away, 43 to 54 where the skin has 60 to 80), and in clay the notch is as dark as in colour.

    The nostril of a drawn face is "sketched with a curve and never filled with black": a shallow dent in a closed, smooth
    surface, small, the same on both sides. So, on the skin at rest, with every shape key taking the same step:

    - A ball is rolled over the skin of the nose base from outside. Where it does not reach the skin is the notch, and there the
      skin comes out to the ball: the hole is closed by a round fillet that leaves the wing of the nose and the lip where the
      ball rests on them. What the ball reaches does not move, so the tip of the nose and its outline in profile stay.
    - The points of the notch, which lay one over another on its walls, are spread evenly over the fillet.
    - The skin round the fillet is evened out with it, so the edge the notch had does not stay as a ridge.
    - A dent the shape of a tear is pressed into the fillet, its narrow end to the tip of the nose: 0.9 mm deep, and one the
      mirror image of the other.

    Run again on its own result, nothing moves: a nostril this shallow is not a notch. On the five the dent is 0.9 to 1.1 mm
    under the same ball, the two sides of a face end 0.000 mm apart, and the foremost place of the skin at each height, 3 cm up
    and down from the tip, is where it was.

    What it leaves (2026-11-12, seen from below): at the far end of Carlos's slit, 6 mm from the middle of the dent and past
    the notch, a ridge of 51 degrees beside a groove of 48, which the evening out reaches with little weight: a pale line 4 mm
    long. And on Dona Marta and Juliana the fold between the wing of the nose and the lip, which was there before, still runs
    from the dent to the lip as a faint crease.

    This file is numpy alone. `shallow_nostrils` takes the skin as a Blender object and reads it through its own attributes; the
    skin step (`parle_nostrils.py`, which has OpenCV and no Blender) finds the same dents on the person's points with
    `closed_and_dented`, to paint the shade where the dent will be.
"""
import numpy as np

# The middle of the face and the tip of the nose, as the skin and lip steps find them: the head is the top 26 cm, its middle the
# median of its points across, and the tip the foremost point (the face looks at -y) within 1 cm of the middle.
HEAD_REACH_M = 0.26
MIDDLE_BAND_M = 0.01
# The window of the nose base, measured from the tip: this far to each side of the middle, below, above and behind. It takes in
# the lip under the nose and the cheek beside it, which the ball rests on.
BASE_HALF_M = 0.03
BASE_BELOW_M = 0.03
BASE_ABOVE_M = 0.016
BASE_BEHIND_M = 0.04
# Points at the same place are one point: the glTF doubles them along the seams of the map.
SAME_M = 1e-5
# The ball. At 4.5 mm it does not go into a notch 6 mm across, and it rests on the wing of the nose on one side and on the slope
# down to the lip on the other. The one size tried, kept because the five came out closed with it: no source.
BALL_M = 0.0045
# The skin is read at places this far apart on its triangles, and a ball is free when no place of the skin is nearer its middle
# than its radius less the second of these (a flat triangle between two points with different normals is that far inside).
STEP_M = 0.0005
FREE_M = 0.0001
# A point is of a notch when the ball laid on it falls short of the skin elsewhere by the first of these, and of the groups of
# such points on one side of the face the seed of the nostril is the one with the most of it among those that are where a
# nostril is: this far from the middle of the face, this far behind the tip and this far above it, no wider than POCKET_REACH_M,
# and looking out of the face, ahead and down (the cosine to SEED_LOOKS). Measured on the five, finished and as the cast build
# has them before the face is evened out: the nostrils are 8 to 14 mm from the middle, 8 to 20 mm behind the tip and 2 to 6 mm
# under it, and look 0.45 to 0.71. The other groups in the window are the corners of the mouth and its lining, 13 to 30 mm under
# the tip, and the hollow between the wing of the nose and the cheek, 15 to 23 mm from the middle, which on two of the five is
# where a nostril may be but carries a tenth of what a nostril does and is 0.3 to 0.9 mm under the ball, too shallow for a
# notch. Only the one seed is tried: when it is not a notch, as on a nose this step has been over, the side is left alone.
# Trying the next groups took a piece of the lining of the mouth for a nostril and moved points 4 to 5 mm.
SEED_SHORT_M = 0.0005
SEED_ACROSS_M = (0.003, 0.016)
SEED_BEHIND_M = (0.002, 0.026)
SEED_HIGH_M = (-0.013, 0.003)
SEED_LOOKS = np.array((0.0, -0.6, -0.8))
SEED_LOOKS_LEAST = 0.3
# The notch is the seed and what hangs to it of the points more than INSIDE_M under the ball, no nearer the middle of the face
# than a nostril is: on the points of the cast build, before the face is evened out, a crease runs from each nostril along the
# foot of the columella to the middle, 0.3 to 0.8 mm under the ball, and the two notches were one (Dona Marta and Juliana,
# 2026-11-12). A point less than INSIDE_M under the ball goes out by a share of it, so that a point just short of the mark on
# one side of the face and just over it on the other end at the same place: taken or left whole, one point of the rim of Lúcia
# ended 0.49 mm from its twin. A notch is between POCKET_LEAST_M and POCKET_MOST_M deep: the dent this step leaves is under
# the first (DENT_DEEP_M has to stay under it), so it is not taken for a notch again.
INSIDE_M = 0.00025
POCKET_LEAST_M = 0.0012
POCKET_MOST_M = 0.006
POCKET_REACH_M = 0.009
# Laid on the ball each at the place nearest it, the points of the two walls of the notch crossed: on Lúcia, Dona Rosa and
# Carlos neighbouring triangles lay folded one on the other (bends of 179 degrees), and the thinnest was 0.004 of its longest
# side tall. Each point goes to the middle of its neighbours and back onto the ball, this many rounds: the sharpest bend left
# round a notch is 47 to 66 degrees (67 to 122 before the step) and the thinnest triangle 0.10 of its longest side.
EVEN_ROUNDS = 40
# The rim: the skin up to the second of these from the closed notch is evened out, whole at the first and less with the
# distance, by rounds of a step forward and a step back as `parle_polish.fair_face` does (Taubin 1995, DOI 10.1145/218380.218473).
# What is within the first of PROFILE_HALF_M of the middle of the face stays, and takes part in full from the second on: the
# outline of the nose in profile is drawn by those points. Chosen by eye, in clay, on Dona Marta, Juliana and Carlos: 12, 40 and
# 80 rounds, over 4 mm and over 6, looked much the same from the front, from three-quarter and from below, and 30 steps forward
# with none back flattened the dent itself. By the numbers 80 rounds over 2 to 6 mm take the ridge and the groove at the end of
# Carlos's slit from 51 and 48 degrees to 25 and 24; that was not looked at in a render.
RIM_REACH_M = (0.0, 0.004)
RIM_STEP, RIM_BACK, RIM_ROUNDS = 0.5, -0.52, 30
PROFILE_HALF_M = (0.0025, 0.005)
# The dent: how deep at its middle, its half length along the nostril and its half width across, which grows by DENT_TAPER
# from the middle to the end away from the tip of the nose and shrinks as much to the other end. Its floor falls as the square
# of what is left to the edge, flat at the edge and at the middle. Only skin within DENT_NEAR_M of the level of its middle is
# dented, not what stands behind it inside the nose (on the five every point it moves is within 2 mm of that level). Chosen by
# eye in clay on Lúcia and Carlos: 1 mm deep over a floor 2 mm wide drew a small hook at the end of the dent, 1.2 mm over a
# floor 3 mm wide a marked comma, and 0.7 to 0.9 mm over this floor a soft dent on both. No source: a design choice to pilot.
DENT_DEEP_M = 0.0009
DENT_LONG_M = 0.0042
DENT_WIDE_M = 0.0027
DENT_TAPER = 0.3
DENT_NEAR_M = 0.003


def smooth(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def nose_tip(points):
    """The middle of the face, across, and the tip of the nose."""
    head = points[points[:, 2] > points[:, 2].max() - HEAD_REACH_M]
    middle = float(np.median(head[:, 0]))
    band = head[np.abs(head[:, 0] - middle) < MIDDLE_BAND_M]
    return middle, band[band[:, 1].argmin()]


def outer_sides(points, triangles):
    """The outer side of the skin at each point: the mean of the normals of its triangles, each by its area."""
    face = np.cross(points[triangles[:, 1]] - points[triangles[:, 0]], points[triangles[:, 2]] - points[triangles[:, 0]])
    sides = np.zeros_like(points)
    for corner in range(3):
        np.add.at(sides, triangles[:, corner], face)
    return sides / np.maximum(np.linalg.norm(sides, axis=1, keepdims=True), 1e-12)


def spread(points, sides, triangles, step):
    """Places on the triangles no further apart than `step`, each with the outer side of the skin there."""
    places, outer = [], []
    longest = np.max([np.linalg.norm(points[triangles[:, a]] - points[triangles[:, b]], axis=1) for a, b in ((0, 1), (1, 2), (2, 0))], axis=0)
    cuts = np.maximum(np.ceil(longest / step).astype(int), 1)
    for count in np.unique(cuts):
        faces = triangles[cuts == count]
        i, j = np.meshgrid(np.arange(count + 1), np.arange(count + 1), indexing="ij")
        keep = i + j <= count
        share = np.stack([i[keep], j[keep], count - i[keep] - j[keep]], axis=1) / count
        places.append(np.einsum("mk,fkd->fmd", share, points[faces]).reshape(-1, 3))
        outer.append(np.einsum("mk,fkd->fmd", share, sides[faces]).reshape(-1, 3))
    places, outer = np.concatenate(places), np.concatenate(outer)
    _, first = np.unique(np.round(places / SAME_M).astype(np.int64), axis=0, return_index=True)
    return places[first], outer[first] / np.maximum(np.linalg.norm(outer[first], axis=1, keepdims=True), 1e-12)


def nearest(these, those, chunk=512):
    """For each of `these`, how far the nearest of `those` is, and which one it is."""
    a, b = these.astype(np.float32), those.astype(np.float32)
    bb = (b * b).sum(axis=1)
    far, which = np.empty(len(a)), np.empty(len(a), int)
    for start in range(0, len(a), chunk):
        part = a[start:start + chunk]
        square = (part * part).sum(axis=1)[:, None] + bb[None, :] - 2 * part @ b.T
        which[start:start + chunk] = square.argmin(axis=1)
        far[start:start + chunk] = np.sqrt(np.maximum(square.min(axis=1), 0))
    return far, which


def middle_of_neighbours(values, edges):
    """The mean of each point's neighbours along the edges; a point with none keeps its own."""
    total = np.zeros_like(values)
    np.add.at(total, edges[:, 0], values[edges[:, 1]])
    np.add.at(total, edges[:, 1], values[edges[:, 0]])
    count = np.bincount(edges.ravel(), minlength=len(values))
    return np.where(count[:, None] > 0, total / np.maximum(count, 1)[:, None], values)


def hanging_together(member, allowed, edges):
    """`member` grown along the edges through the points that are `allowed`."""
    member = member & allowed
    while True:
        touch = allowed[edges[:, 0]] & allowed[edges[:, 1]] & (member[edges[:, 0]] != member[edges[:, 1]])
        if not touch.any():
            return member
        member[edges[touch].ravel()] = True


def parts(member, edges):
    """The groups of `member` that hang together along the edges."""
    left, found = member.copy(), []
    while left.any():
        one = np.zeros(len(member), bool)
        one[np.flatnonzero(left)[0]] = True
        one = hanging_together(one, left, edges)
        found.append(np.flatnonzero(one))
        left &= ~one
    return found


def on_the_ball(places, middles, ball):
    """Each place moved onto the ball whose middle is nearest it, and the way from that middle to it."""
    far, which = nearest(places, middles)
    ray = (places - middles[which]) / np.maximum(far, 1e-12)[:, None]
    return middles[which] + ball * ray, ray


def under_the_nose(places, sides, across):
    """Whether a group of points the ball does not reach is where a nostril is: beside the middle of the face, a little behind
    and under the tip of the nose (the places are measured from the tip), small, and looking out of the face, ahead and down."""
    centre = places.mean(axis=0)
    return (SEED_ACROSS_M[0] < across.mean() < SEED_ACROSS_M[1] and SEED_BEHIND_M[0] < centre[1] < SEED_BEHIND_M[1] and SEED_HIGH_M[0] < centre[2] < SEED_HIGH_M[1]
            and np.linalg.norm(places - centre, axis=1).max() < POCKET_REACH_M and sides.mean(axis=0) @ SEED_LOOKS > SEED_LOOKS_LEAST)


def closed_notch(place, edges, skin, skin_side, seed, beside):
    """One nostril closed. `skin` are the places the skin is read at, each with its outer side; `beside` are the points far enough
    from the middle of the face to be of a nostril. Returns the points of the notch, where the points go (all of them: the rest
    stay), how deep the deepest was and the way into the nose at each point of the notch; None when `seed` is not a notch."""
    centre = place[seed].mean(axis=0)
    reach = np.linalg.norm(place[seed] - centre, axis=1).max()
    middles = skin + BALL_M * skin_side
    middles = middles[np.linalg.norm(middles - centre, axis=1) < reach + 2 * BALL_M]
    wall = skin[np.linalg.norm(skin - centre, axis=1) < reach + 3 * BALL_M]
    free = middles[nearest(middles, wall)[0] > BALL_M - FREE_M]
    near = np.flatnonzero(beside & (np.linalg.norm(place - centre, axis=1) < reach + BALL_M))
    if not len(free) or not len(near):
        return None
    deep = np.zeros(len(place))
    deep[near] = np.maximum(nearest(place[near], free)[0] - BALL_M, 0)
    start = np.zeros(len(place), bool)
    start[seed] = True
    notch = np.flatnonzero(hanging_together(start, deep > INSIDE_M, edges))
    if not len(notch) or not POCKET_LEAST_M < deep[notch].max() < POCKET_MOST_M:
        return None
    now = place.copy()
    now[near] += smooth(deep[near] / INSIDE_M)[:, None] * (on_the_ball(place[near], free, BALL_M)[0] - place[near])
    for _ in range(EVEN_ROUNDS):
        now[notch] = on_the_ball(middle_of_neighbours(now, edges)[notch], free, BALL_M)[0]
    return notch, now, float(deep[notch].max()), on_the_ball(now[notch], free, BALL_M)[1]


def dent_place(places, into):
    """Where the dent of one nostril goes, from the points of its closed notch (measured from the tip of the nose) and the way
    into the nose at each: its middle, the way into the nose there, and the way along the nostril, away from the tip."""
    centre = places.mean(axis=0)
    inward = into.mean(axis=0)
    inward /= np.linalg.norm(inward)
    flat = (places - centre) - np.outer((places - centre) @ inward, inward)
    along = np.linalg.eigh(flat.T @ flat)[1][:, 2]
    return {"centre": centre, "into": inward, "along": along * (1 if along @ centre > 0 else -1)}


def mirrored(vector, plane_x=None):
    """A way, or with `plane_x` a place, seen in the mirror that stands across x."""
    out = np.array(vector, float)
    out[0] = -out[0] if plane_x is None else 2 * plane_x - out[0]
    return out


def same_on_both_sides(dents):
    """The two dents made each the mirror image of the other: the mean of each one and of the other seen in the mirror that
    stands half way between their middles. A dent alone stays as it is."""
    if len(dents) < 2:
        return dents
    plane = (dents[1]["centre"][0] + dents[-1]["centre"][0]) / 2
    unit = lambda vector: vector / np.linalg.norm(vector)
    return {sign: {"centre": (dents[sign]["centre"] + mirrored(dents[-sign]["centre"], plane)) / 2,
                   "into": unit(dents[sign]["into"] + mirrored(dents[-sign]["into"])),
                   "along": unit(dents[sign]["along"] + mirrored(dents[-sign]["along"]))} for sign in (1, -1)}


def dent_share(places, dent, wider=1.0):
    """How deep the dent is at each place, as a share of its deepest, and how far each place stands into the nose from its
    middle. `wider` is for the paint, which goes a little past the dent: the same shape, that many times the size."""
    into = dent["into"]
    along = dent["along"] - (dent["along"] @ into) * into
    along /= np.linalg.norm(along)
    off = places - dent["centre"]
    u = off @ along / (DENT_LONG_M * wider)
    v = off @ np.cross(into, along) / (DENT_WIDE_M * wider * (1 + DENT_TAPER * np.clip(u, -1, 1)))
    return np.clip(1 - u * u - v * v, 0, 1) ** 2, off @ into


def closed_and_dented(points, triangles):
    """The points of a skin that looks at -y, z up, in metres, with the notch of each nostril closed and dented, and the dents:
    {side: dent}, side +1 or -1 by x, each dent with its middle (`centre`), the way into the nose there (`into`), the way along
    the nostril (`along`), how many points its notch had (`points`) and how deep it was (`was_deep`). A skin with no notch
    comes back as it is, with no dents."""
    middle, tip = nose_tip(points)
    off = points - tip
    box = (np.abs(points[:, 0] - middle) < BASE_HALF_M) & (off[:, 2] > -BASE_BELOW_M) & (off[:, 2] < BASE_ABOVE_M) & (off[:, 1] < BASE_BEHIND_M)
    whole = box[triangles].all(axis=1)
    used = np.unique(triangles[whole])
    if not len(used):
        return points, {}
    _, first, group = np.unique(np.round(off[used] / SAME_M).astype(np.int64), axis=0, return_index=True, return_inverse=True)
    group = group.ravel()
    place = off[used][first]
    number = np.full(len(points), -1)
    number[used] = group
    faces = number[triangles[whole]]
    faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 2] != faces[:, 0])]
    edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
    side = outer_sides(place, faces)
    skin, skin_side = spread(place, side, faces, STEP_M)
    short = BALL_M - nearest(place + BALL_M * side, skin)[0]
    across = place[:, 0] + tip[0] - middle
    now, found = place.copy(), {}
    for sign in (1, -1):
        seeds = [seed for seed in parts((np.sign(across) == sign) & (short > SEED_SHORT_M), edges)
                 if short[seed].max() > POCKET_LEAST_M and under_the_nose(place[seed], side[seed], sign * across[seed])]
        made = closed_notch(place, edges, skin, skin_side, max(seeds, key=lambda seed: short[seed].sum()), sign * across > SEED_ACROSS_M[0]) if seeds else None
        if made is not None:
            found[sign] = made
            now += made[1] - place
    if not found:
        return points, {}
    notches = np.concatenate([made[0] for made in found.values()])
    weight = 1 - smooth((nearest(now, now[notches])[0] - RIM_REACH_M[0]) / (RIM_REACH_M[1] - RIM_REACH_M[0]))
    weight *= smooth((np.abs(across) - PROFILE_HALF_M[0]) / (PROFILE_HALF_M[1] - PROFILE_HALF_M[0]))
    cut = number[triangles[box[triangles].any(axis=1) & ~whole]]
    weight[cut[cut >= 0]] = 0.0
    for _ in range(RIM_ROUNDS):
        for step in (RIM_STEP, RIM_BACK):
            now = now + step * weight[:, None] * (middle_of_neighbours(now, edges) - now)
    dents = same_on_both_sides({sign: dent_place(now[made[0]], made[3]) for sign, made in found.items()})
    for sign, dent in dents.items():
        share, over = dent_share(now, dent)
        mine = (np.sign(across) == sign) & (np.abs(over) < DENT_NEAR_M)
        now[mine] += DENT_DEEP_M * share[mine, None] * dent["into"]
        dent.update(centre=dent["centre"] + tip, points=len(found[sign][0]), was_deep=found[sign][2])
    out = points.copy()
    out[used] = now[group] + tip
    return out, dents


def read(block):
    """Points of a shape key, or of `mesh.vertices`."""
    items = block.data if hasattr(block, "relative_key") else block
    points = np.empty(len(items) * 3)
    items.foreach_get("co", points)
    return points.reshape(-1, 3)


def shallow_nostrils(body):
    """The nostrils of the skin of a mounted character closed and dented (`closed_and_dented`), at rest and so in every shape
    key: how many points moved, and how far the one that moved most went, in metres. Run after the face is evened out
    (`parle_polish.fair_face`), which holds the inside of the notch as it finds it."""
    mesh = body.data
    blocks = mesh.shape_keys.key_blocks if mesh.shape_keys else ()
    matrix = np.array(body.matrix_world)
    world = read(blocks[0] if blocks else mesh.vertices) @ matrix[:3, :3].T + matrix[:3, 3]
    mesh.calc_loop_triangles()
    triangles = np.zeros(len(mesh.loop_triangles) * 3, dtype=np.int64)
    mesh.loop_triangles.foreach_get("vertices", triangles)
    moved, dents = closed_and_dented(world, triangles.reshape(-1, 3))
    if not dents:
        return 0, 0.0
    step = (moved - world) @ np.array(body.matrix_world.inverted())[:3, :3].T
    for block in blocks:
        block.data.foreach_set("co", (read(block) + step).ravel())
    mesh.vertices.foreach_set("co", (read(mesh.vertices) + step).ravel())
    mesh.update()
    far = np.linalg.norm(moved - world, axis=1)
    return int((far > 1e-5).sum()), float(far.max())
