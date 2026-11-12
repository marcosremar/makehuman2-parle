"""
    License information: data/licenses/makehuman_license.txt

    The margin of the lids of the Parle cast (`lids_with_a_margin`), run inside Blender on a finished character. The lids of a kit
    end in a rim as thin as paper: "lidless symmetric cut-outs with thin sharp rims" (the judge by sight, round 16, 2026-11-12).
    Raising that rim into a lip was tried and left (`parle_polish`, LID_LIP_M): the skin of a kit has one row of points for the rim,
    a few to an eye, up to 12 mm from one to the next and no two kits alike, and the edge tore or stood in teeth on three faces of
    five. So the margin is drawn with rows of its own: a strip of new faces round each opening, as many points to an eye on every
    kit, added to the skin (same object, same material) and carried by every shape key.

    NOT READY, and not called from `parle_polish.natural_body` (2026-11-12). Applied to five finished characters (Lúcia, Dona
    Marta, Dona Rosa, Juliana, Carlos), each opening gets a whole margin with a thickness, no tear, flap or tooth, and with the
    lids shut the photographs are the ones from before, point for point. But the strip lands on the kit's own skin behind the edge,
    and that skin is ragged there: pale flakes of it, 0.3 to 3 mm long, still show under the lower lid of Lúcia (both eyes), of
    Dona Rosa and of Juliana, one under Dona Marta's left eye, and a bright notch stays at the peak of the fold over Juliana's upper
    lids. Only Carlos is clean. Nothing was seen in the game engine, the step was never run inside `natural_body` (only on finished
    GLB files), and it has no test yet. To make it the last step of `natural_body`, after `blinks_shut`:
        import parle_lid_ring
        added, tall = parle_lid_ring.lids_with_a_margin(meshes)
"""
import bmesh
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.interpolate import poly_3d_calc

import parle_polish

# The strip has this many points round each opening and four rows of them: 384 points on a body of some 12 500.
RING_SAMPLES = 48
# The edge of the opening is read in this many directions round its middle, seen from the middle of the ball. A piece of rim with
# fewer points than this, next to a bigger one, is not of the opening: each face looked at had one or two points of the lining
# of the socket 4 mm under the lower lid that count as rim (2026-11-12).
RING_RAYS = 720
RING_PIECE_LEAST = 4
# Lengths are shares of the radius of the globe (24.9 to 27.5 mm on the women, 19.7 mm on Carlos).
# Row 0 lies on the skin of the lid behind the edge, no further from the line of the crest than RING_FOOT: the skin is looked at
# in RING_FOOT_STEPS places on the way there, and the foot is where the strip, drawn tight from the crest, comes to rest, never
# climbing steeper than RING_TILT. With a foot at a set distance, sunk under the skin, the kit's skin came through the strip in
# flakes wherever it stood higher on the way (2026-11-12).
# Rows 1 and 2 are the crest. It stands level with the highest skin behind it, RING_OVER_SKIN over that; never less than
# RING_STAND off the ball (the lids of the references stand 5 to 6 mm and 3 to 4 mm off theirs, `parle_polish`; these are what
# stayed whole here), and never more than RING_RISE over the edge of the opening: by the skin behind alone, the crest stood
# 5.6 mm off the ball next to the nose, where the skin climbs from the edge like a wall. Row 2 is RING_IN further into the
# opening than row 1. Where no crest stands (a lid that is shut), all four rows lie on one line RING_SINK under the edge of the
# skin, and nothing of the strip is drawn.
# Row 3 is the foot of the wall, inside the ball at RING_DEEP of its radius.
RING_FOOT = 0.07
RING_FOOT_STEPS = 8
RING_TILT = 0.35
RING_SINK = 0.01
RING_STAND = {"upper": 0.11, "lower": 0.07}
RING_OVER_SKIN = 0.01
RING_RISE = 0.05
RING_IN = 0.02
RING_DEEP = 0.96
# Towards the outer corner of the opening, over this share of the length of each lid, the crest comes down to nothing, as the
# fold of `parle_polish.lids_with_a_fold` does. Towards the inner corner, over its own share, it comes down to RING_INNER of
# itself and goes on round the corner: the lids of a kit stand off the ball right up to the pocket of the inner corner, a
# band of skin from its rim down to the globe, and a strip that ended before it ended in a cut face 2 mm wide (2026-11-12).
# The line the crest follows is the edge of the opening evened out this many times, each point taking half of the mean of its
# two neighbours; where less than RING_WHOLE of the crest stands the line goes over to the edge itself, and the wall shortens
# to nothing with it.
RING_CORNER = {"outer": 0.4, "inner": 0.25}
RING_INNER = 0.3
RING_EVEN_ROUNDS = 6
RING_WHOLE = 0.3
# The whole strip of an eye shrinks where a shape brings its two lids nearer each other than the second of these shares of the
# way between them at rest, and is gone at the first: a lid that shuts shuts as `parle_polish.blinks_shut` left it, with no
# crest crossing the other lid's and no wall standing in the slit. The way is read where the lids are at least RING_WIDEST of
# their widest apart at rest. Sample by sample instead, the strip stayed whole round the inner corner, where a blink hardly
# moves the lids, and ended over the shut lid in a wedge; and gone only where the lids touch, 5% of it was left with Dona
# Marta's lids shut, a millimetre apart, and its wall showed in the slit as a pale thread (2026-11-12, both).
RING_SHUT = (0.15, 0.5)
RING_WIDEST = 0.8
# The skin is looked for from this far outside the ball: from further, the line to the inner corner of an eye starts inside the
# nose and meets its far side first (2026-11-12, the feet of row 0 landed 8 mm away on the bridge of the nose). The edge of the
# opening is looked for in the skin no further from the rim than this share of the radius.
RING_PROBE_M = 0.006
RING_NEAR = 0.15
# What covers the globe is looked for in the skin no further from the middle of the ball than this many radii.
RING_BALL_NEAR = 1.3


def even(points, rounds):
    """A closed line evened out: each point takes half of itself and a quarter of each of its two neighbours, `rounds` times."""
    for _ in range(rounds):
        points = points / 2 + (np.roll(points, 1, axis=0) + np.roll(points, -1, axis=0)) / 4
    return points


def spread(values, rounds=2):
    """Along a closed line, each value becomes the greatest of itself and its two neighbours and is then evened out: no value ends
    under what it was."""
    values = np.maximum(values, np.maximum(np.roll(values, 1), np.roll(values, -1)))
    return even(values, rounds)


def to_line(points, line):
    """How far each point is from an open line drawn from one of its points to the next."""
    start, along = line[:-1], line[1:] - line[:-1]
    long = np.maximum((along * along).sum(axis=1), 1e-18)
    share = np.clip(((points[:, None] - start[None]) * along[None]).sum(axis=2) / long[None], 0.0, 1.0)
    return np.sqrt((((start[None] + along[None] * share[:, :, None]) - points[:, None]) ** 2).sum(axis=2).min(axis=1))


def apart(spots, upper, centre, radius):
    """How far each point of the line of the edge is from the other lid's, round the ball: both are looked at from its middle,
    so a lid that lies on the globe and one that stands off it are together when one hides the other. `upper` counts the points
    of the upper lid, which come first, from the outer corner on; the lower lid's start at the inner corner. A corner is of both."""
    spots = spots - centre
    spots = spots / np.linalg.norm(spots, axis=1)[:, None] * radius
    top, bottom = spots[:upper + 1], np.r_[spots[upper:], spots[:1]]
    return np.r_[to_line(spots[:upper], bottom), to_line(spots[upper:], top)]


def opening_line(world, way, rim, edges, group, centre, radius):
    """The edge of one opening as a closed line, in order: for each of RING_RAYS directions round the middle of the opening, the
    place where the skin that shows is met first, seen from the middle of the ball. What comes back, for each place: the two
    points of the skin it lies between, the share of the way from the first to the second, where it is, and where on the ball
    the globe stops being seen in that direction. Nothing when there is no rim. `way` and `rim` are those of
    `parle_polish.lid_ways`.
    The rim of a kit is not a ring of edges: on the five faces looked at (2026-11-12) it had a point tied to three others, a
    triangle of three points of the rim, a spur into the opening, and a point or two of the lining of the socket apart from the
    rest. Nor is it the edge that is seen: over the middle of Lúcia's eye the line from one point of the rim to the next runs
    12 mm under the lid, and the lip of the lid stands 1.5 mm in front of it. So the line is not walked from point to point: it
    is the innermost of the edges of the skin that shows near the rim, and of the lines from each point of the rim to the next
    one round the middle, which close it wherever no edge does."""
    lead = {}
    for index in np.flatnonzero(rim):
        lead.setdefault(int(group[index]), int(index))
    both = rim[edges[:, 0]] & rim[edges[:, 1]] & (group[edges[:, 0]] != group[edges[:, 1]])
    ties = {tuple(sorted((lead[int(group[first])], lead[int(group[second])]))) for first, second in edges[both]}
    piece = {index: index for index in lead.values()}

    def root(index):
        while piece[index] != index:
            index = piece[index]
        return index

    for first, second in ties:
        piece[root(first)] = root(second)
    sizes = {}
    for index in piece:
        sizes[root(index)] = sizes.get(root(index), 0) + 1
    least = min(RING_PIECE_LEAST, max(sizes.values(), default=0))
    points = np.array(sorted(index for index in piece if sizes[root(index)] >= least), dtype=int)
    if len(points) < 8:
        return None
    shown = (way[edges] <= RING_NEAR * radius).all(axis=1)
    shown &= ~(rim[edges].all(axis=1) & ~np.isin(group[edges], group[points]).all(axis=1))
    close = (np.linalg.norm(world[edges] - centre, axis=2) < RING_BALL_NEAR * radius).all(axis=1)
    away = world[points] - centre
    middle = (away / np.linalg.norm(away, axis=1)[:, None]).mean(axis=0)
    for last in (False, True):
        middle /= np.linalg.norm(middle)
        across = np.array((1.0, 0.0, 0.0)) - middle[0] * middle
        across /= np.linalg.norm(across)
        up = np.cross(middle, across)
        order = points[np.argsort(np.arctan2(away @ up, away @ across))]
        ends = np.r_[edges[shown], np.c_[order, np.roll(order, -1)]]
        found, covered = [], []
        for turn in np.arange(RING_RAYS) * 2 * np.pi / RING_RAYS:
            ray = np.cos(turn) * across + np.sin(turn) * up

            def met(ends):
                """Where the lines between those points cross the half plane of this direction: the share of the way along
                each, the place from the middle of the ball, and how far round from the middle of the opening (np.inf: not)."""
                start, along = world[ends[:, 0]] - centre, world[ends[:, 1]] - world[ends[:, 0]]
                side = np.cross(middle, ray)
                first, second = start @ side, (start + along) @ side
                share = first / np.where(first == second, 1.0, first - second)
                at = start + along * share[:, None]
                wide = np.arctan2(at @ ray, at @ middle)
                wide[(first * second > 0) | (first == second) | (at @ ray <= 0) | (at @ middle <= 0)] = np.inf
                return share, at, wide

            share, at, wide = met(ends)
            if not np.isfinite(wide).any():
                continue
            found.append((ends[wide.argmin(), 0], ends[wide.argmin(), 1], share[wide.argmin()], *(at[wide.argmin()] + centre)))
            if last:
                # Any skin outside the ball, shown or not, covers the globe: the globe is seen as far as the innermost of it,
                # or as far as where that skin comes out of the ball, between the place before it and that one.
                _, at, wide = met(edges[close])
                off = np.linalg.norm(at, axis=1) - radius
                out = np.where(off > 0, wide, np.inf)
                if np.isfinite(out).any():
                    here = out.argmin()
                    under = np.where((wide < wide[here]) & (off > -parle_polish.LID_UNDER_BALL_M), wide, -np.inf)
                    reach = wide[here]
                    if np.isfinite(under).any():
                        before = under.argmax()
                        reach = wide[before] + (wide[here] - wide[before]) * (-off[before]) / (off[here] - off[before])
                    covered.append(centre + radius * (np.cos(reach) * middle + np.sin(reach) * ray))
                else:
                    covered.append(np.array(found[-1][3:]))
        found = np.array(found)
        if len(found) < RING_RAYS:
            return None
        # The middle of the opening is the middle of its line, each place weighing the length of line it stands for.
        places = found[:, 3:] - centre
        long = np.linalg.norm(np.roll(places, -1, axis=0) - np.roll(places, 1, axis=0), axis=1)
        if not last:
            middle = ((places / np.linalg.norm(places, axis=1)[:, None]) * long[:, None]).sum(axis=0)
    return found[:, 0].astype(int), found[:, 1].astype(int), found[:, 2], found[:, 3:], np.array(covered)


def even_samples(first, second, share, places, covered, centre):
    """RING_SAMPLES places of the line of the edge (`opening_line`), evenly apart along it. The corners of the opening are the
    two places of the line furthest apart, and both are samples; the outer corner (the one further from the middle of the face)
    is the first, the upper lid comes after it, then the inner corner and the lower lid. What comes back: for each sample the two
    points of the skin it lies between and the share of the way, how many samples the upper lid has, and for each place of the line
    its place among the samples (3.5 is halfway from sample 3 to sample 4), the places of the line and where the globe stops
    being seen (`covered`) in the order of the samples, and for each sample the share of the way along its own lid from the
    outer corner to the inner one."""
    count = len(places)
    gaps = np.linalg.norm(places[:, None] - places[None], axis=2)
    one, other = np.unravel_index(gaps.argmax(), gaps.shape)
    outer, inner = (one, other) if (places[one, 0] - places[other, 0]) * centre[0] > 0 else (other, one)
    turn = (np.arange(count) + outer) % count
    if places[turn[1:(inner - outer) % count], 2].mean() < places[turn[(inner - outer) % count + 1:], 2].mean():
        turn = np.r_[turn[:1], turn[:0:-1]]
    first, second, share, places, covered = first[turn], second[turn], share[turn], places[turn], covered[turn]
    inner = int(np.flatnonzero(turn == inner)[0])
    way = np.r_[0.0, np.cumsum(np.linalg.norm(np.roll(places, -1, axis=0) - places, axis=1))]
    upper = int(np.clip(round(RING_SAMPLES * way[inner] / way[-1]), 4, RING_SAMPLES - 4))
    wanted = np.r_[way[inner] * np.arange(upper) / upper, way[inner] + (way[-1] - way[inner]) * np.arange(RING_SAMPLES - upper) / (RING_SAMPLES - upper)]
    taken = []
    for far in wanted:
        before = min(int(np.searchsorted(way, far, side="right")) - 1, count - 1)
        after = (before + 1) % count
        part = (far - way[before]) / max(way[before + 1] - way[before], 1e-12)
        if (first[before], second[before]) == (first[after], second[after]):
            taken.append((first[before], second[before], share[before] + part * (share[after] - share[before])))
        else:
            near = before if part < 0.5 else after
            taken.append((first[near], second[near], share[near]))
    taken = np.array(taken)
    among = np.interp(way[:-1], np.r_[wanted, way[-1]], np.arange(RING_SAMPLES + 1))
    across = np.r_[np.arange(upper) / upper, 1 - np.arange(RING_SAMPLES - upper) / (RING_SAMPLES - upper)]
    return taken[:, 0].astype(int), taken[:, 1].astype(int), taken[:, 2], upper, among, places, covered, across


def lid_line(spots, centre, radius, plan):
    """The line the crest of one opening follows in one shape of the body, `spots` being the samples of the edge of the opening
    in that shape: that edge evened out where the crest stands, the edge itself where it does not (the corners, and a lid that
    is shut). What comes back, for each sample: its place on the line, the way out from the middle of the ball, the way into the
    opening across the line, the share of the crest that stands, and the share of it that the way between the lids leaves."""
    # Each sample goes where the middle one of the five round it goes: on Lúcia two samples of the upper lid, next to the inner
    # corner, lay on skin the blink moves 1 to 3 mm, between neighbours it moves 8 and 11 mm, and with the lid shut the strip
    # stood up there in a horn (2026-11-12).
    moved = spots - plan["spots"]
    spots = plan["spots"] + np.median([np.roll(moved, turn, axis=0) for turn in (-2, -1, 0, 1, 2)], axis=0)
    # How far open the eye is in this shape, against rest, where the lids are widest apart: the whole strip stands or shrinks
    # by it.
    widest = plan["apart"] >= RING_WIDEST * plan["apart"].max()
    open_share = np.median(apart(spots, plan["upper"], centre, radius)[widest] / (plan["apart"][widest] * radius / plan["radius"]))
    apart_enough = np.full(len(spots), parle_polish.smooth((open_share - RING_SHUT[0]) / (RING_SHUT[1] - RING_SHUT[0])))
    stands = plan["share"] * apart_enough
    line = spots + parle_polish.smooth(stands / RING_WHOLE)[:, None] * (even(spots, RING_EVEN_ROUNDS) - spots)
    out = line - centre
    out /= np.linalg.norm(out, axis=1)[:, None]
    along = np.roll(line, -1, axis=0) - np.roll(line, 1, axis=0)
    along -= (along * out).sum(axis=1)[:, None] * out
    along /= np.maximum(np.linalg.norm(along, axis=1), 1e-12)[:, None]
    return line, out, plan["turn"] * np.cross(out, along), stands, apart_enough


def ring_rows(spots, feet, centre, radius, plan):
    """The four rows of the strip of one opening in one shape of the body: `spots` are the samples of the edge of the opening in
    that shape, `feet` the points of skin row 0 is bound to, and the ball is the globe's in that shape. `plan` is what was
    settled at rest (`ring_plan`): for each sample how much of the crest stands (`share`), how far over the edge (`rise`) and how
    far into the opening (`inward`), in metres on a globe of `plan["radius"]`.
    The crest stands over its line (`lid_line`) along the way out from the middle of the ball; row 2 is the same height further
    into the opening, and row 3 under it inside the ball: the edge of the lid is a wall that runs into the globe. Where no crest
    stands, the four rows lie on one line just under the edge of the skin: the faces between them have no size, and nothing of
    the strip is drawn."""
    scale = radius / plan["radius"]
    line, out, inward, stands, apart_enough = lid_line(spots, centre, radius, plan)
    rise = plan["rise"] * apart_enough * scale
    top = np.linalg.norm(line - centre, axis=1) + rise - np.maximum(RING_SINK * radius - rise, 0.0)
    lean = out + inward * np.tan(plan["inward"] * apart_enough * scale / radius)[:, None]
    lean /= np.linalg.norm(lean, axis=1)[:, None]
    crest = centre + out * top[:, None]
    # Row 0 goes no further from the crest than it is at rest: where a lid slides under skin that stays (the upper lid next to
    # the inner corner, when it shuts), the foot would otherwise stay behind and draw the strip out in a flag (2026-11-12).
    reach = np.linalg.norm(feet - crest, axis=1)
    foot = crest + (feet - crest) * (apart_enough * np.minimum(1.0, plan["reach"] * scale / np.maximum(reach, 1e-9)))[:, None]
    under = out + inward * np.tan(plan["foot"] * apart_enough * scale / radius)[:, None]
    under /= np.linalg.norm(under, axis=1)[:, None]
    deep = top - parle_polish.smooth(stands / RING_WHOLE) * np.maximum(top - RING_DEEP * radius, 0.0)
    return np.stack([foot, crest, centre + lean * top[:, None], centre + under * deep[:, None]])


def ring_plan(world, way, rim, edges, group, centre, radius, skin, faces):
    """What the strip of one opening is at rest, or nothing when the opening has no rim to follow. `skin` is the tree of the
    faces of the skin that show (`faces`, each the numbers of its points) and `way` how far each point is from the rim
    (`parle_polish.lid_ways`). The samples of the edge and the feet of row 0 are kept as points of the skin and shares between
    them, so every shape key can be read through them."""
    found = opening_line(world, way, rim, edges, group, centre, radius)
    if found is None:
        return None
    first, second, share, upper, among, places, covered, across = even_samples(*found, centre)
    spots = world[first] + (world[second] - world[first]) * share[:, None]
    # How much of the crest stands, by the place of each sample along its lid, from the outer corner to the inner one.
    stands = np.minimum(parle_polish.smooth(across / RING_CORNER["outer"]), RING_INNER + (1 - RING_INNER) * parle_polish.smooth((1 - across) / RING_CORNER["inner"]))
    plan = dict(first=first, second=second, part=share, upper=upper, radius=radius, way=way, share=stands, spots=spots, apart=apart(spots, upper, centre, radius), turn=1.0)
    plan.update(rise=np.zeros(RING_SAMPLES), inward=np.zeros(RING_SAMPLES), foot=np.zeros(RING_SAMPLES), reach=np.zeros(RING_SAMPLES))
    # Into the opening is towards the middle of its line.
    line, out, inward, stands, _ = lid_line(spots, centre, radius, plan)
    if ((places.mean(axis=0) - line) * inward).sum() < 0:
        plan["turn"], inward = -1.0, -inward
    far = np.linalg.norm(line - centre, axis=1)

    off = lambda points: np.linalg.norm(np.asarray(points) - centre, axis=1) - radius

    def on_skin(aim, far):
        """The skin on the line from the middle of the ball through `aim`: where, and the face. Nearest to the place `far` from
        the middle on that line when the line meets no skin that shows, or meets it from behind."""
        aim = Vector(aim / np.linalg.norm(aim))
        at, normal, face, _ = skin.ray_cast(Vector(centre) + aim * (radius + RING_PROBE_M), -aim, RING_PROBE_M + parle_polish.LID_UNDER_BALL_M)
        if at is None or normal.dot(aim) <= 0:
            at, _, face, _ = skin.find_nearest(Vector(centre) + aim * far)
        return np.array(at), face

    # The skin of the lid behind the edge, in RING_FOOT_STEPS places as far as RING_FOOT from the line (nearer where less of
    # the crest stands): how far behind the line each is, and how far off the ball.
    behind = [[on_skin(out[sample] - inward[sample] * np.tan(RING_FOOT * plan["share"][sample] * step / RING_FOOT_STEPS), far[sample])
               for step in range(1, RING_FOOT_STEPS + 1)] for sample in range(RING_SAMPLES)]
    high = np.array([[np.linalg.norm(at - centre) - radius for at, _ in row] for row in behind])
    back = np.array([[radius * np.arccos(np.clip((at - centre) @ out[sample] / np.linalg.norm(at - centre), -1.0, 1.0)) for at, _ in row] for sample, row in enumerate(behind)])
    # The crest stands level with the lid behind it, and RING_OVER_SKIN over it: the lid goes on at its own height as far as
    # its edge, and drops from there. Never less than RING_STAND off the ball, and never more than RING_RISE over the edge of
    # the opening, by the highest place of that edge near each sample.
    before = np.floor(among).astype(int) % RING_SAMPLES
    edge = off(spots)
    np.maximum.at(edge, before, off(places))
    np.maximum.at(edge, (before + 1) % RING_SAMPLES, off(places))
    edge = spread(edge)
    least = even(np.where(np.arange(RING_SAMPLES) < upper, RING_STAND["upper"], RING_STAND["lower"]) * radius, 2)
    stand = np.minimum(np.maximum(least, spread(np.maximum(high.max(axis=1), edge)) + RING_OVER_SKIN * radius), edge + RING_RISE * radius)
    plan["rise"] = plan["share"] * np.maximum(stand - (far - radius), 0.0)
    crest = far - radius + plan["rise"] - np.maximum(RING_SINK * radius - plan["rise"], 0.0)
    # Row 0 lies on the skin of the lid where the strip, drawn tight from the crest over the skin behind it, comes to rest:
    # of the places looked at, the one the crest sees highest. The strip never climbs steeper than RING_TILT: past that the
    # nearest place that high is the foot.
    steepness = (high - crest[:, None]) / np.maximum(back, 1e-6)
    feet = [row[int(np.argmax(steep >= RING_TILT)) if (steep >= RING_TILT).any() else int(steep.argmax())] for row, steep in zip(behind, steepness)]
    plan["feet"] = [(faces[face], np.array(poly_3d_calc([Vector(world[index]) for index in faces[face]], Vector(at)))) for at, face in feet]
    plan["reach"] = np.linalg.norm(np.array([at for at, _ in feet]) - (centre + out * (radius + crest)[:, None]), axis=1)
    plan["foot_faces"] = [face for _, face in feet]
    # The wall stands inside the edge of the skin: where a place of that edge is further into the opening than the line of the
    # crest, row 2 goes in by that much more. A place inside the ball is behind the globe and does not count. And the foot of
    # the wall stands inside whatever skin covers the globe (the inside of a lid, the pocket of the inner corner): row 3 goes
    # in as far as the globe stops being seen, and RING_IN further.
    part = among - np.floor(among)
    between = line[before] * (1 - part)[:, None] + line[(before + 1) % RING_SAMPLES] * part[:, None]
    turned = inward[before] * (1 - part)[:, None] + inward[(before + 1) % RING_SAMPLES] * part[:, None]

    def furthest(reach):
        """For each sample, the furthest into the opening of the places of the line next to it, evened along the line."""
        most = np.zeros(RING_SAMPLES)
        np.maximum.at(most, before, reach)
        np.maximum.at(most, (before + 1) % RING_SAMPLES, reach)
        return spread(most)

    whole = parle_polish.smooth(plan["share"] / RING_WHOLE)
    plan["inward"] = plan["share"] * RING_IN * radius + whole * furthest(np.where(off(places) > 0, ((places - between) * turned).sum(axis=1), 0.0))
    plan["foot"] = np.maximum(plan["inward"], whole * (furthest(((covered - between) * turned).sum(axis=1)) + RING_IN * radius))
    return plan


def ball_shapes(eye, balls):
    """The ball of each globe in the shape keys of the globes that move it, by the name of the key: the middle goes with the mean
    of the points of its globe and the radius with how far they spread from it. The globes of the cast have a shape for each
    trait of the face that carries the eyes (their spacing, the width of the head)."""
    blocks = eye.data.shape_keys.key_blocks if eye.data.shape_keys else ()
    rest = parle_polish.rest_points(eye)
    halves = (rest[:, 0] < rest[:, 0].mean(), rest[:, 0] >= rest[:, 0].mean())
    shapes = {}
    for block in blocks[1:]:
        points = parle_polish.to_world(eye, parle_polish.read(block))
        if np.abs(points - rest).max() < 1e-6:
            continue
        moved = []
        for half, (centre, radius) in zip(halves, balls):
            scale = np.sqrt(((points[half] - points[half].mean(axis=0)) ** 2).sum() / ((rest[half] - rest[half].mean(axis=0)) ** 2).sum())
            moved.append((points[half].mean(axis=0) + scale * (np.asarray(centre) - rest[half].mean(axis=0)), radius * scale))
        shapes[block.name] = moved
    return shapes


def lids_with_a_margin(meshes):
    """Each opening of the lids gets a margin with a thickness: how many points were added to the skin, and how far off its ball
    the crest stands at the most, in metres. Nothing moves: the points of the skin, the globes and every shape stay as they are,
    and the strip is read from them in every shape key."""
    body = next((obj for obj in meshes if obj.name == "Body"), None)
    eye = next((obj for obj in meshes if "low-poly" in obj.name), None)
    if body is None or eye is None or not len(body.data.polygons) or not len(eye.data.polygons):
        return 0, 0.0
    mesh = body.data
    blocks = mesh.shape_keys.key_blocks if mesh.shape_keys else None
    local = parle_polish.read(blocks[0]) if blocks else parle_polish.read(mesh.vertices)
    world = parle_polish.to_world(body, local)
    turn = np.array(body.matrix_world)[:3, :3]
    normals = np.array([vertex.normal[:] for vertex in mesh.vertices]) @ turn.T
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    _, group = np.unique(np.round(world / parle_polish.FAIR_SAME_M).astype(np.int64), axis=0, return_inverse=True)
    group = group.ravel()
    edges = np.zeros(len(mesh.edges) * 2, dtype=np.int64)
    mesh.edges.foreach_get("vertices", edges)
    edges = edges.reshape(-1, 2)
    balls = parle_polish.eye_balls(eye)
    moved = ball_shapes(eye, balls)
    corner_normals = np.zeros(len(mesh.loops) * 3)
    mesh.corner_normals.foreach_get("vector", corner_normals)
    corner_normals = corner_normals.reshape(-1, 3)
    uv = mesh.uv_layers.active.data if mesh.uv_layers.active else None
    rings = []
    for number, (centre, radius) in enumerate(balls):
        centre = np.asarray(centre, float)
        way, rim = parle_polish.lid_ways(world, normals, group, edges, centre, radius)
        if rim.sum() < 8:
            continue
        shown = [polygon.index for polygon in mesh.polygons if np.isfinite(way[list(polygon.vertices)]).all()]
        faces = [tuple(mesh.polygons[index].vertices) for index in shown]
        plan = ring_plan(world, way, rim, edges, group, centre, radius, BVHTree.FromPolygons(world.tolist(), faces), faces)
        if plan is None:
            continue
        plan["polygons"] = [shown[face] for face in plan["foot_faces"]]
        plan["ball"] = number
        rings.append((plan, centre, radius))
    if not rings:
        return 0, 0.0

    def rows_in(points, plan, centre, radius):
        spots = points[plan["first"]] + (points[plan["second"]] - points[plan["first"]]) * plan["part"][:, None]
        feet = np.array([shares @ points[list(corners)] for corners, shares in plan["feet"]])
        return ring_rows(spots, feet, centre, radius, plan)

    back = lambda points: parle_polish.to_local(body, points.reshape(-1, 3))
    rest = [rows_in(world, plan, centre, radius) for plan, centre, radius in rings]
    shapes = {}
    for block in blocks or ():
        points = parle_polish.to_world(body, parle_polish.read(block))
        shapes[block.name] = [back(rows_in(points, plan, *moved.get(block.name, balls)[plan["ball"]])) for plan, _, _ in rings]
    tallest = max(float((np.linalg.norm(rows[1] - centre, axis=1) - radius).max()) for rows, (_, centre, radius) in zip(rest, rings))

    had_normals = mesh.has_custom_normals
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    layers = {name: bm.verts.layers.shape[name] for name in bm.verts.layers.shape.keys()}
    weights = bm.verts.layers.deform.active
    picture = bm.loops.layers.uv.active
    skin_points, count, loops = bm.verts[:], len(bm.verts), len(mesh.loops)
    foot_normals = {}
    for ring, ((plan, centre, radius), rows) in enumerate(zip(rings, rest)):
        made = []
        for row, points in enumerate(back(rows).reshape(4, RING_SAMPLES, 3)):
            for sample, point in enumerate(points):
                vertex = bm.verts.new(point)
                for name, layer in layers.items():
                    vertex[layer] = shapes[name][ring][row * RING_SAMPLES + sample]
                first, second, part = skin_points[plan["first"][sample]], skin_points[plan["second"][sample]], plan["part"][sample]
                if weights is not None:
                    for bone in set(first[weights].keys()) | set(second[weights].keys()):
                        vertex[weights][bone] = first[weights].get(bone, 0.0) * (1 - part) + second[weights].get(bone, 0.0) * part
                made.append(vertex)
        # The picture: row 0 has the skin's own place at its foot, and the rows after it go from there to the edge's. The skin
        # of the cast carries a relief (a normal map), which is read along the picture: rows at one place of it would have none.
        places = np.zeros((4, RING_SAMPLES, 2))
        for sample in range(RING_SAMPLES):
            polygon = mesh.polygons[plan["polygons"][sample]]
            shares = plan["feet"][sample][1]
            if uv is not None:
                foot = sum(share * np.array(uv[loop].uv) for share, loop in zip(shares, polygon.loop_indices))
                edge = []
                for index in (plan["first"][sample], plan["second"][sample]):
                    loops_of = [loop for loop in skin_points[index].link_loops if np.isfinite(plan["way"][[v.index for v in loop.face.verts]]).all()] or list(skin_points[index].link_loops)
                    edge.append(np.array(loops_of[0][picture].uv))
                at_rim = edge[0] + (edge[1] - edge[0]) * plan["part"][sample]
                places[:, sample] = [foot + (at_rim - foot) * share for share in (0.0, 0.4, 0.7, 1.0)]
            normal = sum(share * corner_normals[loop] for share, loop in zip(shares, polygon.loop_indices))
            foot_normals[count + ring * 4 * RING_SAMPLES + sample] = normal / max(np.linalg.norm(normal), 1e-12)
        material = mesh.polygons[plan["polygons"][0]].material_index
        # The faces turn their outer side away from the ball on the lid and to the opening on the wall.
        middle = plan["upper"] // 2
        flip = np.cross(rows[0, middle + 1] - rows[0, middle], rows[1, middle] - rows[0, middle]) @ (rows[0, middle] - centre) < 0
        for row in range(3):
            for sample in range(RING_SAMPLES):
                after = (sample + 1) % RING_SAMPLES
                corners = [(row, sample), (row, after), (row + 1, after), (row + 1, sample)]
                for three in ((0, 1, 2), (0, 2, 3)):
                    picked = [corners[index] for index in (three[::-1] if flip else three)]
                    face = bm.faces.new([made[r * RING_SAMPLES + s] for r, s in picked])
                    face.smooth, face.material_index = True, material
                    if picture is not None:
                        for loop, (r, s) in zip(face.loops, picked):
                            loop[picture].uv = places[r, s]
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    # The normals: the skin keeps the ones it had, and row 0 takes the skin's at its foot, so that the light does not show where
    # the strip comes out of the lid.
    now = np.zeros(len(mesh.loops) * 3)
    mesh.corner_normals.foreach_get("vector", now)
    now = now.reshape(-1, 3)
    if had_normals:
        now[:loops] = corner_normals
    of_point = np.zeros(len(mesh.loops), dtype=np.int64)
    mesh.loops.foreach_get("vertex_index", of_point)
    for loop in range(loops, len(mesh.loops)):
        if int(of_point[loop]) in foot_normals:
            now[loop] = foot_normals[int(of_point[loop])]
    mesh.normals_split_custom_set(now.tolist())
    mesh.update()
    return len(rings) * 4 * RING_SAMPLES, tallest
