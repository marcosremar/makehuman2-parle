"""
    License information: data/licenses/makehuman_license.txt

    The folds of the face lit as one smooth skin (2026-11-12): the normals the light is worked out with, not the shape.

    The judge by sight, every round: under smooth, drawn features the skin is still that of "a tired realistic adult": the fold
    from the wing of the nose to the corner of the mouth, the hollow and the bag under each eye, the wings of the nose catching the
    light, "a hard crease under the jaw". The shape was already evened out (`parle_polish.fair_face`); what is left is how those
    planes catch the light, and a drawn face is lit as if they were not there: its normals are copied from a smoother surface, the
    points staying where they are, so the outline does not change.

    Here the smoother surface is the skin itself: in the zones of `fold_zones` each point takes the normals the mesh carries,
    averaged over the skin around it (`smoothed`), whole in the middle of a zone and less towards its edge, and nothing where an
    edge has to stay sharp: the rims of the lids, the lips, the base of the nose with the rims of the nostrils. Run
    `soft_normals` right after `parle_polish.seamless_normals`, on the rest shape, just before the export: it writes the same
    custom split normals, points at the same place take the same normal, so the seams of the picture stay joined, and what
    `parle_face.soften_under_chin` then does to the under side of the chin goes on top of it (the order the pictures of
    2026-11-12 were made in).

    The normals are those of the rest shape. A file exported with normals for its morph targets (the glTF exporter's "Shape Key
    Normals", `export_morph_normal`, on unless it is turned off) carries them along with every shape, and a shape that makes a
    fold lights it again: with the jaw open, or in a smile, the folds beside the mouth were back as deep as before the step
    (2026-11-12, in Blender, where custom normals follow the shape keys). Without them an engine keeps these normals while the
    points move, which is how drawn faces are shipped: the same shapes came out smooth. The cost is a lid that shuts lit as it
    was open: a pale patch with corners on each shut lid (the same day, the shut shape baked and given its rest normals).
"""
import heapq

import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

import parle_normals
import parle_polish

SAME_M = 1e-5
# A place of skin shows when it can be seen from one of these ways without other skin, or an eye globe, in between: the ways a
# face is looked at from, in degrees round it from straight ahead and up from level. Not from behind: the skin is hollow and cut
# open under the clothes, and looked at from all round the skin inside the nose counted as shown (2026-11-12, Lúcia). It has
# to turn its outer side to that way by this much (the cosine), and the look starts this far off the skin, along its normal.
SEEN_ROUND_DEG = (-72, -36, 0, 36, 72)
SEEN_UP_DEG = (-30, 0, 30)
SEEN_FACING = 0.1
SEEN_LIFT_M = 0.0002
SEEN_FAR_M = 0.5
# Every length on the face is a share of the way from the middle of one eye globe to the middle of the other (65 mm on Carlos,
# 66 to 73 mm on the four women looked at, 2026-11-12): the heads of the cast are not of one size (`parle_build.HEAD`).
# The skin around a point is the skin no further than this along the edges, 14 to 16 mm on those five. Chosen by eye: at 0.14
# the pouches under Dona Marta's eyes still showed, at 0.32 her cheeks were lit flat.
RADIUS = 0.22
# The face is looked at this far below the tip of the nose and this far behind it.
FACE_BELOW = 1.9
FACE_BEHIND = 2.4
# The base of the nose (the rims of the nostrils and the skin between them) is the skin this near the tip, across, below and
# behind it, that turns its outer side down by this much (the z of its normal). The nostrils of the kit are shallow: only a
# few points inside each do not show, and with the ways of looking turned by two degrees none on one side of Lúcia's nose
# (2026-11-12), so the base of the nose is not found as an edge of what shows.
NOSE_BASE_REACH = (0.45, 0.3, 0.6)
NOSE_BASE_DOWN = -0.35
# The lips meet in the edge between shown and hidden skin that lies this far below the tip of the nose and this near the middle.
MOUTH_BELOW = (0.2, 0.9)
MOUTH_HALF = 0.6
# The lips are the points their own traits move by this share or more of their widest move (as in `parle_polish.fair_face`).
LIPS = ("faceLipVolume", "faceCupidBow")
LIP_FROM = 0.25
# Nothing changes nearer an edge that stays than the first of these, and the whole change is only taken from the second on
# (1.4 and 4 mm on Lúcia). Chosen by eye: from 0.03 to 0.08 a dark band stayed under each lower lid, what was left of the bag;
# the rim of the lid is lit the same before and after with either, and with 0.01 to 0.04.
EDGE_KEEP = (0.02, 0.06)
# A zone is whole within this share of its reach and fades to nothing at its reach.
ZONE_WHOLE = 0.55
# Under each eye, by the radius of its globe: the middle of the zone lies this far to the side of the middle of the eye (away
# from the nose; none: straight below it) and below it, on the skin, and the zone reaches this far across, up and down, and in
# depth. By eye: shifted a tenth of the radius away from the nose and narrower (1.3), the hollow beside the nose stayed.
UNDER_EYE_AT = (0.0, -0.8)
UNDER_EYE_REACH = (1.5, 0.85, 0.8)
# The fold beside the nose runs on the skin from a point this far to the side of and above the end of the base of the nose, past
# one this far to the side of and above the corner of the mouth, to one this far to the side of and below it. The zone is whole
# within the first distance of that line and nothing from the second on.
FOLD_FROM = (0.07, 0.03)
FOLD_PAST = (0.09, 0.02)
FOLD_TO = (0.06, -0.26)
FOLD_REACH = (0.07, 0.19)
# The wing of the nose: round a point of the skin this far to the side of and above the end of the base of the nose, whole
# within the first distance and nothing from the second on, and never more than this share of the change (by eye: the crease
# round the wing goes, the nose keeps its side).
WING_AT = (0.02, 0.09)
WING_REACH = (0.06, 0.17)
WING = 0.6
# The under side of the jaw is the skin that turns its outer side down by this much (the z of its normal), this far below
# where the lips meet, this near the middle and no further behind the eyes than this. The zone is that skin and what lies
# within the first distance of it, fading to nothing at the second: the edge of the jaw above it and the crease with the neck.
# It does little by itself: the under side is a plane 6 cm deep (Lúcia), dark under a light from above by its own shape, its
# middle the same averaged over RADIUS, and averaged over 0.6 the chin above it went dark with it (2026-11-12). What takes the
# hard edge away is `parle_face.soften_under_chin`, which turns that plane to the front; this zone evens out its borders.
JAW_DOWN = -0.5
JAW_BELOW = (0.35, 1.3)
JAW_HALF = 1.1
JAW_BEHIND = 1.5
JAW_REACH = (0.06, 0.26)
# A place counts as changed from this much of the change on.
CHANGED_FROM = 0.01


def places_of(points):
    """Which place each point is at: points at the same place (the two sides of a seam of the picture) are one."""
    _, place = np.unique(np.round(np.asarray(points) / SAME_M).astype(np.int64), axis=0, return_inverse=True)
    return place.ravel()


def triangles_of(mesh):
    mesh.calc_loop_triangles()
    found = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int64)
    mesh.loop_triangles.foreach_get("vertices", found)
    return found.reshape(-1, 3)


def vector_areas(points, triangles, place):
    """For each place, the sum of the normals of the faces around it, each as long as a third of its face's area: the normal of
    the place, weighing as much as the skin it stands for."""
    across = np.cross(points[triangles[:, 1]] - points[triangles[:, 0]], points[triangles[:, 2]] - points[triangles[:, 0]]) / 6
    total = np.zeros((int(place.max()) + 1, 3))
    for corner in range(3):
        np.add.at(total, place[triangles[:, corner]], across)
    return total


def ties_of(spot, triangles, place, kept):
    """For each place, the places it is tied to by an edge and how far each is; only ties between `kept` places."""
    ends = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    ends = np.unique(np.sort(place[ends], axis=1), axis=0)
    ends = ends[(ends[:, 0] != ends[:, 1]) & kept[ends[:, 0]] & kept[ends[:, 1]]]
    far = np.linalg.norm(spot[ends[:, 0]] - spot[ends[:, 1]], axis=1)
    ties = [[] for _ in range(len(spot))]
    for (first, second), length in zip(ends.tolist(), far.tolist()):
        ties[first].append((second, length))
        ties[second].append((first, length))
    return ties


def ways_from(start, ties, most):
    """How far along the edges each place is from `start`, for the places nearer than `most`."""
    found = {int(start): 0.0}
    heap = [(0.0, int(start))]
    while heap:
        far, at = heapq.heappop(heap)
        if far > found[at]:
            continue
        for other, length in ties[at]:
            reach = far + length
            if reach < most and reach < found.get(other, np.inf):
                found[other] = reach
                heapq.heappush(heap, (reach, other))
    return found


def smoothed(normals, ties, wanted, radius):
    """The normal of each `wanted` place averaged over the skin around it: the sum of `normals` (each as long as the skin its
    place stands for) over the places no further than `radius` along the edges, each counting less the further it is, down to
    nothing at the radius. A fold narrower than the radius has its two sides in the same sum, and what comes out is the normal
    of the skin it folds; a groove as wide as the radius kept 12 of its 29 degrees (2026-11-12, on a sheet)."""
    out = np.zeros_like(normals)
    for at in np.flatnonzero(wanted):
        near = ways_from(at, ties, radius)
        index = np.fromiter(near.keys(), dtype=np.int64, count=len(near))
        share = (1 - (np.fromiter(near.values(), dtype=float, count=len(near)) / radius) ** 2) ** 2
        out[at] = share @ normals[index]
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def seen_from():
    """The ways of SEEN_ROUND_DEG and SEEN_UP_DEG, each from the face to the eye that looks at it. The face looks at -y."""
    ways = []
    for up in np.radians(SEEN_UP_DEG):
        for around in np.radians(SEEN_ROUND_DEG):
            ways.append((np.sin(around) * np.cos(up), -np.cos(around) * np.cos(up), np.sin(up)))
    return np.array(ways)


def shown_places(spot, facing, wanted, cover):
    """Which of the `wanted` places show (see SEEN_ROUND_DEG). `cover` is the tree of all that can hide skin: the skin and the eye
    globes. The lining of the mouth behind closed lips and the socket behind a globe do not show."""
    shown = np.zeros(len(spot), bool)
    ways = seen_from()
    for at in np.flatnonzero(wanted):
        start = Vector(spot[at] + facing[at] * SEEN_LIFT_M)
        for way in ways[facing[at] @ ways.T > SEEN_FACING]:
            if cover.ray_cast(start, Vector(way), SEEN_FAR_M)[0] is None:
                shown[at] = True
                break
    return shown


def far_from(spot, marks):
    """How far each place is from the nearest of the `marks` places, straight."""
    if not marks.any():
        return np.full(len(spot), np.inf)
    tree = KDTree(int(marks.sum()))
    for index in np.flatnonzero(marks):
        tree.insert(spot[index], int(index))
    tree.balance()
    return np.array([tree.find(point)[2] for point in spot])


def fade(far, whole, none):
    """1 as far as `whole`, down to 0 at `none`."""
    return 1 - parle_polish.smooth((far - whole) / (none - whole))


def far_from_line(spot, line):
    """How far each place is from a line drawn from one point to the next."""
    far = np.full(len(spot), np.inf)
    for start, end in zip(line[:-1], line[1:]):
        along = end - start
        share = np.clip((spot - start) @ along / max(along @ along, 1e-18), 0.0, 1.0)
        far = np.minimum(far, np.linalg.norm(spot - (start + share[:, None] * along), axis=1))
    return far


def face_skin(body, globe):
    """The skin of a face, place by place, with what its zones are drawn from: None when the face has no nose base or no mouth
    to find them by. `spot` is where each place is and `facing` its normal, both in the world; `shown` the places that can be
    seen; `sharp` the places of the edges that stay (next to hidden skin, the base of the nose, the lips)."""
    mesh = body.data
    points = parle_polish.to_world(body, parle_polish.read(mesh.vertices))
    place, triangles = places_of(points), triangles_of(mesh)
    areas = vector_areas(points, triangles, place)
    spot = np.zeros_like(areas)
    spot[place] = points
    facing = areas / np.maximum(np.linalg.norm(areas, axis=1, keepdims=True), 1e-12)
    balls = sorted(((np.asarray(centre, float), float(radius)) for centre, radius in parle_polish.eye_balls(globe)), key=lambda ball: ball[0][0])
    unit = float(np.linalg.norm(balls[1][0] - balls[0][0]))
    tip = parle_normals.nose_tip(points)
    face = (spot[:, 2] > tip[2] - FACE_BELOW * unit) & (spot[:, 1] < tip[1] + FACE_BEHIND * unit)
    eye_points = parle_polish.to_world(globe, parle_polish.read(globe.data.vertices))
    cover = BVHTree.FromPolygons(np.concatenate([spot, eye_points]).tolist(), np.concatenate([place[triangles], triangles_of(globe.data) + len(spot)]).tolist())
    shown = shown_places(spot, facing, face, cover)
    rim = np.zeros(len(spot), bool)
    for at, tied in enumerate(ties_of(spot, triangles, place, face)):
        rim[at] = shown[at] and any(not shown[other] for other, _ in tied)
    away = (spot - tip) / unit
    base = shown & (np.abs(away[:, 0]) < NOSE_BASE_REACH[0]) & (away[:, 2] > -NOSE_BASE_REACH[1]) & (away[:, 2] < 0.1)
    base &= (away[:, 1] > 0) & (away[:, 1] < NOSE_BASE_REACH[2]) & (facing[:, 2] < NOSE_BASE_DOWN)
    mouth = rim & (-away[:, 2] > MOUTH_BELOW[0]) & (-away[:, 2] < MOUTH_BELOW[1]) & (np.abs(away[:, 0]) < MOUTH_HALF)
    if not base.any() or not mouth.any():
        return None
    blocks = mesh.shape_keys.key_blocks if mesh.shape_keys else {}
    rest = parle_polish.read(mesh.vertices)
    lips = np.zeros(len(spot), bool)
    for name in LIPS:
        if name in blocks:
            moved = np.linalg.norm(parle_polish.read(blocks[name]) - rest, axis=1)
            lips[place[moved >= LIP_FROM * max(moved.max(), 1e-9)]] = True
    return {"points": points, "place": place, "triangles": triangles, "areas": areas, "spot": spot, "facing": facing, "balls": balls,
            "unit": unit, "tip": tip, "face": face, "shown": shown, "rim": rim, "base": base, "mouth": mouth, "sharp": rim | base | (lips & shown)}


def fold_zones(skin):
    """For each zone of the face that is lit as smooth skin, how much of the change each place takes: whole in the middle of
    the zone, fading to nothing at its edge. Before the edges that stay are taken out (`fold_weights`)."""
    spot, facing, unit, tip = skin["spot"], skin["facing"], skin["unit"], skin["tip"]
    front = skin["shown"] & ~skin["sharp"] & (facing[:, 1] < 0)

    def on_skin(across, up):
        """The point of the skin that is seen from the front at this place across and up."""
        nearest = np.flatnonzero(front)[np.argmin((spot[front, 0] - across) ** 2 + (spot[front, 2] - up) ** 2)]
        return np.array((across, spot[nearest, 1], up))

    zones = {name: np.zeros(len(spot)) for name in ("under eye", "fold", "wing", "jaw")}
    meet = spot[skin["mouth"]]
    for side, (centre, radius) in zip((-1, 1), skin["balls"]):
        middle = on_skin(centre[0] + side * UNDER_EYE_AT[0] * radius, centre[2] + UNDER_EYE_AT[1] * radius)
        far = np.linalg.norm((spot - middle) / (np.array((UNDER_EYE_REACH[0], UNDER_EYE_REACH[2], UNDER_EYE_REACH[1])) * radius), axis=1)
        zones["under eye"] = np.maximum(zones["under eye"], fade(far, ZONE_WHOLE, 1.0))
        mine = skin["base"] & (side * (spot[:, 0] - tip[0]) > 0)
        if not mine.any():
            continue
        end = spot[mine][np.argmax(side * spot[mine, 0])]
        corner = meet[np.argmax(side * meet[:, 0])]
        line = [on_skin(start[0] + side * aside * unit, start[2] + up * unit) for start, (aside, up) in ((end, FOLD_FROM), (corner, FOLD_PAST), (corner, FOLD_TO))]
        zones["fold"] = np.maximum(zones["fold"], fade(far_from_line(spot, np.array(line)), FOLD_REACH[0] * unit, FOLD_REACH[1] * unit))
        wing = on_skin(end[0] + side * WING_AT[0] * unit, end[2] + WING_AT[1] * unit)
        zones["wing"] = np.maximum(zones["wing"], WING * fade(np.linalg.norm(spot - wing, axis=1), WING_REACH[0] * unit, WING_REACH[1] * unit))
    below = (meet[:, 2].mean() - spot[:, 2]) / unit
    under = skin["shown"] & (facing[:, 2] < JAW_DOWN) & (below > JAW_BELOW[0]) & (below < JAW_BELOW[1]) & (np.abs(spot[:, 0] - tip[0]) < JAW_HALF * unit)
    under &= spot[:, 1] < (skin["balls"][0][0][1] + skin["balls"][1][0][1]) / 2 + JAW_BEHIND * unit
    zones["jaw"] = fade(far_from(spot, under), JAW_REACH[0] * unit, JAW_REACH[1] * unit)
    return zones


def fold_weights(skin):
    """How much of the change each place takes: the most any zone gives it (`fold_zones`), nothing on skin that does not show,
    nothing on an edge that stays or near it (EDGE_KEEP)."""
    weight = np.max(list(fold_zones(skin).values()), axis=0) * skin["shown"]
    return weight * (1 - fade(far_from(skin["spot"], skin["sharp"]), EDGE_KEEP[0] * skin["unit"], EDGE_KEEP[1] * skin["unit"]))


def soften(body, weight, radius, left_out=None):
    """Blends into the custom split normals of a mesh, point by point, those same normals averaged over the skin around the
    point (`smoothed`, within `radius`, in metres): `weight` is how much each point takes of it, from 0 to 1, and `left_out`
    the points that are not counted as skin around the others. The normals are the ones the mesh carries, so what an earlier
    step did to them (`parle_polish.seamless_normals`, `parle_face.soften_under_chin`) is kept and evened out, not undone.
    Points at the same place weigh and turn as one. Every corner is written back, the ones that do not change as they were
    read: Blender keeps a custom normal as two 16-bit angles, and those come back up to 0.03 degrees away (2026-11-12, on the
    five faces). Returns how many places changed."""
    mesh = body.data
    points = parle_polish.to_world(body, parle_polish.read(mesh.vertices))
    place, triangles = places_of(points), triangles_of(mesh)
    count = int(place.max()) + 1
    spot = np.zeros((count, 3))
    spot[place] = points
    share = np.zeros(count)
    np.maximum.at(share, place, np.clip(np.asarray(weight, float), 0.0, 1.0))
    counted = np.ones(count, bool)
    if left_out is not None:
        counted[place[np.asarray(left_out, bool)]] = False
    wanted = share >= CHANGED_FROM
    if not wanted.any():
        return 0
    # Normals turn with the inverse of the matrix that carries the points, transposed.
    matrix = np.array(body.matrix_world)[:3, :3]
    point_of = np.empty(len(mesh.loops), dtype=np.int64)
    mesh.loops.foreach_get("vertex_index", point_of)
    corners = np.empty(len(mesh.loops) * 3)
    mesh.corner_normals.foreach_get("vector", corners)
    corners = corners.reshape(-1, 3)
    at = place[point_of]
    world = corners @ np.linalg.inv(matrix)
    world /= np.maximum(np.linalg.norm(world, axis=1, keepdims=True), 1e-12)
    carried = np.zeros((count, 3))
    np.add.at(carried, at, world)
    carried /= np.maximum(np.linalg.norm(carried, axis=1, keepdims=True), 1e-12)
    # Each place weighs as much as the skin it stands for: a third of every face around it.
    faces = np.linalg.norm(np.cross(points[triangles[:, 1]] - points[triangles[:, 0]], points[triangles[:, 2]] - points[triangles[:, 0]]), axis=1) / 6
    skin = np.zeros(count)
    for corner in range(3):
        np.add.at(skin, place[triangles[:, corner]], faces)
    smooth = smoothed(carried * skin[:, None], ties_of(spot, triangles, place, counted | wanted), wanted, radius)
    turned = wanted[at]
    mixed = world[turned] * (1 - share[at][turned, None]) + smooth[at][turned] * share[at][turned, None]
    local = mixed @ matrix
    corners[turned] = local / np.maximum(np.linalg.norm(local, axis=1, keepdims=True), 1e-12)
    mesh.normals_split_custom_set(corners.tolist())
    mesh.update()
    return int(wanted.sum())


def soft_normals(meshes):
    """The folds of the face of a finished character lit as smooth skin, and how many places of its skin changed (some 600 to
    700 of 12 500 points on the five looked at): the fold beside the nose, the hollow and the bag under each eye, the wings of
    the nose and the borders of the under side of the jaw (`fold_zones`). The points stay; only the custom split normals of
    the skin change, and only on the face, from under the eyes to the top of the neck under the jaw. The face looks at -y. No
    eye globes, or no base of the nose or no mouth found: no change."""
    body = next((obj for obj in meshes if obj.name == "Body"), None)
    globe = next((obj for obj in meshes if "low-poly" in obj.name), None)
    if body is None or globe is None or not len(body.data.polygons) or not len(globe.data.polygons):
        return 0
    skin = face_skin(body, globe)
    if skin is None:
        return 0
    left_out = ~skin["shown"] | skin["sharp"]
    return soften(body, fold_weights(skin)[skin["place"]], RADIUS * skin["unit"], left_out[skin["place"]])
