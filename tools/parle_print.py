"""
    License information: data/licenses/makehuman_license.txt

    A print that repeats, laid on a new garment: plaid, flowers, a rib.

    Baked from the game's garment to ours, surface to surface, a print follows every difference between the two cloths. Dona Rosa's
    flowers came out as streaks (2026-11-05), and Carlos's plaid in waves, with the sleeves, which the bake did not reach, in one flat
    colour and a torn edge at the armhole (2026-11-11); the game's own picture of his shirt already carries the plaid in waves. A print
    that repeats needs no bake: it is laid on the cloth itself, straight along the grain of each piece and at the size it has on the
    game's body.

    This file holds the rules, with numpy alone: which picture of a game garment is such a print (`tile_of`), and where each point of a
    piece of ours falls on it (`laid`, `as_in_game`, `mirror_line`). `print_cloth.py` reads the two bodies in Blender and paints the
    picture, and `stamp_tile.py` joins it to the baked one.
"""
import numpy as np

# A piece whose UV runs over more than this many pictures lays its picture again and again: Dona Rosa's dress runs over 5 by 6 of them,
# and a garment with a picture of its own stays inside one.
REPEATS_LEAST = 1.5
# Two islands of a piece on the same part of the picture: it was not drawn for the piece, it is a sheet of print. The two sleeves of
# Carlos's shirt lie on the same plaid, each across the whole width of it.
SHARED_GRID = 64
SHARED_LEAST = 0.5
# A sheet of print meets itself again after a shift: the longest one, past half of the picture, is where it is cut to repeat edge to
# edge. When what is left over after that cut is itself such a shift, the picture already closes on itself and is kept whole. Carlos's
# plaid closes from left to right as it is and, from top to bottom, a few pixels short of its 512.
PERIOD_EDGE_PX = 4
PERIOD_STRENGTH = 0.6
PERIOD_NEAR = 0.02
PERIOD_LINES = 256
# One colour all over is nothing to lay (the print of the game's plain tank tops is a flat picture).
PLAIN_SPREAD = 0.01
# The points of a piece this close to the middle of the body set where the print is centred.
MIDDLE_M = 0.02
# A plaid is its own mirror about the middle of each of its lines: one of them goes on the middle of the body, under the buttons, and
# the two halves of the shirt come out alike. Flowers are the mirror of nothing and go as they are.
MIRROR_LEAST = 0.9


def repeated(uv):
    """Whether a piece lays its picture more than once: its UV, (n, 2), runs past one picture."""
    return bool((uv.max(axis=0) - uv.min(axis=0)).max() > REPEATS_LEAST)


def covered(triangles):
    """The cells of a grid over the picture that the UV triangles of an island, (n, 3, 2), lie on."""
    centres = (np.arange(SHARED_GRID) + 0.5) / SHARED_GRID
    x, y = np.meshgrid(centres, centres)
    grid = np.zeros((SHARED_GRID, SHARED_GRID), bool)
    for a, b, c in triangles:
        wide = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(wide) < 1e-12:
            continue
        first = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / wide
        second = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / wide
        grid |= (first >= 0) & (second >= 0) & (first + second <= 1)
    return grid


def shared(islands):
    """Whether two islands of a piece lie on the same part of its picture. `islands` holds the UV triangles of each, (n, 3, 2)."""
    grids = [covered(triangles) for triangles in islands]
    for index, one in enumerate(grids):
        for other in grids[index + 1:]:
            least = min(one.sum(), other.sum())
            if least and (one & other).sum() / least > SHARED_LEAST:
                return True
    return False


def alike(lines, shift):
    """How alike a picture and itself moved by `shift` along its lines are, from -1 to 1."""
    one, other = lines[:, :lines.shape[1] - shift], lines[:, shift:]
    one, other = one - one.mean(), other - other.mean()
    return float((one * other).mean() / max(float(one.std() * other.std()), 1e-9))


def period(gray, axis):
    """The longest shift along `axis`, in pixels, after which a sheet of print meets itself again, or None when the picture never does
    (a picture drawn for one garment). `gray` is the picture, (rows, columns)."""
    lines = np.moveaxis(gray, axis, 1)
    lines = lines[::max(1, lines.shape[0] // PERIOD_LINES)]
    size = lines.shape[1]
    shifts = list(range(size // 2, size - PERIOD_EDGE_PX + 1))
    found = [alike(lines, shift) for shift in shifts]
    if not found or max(found) < PERIOD_STRENGTH:
        return None
    longest = max(shift for shift, value in zip(shifts, found) if value >= max(found) - PERIOD_NEAR)
    left = size - longest
    return size if left <= PERIOD_EDGE_PX or alike(lines, left) >= PERIOD_STRENGTH else longest


def sheet(pixels):
    """The part of a picture that repeats when laid edge to edge, as (rows, columns) from its first row and column, or None when the
    picture is not a sheet of print: plain, or drawn for one garment. `pixels` is (rows, columns, 3), from 0 to 1."""
    gray = pixels[:, :, :3].mean(axis=2)
    if float(gray.std()) < PLAIN_SPREAD:
        return None
    rows, columns = period(gray, 0), period(gray, 1)
    return None if rows is None or columns is None else (rows, columns)


def tile_of(pixels, uv, islands):
    """The part of the picture of a game piece that is laid edge to edge on our cloth, as (rows, columns) from its first row and column,
    or None when the piece wears no print that repeats. `pixels` is its picture, (rows, columns, 3) from 0 to 1, `uv` where its points
    lie on it, (n, 2), and `islands` the UV triangles of each of its islands. Two kinds of piece wear one: the piece that lays its
    picture again and again (a dress in flowers, a ribbed tank top), all of the picture; and the piece that shares a sheet of print
    among its islands (the sleeves of a plaid shirt), the part of the sheet that closes on itself."""
    if float(pixels[:, :, :3].mean(axis=2).std()) < PLAIN_SPREAD:
        return None
    if repeated(uv):
        return pixels.shape[:2]
    return sheet(pixels) if shared(islands) else None


def mirror_line(pixels):
    """Where a print is its own mirror, in pictures from its first column and its first row: (across, along), each the middle of its
    darkest line of that kind, or 0 when the print mirrors nothing that way. `pixels` is (rows, columns, 3)."""
    gray = pixels[:, :, :3].mean(axis=2)
    found = []
    for axis in (1, 0):
        profile = gray.mean(axis=1 - axis)
        size, best = len(profile), None
        if float(profile.std()) > 1e-4:
            for twice in range(2 * size):
                other = profile[(twice - np.arange(size)) % size]
                alike = float(np.corrcoef(profile, other)[0, 1])
                dark = float(profile[(twice // 2) % size] + profile[((twice + 1) // 2) % size])
                if alike >= MIRROR_LEAST and (best is None or dark < best[0]):
                    best = (dark, twice)
        found.append(0.0 if best is None else (best[1] / 2 + 0.5) / size)
    return tuple(found)


def cloth_frame(corners, flat, grain):
    """How the flat lay of a piece of cloth becomes the cloth itself: the 2x2 matrix that sets the grain along +y and the right of the
    cloth, seen from outside, along +x, and the metres of cloth in one unit of the lay. `corners` are the triangles of the piece in
    space, (n, 3, 3), `flat` the same corners laid flat, (n, 3, 2), at any turn and any size (an unwrap, or the UV of a picture), and
    `grain` the way the cloth hangs in space: up on the trunk, along the arm on a sleeve."""
    grain = np.asarray(grain, float) / np.linalg.norm(grain)
    edges = np.stack([corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]], axis=2)
    lay = np.stack([flat[:, 1] - flat[:, 0], flat[:, 2] - flat[:, 0]], axis=2)
    area = np.linalg.norm(np.cross(edges[:, :, 0], edges[:, :, 1]), axis=1) / 2
    signed = np.linalg.det(lay) / 2
    fine = np.abs(signed) > 1e-12
    edges, lay, area, signed = edges[fine], lay[fine], area[fine], signed[fine]
    # A step on the lay to a step in space, face by face: edges = step @ lay.
    step = edges @ np.linalg.inv(lay)
    # The step on the lay whose step in space is the grain, as far as the face holds it.
    along = np.einsum("nij,j->ni", np.linalg.pinv(step), grain)
    held = np.linalg.norm(np.einsum("nij,nj->ni", step, along), axis=1)
    ways = along / np.maximum(np.linalg.norm(along, axis=1, keepdims=True), 1e-12)
    way = (ways * (area * held)[:, None]).sum(axis=0)
    way /= max(float(np.linalg.norm(way)), 1e-12)
    turn = np.array([[way[1], -way[0]], [way[0], way[1]]])
    # A lay that is the mirror of the cloth (seen from outside, its faces run the other way round) is turned over.
    if float((np.sign(signed) * area).sum()) < 0:
        turn = np.array([[-1.0, 0.0], [0.0, 1.0]]) @ turn
    return turn, float(np.sqrt(area.sum() / np.abs(signed).sum()))


def laid(corners, flat, grain, middle=None, level=0.0):
    """Where each corner of a piece falls on its own cloth, in metres: across the cloth and along the grain, (n, 3, 2). Along the grain
    the cloth is counted as the height along it in space, less `level`, so the print sits at the same height on every piece that hangs
    the same way and the checks of the front meet those of the back at the side. Across, the cloth is centred on the points at
    x = `middle`, the middle of the body, when the piece has any (the buttons of a shirt, the middle of the back), and on the piece
    itself when it has none (a sleeve)."""
    grain = np.asarray(grain, float) / np.linalg.norm(grain)
    turn, metres = cloth_frame(corners, flat, grain)
    cloth = flat @ turn.T * metres
    cloth[:, :, 1] += float(np.mean(corners @ grain - cloth[:, :, 1])) - level
    near = np.abs(corners[:, :, 0] - middle) < MIDDLE_M if middle is not None else np.zeros(corners.shape[:2], bool)
    cloth[:, :, 0] -= float(cloth[:, :, 0][near].mean()) if near.any() else float(cloth[:, :, 0].mean())
    return cloth


def as_in_game(corners, uv, grain):
    """How a picture stands on the cloth of the game's piece: the 2x2 matrix that takes a step on the cloth, in metres (across, along the
    grain), to a step on the picture, in pictures. The turn is the nearest quarter turn: a piece is never cut a few degrees off the
    grain, and the mean over a curved piece is. The size is the one the picture has on the game's body."""
    turn, metres = cloth_frame(corners, uv, grain)
    back = turn.T
    mirror = np.array([[-1.0, 0.0], [0.0, 1.0]]) if np.linalg.det(back) < 0 else np.eye(2)
    plain = back @ mirror
    quarter = np.round(np.arctan2(plain[1, 0], plain[0, 0]) / (np.pi / 2)) * (np.pi / 2)
    square = np.round(np.array([[np.cos(quarter), -np.sin(quarter)], [np.sin(quarter), np.cos(quarter)]]))
    return square @ mirror / metres
