"""
    License information: data/licenses/makehuman_license.txt

    An atlas of their own for the Parle cast's teeth. The stock mesh is 32 separate teeth and two gums, and its UV islands pile up: 76 % of the
    covered texels of the map belong to more than one triangle, three on average. `parle_mouth` shades the map point by point, and over a pile the
    darkest wins, so a molar's shade landed on an incisor and the upper gum came out olive on one side and pink on the other: the grid or braces of the
    scoreboard of 2026-11-10 (q02, q04, q12). Every tooth gets a cell and every gum a strip, the outer faces apart from the inner ones, and no texel is shared.
"""
import numpy as np

SIZE = 512
ARCH_TEETH = 16
TOOTH_W = SIZE // ARCH_TEETH
TOOTH_H = 64
TOOTH_ROWS = 2
GUM_H = (SIZE - TOOTH_ROWS * TOOTH_H) // 2
MARGIN = 0.08
# Sizes of the parts, in welded points: a tooth has 22, a gum over 200.
GUM_LEAST = 100
BEHIND_ARCH_M = 0.03
IVORY = np.array((0.80, 0.90, 0.95))  # BGR
GUM = np.array((112, 116, 196)) / 255  # BGR, the colour of a gum in daylight
# A tooth is a little warmer towards the gum, and a thin darker line keeps it from melting into the next one. Measured by eye in the player: a design choice to pilot.
ROOT_SHADE = 0.9
EDGE_SHADE = 0.8
EDGE_SHARE = 0.1


def parts(points, triangles):
    """A label per point: the points of one tooth or gum share it. The mesh splits points at UV seams, so they are welded by place first."""
    _, place = np.unique(np.round(points, 6), axis=0, return_inverse=True)
    place = place.ravel()
    label = np.arange(place.max() + 1)
    edges = np.concatenate([place[triangles[:, [0, 1]]], place[triangles[:, [1, 2]]], place[triangles[:, [2, 0]]]])
    while True:
        lowest = label.copy()
        np.minimum.at(lowest, edges[:, 0], label[edges[:, 1]])
        np.minimum.at(lowest, edges[:, 1], label[edges[:, 0]])
        if (lowest == label).all():
            return label[place]
        label = lowest


def arch_centre(points):
    """Behind the molars, in the middle: the face looks to -y."""
    return np.array((float(points[:, 0].mean()), float(points[:, 1].max()) + BEHIND_ARCH_M))


def box(values, low, high, inset=MARGIN, members=None):
    """`values` stretched to [low, high], inside the margin that keeps the filter from reading the next cell. The stretch is read off `members`
    only: read off the whole mouth, a tooth took a tenth of its cell and the player's filter mixed it with the background (braces, 2026-11-11)."""
    own = values if members is None else values[members]
    span = max(float(own.max() - own.min()), 1e-9)
    share = (values - own.min()) / span
    return low + (high - low) * (inset + (1 - 2 * inset) * share)


def tooth_cell(arch, order):
    """The cell of a tooth in the map, in pixels from the top left: arch 0 is the upper one."""
    return TOOTH_W * order, TOOTH_H * arch


def gum_strip(arch):
    return TOOTH_ROWS * TOOTH_H + GUM_H * arch


def facing_out(points, triangles, centre):
    """Which triangles look away from the middle of the arch."""
    corner = points[triangles]
    normal = np.cross(corner[:, 1] - corner[:, 0], corner[:, 2] - corner[:, 0])
    out = corner.mean(axis=1)[:, :2] - centre
    return (normal[:, :2] * out).sum(axis=1) >= 0


def atlas(points, triangles):
    """The UV of every corner, as (triangles, 3, 2) with v from the bottom (Blender's). Two halves per tooth and per gum: the outer faces and the inner ones."""
    label = parts(points, triangles)
    centre = arch_centre(points)
    ids, counts = np.unique(label, return_counts=True)
    gums = ids[counts >= GUM_LEAST]
    teeth = ids[counts < GUM_LEAST]
    # The arch of a tooth is the arch of the gum it sits in: by height, a curved arch put the upper and lower incisors in one row (2026-11-11).
    gum_order = sorted(gums, key=lambda part: -points[label == part, 2].mean())
    gum_points = [points[label == part] for part in gum_order]

    def arch_of(part):
        mine = points[label == part]
        return int(np.argmin([np.linalg.norm(mine[:, None] - other[None], axis=2).min(axis=1).mean() for other in gum_points]))
    out = facing_out(points, triangles, centre)
    owner = label[triangles[:, 0]]
    uv = np.zeros((len(triangles), 3, 2))
    pixels = np.zeros((len(triangles), 3, 2))

    def place(chosen, across, up, left, top, width, height):
        corners = triangles[chosen]
        if not len(corners):
            return
        members = np.unique(corners)
        pixels[chosen, :, 0] = box(across, left, left + width, members=members)[corners]
        pixels[chosen, :, 1] = box(-up, top, top + height, members=members)[corners]

    arches = {part: arch_of(part) for part in teeth} if len(gum_points) == TOOTH_ROWS else None
    middle = float(np.median([points[label == part, 2].mean() for part in teeth]))
    for arch in range(TOOTH_ROWS):
        own = [part for part in teeth if (arches[part] == arch if arches else (points[label == part, 2].mean() > middle) == (arch == 0))]
        own.sort(key=lambda part: np.arctan2(points[label == part, 0].mean() - centre[0], centre[1] - points[label == part, 1].mean()))
        for order, part in enumerate(own[:ARCH_TEETH]):
            members = label == part
            radial = points[members, :2].mean(axis=0) - centre
            radial /= max(np.linalg.norm(radial), 1e-9)
            tangent = np.array((radial[1], -radial[0]))
            across = np.zeros(len(points))
            across[members] = points[members, :2] @ tangent
            left, top = tooth_cell(arch, order)
            for half, faces in enumerate((out, ~out)):
                chosen = (owner == part) & faces
                place(chosen, across, points[:, 2], left, top + half * TOOTH_H / 2, TOOTH_W, TOOTH_H / 2)
    for arch, part in enumerate(gum_order):
        members = label == part
        turn = np.arctan2(points[:, 0] - centre[0], centre[1] - points[:, 1])
        for half, faces in enumerate((out, ~out)):
            chosen = (owner == part) & faces
            place(chosen, turn, points[:, 2], 0, gum_strip(arch) + half * GUM_H / 2, SIZE, GUM_H / 2)
    uv[..., 0] = pixels[..., 0] / SIZE
    uv[..., 1] = 1 - pixels[..., 1] / SIZE
    return uv


def tooth_map():
    """The map for that atlas, BGR in 0..1, rows from the top."""
    image = np.empty((SIZE, SIZE, 3))
    image[:TOOTH_ROWS * TOOTH_H] = IVORY
    image[TOOTH_ROWS * TOOTH_H:] = GUM
    xs = (np.arange(TOOTH_W) + 0.5) / TOOTH_W
    edge = np.interp(np.minimum(xs, 1 - xs), (0, EDGE_SHARE), (EDGE_SHADE, 1.0))
    ys = (np.arange(TOOTH_H // 2) + 0.5) / (TOOTH_H // 2)
    for arch in range(TOOTH_ROWS):
        # The root is up on the upper arch and down on the lower one, and each cell keeps up on top.
        root = ys if arch else 1 - ys
        fall = 1 + (ROOT_SHADE - 1) * root
        for order in range(ARCH_TEETH):
            left, top = tooth_cell(arch, order)
            for half in range(2):
                band = int(top + half * TOOTH_H // 2)
                image[band:band + TOOTH_H // 2, left:left + TOOTH_W] = IVORY * (edge * fall[:, None])[:, :, None]
    return image
