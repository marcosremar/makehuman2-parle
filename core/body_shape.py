"""
    License information: data/licenses/makehuman_license.txt

    A defined male torso for makehuman2-parle: the muscles of a sculpture on the proportions of the body.

    Laying the body on the sculpture flattens the anatomy and changes the silhouette, building the
    chest from a formula gives a shelf or a breast, and smoothing the body first collapses the waist.
    So the chest volume comes from the MakeHuman pectoral target, and the sculpture only adds fine
    detail, the way a displacement map does: both torsos go to the same chart, angle around the torso
    and height between the crotch and the pit of the neck, and what is left of the sculpture without
    its low frequencies goes on the body along the normal.

    Works in metres with Z up.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

from core.garment_fit import vertex_normals

ANGLES = 360
HEIGHTS = 420
# Height of the chart in shares of crotch to neck pit: the top of the hips to the collarbones.
CHART_LOW = -0.15
CHART_HIGH = 1.08
# The relief is what is smaller than this, in the same shares.
RELIEF_SHARE = 0.07
GAIN = 0.7
MOST_RELIEF_M = 0.008
PECTORAL = "data/target/hm08/torso/torso-muscle-pectoral-incr.target"
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


def chart(points, marks):
    centre, axis, crotch, pit = marks
    angle = np.arctan2(points[:, 0] - centre, axis - points[:, 1])
    level = (points[:, 2] - crotch) / (pit - crotch)
    radius = np.hypot(points[:, 0] - centre, points[:, 1] - axis)
    return angle, level, radius


def relief_map(sculpt):
    """Muscle relief of the sculpture, in shares of crotch to neck pit."""
    marks = landmarks(sculpt)
    angle, level, radius = chart(sculpt, marks)
    span = marks[3] - marks[2]
    row = ((level - CHART_LOW) / (CHART_HIGH - CHART_LOW) * HEIGHTS).astype(int)
    column = ((angle + np.pi) / (2 * np.pi) * ANGLES).astype(int) % ANGLES
    inside = (row >= 0) & (row < HEIGHTS) & (radius < span * 0.75)
    grid = np.zeros((HEIGHTS, ANGLES))
    np.maximum.at(grid, (row[inside], column[inside]), radius[inside] / span)
    hole = grid == 0
    for _ in range(8):
        grown = maximum_filter(grid, size=3, mode=("nearest", "wrap"))
        grid = np.where(hole, grown, grid)
        hole = grid == 0
    sigma = RELIEF_SHARE / (CHART_HIGH - CHART_LOW) * HEIGHTS
    wide = gaussian_filter(grid, (sigma, sigma * ANGLES / HEIGHTS * 0.9), mode=("nearest", "wrap"))
    return gaussian_filter(grid - wide, 0.6, mode=("nearest", "wrap"))


def sample(grid, angle, level):
    row = np.clip((level - CHART_LOW) / (CHART_HIGH - CHART_LOW) * HEIGHTS - 0.5, 0, HEIGHTS - 1.001)
    column = ((angle + np.pi) / (2 * np.pi) * ANGLES - 0.5) % ANGLES
    r0, c0 = row.astype(int), column.astype(int)
    fr, fc = row - r0, column - c0
    c1 = (c0 + 1) % ANGLES
    return (grid[r0, c0] * (1 - fr) * (1 - fc) + grid[r0 + 1, c0] * fr * (1 - fc) + grid[r0, c1] * (1 - fr) * fc + grid[r0 + 1, c1] * fr * fc)


def read_target(path, count):
    shift = np.zeros((count, 3))
    with open(path) as lines:
        for line in lines:
            if line.startswith("#") or not line.strip():
                continue
            index, x, y, z = line.split()[:4]
            shift[int(index)] = (float(x), -float(z), float(y))
    return shift / DECIMETRES


def pectoral(base, delta, joints):
    """Takes the shelf left by the old delta out of the chest and puts the artist-made MakeHuman pectoral in its place."""
    left, right = joints["LeftArm"], joints["RightArm"]
    centre = (left[0] + right[0]) / 2
    shoulder = abs(left[0] - right[0]) / 2
    across = np.abs(base[:, 0] - centre) / shoulder
    down = (left[2] - base[:, 2]) / shoulder
    area = smoothstep((1.45 - across) / 0.35) * smoothstep((down + 0.45) / 0.3) * smoothstep((1.2 - down) / 0.3)
    return -delta * area[:, None] + read_target(PECTORAL, len(base))


def moulded(points, faces, sculpt, trunk):
    """Fine muscle relief of the sculpture on the body, along the normal; `trunk` is the share of each vertex that belongs to the torso bones."""
    merged, twin, _, welded_faces = welded(points, faces)
    marks = landmarks(merged)
    angle, level, _ = chart(merged, marks)
    own = np.zeros(len(merged))
    own[twin] = trunk
    area = smoothstep((own - 0.6) / 0.35) * smoothstep((level - CHART_LOW) / 0.15) * smoothstep((1.0 - level) / 0.12)
    span = marks[3] - marks[2]
    detail = np.clip(sample(relief_map(sculpt), angle, level) * span * GAIN, -MOST_RELIEF_M, MOST_RELIEF_M) * area
    print("MOLDE marcos", np.round(marks, 3), "relevo", round(float(np.abs(detail).max()) * 1000, 1), "mm")
    return (vertex_normals(merged, welded_faces) * detail[:, None])[twin]
