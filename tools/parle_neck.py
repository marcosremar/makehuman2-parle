"""
    License information: data/licenses/makehuman_license.txt

    The skin of the head where it meets the neck, behind the face.

    Seen from behind, Carlos had a collar of another skin under the hair: a plain line across the nape and down the side of the neck
    (2026-11-11). The head and the neck are two islands of the skin map, and two meshes whose borders stand up to 5 mm apart. The relief
    that shades the face (`skin_style.relief`) takes each border for the edge of the world: it painted a band 22 levels darker round
    the top of the neck, and nothing on the head just above it. `skin_style.even_seams` joins the two sides of a seam over 10 px, a
    centimetre on the head: a line still.

    So behind the face, where there is no face to shade, the relief paints nothing (`behind_face`), and the head takes whatever
    difference is left at the seam and gives it back little by little, over NECK_WIDE_M (`neck_tone`). The body keeps its tone. Under
    the chin and the jaw nothing changes: that shade is the face's.
"""
import cv2
import numpy as np

# The ear sits some 10 cm behind the tip of the nose: behind the face starts a little in front of it and is whole 4 cm further back.
EAR_BEHIND_TIP_M = 0.085
BACK_FADE_M = 0.04
# And under the ear: its lobe ends about level with the tip of the nose, and the ear keeps the volume the relief paints on it.
LOBE_BELOW_TIP_M = 0.02
LOBE_FADE_M = 0.02
# The two borders belong to two meshes and their points do not coincide (1.6 mm apart at the side of the neck, up to 5): each point of
# the head's border takes the nearest one of the body's within this.
ACROSS_M = 0.008
# The difference fades out this far from the seam, counted on the head: up under the hair at the back, short of the jaw at the side.
NECK_WIDE_M = 0.04
SAMPLE_PX = 3
HEAD_BELOW_TOP_M = 0.35


def smooth(value):
    value = np.clip(value, 0, 1)
    return value * value * (3 - 2 * value)


def islands(triangles, count):
    """The island of each point: points joined by a triangle are of the same one."""
    parent = np.arange(count)

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for first, second, third in triangles:
        low = min(root(first), root(second), root(third))
        for index in (first, second, third):
            parent[root(index)] = low
    return np.array([root(index) for index in range(count)])


def open_points(triangles):
    edges = np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1)
    unique, count = np.unique(edges, axis=0, return_counts=True)
    return np.unique(unique[count == 1])


def nose_tip(points):
    """The point at the tip of the nose of a body that looks to -y, z up: the foremost one on the middle of its head, the top 26 cm."""
    head = points[:, 2] > points[:, 2].max() - 0.26
    middle = float(np.median(points[head, 0]))
    line = np.nonzero(head & (np.abs(points[:, 0] - middle) < 0.01))[0]
    return int(line[points[line, 1].argmin()])


def behind_face(points):
    """How far behind the face each point is, from 0 on the face and the ears to 1 behind the ears and below them: the nape and the
    side of the neck."""
    tip = points[nose_tip(points)]
    return smooth((points[:, 1] - (tip[1] + EAR_BEHIND_TIP_M)) / BACK_FADE_M) * smooth((tip[2] - LOBE_BELOW_TIP_M - points[:, 2]) / LOBE_FADE_M)


def neck_tone(image, data, points, size):
    """The skin picture with the head brought to the tone of the neck where the two meet behind the face, and how many points of the
    seam it was read at. `image` is the picture, blue-green-red in floats, `data` the map of the skin (`uv` of each triangle's corners,
    `tri`) and `points` the points of this body, in metres, the face looking at -y. The head is the island of the tip of the nose and
    the body the biggest of the others: the inside of the mouth and the eyelids are islands too, and their colour is theirs."""
    triangles = data["tri"]
    corners = np.stack([data["uv"][:, :, 0] * size, (1 - data["uv"][:, :, 1]) * size], axis=2)
    top = points[:, 2].max()
    tip = nose_tip(points)
    back = behind_face(points)
    label = islands(triangles, len(points))
    used = np.unique(triangles)
    others = label[used][label[used] != label[tip]]
    if not len(others):
        return image, 0
    body = np.bincount(others).argmax()
    place = {}
    for vertex, corner in zip(triangles.ravel(), corners.reshape(-1, 2)):
        place[int(vertex)] = (min(max(int(round(corner[0])), 0), size - 1), min(max(int(round(corner[1])), 0), size - 1))
    border = open_points(triangles)
    border = border[points[border, 2] > top - HEAD_BELOW_TOP_M]
    mine, other = border[label[border] == label[tip]], border[label[border] == body]
    if not len(mine) or not len(other):
        return image, 0
    # Each side is read a little inside its island: at the very border the picture already mixes the two.
    inside = np.zeros(image.shape[:2], np.float32)
    own = np.zeros(image.shape[:2], np.float32)
    area = flat = 0.0
    for triangle, corner in zip(triangles, corners):
        cv2.fillConvexPoly(inside, corner.round().astype(np.int32), 1.0)
        if label[triangle[0]] == label[tip]:
            cv2.fillConvexPoly(own, corner.round().astype(np.int32), float(back[triangle].mean()))
            first, second, third = points[triangle]
            area += float(np.linalg.norm(np.cross(second - first, third - first))) / 2
            across, along = corner[1] - corner[0], corner[2] - corner[0]
            flat += abs(float(across[0] * along[1] - across[1] * along[0])) / 2
    inner = cv2.erode(inside, np.ones((3, 3), np.uint8))
    sample = cv2.GaussianBlur(image * inner[:, :, None], (0, 0), SAMPLE_PX) / np.maximum(cv2.GaussianBlur(inner, (0, 0), SAMPLE_PX), 1e-6)[:, :, None]
    wide = NECK_WIDE_M * float(np.sqrt(flat / max(area, 1e-9)))
    push = np.zeros_like(image)
    weight = np.zeros(image.shape[:2], np.float32)
    seam = np.ones(image.shape[:2], np.uint8)
    read = 0
    for vertex in mine:
        gap = np.linalg.norm(points[other] - points[vertex], axis=1)
        if gap.min() > ACROSS_M or back[vertex] <= 0:
            continue
        u, v = place[int(vertex)]
        u2, v2 = place[int(other[int(gap.argmin())])]
        cv2.circle(push, (u, v), 2, (sample[v2, u2] - sample[v, u]).tolist(), -1)
        cv2.circle(weight, (u, v), 2, 1.0, -1)
        cv2.circle(seam, (u, v), 2, 0, -1)
        read += 1
    if not read:
        return image, 0
    spread = cv2.GaussianBlur(weight, (0, 0), wide / 2)
    field = cv2.GaussianBlur(push, (0, 0), wide / 2) / np.maximum(spread, 1e-6)[:, :, None]
    fade = smooth(1 - cv2.distanceTransform(seam, cv2.DIST_L2, 5) / max(wide, 1.0))
    return image + field * (fade * own)[:, :, None], read
