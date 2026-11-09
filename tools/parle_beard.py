"""The shape of a beard, drawn on the head instead of read from the bake.

The bake only knew the jaw: the beard stopped in a round end in the middle of the cheek, with no sideburn and no moustache (owner, 2026-11-09).
Here it runs from the hair, in front of the ear, down the jaw to the chin, and the moustache meets it at the corners of the mouth. The outlines are
measured on the shared mesh, so they follow every face. Each person keeps the colour of their own beard.
"""
import cv2
import numpy as np

# Seen from the side: (depth behind the nose tip, height above it), in cm. Sideburn in front of the ear, cheek line, then along the jaw.
SIDE_OUTLINE = ((4.0, -3.4), (5.5, -3.0), (7.0, -2.3), (8.4, -1.2), (9.6, 0.2), (9.9, 4.5), (11.6, 4.5), (11.7, -1.0), (11.3, -4.0), (10.4, -7.0), (9.4, -11.5), (4.0, -11.5))
# Seen from the front: (distance from the middle, height), in cm. Moustache under the nose, down past the mouth to the chin.
# It runs past the middle: a border on the middle itself drew a bare line down the chin.
FRONT_OUTLINE = ((-2.0, -1.55), (1.3, -1.55), (2.5, -2.2), (3.4, -3.1), (4.3, -3.5), (5.0, -5.0), (5.0, -11.5), (-2.0, -11.5))
SIDE_FROM_CM = 2.5
FRONT_UP_TO_CM = 7.0
CELLS_PER_CM = 40
FADE_MM = 3.0
RAGGED_MM = 1.0
# The sideburn thins out into the hair instead of ending in a line.
SIDEBURN_FADE_CM = (2.0, 4.5)
# Under the jaw it thins out on the neck: an outline there came out as notches where the mesh folds under the chin.
NECK_FADE_CM = (-8.4, -9.8)
# A moustache parts a little under the nose.
PARTING_CM = 0.35
PARTING_SHARE = 0.35
# The moustache comes down over the top of the lip.
LIP_CLEAR = 0.75
LIP_CLEAR_PX = 1
LIP_SOFT = 1.2
STRAND_PX = 5
STRAND_LIGHT = 0.14
STRAND_SEED = 20261109
MOUTH_LEAST_PX = 2000
MOUTH_MOST_PX = 80000
MOUTH_FROM_M = 0.02
MOUTH_TO_M = 0.06
HAIR_CORE = 0.7


def position_map(data, size):
    """The 3D place of every texel of the head: painted vertex by vertex, the outline came out in the steps of the triangles."""
    points = data["pos"]
    top = points[:, 2].max()
    canvas = np.full((size, size, 3), np.nan, np.float32)
    corners = np.stack([data["uv"][:, :, 0] * size, (1 - data["uv"][:, :, 1]) * size], axis=2)
    for corner, triangle in zip(corners, data["tri"]):
        place = points[triangle]
        if place[:, 2].max() < top - 0.3:
            continue
        low = np.maximum(np.floor(corner.min(axis=0)).astype(int), 0)
        high = np.minimum(np.ceil(corner.max(axis=0)).astype(int) + 1, size)
        if (high <= low).any():
            continue
        v, u = np.mgrid[low[1]:high[1], low[0]:high[0]]
        a, b, c = corner
        area = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
        if abs(area) < 1e-9:
            continue
        w1 = ((u - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (v - a[1])) / area
        w2 = ((b[0] - a[0]) * (v - a[1]) - (u - a[0]) * (b[1] - a[1])) / area
        w0 = 1 - w1 - w2
        inside = (w0 >= -0.02) & (w1 >= -0.02) & (w2 >= -0.02)
        canvas[v[inside], u[inside]] = w0[inside, None] * place[0] + w1[inside, None] * place[1] + w2[inside, None] * place[2]
    return canvas


def inside_mouth(data, size):
    """The texels of the skin island that lines the inside of the mouth, each with how deep it is, 0 at the lips and 1 at the back."""
    place = position_map(data, size)
    tip = nose_tip(data["pos"])
    known = (~np.isnan(place[:, :, 2])).astype(np.uint8)
    count, labels = cv2.connectedComponents(known)
    side, depth, height = np.abs(place[:, :, 0] - tip[0]), place[:, :, 1] - tip[1], place[:, :, 2] - tip[2]
    area = np.zeros((size, size), bool)
    for label in range(1, count):
        island = labels == label
        if MOUTH_LEAST_PX < island.sum() < MOUTH_MOST_PX and np.median(side[island]) < 0.05 and -0.08 < np.median(height[island]) < 0.0 and np.median(depth[island]) > 0.025:
            area |= island
    return area, np.clip((np.nan_to_num(depth) - MOUTH_FROM_M) / (MOUTH_TO_M - MOUTH_FROM_M), 0, 1)


def nose_tip(points):
    head = points[points[:, 2] > points[:, 2].max() - 0.26]
    middle = float(np.median(head[:, 0]))
    tip = head[np.abs(head[:, 0] - middle) < 0.01]
    return tip[tip[:, 1].argmin()]


def signed_mm(outline, x_cm, y_cm):
    """Millimetres inside the outline (negative outside), at each texel."""
    xs, ys = zip(*outline)
    left, bottom = min(xs) - 2.0, min(ys) - 2.0
    width, height = int((max(xs) + 2.0 - left) * CELLS_PER_CM), int((max(ys) + 2.0 - bottom) * CELLS_PER_CM)
    grid = np.zeros((height, width), np.uint8)
    cv2.fillPoly(grid, [np.array([((x - left) * CELLS_PER_CM, (y - bottom) * CELLS_PER_CM) for x, y in outline]).round().astype(np.int32)], 1)
    field = (cv2.distanceTransform(grid, cv2.DIST_L2, 5) - cv2.distanceTransform(1 - grid, cv2.DIST_L2, 5)) * 10.0 / CELLS_PER_CM
    column = np.clip(np.nan_to_num((x_cm - left) * CELLS_PER_CM, nan=0), 0, width - 1).astype(np.int32)
    row = np.clip(np.nan_to_num((y_cm - bottom) * CELLS_PER_CM, nan=0), 0, height - 1).astype(np.int32)
    return field[row, column]


def strands(size):
    """Short hairs hanging down the face, the same for everyone so a rebuild gives the same face."""
    noise = np.random.default_rng(STRAND_SEED).standard_normal((size, size)).astype(np.float32)
    # The head lies on its side in the texture: a hair that hangs down the face runs along the rows.
    kernel = np.ones((1, STRAND_PX), np.float32) / STRAND_PX
    lines = cv2.GaussianBlur(cv2.filter2D(noise, -1, kernel), (0, 0), 0.6)
    return np.clip(lines / max(float(lines.std()), 1e-6) / 2, -1, 1)


def beard_shape(data, lips, size):
    """How much beard each texel has, from 0 to 1."""
    place = position_map(data, size)
    tip = nose_tip(data["pos"])
    side = np.abs(place[:, :, 0] - tip[0]) * 100
    depth = (place[:, :, 1] - tip[1]) * 100
    height = (place[:, :, 2] - tip[2]) * 100
    known = ~np.isnan(height)
    cheek = np.where(side > SIDE_FROM_CM, signed_mm(SIDE_OUTLINE, depth, height), -99.0)
    front = np.where(depth < FRONT_UP_TO_CM, signed_mm(FRONT_OUTLINE, side, height), -99.0)
    inside = np.maximum(cheek, front)
    shape = np.clip((inside + RAGGED_MM * strands(size)) / FADE_MM, 0, 1) * known
    shape *= np.clip((SIDEBURN_FADE_CM[1] - np.nan_to_num(height)) / (SIDEBURN_FADE_CM[1] - SIDEBURN_FADE_CM[0]), 0, 1)
    shape *= np.clip((np.nan_to_num(height) - NECK_FADE_CM[1]) / (NECK_FADE_CM[0] - NECK_FADE_CM[1]), 0, 1)
    parting = np.exp(-((np.nan_to_num(side) / PARTING_CM) ** 2)) * (np.nan_to_num(height) > -3.5)
    shape *= 1 - PARTING_SHARE * parting
    clear = cv2.dilate((lips > LIP_CLEAR).astype(np.uint8), np.ones((2 * LIP_CLEAR_PX + 1, 2 * LIP_CLEAR_PX + 1), np.uint8))
    return shape.astype(np.float32) * (1 - cv2.GaussianBlur(clear.astype(np.float32), (0, 0), LIP_SOFT))


def beard_hair(hair, baked, lips, size):
    """The person's own beard colour, as short hairs: on the new parts of the beard the bake has only skin to go by."""
    core = (baked > HAIR_CORE) & (lips < 0.05)
    colour = np.median(hair[core], axis=0)
    return colour * (1 + STRAND_LIGHT * strands(size))[:, :, None]
