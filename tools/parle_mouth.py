"""
    License information: data/licenses/makehuman_license.txt

    The inside of the mouth is dark. Teeth and tongue were lit like the skin: from the player's distance the mouth showed a bright
    pink tongue, red gums and white molars all at the same brightness, with nothing to say it was a hollow (scoreboard of 2026-11-08,
    lip sync). The shade goes into the colour map, not into vertex colours: the player only reads COLOR_0 on the materials whose
    keyword it turns on, skin and eye, and teeth and tongue are neither.

    Numbers calibrated by eye in the player, no source: a design choice to pilot.
"""
from pathlib import Path

import numpy as np

# Depth back from the front of the teeth (m) -> multiplier. The shade starts just behind the incisors: from 12 mm on the side teeth came out
# brighter than the map already published and floated white in the dark (critic, #2896).
TEETH_DEPTHS = (0.0, 0.004, 0.008, 0.012, 0.016, 0.025, 0.045)
TEETH_SHADES = (1.0, 0.95, 0.82, 0.72, 0.58, 0.4, 0.24)
# Gums are wet flesh behind the lip's shadow. Read off the painted pink in the map, not from the points: the arc behind the upper incisors
# is shallow and stayed bright orange, read as braces (critic, #2896). Pink is red well above blue, in the map's own (sRGB) values.
GUM_SHADE = 0.24
GUM_PINK_FROM = 0.28
GUM_PINK_FULL = 0.42
# The floor is a colour on the wet parts: at luma 48 in the map the inside of the mouth came out black on screen, luma 10 (critic, #2896). It scales
# with the body's own lip, or on dark skin the inside of the mouth came up as bright as the lip (critic, #2896). Teeth keep a plain luma floor, so a
# deep molar does not turn red.
LUMA_FLOOR = 48.0 / 255.0
WINE = np.array((0.46, 0.22, 0.21), np.float32)
# A shaded tooth lit only by the cool room read teal at the corners of the mouth: the deeper, the warmer.
WARM_DEEP = np.array((1.0, 0.9, 0.78), np.float32)
LUMA = np.array((0.2126, 0.7152, 0.0722), np.float32)
LIP_MOVE_SHARE = 0.45
LIP_SHARE = 0.8
# The body's inside of the mouth is never as light as the map it had: on dark skin it was already dark and must stay below it.
BODY_CEILING = 0.85
# A gum is a wide area of the map. The small pink marks painted on the incisors are not, and shaded as gum they read as wine flecks on the teeth.
GUM_OPEN_PX = 10
MARK_INSIDE = 0.5
# The orange-red arc behind the teeth is not gum: it is the body's own inside of the mouth, on the island of the skin map the lips' inner
# band and palate share, lit like the cheek. The same island of the map holds the nape, and how deep the back wall lies changes from body to body;
# a fixed depth darkened the nape or left the back of the mouth orange (critic, #2896). The nape is another piece of the mesh, so the mouth is the
# piece of the island the teeth sit in front of.
BODY_BOX_M = 0.01
BODY_BEHIND_M = 0.001
BODY_ISLAND_U = 0.7
BODY_ISLAND_V = 0.8
TONGUE_FRONT = 0.72
TONGUE_DEEP = 0.52
TONGUE_DARK_M = 0.045
FILL_PX = 8


def clean_teeth(path):
    """The shade multiplies, so it must start from the clean map: `parle_teeth.png` itself has carried a shade since the lash work (b1147bb)."""
    clean = Path(path).with_name("parle_teeth_clean.png")
    return str(clean) if clean.exists() else str(path)


def smoothstep(low, high, value):
    t = np.clip((value - low) / (high - low), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def depth(rest):
    """How far each point lies behind the front of the mesh. The face looks to -y."""
    return rest[:, 1] - rest[:, 1].min()


def teeth_shade(rest):
    """Multiplier of the teeth map at each point."""
    return np.interp(depth(rest), TEETH_DEPTHS, TEETH_SHADES)


def tongue_shade(rest):
    """The whole tongue lies behind the teeth, and its back in the throat."""
    return TONGUE_FRONT + (TONGUE_DEEP - TONGUE_FRONT) * smoothstep(0.0, TONGUE_DARK_M, depth(rest))


def shade_map(width, height, uv_triangles, shade_triangles):
    """Interpolates the points' shade over the UV triangles of a map (row 0 is v = 0). NaN where no triangle falls."""
    canvas = np.full((height, width), np.nan, np.float32)
    size = np.array((width, height))
    for uv, shade in zip(uv_triangles * size, shade_triangles):
        low = np.maximum(np.floor(uv.min(axis=0)).astype(int), 0)
        high = np.minimum(np.ceil(uv.max(axis=0)).astype(int) + 1, size)
        if (high <= low).any():
            continue
        xs, ys = np.meshgrid(np.arange(low[0], high[0]) + 0.5, np.arange(low[1], high[1]) + 0.5)
        ab, ac = uv[1] - uv[0], uv[2] - uv[0]
        area = ab[0] * ac[1] - ab[1] * ac[0]
        if abs(area) < 1e-9:
            continue
        b = ((xs - uv[0, 0]) * ac[1] - (ys - uv[0, 1]) * ac[0]) / area
        c = (ab[0] * (ys - uv[0, 1]) - ab[1] * (xs - uv[0, 0])) / area
        a = 1 - b - c
        inside = (a >= -0.01) & (b >= -0.01) & (c >= -0.01)
        # The UV islands overlap and most texels belong to several triangles: the darkest wins, or a molar kept the shade of the incisor drawn over it.
        region = canvas[low[1]:high[1], low[0]:high[0]]
        region[inside] = np.fmin(region[inside], (a * shade[0] + b * shade[1] + c * shade[2])[inside])
    return canvas


def skin_island(uv):
    """The island of the skin map that holds the inside of the mouth, and the nape. `uv` in glTF convention (v from the top)."""
    return (uv[:, 0] > BODY_ISLAND_U) & (uv[:, 1] > BODY_ISLAND_V)


def connected(count, triangles):
    """A label for each point: points joined through `triangles` share it."""
    label = np.arange(count)
    while True:
        lowest = label[triangles].min(axis=1)
        grown = label.copy()
        for corner in range(3):
            np.minimum.at(grown, triangles[:, corner], lowest)
        if (grown == label).all():
            return label
        label = grown


def body_inside(body, uv, teeth, triangles):
    """Which skin points are the inside of the mouth: in the box of the teeth, behind their front, on the piece of the island that holds the
    largest patch of those points. No depth limit: the piece of the mesh ends where the mouth does."""
    front = teeth[:, 1].min()
    near = (body[:, 0] > teeth[:, 0].min()) & (body[:, 0] < teeth[:, 0].max())
    near &= (body[:, 2] > teeth[:, 2].min() - BODY_BOX_M) & (body[:, 2] < teeth[:, 2].max() + BODY_BOX_M)
    near &= (body[:, 1] > front + BODY_BEHIND_M) & skin_island(uv)
    if not near.any():
        return near
    patches = connected(len(body), triangles[near[triangles].all(axis=1)])
    labels, counts = np.unique(patches[near], return_counts=True)
    seed = np.where(near & (patches == labels[counts.argmax()]))[0][0]
    pieces = connected(len(body), triangles[skin_island(uv)[triangles].all(axis=1)])
    return near & (pieces == pieces[seed])


def body_triangles(triangles, inside, uv):
    """Triangles that take the shade: any corner inside the mouth, none outside its island. With all three corners required, the triangle
    behind the incisors with one point off the box stayed an orange block; the island keeps it from reaching the outer skin."""
    return inside[triangles].any(axis=1) & skin_island(uv)[triangles].all(axis=1)


def grow(canvas, steps=FILL_PX, fill=None):
    """Spreads the shade past the islands' edges: mipmaps read those texels, and a bright or black rim would outline every tooth.
    `fill`: value of whatever stays uncovered; by default the mean, for a map the mesh covers."""
    grown = canvas.copy()
    for _ in range(steps):
        unset = np.isnan(grown)
        if not unset.any():
            break
        around = np.stack([np.roll(grown, 1, 0), np.roll(grown, -1, 0), np.roll(grown, 1, 1), np.roll(grown, -1, 1)])
        known = ~np.isnan(around)
        count = known.sum(axis=0)
        total = np.nan_to_num(around).sum(axis=0)
        reachable = unset & (count > 0)
        grown[reachable] = total[reachable] / count[reachable]
    grown[np.isnan(grown)] = np.nanmean(grown) if fill is None else fill
    return grown


def lip_luma(image, uv, moves):
    """Median luma of the lip in the skin map: the points that move most in the closed-lips visemes. `image` rows from the top, `uv` v from the top."""
    moved = np.maximum.reduce([np.linalg.norm(move, axis=1) for move in moves])
    lip = moved >= moved.max() * LIP_MOVE_SHARE
    height, width = image.shape[:2]
    rows = np.clip((uv[lip, 1] * height).astype(int), 0, height - 1)
    columns = np.clip((uv[lip, 0] * width).astype(int), 0, width - 1)
    return float(np.median(image[rows, columns] @ LUMA))


def wine_for(lip):
    """The wet floor of this body: `WINE`, no lighter than a share of its lip."""
    return WINE * min(1.0, LIP_SHARE * lip / float(WINE @ LUMA))


def wide(mask, radius=GUM_OPEN_PX):
    """`mask` without the parts narrower than the radius (a morphological opening over the four neighbours)."""
    def grown(values, keep):
        around = [np.roll(values, 1, 0), np.roll(values, -1, 0), np.roll(values, 1, 1), np.roll(values, -1, 1)]
        return np.logical_and.reduce([values, *around]) if keep else np.logical_or.reduce([values, *around])

    out = mask.copy()
    for _ in range(radius):
        out = grown(out, True)
    for _ in range(radius):
        out = grown(out, False)
    return out & mask


def gum_of(rgb):
    """How much of each texel is painted gum: `rgb` in 0..1. Only wide areas of pink count."""
    pink = smoothstep(GUM_PINK_FROM, GUM_PINK_FULL, (rgb[:, :, 0] - rgb[:, :, 2]) / np.maximum(rgb[:, :, 0], 1e-3))
    return pink * wide(pink > 0.5).astype(np.float32)


def darken(pixels, shade, gums=False, wine=None, inside=False):
    """`pixels` is the map as rows x columns x RGBA, 0..1, in its own (sRGB) values. `gums`: painted gums go down too.
    `wine`: the floor of the wet texels, a colour. `inside`: the texels with shade under 1 take the colour of the body's inside of the mouth
    (the floor, or the map they had, a little darker, whichever is darker); `shade` only says which texels."""
    out = pixels.copy()
    rgb = out[:, :, :3]
    if inside:
        covered = (shade < 1.0)[:, :, None]
        out[:, :, :3] = np.where(covered, np.minimum(rgb * BODY_CEILING, wine), rgb)
        return out
    gum = gum_of(rgb) if gums else None
    if gums:
        shade = shade * (1 - (1 - GUM_SHADE) * gum)
    luma = rgb @ LUMA
    kept = np.minimum(1.0, LUMA_FLOOR / np.maximum(luma, 1e-3))
    shaded = rgb * np.maximum(shade, kept)[:, :, None]
    if gums:
        shaded = shaded * (1 + (WARM_DEEP - 1) * (1 - np.clip(shade, 0, 1))[:, :, None])
    if wine is not None and gums:
        shaded = shaded + (np.maximum(shaded, wine) - shaded) * gum[:, :, None]
    out[:, :, :3] = shaded
    return out
