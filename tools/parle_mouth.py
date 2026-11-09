"""
    License information: data/licenses/makehuman_license.txt

    The inside of the mouth is dark. Teeth and tongue were lit like the skin: from the player's distance the mouth showed a bright
    pink tongue, red gums and white molars all at the same brightness, with nothing to say it was a hollow (scoreboard of 2026-11-08,
    lip sync). The shade goes into the colour map, not into vertex colours: the player only reads COLOR_0 on the materials whose
    keyword it turns on, skin and eye, and teeth and tongue are neither.

    Numbers calibrated by eye in the player, no source: a design choice to pilot.
"""
import numpy as np

# Depth is measured back from the front of the teeth: the incisors stay lit, and by the molars the mouth is almost closed off.
TEETH_LIT_M = 0.012
TEETH_DARK_M = 0.045
TEETH_DEEP = 0.32
# Gums are wet flesh behind the lip's shadow. Read off the painted pink in the map, not from the points: the arc behind the upper incisors
# is shallow and stayed bright orange, read as braces (critic, #2896). Pink is red well above blue, in the map's own (sRGB) values.
GUM_SHADE = 0.24
GUM_PINK_FROM = 0.28
GUM_PINK_FULL = 0.42
# The black painted between the teeth (luma 33) went to luma 9 and read as missing teeth at the player's size (critic, #2896): shade
# never takes a texel below this luma, and a texel darker than it is left alone.
LUMA_FLOOR = 48.0 / 255.0
LUMA = np.array((0.2126, 0.7152, 0.0722), np.float32)
# The orange-red arc behind the teeth is not gum: it is the body's own inside of the mouth, on the island of the skin map the lips' inner
# band and palate share, lit like the cheek. Those points are behind the front of the teeth and inside their extent.
BODY_INSIDE = 0.3
BODY_BOX_M = 0.01
BODY_BEHIND_M = 0.001
BODY_ISLAND_U = 0.7
BODY_ISLAND_V = 0.8
TONGUE_FRONT = 0.62
TONGUE_DEEP = 0.3
TONGUE_DARK_M = 0.045
FILL_PX = 8


def smoothstep(low, high, value):
    t = np.clip((value - low) / (high - low), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def depth(rest):
    """How far each point lies behind the front of the mesh. The face looks to -y."""
    return rest[:, 1] - rest[:, 1].min()


def teeth_shade(rest):
    """Multiplier of the teeth map at each point."""
    return 1 - (1 - TEETH_DEEP) * smoothstep(TEETH_LIT_M, TEETH_DARK_M, depth(rest))


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


def body_inside(body, uv, teeth):
    """Which skin points are the inside of the mouth. `uv` in glTF convention (v from the top); a point must be in the island and in the box."""
    near = (body[:, 0] > teeth[:, 0].min()) & (body[:, 0] < teeth[:, 0].max())
    near &= (body[:, 2] > teeth[:, 2].min() - BODY_BOX_M) & (body[:, 2] < teeth[:, 2].max() + BODY_BOX_M)
    near &= body[:, 1] > teeth[:, 1].min() + BODY_BEHIND_M
    return near & (uv[:, 0] > BODY_ISLAND_U) & (uv[:, 1] > BODY_ISLAND_V)


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


def gum_of(rgb):
    """How much of each texel is painted gum: `rgb` in 0..1."""
    return smoothstep(GUM_PINK_FROM, GUM_PINK_FULL, (rgb[:, :, 0] - rgb[:, :, 2]) / np.maximum(rgb[:, :, 0], 1e-3))


def darken(pixels, shade, gums=False):
    """`pixels` is the map as rows x columns x RGBA, 0..1, in its own (sRGB) values. `gums`: the map has painted gums to take down too."""
    out = pixels.copy()
    rgb = out[:, :, :3]
    if gums:
        shade = shade * (1 - (1 - GUM_SHADE) * gum_of(rgb))
    luma = rgb @ LUMA
    kept = np.minimum(1.0, LUMA_FLOOR / np.maximum(luma, 1e-3))
    out[:, :, :3] = rgb * np.maximum(shade, kept)[:, :, None]
    return out
