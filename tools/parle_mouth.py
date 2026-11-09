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
# Gums are wet flesh behind the lip's shadow; at the teeth's brightness the open mouth showed a red band.
GUM_SHADE = 0.6
GUM_FROM_TIP_M = 0.0095
GUM_FADE_M = 0.003
TONGUE_FRONT = 0.62
TONGUE_DEEP = 0.3
TONGUE_DARK_M = 0.045
FILL_PX = 8
# The upper incisors show under the lip on an open "a": at the player's widest (`viseme_aa` 0.55 in LipSync.MaxAmplitude and what
# JawRoom leaves of `jawOpen`) a few bodies had the tip at the lip's edge, or behind it, and the open mouth showed only lower teeth.
AA_WEIGHT = 0.55
AA_JAW = 0.205
SHOW_MIN_M = 0.001
SHOW_TARGET_M = 0.002
INCISOR_HALF_M = 0.008
INCISOR_FRONT_M = 0.004
LIP_COLUMN_M = 0.003
LIP_JAW_SHARE = 0.3
LIP_REACH_M = 0.008


def smoothstep(low, high, value):
    t = np.clip((value - low) / (high - low), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def depth(rest):
    """How far each point lies behind the front of the mesh. The face looks to -y."""
    return rest[:, 1] - rest[:, 1].min()


def gum_share(rest, lower):
    """How much of each point is gum: more than a crown away from the tip of its row, as in `teeth_gum.py`."""
    tip_up = np.percentile(rest[~lower, 2], 2)
    top_low = np.percentile(rest[lower, 2], 98)
    away = np.where(lower, top_low - rest[:, 2], rest[:, 2] - tip_up)
    return smoothstep(GUM_FROM_TIP_M, GUM_FROM_TIP_M + GUM_FADE_M, away)


def teeth_shade(rest, lower):
    """Multiplier of the teeth map at each point; `lower` are the points the jaw carries."""
    deep = 1 - (1 - TEETH_DEEP) * smoothstep(TEETH_LIT_M, TEETH_DARK_M, depth(rest))
    return deep * (1 - (1 - GUM_SHADE) * gum_share(rest, lower))


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
        canvas[low[1]:high[1], low[0]:high[0]][inside] = (a * shade[0] + b * shade[1] + c * shade[2])[inside]
    return canvas


def grow(canvas, steps=FILL_PX):
    """Spreads the shade past the islands' edges: mipmaps read those texels, and a bright or black rim would outline every tooth."""
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
    grown[np.isnan(grown)] = np.nanmean(grown)
    return grown


def darken(pixels, shade):
    """`pixels` is the map as rows x columns x RGBA in its own (sRGB) values."""
    out = pixels.copy()
    out[:, :, :3] = out[:, :, :3] * shade[:, :, None]
    return out


def posed(rest, aa_moves, jaw_moves):
    return rest + aa_moves * AA_WEIGHT + jaw_moves * AA_JAW


def incisors(rest, lower):
    """Front points of the upper arch, around the mouth's middle (x), in the rest pose."""
    front = rest[:, 1].min()
    middle = np.median(rest[:, 0])
    return (~lower) & (rest[:, 1] < front + INCISOR_FRONT_M) & (np.abs(rest[:, 0] - middle) < INCISOR_HALF_M)


def incisor_show(teeth, teeth_aa, teeth_jaw, lower, body, body_aa, body_jaw):
    """Millimetres of upper incisor below the lip's edge on the widest "a", in metres; or None without a lip to measure."""
    tip = posed(teeth, teeth_aa, teeth_jaw)[incisors(teeth, lower), 2].min()
    middle = np.median(teeth[incisors(teeth, lower), 0])
    front = teeth[:, 1].min()
    pose = posed(body, body_aa, body_jaw)
    jaw_move = np.linalg.norm(body_jaw, axis=1)
    near = (np.abs(body[:, 0] - middle) < LIP_COLUMN_M) & (body[:, 1] < front) & (np.abs(pose[:, 2] - tip) < LIP_REACH_M)
    if not near.any():
        return None
    still = near & (jaw_move < jaw_move.max() * LIP_JAW_SHARE)
    return float(pose[still, 2].min() - tip) if still.any() else None


def incisor_drop(show):
    """How much lower the upper arch must sit on the widest "a"; zero when the tooth already shows."""
    return 0.0 if show is None or show >= SHOW_MIN_M else SHOW_TARGET_M - show
