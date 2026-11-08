"""
    License information: data/licenses/makehuman_license.txt

    The tongue rides the jaw. The kit's mouth shapes come from body targets that leave the tongue where it is: at the widest
    opening the lower teeth dropped 17 mm and the tongue tip 0.0 mm, so the tongue hung against the upper teeth in the player
    (Carlos, 2026-11-08). Each mouth shape now carries the tongue by what it does to the lower teeth: most at the tip, little at
    the root, which is fixed in the throat.
"""
import numpy as np

TIP_SHARE = 0.85
ROOT_SHARE = 0.25
LOWER_TEETH = 0.5
MIN_CARRY_M = 1e-4
MOUTH_KEYS = ("viseme_", "jawOpen")


def tongue_share(rest):
    """How much of the jaw each tongue point takes. The front is -y."""
    front, back = rest[:, 1].min(), rest[:, 1].max()
    return ROOT_SHARE + (TIP_SHARE - ROOT_SHARE) * (back - rest[:, 1]) / max(back - front, 1e-6)


def lower_teeth(jaw_moves):
    """The teeth the jaw carries are the ones that move at least half as far as the farthest."""
    distance = np.linalg.norm(jaw_moves, axis=1)
    return distance >= LOWER_TEETH * distance.max()


def carried(tongue_rest, teeth_moves, lower):
    """Tongue move for one mouth shape, or None when the shape does not move the jaw."""
    mean = teeth_moves[lower].mean(axis=0)
    if np.linalg.norm(mean) < MIN_CARRY_M:
        return None
    return tongue_share(tongue_rest)[:, None] * mean


def is_mouth_key(name):
    return name.startswith(MOUTH_KEYS)
