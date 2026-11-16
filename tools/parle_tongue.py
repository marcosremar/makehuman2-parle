"""
    License information: data/licenses/makehuman_license.txt

    The tongue rides the jaw. The kit's mouth shapes come from body targets that leave the tongue where it is: at the widest
    opening the lower teeth dropped 17 mm and the tongue tip 0.0 mm, so the tongue hung against the upper teeth in the player
    (Carlos, 2026-11-08). Each mouth shape now carries the tongue by what it does to the lower teeth: most at the tip, little at
    the root, which is fixed in the throat.
"""
import numpy as np

# The tip goes with the lower teeth almost whole. At 0.85, once the cast's mouth opened as the leads' do (6cfa08a, a smaller jaw),
# the tip dropped 7.6 to 7.7 mm at full `jawOpen` on eight of the men, under the 8 mm babylon-cinema asks
# (`test/npcs/cast-tongue-follows-jaw.spec.ts`, 2026-11-12).
TIP_SHARE = 0.95
ROOT_SHARE = 0.25
LOWER_TEETH = 0.5
MIN_CARRY_M = 1e-4
MOUTH_KEYS = ("viseme_", "jawOpen")
# The lower teeth sit a little lower and further back than the upper ones, behind the lower lip: in every open "a" and "o" both
# rows showed, "dente cerrado" where the series the owner chose shows the upper row over a dark mouth (both judges by sight, rounds
# 44 to 51 of 2026-11-16). The upper row stays where the owner had it shown (`parle_face.TEETH_DOWN_M`, 2026-11-08), and the mouth
# opens no wider (the owner kept Rafa's limit on 2026-11-16).
# Round 53: at 1.5 mm both rows still showed ("dentadura"): 3 mm.
LOWER_TEETH_TUCK = (0.0, 0.0012, -0.003)


def tongue_share(rest):
    """How much of the jaw each tongue point takes. The front is -y."""
    front, back = rest[:, 1].min(), rest[:, 1].max()
    return ROOT_SHARE + (TIP_SHARE - ROOT_SHARE) * (back - rest[:, 1]) / max(back - front, 1e-6)


def lower_teeth(jaw_moves):
    """The teeth the jaw carries are the ones that move at least half as far as the farthest."""
    distance = np.linalg.norm(jaw_moves, axis=1)
    return distance >= LOWER_TEETH * distance.max()


def tucked(jaw_moves):
    """The move of each tooth that tucks the lower row behind the lower lip (LOWER_TEETH_TUCK), none on the upper row."""
    return np.where(lower_teeth(jaw_moves)[:, None], np.asarray(LOWER_TEETH_TUCK, float)[None, :], 0.0)


def carried(tongue_rest, teeth_moves, lower):
    """Tongue move for one mouth shape, or None when the shape does not move the jaw."""
    mean = teeth_moves[lower].mean(axis=0)
    if np.linalg.norm(mean) < MIN_CARRY_M:
        return None
    return tongue_share(tongue_rest)[:, None] * mean


def is_mouth_key(name):
    return name.startswith(MOUTH_KEYS)
