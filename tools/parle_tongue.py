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
# Round 54: the old woman's "a" still showed both rows ("dentes de baixo 50 % cobertos pelo lábio"): 4 mm.
# Round 88 (four judges of four on round 86: "sem a fileira de baixo", "dentadura"): 6 mm down and 2 mm back.
LOWER_TEETH_TUCK = (0.0, 0.002, -0.006)
# The smile parts the lips with no shape of its own on the teeth, and the lower row showed under the upper one (round 86, all eight):
# a smile shape on the teeth that takes the lower row down and back by SMILE_LOWER_TUCK, the upper row where it is.
SMILE_TEETH_KEYS = ("mouthSmileLeft", "mouthSmileRight")
SMILE_LOWER_TUCK = (0.0, 0.0015, -0.0035)

# Round 70 (owner, 2026-11-17: the mouth first of what is ugly; both judges, rounds 66 to 69: the "o" "um bico em anel com os dentes à
# mostra"): in the rounded shapes the series shows a small dark ring and no teeth. In those shapes the upper row goes up behind
# the upper lip, back to where the kit had it (it sits `parle_face.TEETH_DOWN_M` lower to show in the "a"), and the lower row down
# behind the lower lip.
ROUND_KEYS = ("viseme_O", "viseme_U")
ROUND_UPPER_UP_M = 0.0035
ROUND_LOWER_DOWN_M = 0.002


def smile_tuck(jaw_moves, share=1.0):
    """The move of each tooth in a smile shape (SMILE_TEETH_KEYS): the lower row down and back by `share` of SMILE_LOWER_TUCK, the
    upper row still. One side of the smile takes half (`share` 0.5): both sides at once take it whole."""
    lower = lower_teeth(jaw_moves)
    return np.where(lower[:, None], np.asarray(SMILE_LOWER_TUCK, float) * share, 0.0)


def hidden_in_round(jaw_moves):
    """The move of each tooth in a rounded shape (ROUND_KEYS) that hides both rows behind the lips: the upper up, the lower down."""
    lower = lower_teeth(jaw_moves)
    move = np.zeros((len(lower), 3))
    move[:, 2] = np.where(lower, -ROUND_LOWER_DOWN_M, ROUND_UPPER_UP_M)
    return move


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
