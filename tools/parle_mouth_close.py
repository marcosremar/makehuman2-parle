"""
    License information: data/licenses/makehuman_license.txt

    The lips' own closing shape, `mouthClose`, on the outer skin of the Parle cast (2026-11-11).

    The kit has `jawOpen` and the visemes but nothing that closes the lips over an open jaw. The game closed /p b m/ by
    closing the jaw instead (`map/copacabana/npc/npc-jaw-seal.lua` in babylon-cinema), and in the middle of each /m/ a
    thread of tooth still showed between the lips: the seal came from the jaw, not the lips. With `mouthClose` on the
    skin the player keeps the jaw where the voice puts it and writes `mouthClose = jawOpen x seal`
    (`FaceMorphs.ApplyVisemes`), as in the ARKit set it is named after.

    The shape is the jaw's own move on the lower lip, backwards: at the same weight as `jawOpen` it brings the lower lip
    back to where it rests, and the rest already has the lips meeting (`CALM_LOOK` in `parle_face`). It fades over the
    skin under the lip, so the chin and the teeth stay with the jaw. The upper lip does not move: in `jawOpen` it hardly
    does either. Only the outer skin gets it; teeth and tongue have no `mouthClose` and keep following the jaw.

      Inside Blender, after the rest-shape rules: `add_mouth_close(body)` (called by `parle_face.clean_face`).
"""
import numpy as np

# The lower lip and the skin it pulls: the whole move within this of the lip, none past the fade (metres).
FULL_M = 0.004
FADE_M = 0.012
# A vertex the jaw carries at least this far down belongs to the lower face (as in `parle_face.carried_lip`).
CARRIED_M = 0.001
# The lip: what the closed-lip and the kiss visemes move by at least this share of their largest move.
LIP_SHARE = 0.45


def lower_lip(rest, jaw, lip_moves):
    """Indices of the lower lip: moved by the lip visemes and carried down by the jaw."""
    if not lip_moves:
        return np.array([], int)
    moved = np.maximum.reduce(lip_moves)
    return np.where((moved >= moved.max() * LIP_SHARE) & (jaw[:, 2] < -CARRIED_M))[0]


def closing_weight(rest, jaw, lip):
    """1 on the lower lip and the skin within `FULL_M` of it, fading to 0 at `FADE_M`; 0 on what the jaw does not carry."""
    weight = np.zeros(len(rest))
    if len(lip) == 0:
        return weight
    carried = np.where(jaw[:, 2] < -CARRIED_M)[0]
    points = rest[lip]
    for index in carried:
        near = np.min(np.linalg.norm(points - rest[index], axis=1))
        weight[index] = np.clip((FADE_M - near) / (FADE_M - FULL_M), 0.0, 1.0)
    return weight


def mouth_close(rest, jaw, lip_moves):
    """The `mouthClose` move of each vertex: minus the jaw's move, weighted by `closing_weight`."""
    weight = closing_weight(rest, jaw, lower_lip(rest, jaw, lip_moves))
    return -jaw * weight[:, None]


def add_mouth_close(body):
    """Adds (or replaces) `mouthClose` on the body; returns how many vertices it moves."""
    keys = body.data.shape_keys.key_blocks if body.data.shape_keys else None
    if keys is None or "jawOpen" not in keys:
        return 0
    points = lambda block: np.array([point.co[:] for point in block.data])
    rest = points(keys[0])
    jaw = points(keys["jawOpen"]) - rest
    lip_moves = [np.linalg.norm(points(keys[name]) - rest, axis=1) for name in ("viseme_U", "viseme_PP") if name in keys]
    move = mouth_close(rest, jaw, lip_moves)
    if "mouthClose" in keys:
        body.shape_key_remove(keys["mouthClose"])
    block = body.shape_key_add(name="mouthClose", from_mix=False)
    block.data.foreach_set("co", (rest + move).ravel())
    moved = int(np.count_nonzero(np.linalg.norm(move, axis=1) > 1e-6))
    print("BOCA mouthClose:", moved, "vertices")
    return moved
