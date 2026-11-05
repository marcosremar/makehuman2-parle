"""
    License information: data/licenses/makehuman_license.txt

    A defined male chest for makehuman2-parle.

    A sculpted delta left the muscled body with a chest hanging like a shelf. Pushing or smoothing that
    surface by hand gave a breast or a dented torso, so this takes the delta out of the chest and uses
    the artist-made pectoral target MakeHuman ships instead, which already follows the anatomy: a thin
    box tied to the clavicle with a level lower border (Proko, How to Draw Pecs).

    Works in metres with Z up, on the base mesh, and hands the result to the rigged body by position.
"""
import numpy as np
from scipy.spatial import cKDTree

PECTORAL = "data/target/hm08/torso/torso-muscle-pectoral-incr.target"
DECIMETRES = 10.0


def smoothstep(value):
    value = np.clip(value, 0, 1)
    return value * value * (3 - 2 * value)


def read_target(path, count):
    shift = np.zeros((count, 3))
    with open(path) as lines:
        for line in lines:
            if line.startswith("#") or not line.strip():
                continue
            index, x, y, z = line.split()[:4]
            shift[int(index)] = (float(x), -float(z), float(y))
    return shift / DECIMETRES


def chest_shift(body, base, sculpt, joints, strength):
    """`base` is the body on the base mesh and `sculpt` the delta it carries; `body` is the rigged mesh, in the same place."""
    left, right = joints["LeftArm"], joints["RightArm"]
    centre = (left[0] + right[0]) / 2
    shoulder = abs(left[0] - right[0]) / 2
    across = np.abs(base[:, 0] - centre) / shoulder
    down = (left[2] - base[:, 2]) / shoulder
    # The whole chest and the top of the belly, fading out before the arm, the neck and the abs.
    area = smoothstep((1.45 - across) / 0.35) * smoothstep((down + 0.45) / 0.3) * smoothstep((1.2 - down) / 0.3)
    move = -sculpt * area[:, None] + read_target(PECTORAL, len(base)) * strength
    _, near = cKDTree(base).query(body)
    return move[near], move
