"""
    License information: data/licenses/makehuman_license.txt

    Where a balding man has no hair. The same rule clears the painted scalp on the skin map and cuts the hair mesh,
    so the two stop at the same line: it drops at the temple, then climbs to the back
    of the crown, leaving hair on the sides and the back.
"""
import numpy as np

HEAD_REACH_M = 0.26
# Fraction of the head from front to back, and how far under the top the hair starts there: steep at the temple so the front corner comes out cut
# on the slant, then climbing to the back of the crown. Each stretch must be less steep than the one before it, or the bare area stops being convex.
ALONG = (0.3, 0.45, 1.0)
DROP_M = (0.13, 0.075, 0.015)


def measures(reference):
    """Top of the head, its front (-y) and its depth, on `reference`."""
    top = float(reference[:, 2].max())
    head = reference[reference[:, 2] > top - HEAD_REACH_M]
    front = float(head[:, 1].min())
    return top, front, max(float(head[:, 1].max()) - front, 1e-6)


def bare_crown(points, reference):
    """True where `points` lie above the balding line of the head measured on `reference`. Front of the head is -y, up is +z."""
    top, front, depth = measures(reference)
    return points[:, 2] > top - np.interp((points[:, 1] - front) / depth, ALONG, DROP_M)


def line_planes(reference):
    """The planes (point, normal) the line is made of, one per stretch."""
    top, front, depth = measures(reference)
    knots = [(0.0, front + share * depth, top - drop) for share, drop in zip(ALONG, DROP_M)]
    return [(here, (0.0, -(there[2] - here[2]), there[1] - here[1])) for here, there in zip(knots, knots[1:])]
