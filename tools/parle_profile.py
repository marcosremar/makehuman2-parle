"""
    License information: data/licenses/makehuman_license.txt

    The lower face of a body seen from the side: nose, lips and chin, read on its silhouette.

    Owner, 2026-11-11: the faces of the cast were still "muito feio". From the front the women's mouth read as a lower lip alone; from
    the side the reason showed: the lower lip stood out in front of the upper one like a shelf, and the mouth sat right under the nose.
    On a face at rest the upper lip leads: it stands some 3 mm ahead of the line from under the nose to the chin and the lower lip some
    2 mm (Burstone 1967, "Lip posture and its significance in treatment planning", American Journal of Orthodontics 53(4):262-284, DOI
    10.1016/0002-9416(67)90022-X; Legan & Burstone 1980, "Soft tissue cephalometric analysis for orthognathic surgery", Journal of Oral
    Surgery 38(10):744-751), and both lie behind the line from the tip of the nose to the chin, the lower one nearer to it (Ricketts
    1968, "Esthetics, environment, and the law of lip relation", American Journal of Orthodontics 54(4):272-289, DOI
    10.1016/S0002-9416(68)90278-9). Norms of real faces, in millimetres, taken from the standard analyses and not read in full.

    The cast is drawn, so what it aims at is read the same way on the two faces the owner gave as the aim and on four faces of the
    game (2026-11-11): as a share of the height from the tip of the nose to where the lips meet, the upper lip leads the lower one by
    0.07 (Rafa) to 0.11 (Sofia), 2.6 to 4 mm. On the five people of the cast measured that day the lower lip led, by up to 0.16, and
    one had no upper lip in profile at all. See `AIM`. How far under the nose the mouth sits is given too, but not marked: as a share
    of that same height it tells a mouth ridden up from one in its place poorly, and from the front the length of the chin tells it.

    The silhouette is any picture of the head from the side with the face looking to the left, as a table of true where there is body.
    Numpy alone.
"""
import numpy as np

# Where on the picture the tip of the nose is looked for, as shares of its height, and how far down each next turn may be.
NOSE_BAND = (0.42, 0.8)
NEXT_WITHIN = 0.22
# A turn counts once the edge has come back this many pixels from it; the edge is smoothed over this many rows first.
TURN_PX = 0.8
SMOOTH_PX = 2.0
MARKS = ("tip", "under", "upper", "meet", "lower", "hollow", "chin")
# What the cast aims at, by sex, and how far from it scores nothing (`parle_beauty.mark`, way `up`: full from the aim to a reach past it).
AIM = {"female": {"upper_leads": (0.095, 0.12, "up")}, "male": {"upper_leads": (0.07, 0.12, "up")}}
NAMES = {"upper_leads": "lábio de cima à frente do de baixo (perfil)"}


def edge_of(solid):
    """For each row of a silhouette whose face looks to the left, the first column that is body, smoothed down the rows; infinite
    where the row is empty."""
    solid = np.asarray(solid, bool)
    first = np.where(solid.any(axis=1), solid.argmax(axis=1), -1).astype(float)
    reach = int(np.ceil(SMOOTH_PX * 3))
    weights = np.exp(-0.5 * (np.arange(-reach, reach + 1) / SMOOTH_PX) ** 2)
    filled = np.where(first < 0, np.nan, first)
    padded = np.pad(filled, reach, mode="edge")
    stack = np.stack([padded[index:index + len(first)] for index in range(2 * reach + 1)])
    seen = ~np.isnan(stack)
    smooth = (np.where(seen, stack, 0) * weights[:, None]).sum(axis=0) / np.maximum((seen * weights[:, None]).sum(axis=0), 1e-9)
    return np.where(first < 0, np.inf, smooth)


def marks(edge):
    """The rows of the seven turns of the lower face, from the top: the tip of the nose, under the nose, the upper lip, where the lips
    meet, the lower lip, the hollow under it and the chin. None when the edge does not turn seven times: a face whose lower lip covers
    the upper one has no upper lip in profile."""
    high = len(edge)
    rows = np.arange(high)
    band = (rows > NOSE_BAND[0] * high) & (rows < NOSE_BAND[1] * high) & np.isfinite(edge)
    if not band.any():
        return None
    found = [int(rows[band][edge[band].argmin()])]
    inward = True
    while len(found) < len(MARKS):
        row = best = found[-1]
        for step in range(row + 1, min(high, row + int(NEXT_WITHIN * high))):
            if not np.isfinite(edge[step]):
                break
            if (edge[step] > edge[best]) if inward else (edge[step] < edge[best]):
                best = step
            elif abs(edge[step] - edge[best]) > TURN_PX and step - best > 2:
                break
        if best == row:
            return None
        found.append(best)
        inward = not inward
    return dict(zip(MARKS, found))


def measures(edge, found):
    """The measures of a profile, each a share of the height from the tip of the nose to where the lips meet: how far the upper lip
    stands ahead of the lower one (less than nothing when the lower lip leads), how far under the nose's base the mouth sits, and how
    far each lip stands ahead of the line from under the nose to the chin."""
    at = lambda name: np.array([edge[found[name]], found[name]], float)
    tall = float(found["meet"] - found["tip"])
    under, chin = at("under"), at("chin")
    line = chin - under
    out = np.array([-line[1], line[0]]) / np.linalg.norm(line)
    out = out if out[0] < 0 else -out
    return {
        "upper_leads": float(at("lower")[0] - at("upper")[0]) / tall,
        "mouth_low": float(found["meet"] - found["under"]) / tall,
        "upper_ahead": float((at("upper") - under) @ out) / tall,
        "lower_ahead": float((at("lower") - under) @ out) / tall,
        "chin_back": float(chin[0] - at("lower")[0]) / tall,
    }


def read(solid):
    """The measures of the profile on a silhouette, or None when the lower face has not its seven turns."""
    edge = edge_of(solid)
    found = marks(edge)
    return None if found is None else dict(measures(edge, found), marks=found)
