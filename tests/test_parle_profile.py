"""Regression for the mouth of the cast seen from the side (2026-11-11): run with `.local/venv/bin/python tests/test_parle_profile.py`.

The owner found the faces "muito feio". The jaw rested a quarter closed and carried the lower lip up over the upper one and out in front of
it: 0.4 to 3.8 mm ahead on the five people measured, where on Sofia, Rafa and the faces of the game the upper lip leads by 2.6 to 4 mm.
`parle_profile` reads nose, lips and chin on the silhouette of a head from the side.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import parle_beauty  # noqa: E402
import parle_profile  # noqa: E402


def head(upper=20.0, lower=12.0, mouth=70.0):
    """The silhouette of a lower face 600 px high, looking left: the tip of the nose at row 300, the nose's base 40 rows under it, the
    lips meeting `mouth` rows under the tip, each lip a bump standing `upper` and `lower` px out of the face, a chin under them."""
    rows = np.arange(600, dtype=float)
    bump = lambda at, wide, tall: tall * np.exp(-0.5 * ((rows - at) / wide) ** 2)
    front = 300 - bump(300, 14, 80) - bump(300 + mouth - 14, 7, upper) - bump(300 + mouth + 16, 8, lower) - bump(300 + mouth + 75, 18, 14)
    return np.arange(700)[None, :] >= front[:, None]


found = parle_profile.read(head())
assert found is not None and list(found["marks"]) == list(parle_profile.MARKS), f"the seven turns of the lower face are not found: {found}"
assert abs(found["marks"]["tip"] - 300) <= 2 and abs(found["marks"]["meet"] - 370) <= 4, f"the tip of the nose or the meeting of the lips is elsewhere: {found['marks']}"
assert 0.06 < found["upper_leads"] < 0.16, f"an upper lip 8 px ahead of the lower one over 70 px of height: {found['upper_leads']}"
assert 0.3 < found["mouth_low"] < 0.7 and found["upper_ahead"] > found["lower_ahead"], f"where the mouth sits, or which lip stands further from the line to the chin: {found}"
# The same face twice as big reads the same: the measures are shares of the face.
big = parle_profile.read(np.kron(head(), np.ones((2, 2), bool)))
assert big is not None and abs(big["upper_leads"] - found["upper_leads"]) < 0.02 and abs(big["mouth_low"] - found["mouth_low"]) < 0.03, f"the size of the picture counts: {big}"
# The defect: the lower lip out in front of the upper one reads as less than nothing, and scores nothing against the aim.
shelf = parle_profile.read(head(upper=10.0, lower=22.0))
assert shelf is not None and shelf["upper_leads"] < -0.05, f"a lower lip 12 px ahead of the upper one is not read as leading: {shelf}"
for sex, aim in parle_profile.AIM.items():
    assert parle_beauty.mark(found["upper_leads"], *aim["upper_leads"]) == 1, f"an upper lip that leads loses marks on the {sex} aim"
    assert parle_beauty.mark(shelf["upper_leads"], *aim["upper_leads"]) == 0, f"a lower lip like a shelf scores on the {sex} aim"
    assert set(aim) <= set(parle_profile.NAMES)
# A lower lip that covers the upper one leaves no upper lip in profile: no measure, not a wrong one.
assert parle_profile.read(head(upper=0.0, lower=30.0)) is None, "a face with one lip in profile is measured as if it had two"
assert parle_profile.read(np.zeros((600, 700), bool)) is None, "an empty picture has a profile"
print("ok")
