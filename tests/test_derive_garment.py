"""The clothes of who serves at a counter, cut by `tools/derive_garment.py` (gauntlet of the shop interiors, 2026-11-17): run with
`.local/venv/bin/python tests/test_derive_garment.py`. Needs the MakeHuman assets in `.local/mhd/data/clothes/hm08`.

- The apron keeps the bib, two straps on the back, the waist band and the front of the skirt. Cut from a woman's dress, its bib kept
  the dress's cups and the baker took two mounds on his chest: from the hips up it lies at a fixed gap over the body it is bound to.
- The jacket is the biggest piece of a suit: no skirt, blouse or buttons, and it keeps both sleeves (the pharmacist's white coat).
- The badge is a small patch on the left of the chest, in front.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import derive_garment as derive  # noqa: E402

CLOTHES = ROOT / ".local" / "mhd" / "data" / "clothes" / "hm08"
bones = derive.Bones(str(ROOT / "data"))
dm = derive.DECIMETRES


def island_count(garment):
    owner = list(range(len(garment.points)))

    def find(item):
        while owner[item] != item:
            owner[item] = owner[owner[item]]
            item = owner[item]
        return item

    for face in garment.faces:
        for corner in face[1:]:
            owner[find(corner[0])] = find(face[0][0])
    return len({find(face[0][0]) for face in garment.faces})


def used(garment):
    return sorted({corner[0] for face in garment.faces for corner in face})


# The apron.
apron = derive.Garment(str(CLOTHES / "toigo_shift_dress"), "toigo_shift_dress")
before = len(apron.faces)
derive.apron(apron, bones)
assert 1000 < len(apron.faces) < before, len(apron.faces)
low, high = bones.head("spine03")[1], bones.head("spine02")[1]
waist = low + (high - low) * derive.APRON_WAIST_SHARE
band = derive.APRON_BAND_M * dm
over = derive.APRON_OVER_M * dm
for face in apron.faces:
    corners = apron.points[[corner[0] for corner in face]]
    middle = corners.mean(axis=0)
    if (corners[:, 1] < waist - band - over).all():
        # The skirt keeps the front only (the cut moved with the cloth by at most the gap over the shirt).
        assert (corners[:, 2] >= derive.APRON_SIDE_M * dm - 2 * over).all(), corners
    elif (corners[:, 1] > waist + band + over).all() and middle[2] >= derive.APRON_FRONT_M * dm + over:
        assert (np.abs(corners[:, 0]) <= derive.APRON_BIB_HALF_M * dm + 2 * over).all(), corners
# No cups: from the hips up no point stands off the body more than the gap and the lift over the shirt, as the dress's cups did.
hip = bones.head("upperleg01.L")[1] + derive.APRON_BIB_FADE_M * dm
expected = derive.APRON_BIB_GAP_M * dm


def lifts(garment):
    points = [index for index in used(garment) if garment.points[index, 1] > hip + 2 * over]
    return (garment.offsets[points] * derive.outward(garment, bones)[points]).sum(axis=1)


dress = derive.Garment(str(CLOTHES / "toigo_shift_dress"), "toigo_shift_dress")
assert np.percentile(lifts(dress), 99) > expected + 2 * over, "the dress has cups to take away"
lift = lifts(apron)
assert len(lift) > 500, len(lift)
assert np.percentile(lift, 99) < expected + 1.5 * over, np.percentile(lift, 99)
assert np.median(lift) > expected + 0.5 * over, np.median(lift)

# The jacket.
jacket = derive.Garment(str(CLOTHES / "toigo_female_suit"), "toigo_female_suit")
derive.jacket(jacket)
assert island_count(jacket) == 1
reach = jacket.points[used(jacket)]
assert np.ptp(reach[:, 0]) > 6.0, np.ptp(reach[:, 0])
assert reach[:, 1].max() > bones.head("spine01")[1] and reach[:, 1].min() < bones.head("spine04")[1], (reach[:, 1].min(), reach[:, 1].max())

# The badge.
badge = derive.Garment(str(CLOTHES / "toigo_shift_dress"), "toigo_shift_dress")
derive.badge(badge, bones)
patch = badge.points[used(badge)]
half_x, half_y = (item * dm for item in derive.BADGE_HALF_M)
assert len(badge.faces) >= 20, len(badge.faces)
# The lift over the clothes spreads the patch a little on the curve of the chest.
spread = derive.BADGE_OVER_M * dm
assert np.ptp(patch[:, 0]) <= 2 * half_x + spread and np.ptp(patch[:, 1]) <= 2 * half_y + spread, (np.ptp(patch[:, 0]), np.ptp(patch[:, 1]))
assert patch[:, 0].mean() > 0 and (patch[:, 2] > 0).all()
print("ok")
