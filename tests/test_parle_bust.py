"""Regression for the bust under a top (2026-11-11): run with `.local/venv/bin/python tests/test_parle_bust.py`.

The breast of the body comes to a point, and a T-shirt fitted over it ended in a cone (owner: "o bico do peito da mulher está muito pontudo
na roupa", "com roupa deve parecer que está com sutiã"). Under a top each breast takes the shape of a cup: the tip round, the sides full,
the rim where it was, and the skin under the cloth.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import parle_bust  # noqa: E402

# A chest: a wall at y = 0 with a pointed breast on each side, 4.5 cm tall and 5.7 cm wide at the rim, the face looking to -y.
HEIGHT, REACH, TALL = 1.62, 0.035 * 1.62, 0.045
HIP, SHOULDER, MIDDLE, JOINT = 0.85, 1.32, 0.0, 0.14
centre = np.array([0.085, 0.0, 1.20])
grid = np.stack(np.meshgrid(np.linspace(-0.2, 0.2, 161), np.linspace(1.0, 1.4, 161)), axis=-1).reshape(-1, 2)


def chest(lift=0.0):
    """The cloth (or, `lift` lower, the skin under it) as points: a cone on each side of the middle."""
    out = np.zeros((len(grid), 3))
    out[:, 0], out[:, 2] = grid[:, 0], grid[:, 1]
    for side in (1, -1):
        away = np.hypot(grid[:, 0] - side * centre[0], grid[:, 1] - centre[2])
        out[:, 1] = np.minimum(out[:, 1], -TALL * np.clip(1 - away / REACH, 0, None))
    out[:, 1] += lift
    return out


cloth, skin = chest(), chest(0.004)
tip = parle_bust.tip_of(cloth, HIP, SHOULDER, MIDDLE, JOINT, 1)
assert np.linalg.norm(tip - (centre + (0, -TALL, 0))) < 0.004, f"the tip of the breast is not the foremost point of the top on that side: {tip}"
assert parle_bust.tip_of(cloth, HIP, SHOULDER, MIDDLE, JOINT, -1)[0] < 0, "the other side has the same tip"
assert parle_bust.tip_of(cloth[cloth[:, 2] < 1.05], HIP, SHOULDER, MIDDLE, JOINT, 1) is None, "a top that does not cover the bust has a tip"
axis, tall = parle_bust.rim_of(cloth, tip, REACH)
assert axis @ (0, -1, 0) > 0.99 and abs(tall - TALL) < 0.006, f"the axis does not point out of the body, or the tip is not that far over the rim: {axis}, {tall}"

moved = cloth + parle_bust.cupped(cloth, tip, axis, tall, REACH)
at_tip = np.linalg.norm(cloth - tip, axis=1).argmin()
assert abs((moved[at_tip, 1] - cloth[at_tip, 1]) - (1 - parle_bust.CUP) * tall) < 1e-4, f"the tip did not come back by its share of the height: {moved[at_tip] - cloth[at_tip]}"
assert parle_bust.CUP <= 0.75 and parle_bust.ROUND < 1, "the bust under a top is pointed again: a flat tip on straight sides was still a cone (owner, 2026-11-11)"
away = np.hypot(cloth[:, 0] - centre[0], cloth[:, 2] - centre[2])
assert np.abs(moved[away >= REACH] - cloth[away >= REACH]).max() < 1e-9, "the rim of the breast, the wall or the other breast moved"
# Seen from the side the tip was a cone; now it is round: a centimetre from the axis the cloth stands nearly as far out as at the tip.
ring = np.abs(away - 0.01) < 0.002
before, after = cloth[ring, 1].mean() - cloth[at_tip, 1], moved[ring, 1].mean() - moved[at_tip, 1]
assert before > 0.007 and after < 0.5 * before, f"the tip is as pointed as it was: {before * 1000:.1f} mm -> {after * 1000:.1f} mm over a centimetre"
# And the sides are full: half way from the rim to the axis the cloth does not come back with the tip, it goes out, and nearer the rim
# it goes out more than it did when the cup was a plain quarter of a sine wave.
half = np.abs(away - REACH / 2) < 0.002
assert (cloth[half, 1] - moved[half, 1]).mean() > 0.001, f"the sides of the cup are not fuller: {(cloth[half, 1] - moved[half, 1]).mean() * 1000:.1f} mm"
low = np.abs(away - 0.8 * REACH) < 0.002
assert (cloth[low, 1] - moved[low, 1]).mean() > 0.002, f"near the rim the sides still fall straight: {(cloth[low, 1] - moved[low, 1]).mean() * 1000:.1f} mm"
# The skin goes by the same rule and stays under the cloth, and no point passes another on its way.
under = skin + parle_bust.cupped(skin, tip, axis, tall, REACH)
assert (under[:, 1] - moved[:, 1]).min() > 0.0005, f"the skin came through the cloth: {(under[:, 1] - moved[:, 1]).min() * 1000:.2f} mm"
line = np.stack([np.full(50, tip[0]), np.linspace(tip[1], tip[1] + 0.06, 50), np.full(50, tip[2])], axis=1)
went = line + parle_bust.cupped(line, tip, axis, tall, REACH)
assert np.all(np.diff(went[:, 1]) > 0), "a point behind another came out in front of it"
# A flat chest has nothing to round.
flat = np.zeros((len(grid), 3))
flat[:, 0], flat[:, 2] = grid[:, 0], grid[:, 1]
found = parle_bust.rim_of(flat, parle_bust.tip_of(flat, HIP, SHOULDER, MIDDLE, JOINT, 1), REACH)
assert found is None or found[1] < parle_bust.TALL_LEAST_M, "a flat chest passed for a breast"
# The cast's own step shapes the bust, on the women alone and on the top and the skin together: Blender is not here, the source is read.
polish = (Path(__file__).parent.parent / "tools" / "parle_polish.py").read_text()
assert 'cups = round_bust(meshes, rig) if sex == "female" else []' in polish, "the body's step no longer shapes the bust of the women, or shapes the men's"
assert "parle_bust.tip_of(cloth" in polish and "parle_bust.cupped(points" in polish, "the tip is no longer read on the cloth, or the cup is not the rule's"
print("ok")
