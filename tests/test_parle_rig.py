"""Regression for the claw hand of the cast (2026-11-08): run with Blender's Python, `blender -b --python-exit-code 1 -P tests/test_parle_rig.py`."""
import sys
from pathlib import Path

from mathutils import Vector

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
from parle_rig import rest_turn  # noqa: E402

# The game's index finger rests straight along the hand; ours rests half curled, 30 degrees down.
game = {f"LeftHandIndex{joint}": Vector((1, 0, 0)) for joint in (1, 2, 3)}
ours = {f"LeftHandIndex{joint}": Vector((1, 0, -0.58)).normalized() for joint in (1, 2, 3)}
for joint in (1, 2, 3):
    name = f"LeftHandIndex{joint}"
    turn = rest_turn(name, game, ours)
    assert turn is not None, f"{name} keeps its curled rest, and the clip's curl lands on top of it"
    assert (turn @ ours[name] - game[name]).length < 1e-5, name
# The tip has no child, so no direction of its own: it follows the phalanx before it.
assert rest_turn("LeftHandIndex4", game, ours) is None
# The hand and the head keep their own rest.
assert rest_turn("LeftHand", {"LeftHand": Vector((1, 0, 0))}, {"LeftHand": Vector((0, 1, 0))}) is None
# The clavicle ends further back than the game's: standing still, the tips of the shoulders curved forward (owner, 2026-11-09).
game = {"LeftShoulder": Vector((1, 0, 0))}
ours = {"LeftShoulder": Vector((1, -0.2, 0)).normalized()}
assert (rest_turn("LeftShoulder", game, ours) @ ours["LeftShoulder"]).y > 0.1

# The arms move as Rafa's (owner, 2026-11-11): standing still, the cast held the arms 3 to 20 degrees further out than he does in the same clip,
# because each clip came with the opening the game had tuned to a body that is gone. The arm takes his turn, counted from the chest.
from mathutils import Quaternion  # noqa: E402

from parle_rig import ARM_ROOT, arm_turn, moves_with_arm, same_clip  # noqa: E402

for name in ("LeftShoulder", "RightArm", "LeftForeArm", "RightHand", "LeftHandIndex2", "RightHandThumb4"):
    assert moves_with_arm(name), f"{name} keeps the old body's motion"
for name in ("Spine2", "Neck", "Head", "LeftUpLeg", "RightFoot", "Hips"):
    assert not moves_with_arm(name), f"{name} takes Rafa's motion, and only the arms should"
assert ARM_ROOT == "Spine2"
assert same_clip((0, 200), (0, 201)) and not same_clip((0, 12), (0, 200)), "a clip Rafa holds as a copy of his idle counts as his"
out = Quaternion((0, 1, 0), 0.3)  # Rafa's arm, 0.3 rad out from his chest
lean = Quaternion((1, 0, 0), 0.2)  # his chest, leaning
ours = Quaternion((0, 0, 1), 0.5)  # our chest, turned another way
got = arm_turn(ours, lean, lean @ out)
assert got.rotation_difference(ours @ out).angle < 1e-5, "the arm is not as far from our chest as Rafa's is from his"
still = arm_turn(Quaternion(), Quaternion(), Quaternion())
assert still.angle < 1e-6, "an arm at rest on Rafa is not at rest on us"
gap = Quaternion((0, 1, 0), 0.1)
assert arm_turn(ours, lean, lean @ out, gap).rotation_difference(ours @ out @ gap).angle < 1e-5

# Each body keeps Rafa's gap between the arm and the body (2026-11-11): the muscled man's forearm sank into the trunk and Lúcia's hand brushed the hip.
from parle_rig import ARM_GAP_M, ARM_OPENING_MOST, arm_opening, arm_spots, opened, skin_gap  # noqa: E402

assert arm_opening(ARM_GAP_M, 0.55) == 0 and arm_opening(0.1, 0.55) == 0, "an arm that is clear opens further"
assert abs(arm_opening(-0.008, 0.55) - 0.06) < 1e-6, "an arm 8 mm inside opens by what brings the gap back"
assert arm_opening(-1.0, 0.55) == ARM_OPENING_MOST, "the opening has no ceiling"
point = Vector((0.2, 0, 1))
assert abs(skin_gap(point, (Vector((0.15, 0, 1)), Vector((1, 0, 0)), 0.05), 0.03) - 0.02) < 1e-9, "outside the body, the gap is the way to it less the arm's own skin"
assert abs(skin_gap(point, (Vector((0.21, 0, 1)), Vector((1, 0, 0)), 0.01), 0.03) + 0.04) < 1e-9, "inside the body the gap is negative"
assert skin_gap(point, (Vector((0.5, 0, 1)), Vector((1, 0, 0)), 0.3), 0.03) > 0, "a face far away, looking the other way, is not a body the arm is in"
assert len(arm_spots(Vector((0, 0, 1)), Vector((0, 0, 0.7)), Vector((0, 0, 0.6)))) == 5
assert abs(skin_gap(point, (Vector((0.15, 0, 1)), Vector((1, 0, 0)), 0.05), 0.002) - 0.04) < 1e-9, "the hand is read as thin as the skin over its knuckle"
# An arm hanging straight down on +x swings out to +x, the other to -x, and by the angle asked.
down = Vector((0, 0, -1))
for outward in (1, -1):
    swung = opened(Quaternion(), Quaternion(), outward, 0.1) @ down
    assert swung.x * outward > 0.09 and abs(down.angle(swung) - 0.1) < 1e-5, f"the arm on side {outward} does not open outwards: {swung}"
rest = Quaternion((1, 0, 0), 0.7)
assert ((rest @ opened(Quaternion(), rest, 1, 0.1)) @ rest.inverted() @ down).x > 0.09, "a bone that rests turned does not open about the skeleton's axis"
print("ok")
