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

# The arm of either side, for the gap between the arm and the body: the clavicle, the arm, the forearm, the hand and the fingers.
from mathutils import Quaternion  # noqa: E402

from parle_rig import moves_with_arm, same_clip  # noqa: E402

for name in ("LeftShoulder", "RightArm", "LeftForeArm", "RightHand", "LeftHandIndex2", "RightHandThumb4"):
    assert moves_with_arm(name), f"{name} is not the arm's"
for name in ("Spine2", "Neck", "Head", "LeftUpLeg", "RightFoot", "Hips"):
    assert not moves_with_arm(name), f"{name} passed for a bone of the arm"
assert same_clip((0, 200), (0, 201)) and not same_clip((0, 12), (0, 200)), "a clip Rafa holds as a copy of his idle counts as his"

# Standing still, the clavicle rests where it was modelled (2026-11-11): the turn left over from the old clips held it 7 to 16 degrees back
# and up, and the shoulder stood as a square mass with a groove in front of the arm.
ours = {"LeftShoulder": Vector((1, -0.05, -0.5)).normalized()}
game = {"LeftShoulder": Vector((1, 0.25, -0.05)).normalized()}
clip = Quaternion((0, 0, 1), -0.28) @ Quaternion((0, 1, 0), 0.2)  # what the clip turns the clavicle by when Rafa stands still
turn = rest_turn("LeftShoulder", game, ours, clip)
assert (clip @ (turn @ ours["LeftShoulder"]) - ours["LeftShoulder"]).length < 1e-5, "standing still, the clavicle is not where it was modelled"
assert (rest_turn("LeftShoulder", game, ours) @ ours["LeftShoulder"]).y > 0.1, "without a body to copy, the clavicle no longer points as in the game"
assert rest_turn("LeftArm", {"LeftArm": Vector((1, 0, 0))}, {"LeftArm": Vector((0, 0, -1))}, clip).angle > 1.5, "the arm stopped pointing as in the game"

from parle_rig import clavicle_as_modelled  # noqa: E402

assert clavicle_as_modelled("male-slim") and clavicle_as_modelled("male-muscled"), "the men's shoulder is square again"
# The women's (2026-11-11). Where it was modelled, their clavicle stands 1.5 to 1.8 cm lower and 2.6 to 3.6 cm further to the front than
# the clips leave it. All of it undone, the arms came forward round the chest and the breasts looked higher and flatter; none of it undone,
# the shoulder stood high ("uma curva muito alta na parte de cima"). It comes down to the height it was modelled at and stays as far back
# as the clip leaves it.
import math  # noqa: E402

left_over = clip @ (rest_turn("LeftShoulder", game, ours) @ ours["LeftShoulder"])
stands = clip @ (rest_turn("LeftShoulder", game, ours, clip, level=True) @ ours["LeftShoulder"])
assert abs(stands.z - ours["LeftShoulder"].z) < 1e-5, f"standing still, the women's clavicle is not as high as it was modelled: {stands.z}"
assert left_over.z > ours["LeftShoulder"].z + 0.03, "the turn left over no longer holds the clavicle up: this test checks nothing"
assert abs(math.atan2(stands.y, stands.x) - math.atan2(left_over.y, left_over.x)) < 1e-5, "the women's clavicle is not as far back as the clip leaves it"
assert abs(stands.length - 1) < 1e-5 and (rest_turn("LeftShoulder", game, ours, None, level=True) @ ours["LeftShoulder"]).y > 0.1, "with no body to copy"
assert not clavicle_as_modelled("female-slim") and not clavicle_as_modelled("female-normal"), "the women's breasts sit higher and flatter again"

# On the slim man the front of the chest follows the spine alone (2026-11-11): with the arms hanging, the skin around the nipple followed the
# arm, sank a centimetre and left a mound 9.5 mm tall under each nipple.
from parle_rig import CHEST_KITS, chest_share, chest_weights  # noqa: E402

assert chest_share(0.75, 0.45, 0.1) == 1.0, "the nipple still follows the arm"
assert chest_share(0.75, 0.95, 0.1) == 0.0 and chest_share(0.75, 0.45, 0.0) == 0.0, "the armpit or the back lost the arm"
assert chest_share(0.2, 0.45, 0.1) == 0.0 and chest_share(1.2, 0.45, 0.1) == 0.0, "the belly or the neck was taken for the chest"
assert 0 < chest_share(0.75, 0.75, 0.1) < 1, "the weight comes back in one step, a crease on the chest"
moved = chest_weights({"Spine2": 0.6, "Spine1": 0.3, "LeftArm": 0.06, "LeftShoulder": 0.04}, 1.0)
assert set(moved) == {"Spine2", "Spine1"} and abs(sum(moved.values()) - 1.0) < 1e-9, f"the weight of the arm was lost or kept: {moved}"
assert abs(moved["Spine2"] / moved["Spine1"] - 2.0) < 1e-9, "the spine did not take it in the measure it already held"
kept = {"Spine2": 0.6, "LeftArm": 0.4}
assert chest_weights(kept, 0.0) == kept and chest_weights({"Head": 1.0}, 1.0) == {"Head": 1.0}
assert chest_weights({"LeftArm": 1.0}, 1.0) == {"Spine2": 1.0}, "a point the spine did not hold has nowhere to go"
assert CHEST_KITS == ("male-slim",), "the owner asked for the slim man only"

# The feet stand as far apart as Rafa's, in shares of the height (2026-11-11): on wider hips, with the legs pointed as his, they stood 6 cm wider.
from parle_rig import STANCE_MOST, even_joints, stance_turn  # noqa: E402

legs = stance_turn(0.177, 0.141, 0.48)
down = Vector((0, 0, -1))
assert (legs["Left"] @ down).x < 0 < (legs["Right"] @ down).x, "the legs lean out, not in"
assert abs((legs["Left"] @ down).x * 0.48 * 2 + (0.177 - 0.141)) < 1e-4, "the ankles do not come as close as the reference's"
assert stance_turn(0.141, 0.141, 0.48)["Left"].angle < 1e-3 and abs(stance_turn(0.9, 0.1, 0.48)["Left"].angle - STANCE_MOST) < 1e-3
# The clavicles of the game rig start 2.3 cm from each other's mirror: they go half way each.
left, right = even_joints(Vector((0.0659, -0.013, 1.5075)), Vector((-0.0556, -0.0068, 1.4879)))
assert abs(left.x + right.x) < 1e-6 and abs(left.y - right.y) < 1e-6 and abs(left.z - right.z) < 1e-6, "the clavicles are not mirrored"
assert abs(left.x - 0.06075) < 1e-5 and abs(left.z - 1.4977) < 1e-4, "the joints did not meet half way"

# Standing, the head sits over the ankles (2026-11-11): with the game's clips on our spine the cast stooped, the head 5 cm in front of the ankles.
import math  # noqa: E402

from parle_rig import HEAD_AHEAD, LEAN_MOST, trunk_lean  # noqa: E402

reach = 0.3
lean = trunk_lean(0.029, reach)
assert list(lean) == ["Spine", "Spine1", "Spine2"]
whole = lean["Spine2"].angle
assert abs(math.tan(whole) * reach - (0.029 - HEAD_AHEAD)) < 2e-4, "the lean does not bring the head back to its mark"
assert abs(lean["Spine"].angle * 3 - whole) < 1e-3 and abs(lean["Spine1"].angle * 3 - whole * 2) < 1e-3, "the three bones do not share the turn"
assert (lean["Spine2"] @ Vector((0, 0, 1))).y > 0, "the trunk leans forward"
assert trunk_lean(HEAD_AHEAD - 0.01, reach)["Spine2"].angle < 1e-3, "a body that already stands behind the mark was pushed"
assert abs(trunk_lean(0.5, reach)["Spine2"].angle - LEAN_MOST) < 1e-3, "the lean has no ceiling"

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
