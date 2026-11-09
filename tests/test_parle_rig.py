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
print("ok")
