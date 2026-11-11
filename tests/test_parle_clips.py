"""Regression for the clips of the cast (2026-11-11): run with Blender's Python, `blender -b --python-exit-code 1 -P tests/test_parle_clips.py`.

How a clip of the game reaches our skeleton, whose arms it copies, the gap of the arms and the stance were written in the game's own script
(tools/npc-editor/cast-assemble.py) and moved here, where every change to the cast belongs (owner, 2026-11-05 and 2026-11-11). This checks
the pieces that need no body, and that the whole of it is here.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

from mathutils import Quaternion

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import parle_clips  # noqa: E402

clip = lambda name, last: SimpleNamespace(name=name, frame_range=(0, last))
# Blender names the clip of a second file `idle.001` and the importer's `idle_Armature`: both are the idle clip.
assert parle_clips.clip_name(clip("idle.001", 200)) == "idle" and parle_clips.clip_name(clip("idle_Armature", 200)) == "idle"
assert parle_clips.clip_name(clip("talkArmSide", 80)) == "talkArmSide"

# A man copies Rafa first and takes from Sofia what he lacks; a clip of another length is another motion and is copied from nobody.
rafa = dict(actions={"idle": clip("idle.001", 200), "thinking": clip("thinking.001", 200)})
sofia = dict(actions={"idle": clip("idle.002", 200), "idleFemale": clip("idleFemale.001", 240)})
assert parle_clips.arm_lead([rafa, sofia], clip("idle", 200)) is rafa and parle_clips.arm_lead([sofia, rafa], clip("idle", 200)) is sofia
assert parle_clips.arm_lead([rafa, sofia], clip("idleFemale", 240)) is sofia
assert parle_clips.arm_lead([rafa, sofia], clip("thinking", 100)) is None, "a copy of the idle passed for the clip"
assert parle_clips.arm_lead([], clip("idle", 200)) is None and parle_clips.arm_lead(None, clip("idle", 200)) is None

# A bone with no curve stays at its rest, and a child goes with its parent.
turn = Quaternion((0, 0, 1), 0.5)
curve = lambda value: SimpleNamespace(evaluate=lambda frame: value)
table = {"Spine": {"rotation_quaternion": {index: curve(turn[index]) for index in range(4)}}}
world = parle_clips.posed(["Spine", "Neck"], {"Spine": None, "Neck": "Spine"}, {"Spine": Quaternion(), "Neck": Quaternion()}, table, 10)
assert world["Spine"].rotation_difference(turn).angle < 1e-5 and world["Neck"].rotation_difference(turn).angle < 1e-5

source = (ROOT / "tools" / "parle_clips.py").read_text()
for piece in ("def clips_on(", "def retarget(", "def aligned_rest(", "def standing_lean(", "def still_turns(", "def clear_arms(", "def open_arms(", "def standing_gap("):
    assert piece in source, f"{piece} left the fork"
for rule in ("parle_rig.arm_turn(", "parle_rig.rest_turn(", "parle_rig.trunk_lean(", "parle_rig.clavicle_as_modelled(", "parle_rig.arm_opening(", "parle_rig.skin_gap("):
    assert rule in source, f"the clips no longer ask parle_rig: {rule}"
print("ok")
