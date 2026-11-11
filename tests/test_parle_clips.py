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

# The whole body copies Rafa first on a man and takes from Sofia what he lacks; a clip of another length is another motion, and one that
# neither has comes from the game's own body of the character (2026-11-11: copying the arms alone left them wrong whenever the trunk of
# the old body leaned another way than Rafa's, and the old clips swayed the hips 17 cm and slid the planted foot 10 cm).
own = dict(actions={"idle": clip("idle", 200), "thinking": clip("thinking", 100), "idleFemale": clip("idleFemale", 240)})
rafa = dict(actions={"idle": clip("idle.001", 200), "thinking": clip("thinking.001", 200)})
sofia = dict(actions={"idle": clip("idle.002", 200), "idleFemale": clip("idleFemale.001", 240)})
assert parle_clips.clip_source(own, [rafa, sofia], own["actions"]["idle"]) == (rafa, rafa["actions"]["idle"])
assert parle_clips.clip_source(own, [sofia, rafa], own["actions"]["idle"]) == (sofia, sofia["actions"]["idle"])
assert parle_clips.clip_source(own, [rafa, sofia], own["actions"]["idleFemale"]) == (sofia, sofia["actions"]["idleFemale"])
thinking = own["actions"]["thinking"]
assert parle_clips.clip_source(own, [rafa, sofia], thinking) == (own, thinking), "a copy of the idle passed for the clip"
assert parle_clips.clip_source(own, [], own["actions"]["idle"]) == (own, own["actions"]["idle"])
assert parle_clips.clip_source(own, None, own["actions"]["idle"])[0] is own

# Whose arms a kit copies first is decided here, not by who calls: a woman copies Sofia, a man Rafa.
import parle_rig  # noqa: E402

assert parle_rig.lead_order("female-slim") == ("sofia", "rafa"), "a woman copies Sofia first"
assert parle_rig.lead_order("male-muscled") == ("rafa",), "a man moves as Sofia where Rafa has no clip"
assert parle_clips.lead_files("rafa=/a/rafa.glb,sofia=/b/sofia.glb") == {"rafa": "/a/rafa.glb", "sofia": "/b/sofia.glb"} and parle_clips.lead_files("") == {}

# A bone with no curve stays at its rest, and a child goes with its parent.
turn = Quaternion((0, 0, 1), 0.5)
curve = lambda value: SimpleNamespace(evaluate=lambda frame: value)
table = {"Spine": {"rotation_quaternion": {index: curve(turn[index]) for index in range(4)}}}
world = parle_clips.posed(["Spine", "Neck"], {"Spine": None, "Neck": "Spine"}, {"Spine": Quaternion(), "Neck": Quaternion()}, table, 10)
assert world["Spine"].rotation_difference(turn).angle < 1e-5 and world["Neck"].rotation_difference(turn).angle < 1e-5

source = (ROOT / "tools" / "parle_clips.py").read_text()
PIECES = ("def clips_on(", "def retarget(", "def aligned_rest(", "def standing(", "def still_turns(", "def clear_arms(", "def open_arms(",
          "def standing_gap(")
for piece in PIECES:
    assert piece in source, f"{piece} left the fork"
RULES = ("parle_rig.lead_order(", "parle_rig.same_clip(", "parle_rig.rest_turn(", "parle_rig.trunk_lean(", "parle_rig.clavicle_as_modelled(",
         "parle_rig.arm_opening(", "parle_rig.skin_gap(", "parle_rig.stance_turn(", "parle_rig.even_joints(")
for rule in RULES:
    assert rule in source, f"the clips no longer ask parle_rig: {rule}"
print("ok")
