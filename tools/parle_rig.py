"""
    License information: data/licenses/makehuman_license.txt

    How a MakeHuman body takes the Parle game's motion clips. The two skeletons rest differently, so each
    bone is first pointed the way the game's bone points, and only then does the clip turn it.

    - Arms and legs point exactly as in the game: otherwise the clip opened the legs and crossed the hands.
    - The clavicle points front to back as in the game: the MakeHuman one points forward, and standing
      still the shoulders sat 2 cm in front of the neck (owner, 2026-11-07). In height it goes only half
      way, because all the way the shoulders came up 2.5 cm and the person looked hunched.
    - Each finger phalanx points as in the game: MakeHuman rests with the fingers half curled and spread, the
      game (Mixamo) with them straight, and the clip's curl came on top of ours. Pointing, the hand closed into a
      claw: 64 degrees of curl against 38, and the index and little finger 52 degrees apart against 18 (2026-11-08,
      measured in the Unity player on Dona Rosa). The tip (Thumb4, Index4...) has no child and follows its parent.
"""
from mathutils import Vector

FINGERS = tuple(f"Hand{finger}{joint}" for finger in ("Thumb", "Index", "Middle", "Ring", "Pinky") for joint in (1, 2, 3))
ALIGNED = ("Shoulder", "Arm", "ForeArm", "UpLeg", "Leg") + FINGERS
SHOULDER_HEIGHT_SHARE = 0.5


def rest_turn(name, old_dir, new_dir):
    """The turn that points our bone like the game's, or None when this bone keeps its own rest."""
    if name not in old_dir or name not in new_dir or not name.endswith(ALIGNED):
        return None
    wanted = old_dir[name]
    if name.endswith("Shoulder"):
        wanted = Vector((wanted.x, wanted.y, new_dir[name].z + (wanted.z - new_dir[name].z) * SHOULDER_HEIGHT_SHARE)).normalized()
    return new_dir[name].rotation_difference(wanted)
