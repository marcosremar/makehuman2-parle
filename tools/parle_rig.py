"""
    License information: data/licenses/makehuman_license.txt

    How a MakeHuman body takes the Parle game's motion clips. The two skeletons rest differently, so each
    bone is first pointed the way the game's bone points, and only then does the clip turn it.

    - Arms and legs point exactly as in the game: otherwise the clip opened the legs and crossed the hands.
    - The clavicle points front to back as in the game: the MakeHuman one points forward, and standing
      still the shoulders sat 2 cm in front of the neck (owner, 2026-11-07). In height it goes only half
      way, because all the way the shoulders came up 2.5 cm and the person looked hunched.
"""
from mathutils import Vector

ALIGNED = ("Shoulder", "Arm", "ForeArm", "UpLeg", "Leg")
SHOULDER_HEIGHT_SHARE = 0.5


def rest_turn(name, old_dir, new_dir):
    """The turn that points our bone like the game's, or None when this bone keeps its own rest."""
    if name not in old_dir or name not in new_dir or not name.endswith(ALIGNED):
        return None
    wanted = old_dir[name]
    if name.endswith("Shoulder"):
        wanted = Vector((wanted.x, wanted.y, new_dir[name].z + (wanted.z - new_dir[name].z) * SHOULDER_HEIGHT_SHARE)).normalized()
    return new_dir[name].rotation_difference(wanted)
