"""Regression for the lengths of the body on a skeleton with its clips (2026-11-14): run with
`Blender -b --factory-startup --python-exit-code 1 -P tests/test_parle_lengths_rig.py`.

The lengths of a kit's profile go on a finished character, after its clips are on it (`parle_polish.natural_lengths`). A skeleton
of the cast's bones, a skin of loose points and a clip of five poses stand for one here. Checked: every bone rests with the heading
it had, so the clips turn it as before; the joints of the legs are where the ratios put them and the feet where they were; the
stature stays; and in every pose of the clip the lowest point of the soles is as high over the floor as it was, read from Blender's
own pose and not from the sums the step makes.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import bpy
import numpy as np
from mathutils import Quaternion, Vector

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import parle_build  # noqa: E402
import parle_clips  # noqa: E402
import parle_polish  # noqa: E402

HEIGHT, UNIT, FLOOR = 1.74, 0.87, 0.14
RAW = dict(Hips=(0, 0, 1.008), Spine=(0, -0.026, 1.094), Spine1=(0, 0, 1.199), Spine2=(0, 0.069, 1.324), Neck=(0, -0.012, 1.534), Head=(0, -0.026, 1.59),
           LeftArm=(0.197, -0.019, 1.428), LeftForeArm=(0.407, -0.017, 1.189), LeftHand=(0.531, -0.202, 1.069),
           LeftUpLeg=(0.106, 0.007, 0.967), LeftLeg=(0.166, -0.027, 0.471), LeftFoot=(0.222, -0.014, -0.067))
JOINTS = {name: (np.array(point) + (0, 0, FLOOR)) * UNIT for name, point in RAW.items()}
JOINTS["LeftToeBase"] = JOINTS["LeftFoot"] + (0, -0.13, -0.035)
JOINTS["LeftShoulder"] = JOINTS["Neck"] + (0.05, 0.0, -0.03)
JOINTS.update({name.replace("Left", "Right"): point * (-1, 1, 1) for name, point in JOINTS.items() if name.startswith("Left")})

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
rig = bpy.data.objects.new("rig", bpy.data.armatures.new("rig"))
scene.collection.objects.link(rig)
bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode="EDIT")


def bone(name, tail, parent=None, joined=False):
    made = rig.data.edit_bones.new(name)
    made.head, made.tail = JOINTS[name], tail
    if parent:
        made.parent, made.use_connect = rig.data.edit_bones[parent], joined
    return made


bone("Hips", JOINTS["Spine"])
for name, child, parent in (("Spine", "Spine1", "Hips"), ("Spine1", "Spine2", "Spine"), ("Spine2", "Neck", "Spine1"), ("Neck", "Head", "Spine2")):
    bone(name, JOINTS[child], parent, True)
bone("Head", JOINTS["Head"] + (0, 0, 0.15), "Neck", True)
for side in parle_build.SIDES:
    bone(side + "Shoulder", JOINTS[side + "Arm"], "Spine2")
    bone(side + "Arm", JOINTS[side + "ForeArm"], side + "Shoulder", True)
    bone(side + "ForeArm", JOINTS[side + "Hand"], side + "Arm", True)
    bone(side + "Hand", JOINTS[side + "Hand"] + 0.08 * (JOINTS[side + "Hand"] - JOINTS[side + "ForeArm"]), side + "ForeArm", True)
    bone(side + "UpLeg", JOINTS[side + "Leg"], "Hips")
    bone(side + "Leg", JOINTS[side + "Foot"], side + "UpLeg", True)
    bone(side + "Foot", JOINTS[side + "ToeBase"], side + "Leg", True)
    bone(side + "ToeBase", JOINTS[side + "ToeBase"] + (0, -0.06, 0), side + "Foot", True)
bpy.ops.object.mode_set(mode="OBJECT")

# A skin of loose points, each wholly of one bone: down the trunk, down each leg and each arm, the soles and the top of the head.
spots = []
for name, child in (("Hips", "Spine"), ("Spine", "Spine1"), ("Spine1", "Spine2"), ("Spine2", "Neck"), ("Neck", "Head")):
    spots += [(name, JOINTS[name] + share * (JOINTS[child] - JOINTS[name]) + (0.1, 0.05, 0)) for share in (0.0, 0.5)]
spots += [("Head", np.array([0.0, 0.0, HEIGHT])), ("Hips", JOINTS["LeftUpLeg"] * (0, 1, 1) + (0, 0, -0.05))]
for side in parle_build.SIDES:
    for name, child in ((side + "UpLeg", side + "Leg"), (side + "Leg", side + "Foot"), (side + "Arm", side + "ForeArm"), (side + "ForeArm", side + "Hand")):
        spots += [(name, JOINTS[name] + share * (JOINTS[child] - JOINTS[name]) + (0.0, 0.03, 0)) for share in (0.25, 0.5, 0.75)]
    spots += [(side + "Foot", point) for point in parle_build.sole_points(JOINTS[side + "Foot"], JOINTS[side + "ToeBase"], 0.0)]
mesh = bpy.data.meshes.new("Body")
mesh.from_pydata([tuple(point) for _, point in spots], [], [])
body = bpy.data.objects.new("Body", mesh)
scene.collection.objects.link(body)
body.parent = rig
for index, (name, _) in enumerate(spots):
    (body.vertex_groups.get(name) or body.vertex_groups.new(name=name)).add([index], 1.0, "REPLACE")

# A clip of five poses: standing, a stride each way, a crouch and a turn on the spot, the hips where a walker's would be.
turn = lambda axis, degrees: Quaternion(Vector(axis), np.radians(degrees))
POSES = [
    dict(place=(0.0, 0.0, 0.0)),
    dict(place=(0.01, -0.02, -0.03), LeftUpLeg=turn((1, 0, 0), 28), LeftLeg=turn((1, 0, 0), -6), LeftFoot=turn((1, 0, 0), 14), RightUpLeg=turn((1, 0, 0), -22), RightLeg=turn((1, 0, 0), -30), RightFoot=turn((1, 0, 0), -35)),
    dict(place=(-0.01, 0.03, -0.02), RightUpLeg=turn((1, 0, 0), 30), RightLeg=turn((1, 0, 0), -10), LeftUpLeg=turn((1, 0, 0), -18), LeftLeg=turn((1, 0, 0), -40), LeftFoot=turn((1, 0, 0), -30)),
    dict(place=(0.0, 0.05, -0.25), Hips=turn((1, 0, 0), 10), LeftUpLeg=turn((1, 0, 0), 60), LeftLeg=turn((1, 0, 0), -95), RightUpLeg=turn((1, 0, 0), 60), RightLeg=turn((1, 0, 0), -95)),
    dict(place=(0.04, 0.0, -0.01), Hips=turn((0, 0, 1), 170), LeftUpLeg=turn((0, 1, 0), 8)),
]
rig.animation_data_create()
action = bpy.data.actions.new("walk")
slot = action.slots.new(id_type="OBJECT", name=rig.name)
bag = action.layers.new("clipe").strips.new(type="KEYFRAME").channelbag(slot, ensure=True)
chain = ["Hips"] + [side + part for side in parle_build.SIDES for part in parle_build.LEG]
for name in chain:
    rest = rig.data.bones[name].matrix_local.to_quaternion()
    for axis in range(4):
        curve = bag.fcurves.new(f'pose.bones["{name}"].rotation_quaternion', index=axis)
        curve.keyframe_points.add(len(POSES))
        # A turn in the frame of the skeleton, written in the axes of the bone as a clip keeps it.
        values = [(rest.inverted() @ pose.get(name, Quaternion()) @ rest)[axis] for pose in POSES]
        curve.keyframe_points.foreach_set("co", [value for frame, item in enumerate(values, 1) for value in (frame, item)])
        curve.update()
to_bone = rig.data.bones["Hips"].matrix_local.to_quaternion().inverted()
for axis in range(3):
    curve = bag.fcurves.new('pose.bones["Hips"].location', index=axis)
    curve.keyframe_points.add(len(POSES))
    curve.keyframe_points.foreach_set("co", [value for frame, pose in enumerate(POSES, 1) for value in (frame, (to_bone @ Vector(pose["place"]))[axis])])
    curve.update()
track = rig.animation_data.nla_tracks.new()
track.strips.new(action.name, 1, action).action_slot = slot
for pose_bone in rig.pose.bones:
    pose_bone.rotation_mode = "QUATERNION"


def soles():
    """The ball and the heel of each foot where they rest now, in the frame of the skeleton: they go with the foot."""
    return {side: parle_build.sole_points(np.array(rig.data.bones[side + "Foot"].head_local), np.array(rig.data.bones[side + "ToeBase"].head_local), 0.0) for side in parle_build.SIDES}


def lowest(points):
    """The lowest point of the soles in each pose of the clip, read from Blender's pose, and where the hips are."""
    lows, hips = [], []
    for frame in range(1, len(POSES) + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        placed = [rig.pose.bones[side + "Foot"].matrix @ rig.data.bones[side + "Foot"].matrix_local.inverted() @ Vector(point) for side in parle_build.SIDES for point in points[side]]
        lows.append(min(point.z for point in placed))
        hips.append(np.array(rig.pose.bones["Hips"].head))
    return np.array(lows), np.array(hips)


headings = lambda: {item.name: np.array(item.matrix_local.to_3x3()) for item in rig.data.bones}
joints = lambda: {item.name: np.array(item.head_local) for item in rig.data.bones}
points = lambda: parle_polish.rest_points(body)
folder = tempfile.mkdtemp()
plain, short = os.path.join(folder, "plain.json"), os.path.join(folder, "short.json")
parle_build.save_profile(parle_build.identity("test"), plain)
parle_build.save_profile(dict(parle_build.identity("test"), lengths=dict(thigh=0.92, shank=0.84)), short)



def posed(coming=None):
    """Where every joint is in each pose of the clip; with `coming`, a profile, where it will be once that profile's lengths are on."""
    out = []
    for frame in range(1, len(POSES) + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        gone = parle_clips.lengths_to_come(rig, coming, HEIGHT) if coming else {}
        out.append({item.name: np.array(rig.matrix_world @ item.head) + np.array(gone.get(item.name, (0, 0, 0))) for item in rig.pose.bones})
    return out


was_headings, was_joints, was_points, (was_low, was_hips) = headings(), joints(), points(), lowest(soles())
assert np.abs(was_low[0]).max() < 1e-6, f"standing, the soles of the test body are not on the floor: {was_low}"
# Before the lengths are on, the clips are told where the joints will be (`parle_clips.lengths_to_come`): nothing for a kit with none.
scene.frame_set(2)
assert parle_clips.lengths_to_come(rig, plain, HEIGHT) == {}, "a kit with no lengths has joints to come"
foretold = posed(short)
# A profile with no lengths: nothing happens.
assert parle_polish.natural_lengths([body], rig, HEIGHT, plain) is None and np.abs(points() - was_points).max() == 0 and all(np.array_equal(joints()[name], was_joints[name]) for name in was_joints)

drop, clips = parle_polish.natural_lengths([body], rig, HEIGHT, short)
now_headings, now_joints, now_points, (now_low, now_hips) = headings(), joints(), points(), lowest(soles())
assert clips == 1 and drop > 0.08, f"the clip was not set again, or the hips did not come down: {clips}, {drop}"
# Every bone rests with the heading it had: the clips only turn the bones.
worst = max(float(np.abs(now_headings[name] - was_headings[name]).max()) for name in was_headings)
assert worst < 1e-5, f"a bone rests with another heading: {worst}"
# The joints: the hips down by what the legs lose, each bone of the leg its ratio long, the feet at their height, the shoulders, the neck and the head where they were.
for side in parle_build.SIDES:
    hip, knee, foot = (now_joints[side + part] for part in parle_build.LEG)
    old_hip, old_knee, old_foot = (was_joints[side + part] for part in parle_build.LEG)
    assert abs(np.linalg.norm(knee - hip) / np.linalg.norm(old_knee - old_hip) - 0.92) < 1e-5 and abs(np.linalg.norm(foot - knee) / np.linalg.norm(old_foot - old_knee) - 0.84) < 1e-5, "a bone of the leg is not its ratio long"
    assert abs((old_hip - hip)[2] - drop) < 1e-5 and abs(foot[2] - old_foot[2]) < 1e-5 and np.allclose(now_joints[side + "ToeBase"] - foot, was_joints[side + "ToeBase"] - old_foot, atol=1e-6), "the hips, the ankle or the foot are not where the lengths put them"
    assert np.allclose(now_joints[side + "Arm"], was_joints[side + "Arm"], atol=1e-6) and np.allclose(now_joints[side + "Hand"], was_joints[side + "Hand"], atol=1e-6), "an arm moved"
    # A bone of the leg still ends where the next one starts, though the two are no longer joined.
    assert not rig.data.bones[side + "Leg"].use_connect and np.allclose(np.array(rig.data.bones[side + "UpLeg"].tail_local), knee, atol=1e-6), "the thigh does not end at the knee"
assert all(np.allclose(now_joints[name], was_joints[name], atol=1e-6) for name in ("Neck", "Head")), "the neck or the head moved"
assert was_joints["Hips"][2] - drop < now_joints["Hips"][2] < was_joints["Hips"][2], "the hips' own bone, over the joints of the legs, did not come part of the way down"
# The skin: the stature stays, the soles stay on the floor, the trunk has no level over another it was under.
assert abs(now_points[:, 2].max() - HEIGHT) < 1e-6 and abs(now_points[:, 2].min()) < 1e-6, f"the stature changed: {now_points[:, 2].min()}, {now_points[:, 2].max()}"
trunk = [index for index, (name, _) in enumerate(spots) if name in ("Hips", "Spine", "Spine1", "Spine2")]
order = np.argsort(was_points[trunk, 2])
assert (np.diff(now_points[trunk][order][:, 2]) > 0).all(), "a level of the trunk overtook another"
arms = [index for index, (name, _) in enumerate(spots) if "Arm" in name]
assert np.abs(now_points[arms] - was_points[arms]).max() < 1e-6, "the skin of an arm moved"
# The clip: in every pose the lowest point of the soles is as high as it was, and the hips went across by the ratio of the legs.
assert np.abs(now_low - was_low).max() < 1e-5, f"a foot meets the floor at another height: {was_low} -> {now_low}"
across = (0.92 * np.linalg.norm(was_joints["LeftLeg"] - was_joints["LeftUpLeg"]) + 0.84 * np.linalg.norm(was_joints["LeftFoot"] - was_joints["LeftLeg"])) / (
    np.linalg.norm(was_joints["LeftLeg"] - was_joints["LeftUpLeg"]) + np.linalg.norm(was_joints["LeftFoot"] - was_joints["LeftLeg"]))
assert np.allclose(now_hips[:, :2], across * was_hips[:, :2], atol=1e-5), f"the hips do not go across by the ratio of the legs: {was_hips[:, :2]} -> {now_hips[:, :2]}"
assert (now_hips[:, 2] < was_hips[:, 2] - 0.05).all(), "the hips of a pose did not come down with the legs"
# And every joint is where the clips were told it would be, in every pose, counted from the hips; standing, from the floor too.
for told, got in zip(foretold, posed()):
    off = max(float(np.abs((told[name] - told["Hips"]) - (got[name] - got["Hips"])).max()) for name in told)
    assert off < 1e-5, f"a joint is not where the clips were told it would be: {off}"
assert max(float(np.abs(foretold[0][name] - posed()[0][name]).max()) for name in foretold[0]) < 1e-5, "standing, the body is not where the clips were told it would be"
print("ok")
