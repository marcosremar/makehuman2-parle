"""
    License information: data/licenses/makehuman_license.txt

    Finishing touches of the Parle cast's look, run inside Blender on a mounted character, what was judged an improvement
    next to the player and Rafa (owner, 2026-11-08). Every shape change moves all shape keys, so blinking and speaking keep working.
"""
import glob
import os

import bmesh
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

import parle_build
import parle_bust
import parle_rig

CLOTHES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".local/mhd/data/clothes/hm08")

# 12% with a staring look all round; 10% reads as awake next to the player.
EYE_SCALE = 1.1
# The women's eyes grow a little more: bigger eyes are what most sets apart a woman's face that is found attractive (eye height
# against the length of the face, r = +.50: Cunningham 1986, Journal of Personality and Social Psychology 50(5):925-935, DOI
# 10.1037/0022-3514.50.5.925), and next to Sofia, the face the cast aims at, theirs showed 6 to 7% of the box of the face against 10%
# (`parle_beauty.eye_shown`, 2026-11-11). The men's show as much as Rafa's and stay. How far is bounded by the head they sit
# on, which has the proportions of a real one: on photographs, eyes are judged too large from 116.3% of their size on (SD 9.1; Seyama &
# Nagayama 2009, Presence 18(5):321-339, DOI 10.1162/pres.18.5.321), and eyes 50% bigger on a computer face of human proportions were
# eerie at every level of skin (MacDorman, Green, Ho & Koch 2009, Computers in Human Behavior 25(3):695-710, DOI
# 10.1016/j.chb.2008.12.026). Tried on 2026-11-11: at 1.22 the face is the same person with bigger eyes; at 1.4 the skin round each eye
# stood as a bag, and grown in the plane of the face alone (1.35) the globe came out through the lids.
# On 2026-11-12, with the eye read as the opening between the lids (`parle_beauty.eye_shown`), the women's at 1.2 already showed as
# much as Sofia's (2.4% of the box of the face): what Sofia's showed beyond that was the line drawn on her lids. The measure no longer
# asks for more. The eye does: the head is no longer of real proportions (a quarter bigger on the women, `parle_build.HEAD`) nor its
# skin a real one's (`fair_face`), and on a drawn face bigger eyes are not the fault they are on a photograph (eyes half as big
# again were unpleasant on a realistic face and hardly mattered on a drawn one: Seyama & Nagayama 2007, Presence 16(4):337-351, DOI
# 10.1162/pres.16.4.337). At 1.3, with the skin round them given more room (EYE_REACH_WIDE_M), no bag stands under the eye. Not
# further this way: at 1.4 the skin has room only with a reach that carries the bridge of the nose 4 mm forward, each eye pushing
# it from its side, and with the bridge kept out of the move the skin folded between it and the inner corner of each eye
# (2026-11-12, both tried). What the eyes still lack next to Sofia's is not size by this field.
EYE_SCALE_BY_SEX = {"female": 1.3}
# What a kit's eyes grow past EYE_SCALE has more skin to settle in: to this far from the middle of the eye.
EYE_REACH_WIDE_M = 0.038
EYE_CORE_M = 0.017
EYE_REACH_M = 0.032
# The skin between the head and the chest shares out what a shorter neck loses (`even_follow`): a point goes whole with the head
# from this much of its weight on it, follows neither head nor neck below this much on both, and the rest is evened in these rounds.
NECK_WHOLE = 0.999
NECK_NONE = 0.05
NECK_EVEN_ROUNDS = 400
NECK_SAME_M = 1e-5
# The pieces whose openings for the legs end on a level line. A hem is in the lowest share of the bottoms, where the open edge leans
# no more than this from level (the sine); an opening has at least these points, and the plane that fits it stands at least this
# upright. A skirt's hem has points at the middle (this near it) from the front to the back, this share of the hem's depth.
HEM_PIECES = ("Outfit_Bottom",)
HEM_LOW_SHARE = 0.25
HEM_LEVEL = 0.8
HEM_LEAST_POINTS = 8
HEM_UPRIGHT = 0.7
HEM_MIDDLE_M = 0.02
HEM_SKIRT_DEEP = 0.6
# A point of the globe that is not seen goes this far behind the skin, when the skin is no further behind it than the reach.
EYE_BEHIND_M = 0.001
EYE_BEHIND_REACH_M = 0.02
# The faces of the globe are looked at in a grid of this many steps along each side. A place of a face is the globe coming through
# when the skin is no further behind it than this, and that skin turns its outer side to the front by this much (the cosine): at the
# corner of Carlos's eye the white that is meant to show has the inside of the lid 1 mm behind it, turned away (-0.76), where the
# specks under Dona Marta's eyes had the cheek 0.3 mm behind, turned to the front (0.7 to 0.9). And skin comes first in at least this
# many of the places looked at on the way to the middle of the eye: a strip of lid, not one stray place (Carlos had one, Dona Marta
# three or four). No corner goes back more than the most in all, over these rounds.
EYE_FACE_STEPS = 8
# A face smaller than the kit's is looked at in fewer steps, one for each of these along its longest side: the places stay as far
# apart on the rounded globe (`round_globes`), whose faces are a sixteenth of the kit's, as on the kit's own.
EYE_FACE_SPACING_M = 0.0016
# The globe of the kit is a ball of eight sides: between two of its points a face stands up to 1.4 mm inside the ball, the rim of
# a lid floats over it, and where a point under the lid goes back behind the skin (`globes_behind_skin`) the whole face slopes back
# with it, a centimetre wide: "ragged lid edges, a torn line under the lower lid, a jag at the outer corner" (the judge by sight,
# 2026-11-12, round 14). Each face becomes four this many times, its new points out on the ball: 32 sides, the same face 0.1 mm
# inside. 96 points become 1410 and 172 faces 2752, on a body of some 12 500 points.
EYE_ROUND_LEVELS = 2
EYE_THROUGH_M = 0.002
EYE_SKIN_FACING = 0.3
EYE_WAY_CUT = 2
EYE_BEHIND_MOST_M = 0.006
EYE_BEHIND_ROUNDS = 6
# The way from a point of the globe to the middle of the eye is looked at in these places, as shares of the way: the strip of lid
# between a point under it and the opening is a millimetre or two wide.
EYE_WAY_SHARES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
# The skin of the face evened out (`fair_face`). The traits of the kit and the shapes of speech whose reach is the soft of the face
# (the cheeks, the jaw, the chin, round the mouth and beside the nose), and those whose reach keeps its points: the lids and the
# nose, each with the share of its widest move from which a point is held whole.
# The skin round the eyes is of the zone too, by the trait that sets the eyes apart, which carries the whole of it: left out, it
# kept the bag and the furrow under each eye and a fold running out from the outer corner, and with the rest of the face smooth
# they were what aged it most ("bags and a tear-trough crease under the eyes", the judge by sight, 2026-11-12, round 14; on Dona
# Marta "tired and fifty"). The lids themselves stay: what blinks, and what the trait that opens the eye moves by half its widest.
FAIR_SOFT = ("faceCheek", "faceCheekVolume", "faceJaw", "faceJawSoft", "faceChin", "faceVChin", "faceMouthWidth", "cheekSquintLeft", "cheekSquintRight",
             "mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft", "mouthFrownRight", "faceLipVolume", "faceNoseWidth", "viseme_PP", "viseme_U", "viseme_O",
             "faceEyeSpacing")
FAIR_KEPT = {"eyeBlinkLeft": 0.3, "eyeBlinkRight": 0.3, "faceEyeOpen": 0.5}
# The nose kept every point until 2026-11-12, with the skin round it evened out: the rims of the nostrils and the wings stood as
# they came, pinched and uneven from one point to the next, and a small fold ran from each wing where held points met free ones ("a
# torn, lopsided M at the base of the nose", the judge by sight, rounds 13 and 14). Like the lips, the nose now takes part with a
# small weight, which takes out what goes from one point to the next and leaves its form: the points its own traits move by these
# shares or more of their widest move.
FAIR_NOSE_BY = {"faceNoseTip": 0.2, "faceNoseWidth": 0.7}
FAIR_NOSE = 0.3
# The skin of the lids lies on the eye globes, and evened out with nothing in its way it sank into them: at the outer corner of
# each eye the white of the globe came out through the skin (2026-11-12, the first body with the skin round the eyes in the zone).
# A point of skin that starts outside the ball of a globe, or no more than the first of these inside it, never ends nearer the
# middle of that ball than it started, or than the second of these off the ball when it started further; what starts deeper inside
# is the socket behind the globe, which no one sees, and stays. Skin further from the ball than the third is not looked at.
FAIR_ON_BALL_M = 0.002
FAIR_OFF_BALL_M = 0.001
FAIR_NEAR_BALL_M = 0.02
# The lids lie over the eye globes (`lids_over_globes`). The kit fits its lids to a globe of eight sides, whose faces stand up to
# 1.4 mm inside the ball; on the rounded globe (`round_globes`) the ball came out through the skin beyond the outer corner of each
# eye, the white showing in a ragged flag with the skin 0.2 to 1.4 mm behind it (2026-11-12, measured on the body). Skin that
# turns its outer side away from the middle of a ball by the cosine below, and is no further inside the ball than the first of
# these, goes out to the second of these off the ball. Deeper inside is the socket. All of the way out as far round the ball from
# the front as the outer corner of the opening (37 degrees, the second of the two last cosines), less from there, and not at all
# past 43 degrees (the first): on the body measured, all the skin inside the ball was on the side of the temple, 30 to 60 degrees
# round, most of it past the corner and up to 2 mm deep, and laid over the ball there (as far as 70 degrees, then 60) it stood
# beside each eye in a mound with the outline of the ball, seen from the side (2026-11-12). Past the corner the globe goes back
# instead (`globes_behind_skin`, EYE_SIDE_FRONT).
LID_UNDER_BALL_M = 0.003
LID_OFF_BALL_M = 0.0004
LID_FACING = 0.3
LID_ROUND_FRONT = (0.731, 0.799)
# A place of the globe further round its ball from the front than this cosine (40 degrees) is past the outer corner of the
# opening: nothing of the globe is meant to show there, so one that shows with skin no further behind it than the reach below goes
# back with no strip of lid asked for between it and the middle of the eye.
EYE_SIDE_FRONT = 0.766
EYE_SIDE_REACH_M = 0.004
# The upper lid has a fold above it (`lids_with_a_fold`). Cut through the middle of an eye (2026-11-12), the upper lid of Rafa
# ends in a lip 5.5 mm in front of the globe and the skin above it goes in again, 2.4 mm behind the lip 4.5 mm up; Sofia's lip
# stands 5.4 to 6.5 mm off, the skin 4.3 mm behind it 7.6 mm up. Lúcia's lip stood 2 to 3 mm off the globe and the skin went from
# it to the brow in one flat wall, the brow 1.5 mm in front of the lip: "lidless symmetric cut-outs with thin sharp rims" (the
# judge by sight, round 16).
# The rim of the opening is the skin that shows next to skin that does not: skin in the shadow of the ball, seen from the front,
# that turns its outer side to the ball (less than LID_FACING away from its middle) or stands deeper inside it than
# LID_UNDER_BALL_M. The skin of the temple that stands inside the rounded ball is shallower and turns away from it: it is not the
# socket. Rim counts no further off the ball than the reach, no further round the ball from the front than the cosine (47
# degrees), and within so many radii of the middle of the ball, where the skin is looked at.
LID_NEAR_BALLS = 1.7
LID_RIM_REACH_M = 0.004
LID_RIM_FRONT = 0.68
# The rim stays where the kit has it. In the middle of the upper lid, the skin as far from the rim as the first distance stands
# at least LID_LIP_M off the ball (nearer the rim, by less: nothing at the rim itself); the lid keeps the share of that height as
# far as the second; and the fold is deepest at the third, the skin going in by LID_FOLD_M there and by less either side, over
# the two reaches (towards the lip, and towards the brow). Towards the corners of the opening, over these shares of its width
# from each (the outer one is the one further from the middle of the face), the height and the fold go down to nothing and the
# distances to this share of themselves: the fold runs into the corners. Skin further from the rim than the last is not looked at.
# The rim raised into a lip with a wall behind it, as the lids of Rafa and of Sofia are made (the edge of the lid runs into the
# globe), was tried at 4.2, 3.4, 3 and 2.6 mm and left: the skin of a kit has one row of points for the rim and the lip, a few
# to an eye and no two kits alike, and what does not show behind the rim came out in torn flaps or left the edge in teeth, on
# three of the five faces looked at (2026-11-12). It asks for a lid drawn with rows of its own.
LID_LIP_M = 0.0026
LID_LIP_AT_M = 0.0016
LID_PLATE_M = 0.0034
LID_PLATE_SHARE = 0.85
LID_FOLD_M = 0.0035
LID_FOLD_AT_M = 0.0052
LID_FOLD_REACH_M = (0.002, 0.0036)
LID_CORNER = {"outer": 0.4, "inner": 0.25}
LID_SHORT = 0.6
LID_WAY_MOST_M = 0.012
# The shapes that shut the lids: a point one of them moves this far or more is where the kit shuts it, whatever moved the rest
# shape (`lids_with_a_fold` here, `parle_face.blink_still_closes` for the traits of the look).
BLINKS = ("eyeBlinkLeft", "eyeBlinkRight")
BLINK_WHOLE_M = 0.001
# With a lid shut nothing of the globe shows (`blinks_shut`). The blink of a kit stops the upper lid a millimetre above the lower
# one: on the head of the kit the lashes cover the slit, and the cast has none (owner: "sem cílios"), so every blink showed a dark
# slit with a thread of white in it. Measured by rays from the front with the lid shut (2026-11-12), the globe showed in a strip
# 17 to 22 mm wide and 1 to 2.4 mm tall, on the head of the kit and on the finished body alike; only Dona Marta's lids, less open
# than the kit's at rest, already met.
# A blink that moves no point as far as the first of these is not looked at. Shut, the skin of the upper lid that stands in the
# ball by no more than LID_UNDER_BALL_M comes out to LID_OFF_BALL_M off it (deeper is the inside of the lid). Then the face is
# looked at from the front in a grid of this step, as far from the middle of the ball as these shares of its radius across and
# up: in each column where the globe shows before any skin, the upper lid comes down by the height that shows (the tallest within
# the reach either side) and this much over, a point by the share of the way it travels in the blink (whole from this share of
# the longest way), over these rounds. Never by more, in all, than this many times what its column asked at first, nor than the
# most, times that share: a column where coming down does not hide the globe is not chased (on a sheet of skin that shut across
# the whole ball, the side of it went 4 mm down and into the ball after a slit that was past its edge).
BLINK_SHUT_LEAST_M = 0.002
BLINK_SHUT_STEP_M = 0.0004
BLINK_SHUT_ACROSS = 0.8
BLINK_SHUT_UP = 0.6
BLINK_SHUT_REACH_M = 0.0015
BLINK_SHUT_OVER_M = 0.0004
BLINK_SHUT_WHOLE = 0.6
BLINK_SHUT_ROUNDS = 4
BLINK_SHUT_PAST = 1.5
BLINK_SHUT_MOST_M = 0.004
FAIR_SOFT_FROM = 0.15
# The lips are the points their own traits move by this share or more. They take part with a small weight: point against point
# their sculpted form came out rough, a row of ridges along the upper lip (2026-11-12, in clay), and a small weight takes out what
# goes from one point to the next and leaves the two forms of the lips. Where the two lips meet they stay: a lip is of the lower jaw
# when the open jaw carries it this share of how far it carries the lip it carries most (it drags the upper lip a millimetre or
# two as well), and a point this near the other lip is at the meeting.
FAIR_LIPS = ("faceLipVolume", "faceCupidBow")
FAIR_LIP_FROM = 0.25
FAIR_LIP = 0.35
FAIR_JAW_SHARE = 0.3
FAIR_MEET_M = 0.003
# One round is a step of this share toward the middle of a point's neighbours and a step back of that one: what is narrower than a
# few centimetres goes and the shape of the head stays, where steps forward alone would shrink it (Taubin 1995, "A signal processing
# approach to fair surface design", SIGGRAPH '95, pp. 351-358, DOI 10.1145/218380.218473).
FAIR_STEP, FAIR_BACK, FAIR_ROUNDS = 0.5, -0.52, 80
# A point with skin straight ahead of it, anywhere in the depth of a head, is inside the head (the lining of the mouth, down to the
# throat) and stays. Looked for 5 cm ahead only, the back of Carlos's mouth, 10 cm behind his nose, was smoothed with the skin and
# went 18 mm (2026-11-12).
FAIR_INSIDE_M = 0.25
# The edge of the zone is itself evened over this many rounds: the skin does not step where the zone begins. Over six the weight
# rose to whole only some 2 cm from the nose and the lips, and the folds beside them stayed.
FAIR_EDGE_ROUNDS = 2
FAIR_SAME_M = 1e-5
FAIR_HEAD_FROM = 0.5
# The eyebrows lie this far off the skin (`brows_on_skin`).
BROW_OFF_SKIN_M = 0.0006
# The inside of the mouth stays behind the skin of the face (`lining_behind_skin`): this far behind it at the least, looked for
# this far ahead of a point and behind it, and no point goes back more than this.
LINING_BEHIND_M = 0.002
LINING_REACH_M = 0.02
LINING_MOST_M = 0.012
LINING_SKIN_FACING = 0.3
LINING_ROUNDS = 4
# The places of a face of the lining that are looked at: its corners, the middle of its sides and its own middle.
LINING_PLACES = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (0.5, 0.5, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (1 / 3, 1 / 3, 1 / 3))
NORMAL_PX = 512
# Arms (owner, 2026-11-10), as a share of the height (body without hair or shoes) from joint centre to joint centre. The yardstick that hair, hat and
# heels do not move is the arm against the leg (upper arm + forearm over thigh + shank): 0.58, 0.58 and 0.60 on the three rigged whole-body models the
# owner gave as reference, 0.51 to 0.56 on the cast, whose forearm was short in both sexes (12.4 to 12.9% of the height) and whose upper arm was short
# in the women (13.6%). De Leva (1996, Journal of Biomechanics 29(9), 1223-1230, DOI 10.1016/0021-9290(95)00178-6) measures between joint centres
# 16.2 and 15.4% in men and 15.9 and 15.2% in women, 0.64 of the leg in men. The shares below sit between the references and that measure and put the
# cast at 0.58 of the leg. One bone stretches 16% at most, which is what the cast needs.
ARM_UPPER_SHARE = {"female": 0.157, "male": 0.162}
ARM_FORE_SHARE = {"female": 0.142, "male": 0.148}
ARM_SCALE_LIMITS = (0.94, 1.16)
ARM_STILL_SHARE = 0.1
ARM_BONES = ("Arm", "ForeArm", "Hand")
# The top hangs outside the bottom. The build of the kit moves the trunk one way and the legs another, and the two meet at the hip, where
# the hem of a shirt is: after it the front of the hem of Carlos's shirt stood up to 11 mm inside his trousers, and the waistband showed
# through the plaid in two slits beside the last button (2026-11-11).
OVER_GAP_M = 0.005
OVER_REACH_M = 0.03
# Cloth goes the way of the skin under it. The build moves a point by where it is and by how much of it follows each arm, and a garment
# has weights of its own: where its share of the arm is not the skin's beneath, the two part. It showed once the shoulders of the slim
# woman came in while her chest stayed (2026-11-11): at the back of the armhole the shirt went a centimetre further in than the skin,
# and the skin of the trunk showed under the sleeve when the arm swung back. Lying on the skin, a point of cloth takes the skin's share;
# from SKIN_FAR_M on it keeps its own, as a loose sleeve does.
SKIN_NEAR_M = 0.015
SKIN_FAR_M = 0.04
OVER_STEP_M = 0.01
OVER_ROUNDS = 6
OVER_SHARED = 2


def smooth(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def read(block):
    """Points of a shape key, or of `mesh.vertices`."""
    items = block.data if hasattr(block, "relative_key") else block
    points = np.empty(len(items) * 3)
    items.foreach_get("co", points)
    return points.reshape(-1, 3)


def to_world(obj, points):
    matrix = np.array(obj.matrix_world)
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def to_local(obj, points):
    matrix = np.array(obj.matrix_world.inverted())
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def rest_points(obj):
    return to_world(obj, read(obj.data.vertices))


def warp(obj, field):
    """`field` takes world points; every key is absolute, so a stretched lid still closes all the way."""
    blocks = obj.data.shape_keys.key_blocks if obj.data.shape_keys else []
    for block in blocks:
        block.data.foreach_set("co", to_local(obj, field(to_world(obj, read(block)))).ravel())
    obj.data.vertices.foreach_set("co", to_local(obj, field(rest_points(obj))).ravel())
    obj.data.update()


def eye_radius(eye):
    points = rest_points(eye)
    return float(np.ptp(points[points[:, 0] < points[:, 0].mean(), 0]) / 2)


def eye_centres(eye):
    points = rest_points(eye)
    middle = points[:, 0].mean()
    centres = []
    for side in (points[:, 0] < middle, points[:, 0] >= middle):
        part = points[side]
        radius = np.ptp(part[:, 0]) / 2
        centres.append(np.array([(part[:, 0].min() + part[:, 0].max()) / 2, part[:, 1].min() + radius, (part[:, 2].min() + part[:, 2].max()) / 2]))
    return centres


def hem_points(obj, points, low, tall):
    """The points of a piece on its hem: the ends of the open edges that lie in the lowest part of the bottoms and run across the leg,
    not down it (a seam between two panels is an open edge too). The corner where a hem meets a seam is of the hem."""
    count = {}
    for polygon in obj.data.polygons:
        corners = list(polygon.vertices)
        for first, second in zip(corners, corners[1:] + corners[:1]):
            key = (min(first, second), max(first, second))
            count[key] = count.get(key, 0) + 1
    found = set()
    for (first, second), times in count.items():
        along = points[second] - points[first]
        if times == 1 and max(points[first][2], points[second][2]) < low + HEM_LOW_SHARE * tall and abs(along[2]) < HEM_LEVEL * np.linalg.norm(along):
            found |= {first, second}
    return sorted(found)


def level_hems(meshes):
    """The leg of a pair of shorts, of trousers or of a skirt ends on a level line, and how many openings were levelled. The bottoms of
    a kit have few faces, and the hem of the pharmacy attendant's shorts went round each thigh in teeth of 2 cm, torn-looking
    (2026-11-11). A bottom is sewn of panels, front and back, so a leg's opening is the hems of both: its points, of every panel, go
    onto the plane that fits them best, along that plane's normal, which is the way of the leg, so none comes nearer the skin. One
    opening for each leg; one for a skirt, whose hem goes right across the middle from front to back. Waist and seams stay."""
    pieces = [obj for obj in meshes if obj.name.startswith(HEM_PIECES) and len(obj.data.vertices)]
    if not pieces:
        return 0
    clouds = {obj.name: rest_points(obj) for obj in pieces}
    every = np.concatenate(list(clouds.values()))
    low, tall, middle = every[:, 2].min(), np.ptp(every[:, 2]), every[:, 0].mean()
    found = [(obj.name, index) for obj in pieces for index in hem_points(obj, clouds[obj.name], low, tall)]
    if len(found) < HEM_LEAST_POINTS:
        return 0
    spots = np.array([clouds[name][index] for name, index in found])
    between = np.abs(spots[:, 0] - middle) < HEM_MIDDLE_M
    skirt = between.sum() >= 2 and np.ptp(spots[between, 1]) > HEM_SKIRT_DEEP * np.ptp(spots[:, 1])
    moves = {name: np.zeros_like(cloud) for name, cloud in clouds.items()}
    levelled = 0
    for side in ([np.ones(len(spots), bool)] if skirt else [spots[:, 0] < middle, spots[:, 0] >= middle]):
        if side.sum() < HEM_LEAST_POINTS:
            continue
        centre = spots[side].mean(axis=0)
        normal = np.linalg.svd(spots[side] - centre)[2][-1]
        if abs(normal[2]) < HEM_UPRIGHT:
            continue
        for (name, index), spot in zip([item for item, mine in zip(found, side) if mine], spots[side]):
            moves[name][index] = -((spot - centre) @ normal) * normal
        levelled += 1
    for obj in pieces:
        if moves[obj.name].any():
            warp(obj, lambda world, move=moves[obj.name]: world + move)
    return levelled


def under_the_lid(skin, at, centre, radius, front):
    """Whether, seen from the front, skin lies over the globe somewhere between one of its points and the middle of the eye: the point
    is then outside the opening of the lids on this face. The globe is taken as its ball, and the skin as what a ray from the front
    meets first: where the lids are open it meets the socket, behind the ball."""
    for share in EYE_WAY_SHARES:
        spot = at.lerp(centre, share)
        deep = radius * radius - (spot.x - centre.x) ** 2 - (spot.z - centre.z) ** 2
        if deep <= 0:
            continue
        hit = skin.ray_cast(Vector((spot.x, centre.y, spot.z)) + front * 0.05, -front, 0.1)[0]
        if hit is not None and (hit - centre).dot(front) > deep ** 0.5:
            return True
    return False


def face_steps(corners):
    """In how many steps along each side a face of the globe is looked at: one for each EYE_FACE_SPACING_M of its longest side."""
    longest = max(float(np.linalg.norm(corners[first] - corners[second])) for first, second in ((0, 1), (1, 2), (2, 0)))
    return int(min(EYE_FACE_STEPS, max(1, np.ceil(longest / EYE_FACE_SPACING_M))))


def round_globes(eye, levels=EYE_ROUND_LEVELS):
    """Each face of the eye globes becomes four, `levels` times over, and each new point goes out to the ball of the two points it
    was born between: as far from the middle of its eye as they are, the mean of the two. Shape keys, UVs and bone weights go with
    the faces; a new point takes in every key the step it takes at rest. The points of the kit keep their numbers, and what comes
    back is, for each new point in the order of its number, the two it was born between."""
    mesh = eye.data
    parents = []
    for _ in range(levels):
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bm.verts.ensure_lookup_table()
        shapes = [bm.verts.layers.shape[name] for name in bm.verts.layers.shape.keys()]
        points = np.array([vertex.co[:] for vertex in bm.verts])
        side = to_world(eye, points)[:, 0]
        centres = []
        for half in (side < side.mean(), side >= side.mean()):
            # The ball through the points of one eye, by least squares: |p|^2 = 2 p.c + (r^2 - |c|^2).
            centres.append(np.linalg.lstsq(np.c_[2 * points[half], np.ones(int(half.sum()))], (points[half] ** 2).sum(axis=1), rcond=None)[0][:3])
        ends = [(edge.verts[0].index, edge.verts[1].index) for edge in bm.edges]
        born = KDTree(len(ends))
        for number, (first, second) in enumerate(ends):
            born.insert((points[first] + points[second]) / 2, number)
        born.balance()
        count = len(bm.verts)
        bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=1, use_grid_fill=True)
        bmesh.ops.triangulate(bm, faces=[face for face in bm.faces if len(face.verts) > 3])
        bm.verts.index_update()
        bm.verts.ensure_lookup_table()
        for vertex in bm.verts[count:]:
            first, second = ends[born.find(vertex.co)[1]]
            at = np.array(vertex.co[:])
            centre = min(centres, key=lambda middle: float(np.linalg.norm(at - middle)))
            away = at - centre
            far = (np.linalg.norm(points[first] - centre) + np.linalg.norm(points[second] - centre)) / 2
            step = Vector(away / max(np.linalg.norm(away), 1e-12) * far - away)
            vertex.co += step
            for layer in shapes:
                vertex[layer] = vertex[layer] + step
            parents.append((first, second))
        bm.to_mesh(mesh)
        bm.free()
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    mesh.update()
    return parents


def globes_behind_skin(meshes):
    """What of the eye globes is not seen through the lids goes a millimetre behind the skin, and how many points went. The last touch
    on the eyes: the head of a kit grows by its bones and its skin point by point, and under each of Dona Marta's eyes the globe came
    through the cheek as a white speck (2026-11-11), after the step that tucks the globes (`parle_face.tuck_eyes`) had run. Not a
    point of it: the globe has few faces, and the middle of one stood in front of the skin where the skin dips, its corners all behind.
    So the faces are looked at too, in the places of `EYE_FACE_STEPS`. A place of the globe is coming through when, seen from the
    front, it shows, the skin is close behind it, and skin shows somewhere between it and the middle of the eye: it is not part of
    what the lids open on. On the side of the ball (EYE_SIDE_FRONT) the first two are enough. The corners of its face that are not
    seen go back until it is behind. The face looks at -y."""
    body = next((obj for obj in meshes if obj.name == "Body"), None)
    eye = next((obj for obj in meshes if "low-poly" in obj.name), None)
    if body is None or eye is None or not len(body.data.polygons):
        return 0
    skin = BVHTree.FromPolygons([tuple(point) for point in rest_points(body)], [tuple(polygon.vertices) for polygon in body.data.polygons])
    front, back = Vector((0, -1, 0)), Vector((0, 1, 0))
    centres = [Vector(centre) for centre in eye_centres(eye)]
    faces = [tuple(polygon.vertices) for polygon in eye.data.polygons]
    triangles = [(face[0], face[index], face[index + 1]) for face in faces for index in range(1, len(face) - 1)]
    grids = {count: [(a / count, b / count) for a in range(count + 1) for b in range(count + 1 - a)] for count in range(1, EYE_FACE_STEPS + 1)}
    gone = np.zeros(len(eye.data.vertices))
    for _ in range(EYE_BEHIND_ROUNDS):
        globe = rest_points(eye)
        ball = BVHTree.FromPolygons([tuple(point) for point in globe], faces)

        def skin_first(x, z, y):
            """Whether a ray from the front meets skin before the globe at this place."""
            start = Vector((x, y - 0.05, z))
            on_skin, on_ball = skin.ray_cast(start, back, 0.1)[3], ball.ray_cast(start, back, 0.1)[3]
            return on_skin is not None and (on_ball is None or on_skin < on_ball - 1e-5)

        def through(spot, reach):
            """How far behind a place of the globe the skin is, when the place has come through it: it shows, the skin behind it
            turns its outer side to the front, and skin cuts it off from the middle of the eye. Else None."""
            at = Vector(spot)
            if skin.ray_cast(at + front * 1e-4, front, 0.05)[0] is not None:
                return None
            hit, normal, _, far = skin.ray_cast(at, back, reach)
            if hit is None or normal.dot(front) < EYE_SKIN_FACING:
                return None
            centre = min(centres, key=lambda middle: (middle - at).length)
            if far <= EYE_SIDE_REACH_M and (at - centre).normalized().dot(front) < EYE_SIDE_FRONT:
                return far
            cut = sum(skin_first(at.x + (centre.x - at.x) * share, at.z + (centre.z - at.z) * share, centre.y) for share in EYE_WAY_SHARES)
            return far if cut >= EYE_WAY_CUT else None

        move = np.zeros_like(globe)
        seen = np.zeros(len(globe), bool)
        for index, point in enumerate(globe):
            deep = through(point, EYE_BEHIND_REACH_M)
            if deep is not None:
                move[index][1] = deep + EYE_BEHIND_M
            else:
                seen[index] = skin.ray_cast(Vector(point) + front * 1e-4, front, 0.05)[0] is None
        for corners in triangles:
            hidden = [index for index in corners if not seen[index]]
            if not hidden:
                continue
            for first, second in grids[face_steps(globe[list(corners)])]:
                shares = np.array((1 - first - second, first, second))
                carried = sum(share for share, index in zip(shares, corners) if not seen[index])
                if carried < 0.2:
                    continue
                deep = through(shares @ globe[list(corners)], EYE_THROUGH_M)
                if deep is None:
                    continue
                for index in hidden:
                    move[index][1] = max(move[index][1], (deep + EYE_BEHIND_M) / carried)
        move[:, 1] = np.minimum(move[:, 1], EYE_BEHIND_MOST_M - gone)
        if not (move[:, 1] > 1e-6).any():
            break
        warp(eye, lambda world: world + move)
        gone += move[:, 1]
    return int((gone > 1e-6).sum())


def eye_scale(kit):
    """How many times their own size the eyes of a kit are."""
    return next((scale for sex, scale in EYE_SCALE_BY_SEX.items() if str(kit).startswith(sex)), EYE_SCALE)


def bigger_eyes(meshes, eye, scale=EYE_SCALE):
    """Globe and lids grow together around each eye, fading out before the nose and the temple; further out when they grow past
    EYE_SCALE, so that the skin round them is not pressed into a fold."""
    centres = eye_centres(eye)
    reach = EYE_REACH_M if scale <= EYE_SCALE else EYE_REACH_WIDE_M

    def field(points):
        out = points.copy()
        for centre in centres:
            away = points - centre
            weight = smooth((reach - np.linalg.norm(away, axis=1)) / (reach - EYE_CORE_M))
            out += away * (scale - 1) * weight[:, None]
        return out

    for obj in meshes:
        warp(obj, field)


def arm_scales(height, upper_m, fore_m, sex):
    """How much the upper arm and the forearm stretch to reach the natural share of this height."""
    low, high = ARM_SCALE_LIMITS
    return float(np.clip(ARM_UPPER_SHARE[sex] * height / upper_m, low, high)), float(np.clip(ARM_FORE_SHARE[sex] * height / fore_m, low, high))


def arm_shift(joint, elbow, wrist, upper, fore):
    """Takes world points, gives how far each moves: nothing at the shoulder, an even stretch down each bone, the hand carried whole."""
    upper_way, fore_way = elbow - joint, wrist - elbow
    upper_m, fore_m = np.linalg.norm(upper_way), np.linalg.norm(fore_way)

    def shift(points):
        on_upper = np.clip(((points - joint) @ upper_way / upper_m**2 - ARM_STILL_SHARE) / (1 - ARM_STILL_SHARE), 0, 1)
        on_fore = np.clip((points - elbow) @ fore_way / fore_m**2, 0, 1)
        return np.outer(on_upper, (upper - 1) * upper_way) + np.outer(on_fore, (fore - 1) * fore_way)

    return shift


def arm_share(obj, side):
    """How much of each vertex follows that arm, from its weights: the flank beside a hanging arm stays, the sleeve goes."""
    groups = {group.index for group in obj.vertex_groups if group.name.startswith(side) and any(part in group.name for part in ARM_BONES)}
    share = np.zeros(len(obj.data.vertices))
    for vertex in obj.data.vertices if groups else ():
        share[vertex.index] = sum(item.weight for item in vertex.groups if item.group in groups)
    return np.clip(share * 2, 0, 1)


def chest_follows_spine(meshes, rig, kit):
    """On the kits of `parle_rig.CHEST_KITS`, the front of the chest gives the weight of the arm and of the clavicle to the spine
    (`parle_rig.chest_share`): skin and cloth alike, by where each point is. Run before anything reads the poses of the body."""
    if str(kit) not in parle_rig.CHEST_KITS:
        return 0
    bpy.context.view_layer.update()
    joint = lambda name: np.array(rig.matrix_world @ rig.data.bones[name].head_local)
    shoulder, hip = (joint("LeftArm") + joint("RightArm")) / 2, (joint("LeftUpLeg") + joint("RightUpLeg")) / 2
    half = abs(joint("LeftArm")[0] - shoulder[0])
    spine = np.array(sorted((joint(name) for name in ("Hips", "Spine", "Spine1", "Spine2", "Neck")), key=lambda point: point[2]))
    height = joint("Head")[2] * 1.08 - min(joint("LeftToeBase")[2], joint("RightToeBase")[2])
    changed = 0
    for obj in meshes:
        if not obj.vertex_groups:
            continue
        names = [group.name for group in obj.vertex_groups]
        points = rest_points(obj)
        level = (points[:, 2] - hip[2]) / (shoulder[2] - hip[2])
        side = np.abs(points[:, 0] - shoulder[0]) / half
        ahead = (np.interp(points[:, 2], spine[:, 2], spine[:, 1]) - points[:, 1]) / height
        for vertex in obj.data.vertices:
            share = parle_rig.chest_share(level[vertex.index], side[vertex.index], ahead[vertex.index])
            if share <= 0:
                continue
            before = {names[item.group]: item.weight for item in vertex.groups}
            after = parle_rig.chest_weights(before, share)
            if after == before:
                continue
            changed += 1
            for name in set(before) | set(after):
                if name not in after:
                    obj.vertex_groups[name].remove([vertex.index])
                elif name in names:
                    obj.vertex_groups[name].add([vertex.index], after[name], "REPLACE")
                else:
                    obj.vertex_groups.new(name=name).add([vertex.index], after[name], "REPLACE")
                    names.append(name)
    return changed


def natural_arms(meshes, rig, height, sex):
    """Run on a finished character, in its final size. Skin, cloth and bones move by the same field, and the clips only turn the bones, so every clip still plays."""
    bpy.context.view_layer.update()
    world, back = np.array(rig.matrix_world), np.array(rig.matrix_world.inverted())
    joint = lambda name: np.array(rig.matrix_world @ rig.data.bones[name].head_local)
    shifts = {}
    for side in ("Left", "Right"):
        ends = [joint(side + part) for part in ARM_BONES]
        shifts[side] = arm_shift(*ends, *arm_scales(height, np.linalg.norm(ends[1] - ends[0]), np.linalg.norm(ends[2] - ends[1]), sex))
    for obj in meshes:
        shares = {side: arm_share(obj, side) for side in shifts}
        if not any(share.any() for share in shares.values()):
            continue
        warp(obj, lambda points: points + sum(shares[side][:, None] * shifts[side](points) for side in shifts))
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    for side, shift in shifts.items():
        upper = rig.data.edit_bones[side + "Arm"]
        for bone, ends in [(upper, ("tail",))] + [(child, ("head", "tail")) for child in upper.children_recursive]:
            for end in ends:
                point = np.array(getattr(bone, end)) @ world[:3, :3].T + world[:3, 3]
                setattr(bone, end, (point + shift(point[None])[0]) @ back[:3, :3].T + back[:3, 3])
    bpy.ops.object.mode_set(mode="OBJECT")


def group_share(obj, names):
    """How much of each vertex follows those bones, from its weights."""
    groups = {group.index for group in obj.vertex_groups if group.name in names}
    share = np.zeros(len(obj.data.vertices))
    for vertex in obj.data.vertices if groups else ():
        share[vertex.index] = sum(item.weight for item in vertex.groups if item.group in groups)
    return np.clip(share, 0, 1)


def even_follow(obj, follow, rigid, loose):
    """`follow` (how much of the neck's loss each point of a skin comes down by) made even over the mesh: the points that go whole
    with the head (`rigid` all of their weight) keep 1, those that follow neither the head nor the neck (`rigid` and `loose` nothing)
    keep 0, and every other takes the mean of its neighbours until the way from one to the other is even. By their weights alone the
    points step from the neck's share to the head's in a ring or two under the jaw and at the nape, and a neck a quarter shorter
    closed up all there, in a dark fold (2026-11-11). Points at the same place are one point."""
    points, follow = rest_points(obj), np.asarray(follow, float)
    place = {}
    for index, point in enumerate(points):
        place.setdefault(tuple(np.round(point / NECK_SAME_M).astype(int)), []).append(index)
    group = np.empty(len(points), int)
    for number, members in enumerate(place.values()):
        group[members] = number
    count = np.bincount(group, minlength=len(place)).astype(float)
    value = np.bincount(group, weights=follow, minlength=len(place)) / count
    whole = np.bincount(group, weights=np.asarray(rigid, float), minlength=len(place)) / count
    none = np.bincount(group, weights=np.asarray(rigid, float) + np.asarray(loose, float), minlength=len(place)) / count
    value[whole >= NECK_WHOLE] = 1.0
    free = (whole < NECK_WHOLE) & (none > NECK_NONE)
    edges = np.unique(np.sort(group[np.array([tuple(edge.vertices) for edge in obj.data.edges], int).reshape(-1, 2)], axis=1), axis=0)
    edges = edges[(edges[:, 0] != edges[:, 1]) & (free[edges[:, 0]] | free[edges[:, 1]])]
    near = np.bincount(edges.ravel(), minlength=len(place)).astype(float)
    for _ in range(NECK_EVEN_ROUNDS if free.any() and len(edges) else 0):
        total = np.bincount(edges[:, 0], weights=value[edges[:, 1]], minlength=len(place)) + np.bincount(edges[:, 1], weights=value[edges[:, 0]], minlength=len(place))
        value = np.where(free & (near > 0), total / np.maximum(near, 1), value)
    return value[group]


def over_skin(obj, own, skin, skin_arms):
    """The share of each arm for the points of a garment: the skin's where the cloth lies on it, its own away from it (SKIN_NEAR_M,
    SKIN_FAR_M). `skin` is the tree of the points of the skin at rest and `skin_arms` their share of each arm."""
    found = [skin.find(Vector(point)) for point in rest_points(obj)]
    # A piece with weights but no points (the short hair of the cast) gives empty lists, which numpy would read as floats.
    nearest, away = np.array([index for _, index, _ in found], dtype=int), np.array([distance for _, _, distance in found], dtype=float)
    return {side: parle_build.share_over_skin(own[side], skin_arms[side][nearest], away, SKIN_NEAR_M, SKIN_FAR_M) for side in own}


def natural_build(meshes, rig, height, kit):
    """The build of the kit's profile (`parle_build`): trunk, arms and legs get the widths found against a reference body. Run last, after the arms.
    The head keeps its shape, the spine stays, and each arm and leg goes as a whole with its bones, so every clip still plays. Cloth goes
    the way of the skin under it (`over_skin`). On the kits whose head grows against the body (`parle_build.head_ratio`), its bones grow
    with it, and where the neck is shorter (`parle_build.neck_ratio`) the head and its bones come down; the body comes back to the
    stature it had. No profile, no change."""
    profile = parle_build.load_profile(kit)
    if profile is None:
        return False
    bpy.context.view_layer.update()
    back = np.array(rig.matrix_world.inverted())[:3, :3]
    joints = {name: np.array(rig.matrix_world @ rig.data.bones[name].head_local) for name in parle_build.BONES}
    field, rides = parle_build.build_field(profile, joints, height)
    head = {"Head"} | {bone.name for bone in rig.data.bones["Head"].children_recursive}
    body = next((obj for obj in meshes if obj.name.split(".")[0] == "Body" and obj.vertex_groups), None)
    grow, stature = parle_build.head_ratio(profile.get("kit", "")), float(np.ptp(rest_points(body)[:, 2])) if body else 0.0
    short = parle_build.neck_ratio(profile.get("kit", ""))
    # The head comes down by what the neck loses, in the rig's own frame for its bones.
    down = -(1 - short) * (joints["Head"] - joints["Neck"]) @ back.T
    skin, skin_arms = KDTree(len(body.data.vertices) if body else 0), {side: arm_share(body, side) for side in parle_build.SIDES} if body else {}
    for index, point in enumerate(rest_points(body) if body else ()):
        skin.insert(point, index)
    skin.balance()
    follows = lambda obj: parle_build.neck_follow(rest_points(obj), group_share(obj, head), group_share(obj, {"Neck"}), joints)
    # The skin shares the neck's loss out evenly, and cloth goes the way of the skin under it; hair, glasses and the rest go by their own weights.
    skin_follow = even_follow(body, follows(body), group_share(body, head), group_share(body, {"Neck"})) if body is not None and short != 1 else None
    for obj in meshes:
        if not obj.vertex_groups:
            continue
        arms, top, follow = {side: arm_share(obj, side) for side in parle_build.SIDES}, group_share(obj, head), follows(obj)
        if body is not None and obj is not body:
            arms = over_skin(obj, arms, skin, skin_arms)
        if skin_follow is not None:
            follow = skin_follow if obj is body else over_skin(obj, {"neck": follow}, skin, {"neck": skin_follow})["neck"] if obj.name.startswith("Outfit_") else follow
        warp(obj, lambda points: points + field(points, arms, top, follow))
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    moves = []
    for name, ride in rides.items():
        first, local = rig.data.edit_bones[name], ride @ back.T
        # The collar bone ends where the arm starts.
        moves += [(first.parent, "tail", np.array(first.parent.tail) + local)] if name.endswith("Arm") else []
        for bone in [first] + list(first.children_recursive):
            moves += [(bone, end, np.array(getattr(bone, end)) + local) for end in ("head", "tail")]
    # The bones of the head grow with it, about its joint.
    crown = rig.data.edit_bones["Head"]
    base = np.array(crown.head)
    for bone, ends in [(crown, ("tail",))] + [(child, ("head", "tail")) for child in crown.children_recursive]:
        moves += [(bone, end, base + grow * (np.array(getattr(bone, end)) - base) + down) for end in ends]
    # And come down with it where the neck is shorter: the joint of the head, which is the end of the bone of the neck.
    moves += [(crown, "head", base + down), (rig.data.edit_bones["Neck"], "tail", base + down)]
    # Every end is read before any is written: a bone joined to its parent would otherwise move twice.
    for bone, end, point in moves:
        setattr(bone, end, point)
    bpy.ops.object.mode_set(mode="OBJECT")
    if (grow != 1 or short != 1) and body is not None:
        # A bigger head on the same body stands taller, and a shorter neck lower: the whole comes back to the stature it had.
        rig.scale = [rig.scale[0] * stature / float(np.ptp(rest_points(body)[:, 2]))] * 3
        bpy.context.view_layer.update()
    return True


def top_over_bottom(tops, bottoms):
    """At rest, moves out the points of the top that stand inside the bottom, or closer to it than OVER_GAP_M. In each round the push is
    shared with the points around, so the hem keeps its line, and the halves of a point that a seam of the picture splits go together;
    a last pass sets whatever is still short of the gap. Returns how many points moved."""
    if not tops or not bottoms:
        return 0
    points, faces = [], []
    for obj in bottoms:
        obj.data.calc_loop_triangles()
        start = len(points)
        points.extend(map(tuple, rest_points(obj)))
        faces.extend(tuple(int(index) + start for index in triangle.vertices) for triangle in obj.data.loop_triangles)
    tree = BVHTree.FromPolygons(points, faces)

    def short(point):
        """How far a point of the top has to go, and which way, to stand OVER_GAP_M outside the bottom."""
        location, normal, _, _ = tree.find_nearest(Vector(point), OVER_REACH_M)
        need = 0.0 if location is None else OVER_GAP_M - (Vector(point) - location).dot(normal)
        return (np.array(normal) * need) if need > 0 else np.zeros(3)

    count = 0
    for obj in tops:
        rest = rest_points(obj)
        around = [[] for _ in rest]
        for edge in obj.data.edges:
            first, second = edge.vertices
            around[first].append(second)
            around[second].append(first)
        spots = {}
        twin = np.array([spots.setdefault(tuple(np.round(point, 4)), len(spots)) for point in rest])
        halves = np.bincount(twin, minlength=len(spots))[:, None]
        moved = np.zeros_like(rest)
        for _ in range(OVER_ROUNDS):
            step = np.array([short(point) for point in rest + moved])
            lengths = np.linalg.norm(step, axis=1, keepdims=True)
            step = step * np.minimum(1.0, OVER_STEP_M / np.maximum(lengths, 1e-9))
            if not lengths.any():
                break
            for _ in range(OVER_SHARED):
                step = np.array([(step[index] + step[near].mean(axis=0)) / 2 if near else step[index] for index, near in enumerate(around)])
            together = np.zeros((len(spots), 3))
            np.add.at(together, twin, step)
            moved += (together / halves)[twin]
        last = np.array([short(point) for point in rest + moved])
        together = np.zeros((len(spots), 3))
        np.add.at(together, twin, last)
        moved += (together / halves)[twin]
        gone = np.linalg.norm(moved, axis=1) > 1e-5
        if gone.any():
            count += int(gone.sum())
            warp(obj, lambda where, moved=moved: where + moved)
    return count


def seamless_normals(obj):
    """Gives the points of a mesh that stand at the same place the same normal, and returns how many places had more than one. The skin
    comes from a glTF, which splits a point wherever the picture has a seam: two rings of points at the same place, each lit by its own
    faces alone. Where the head meets the neck the two normals stood 4 to 20 degrees apart, and the light drew the seam as a line across
    the nape and down the side of the neck, a collar of another skin (Carlos, 2026-11-11); the colour was the same on both sides."""
    mesh = obj.data
    points = np.round(read(mesh.vertices), 5)
    _, place, count = np.unique(points, axis=0, return_inverse=True, return_counts=True)
    place = place.ravel()
    point_of = np.empty(len(mesh.loops), dtype=np.int64)
    mesh.loops.foreach_get("vertex_index", point_of)
    corners = np.empty(len(mesh.loops) * 3)
    mesh.corner_normals.foreach_get("vector", corners)
    corners = corners.reshape(-1, 3)
    # Each corner of a face brings the normal of its point: a place with more faces around it on one side weighs more on that side.
    total = np.zeros((len(count), 3))
    np.add.at(total, place[point_of], corners)
    total /= np.maximum(np.linalg.norm(total, axis=1, keepdims=True), 1e-9)
    shared = count[place[point_of]] > 1
    corners[shared] = total[place[point_of]][shared]
    mesh.normals_split_custom_set(corners.tolist())
    mesh.update()
    return int((count > 1).sum())


def round_bust(meshes, rig):
    """Under a top each breast of a woman takes the shape of a cup, as in a bra (`parle_bust`): the pieces of the top, and the skin
    with them. Run at rest on the finished body. Returns how far each tip came back, in metres; nothing when no top covers the bust."""
    tops = [obj for obj in meshes if obj.name.startswith("Outfit_Top")]
    if not tops:
        return []
    joint = lambda name: np.array(rig.matrix_world @ rig.data.bones[name].head_local)
    hip, shoulder = (joint("LeftUpLeg")[2] + joint("RightUpLeg")[2]) / 2, (joint("LeftArm")[2] + joint("RightArm")[2]) / 2
    middle = (joint("LeftArm")[0] + joint("RightArm")[0]) / 2
    cloth = np.concatenate([rest_points(obj) for obj in tops])
    reach, cups = parle_bust.REACH * float(np.ptp(np.concatenate([rest_points(obj) for obj in meshes if obj.name.split(".")[0] == "Body"])[:, 2])), []
    for side in (1, -1):
        tip = parle_bust.tip_of(cloth, hip, shoulder, middle, abs(joint("LeftArm")[0] - middle), side)
        found = parle_bust.rim_of(cloth, tip, reach) if tip is not None else None
        if found is not None and found[1] >= parle_bust.TALL_LEAST_M:
            cups.append((tip, *found))
    for obj in [obj for obj in meshes if obj in tops or obj.name.split(".")[0] == "Body"] if cups else ():
        warp(obj, lambda points: points + sum(parle_bust.cupped(points, tip, axis, tall, reach) for tip, axis, tall in cups))
    return [(1 - parle_bust.CUP) * tall for _, _, tall in cups]


def eye_balls(eye):
    """The ball of each eye globe, middle and radius, by least squares through its points: |p|^2 = 2 p.c + (r^2 - |c|^2). Twice,
    the second time without the points that stand off the first ball (the ones tucked behind the skin)."""
    points = rest_points(eye)
    balls = []
    for half in (points[:, 0] < points[:, 0].mean(), points[:, 0] >= points[:, 0].mean()):
        mine = points[half]
        for _ in range(2):
            found = np.linalg.lstsq(np.c_[2 * mine, np.ones(len(mine))], (mine ** 2).sum(axis=1), rcond=None)[0]
            centre, radius = found[:3], float(np.sqrt(found[3] + found[:3] @ found[:3]))
            near = np.abs(np.linalg.norm(mine - centre, axis=1) - radius) < 0.0005
            if near.sum() < 8:
                break
            mine = mine[near]
        balls.append((centre, radius))
    return balls


def lids_over_globes(body, balls):
    """The skin of the lids out of the eye globes, `balls` being the middle and the radius of each (`eye_balls`): how many points
    went, and how far the one that went furthest. Every shape key takes the move. Points at the same place are one point."""
    blocks = body.data.shape_keys.key_blocks if body.data.shape_keys else None
    rest = read(blocks[0]) if blocks else read(body.data.vertices)
    world = to_world(body, rest)
    turn = np.array(body.matrix_world)[:3, :3]
    normals = np.array([vertex.normal[:] for vertex in body.data.vertices]) @ turn.T
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    _, group = np.unique(np.round(world / FAIR_SAME_M).astype(np.int64), axis=0, return_inverse=True)
    group = group.ravel()
    move = np.zeros_like(world)
    for centre, radius in balls:
        away = world - np.asarray(centre, float)
        far = np.maximum(np.linalg.norm(away, axis=1), 1e-12)
        out = away / far[:, None]
        share = smooth((-out[:, 1] - LID_ROUND_FRONT[0]) / (LID_ROUND_FRONT[1] - LID_ROUND_FRONT[0]))
        lift = (far > radius - LID_UNDER_BALL_M) & (far < radius + LID_OFF_BALL_M) & ((normals * out).sum(axis=1) > LID_FACING) & (share > 0)
        lift = np.bincount(group, weights=lift.astype(float), minlength=group.max() + 1)[group] > 0
        lift &= (far > radius - LID_UNDER_BALL_M) & (far < radius + LID_OFF_BALL_M) & (share > 0)
        move[lift] = out[lift] * ((radius + LID_OFF_BALL_M - far[lift]) * share[lift])[:, None]
    went = np.linalg.norm(move, axis=1)
    if not (went > 1e-7).any():
        return 0, 0.0
    local = move @ np.linalg.inv(turn).T
    for block in blocks or ():
        block.data.foreach_set("co", (read(block) + local).ravel())
    body.data.vertices.foreach_set("co", (read(body.data.vertices) + local).ravel())
    body.data.update()
    return int((went > 1e-7).sum()), float(went.max())


def lid_ways(world, normals, group, edges, centre, radius):
    """Round one ball: how far each point of skin that shows is from the rim of the opening of the lids (np.inf where it does not
    show, or is further than LID_WAY_MOST_M), and which points are that rim. `group` numbers the places: points at the same place
    are one. The way is the straight one to the line of the rim, drawn from one point of it to the next: the skin of a kit has
    few points there, the rim of Lúcia's upper lid ran 12 mm from one to the next over the middle of the eye, and counted along
    the edges the lip that stands 1 mm from that line was 5.5 mm from the rim."""
    away = world - centre
    far = np.maximum(np.linalg.norm(away, axis=1), 1e-12)
    near = far < radius * LID_NEAR_BALLS
    aside = away[:, 0] ** 2 + away[:, 2] ** 2
    shadowed = near & (aside < radius * radius) & (away[:, 1] > -np.sqrt(np.maximum(radius * radius - aside, 0.0)))
    places = int(group.max()) + 1
    facing = np.bincount(group, weights=(normals * away).sum(axis=1) / far, minlength=places) / np.maximum(np.bincount(group, minlength=places), 1)
    hidden = np.zeros(places, bool)
    hidden[group[shadowed]] = True
    deep = np.zeros(places, bool)
    deep[group[far < radius - LID_UNDER_BALL_M]] = True
    hidden &= (facing < LID_FACING) | deep
    ends = edges[near[edges[:, 0]] & near[edges[:, 1]]]
    ends = np.r_[ends, ends[:, ::-1]]
    first, second = group[ends[:, 0]], group[ends[:, 1]]
    spot = np.zeros((places, 3))
    spot[group] = world
    off = np.maximum(np.linalg.norm(spot - centre, axis=1), 1e-12)
    rim = np.zeros(places, bool)
    rim[first[~hidden[first] & hidden[second]]] = True
    rim &= (off < radius + LID_RIM_REACH_M) & (-(spot - centre)[:, 1] / off > LID_RIM_FRONT)
    line = rim[first] & rim[second]
    way = np.full(len(world), np.inf)
    if not line.any():
        return way, rim[group]
    start, along = spot[first[line]], spot[second[line]] - spot[first[line]]
    long = np.maximum((along * along).sum(axis=1), 1e-18)
    for index in np.flatnonzero(near & ~hidden[group]):
        share = np.clip(((world[index] - start) * along).sum(axis=1) / long, 0.0, 1.0)
        way[index] = np.sqrt((((start + along * share[:, None]) - world[index]) ** 2).sum(axis=1).min())
    way[way > LID_WAY_MOST_M] = np.inf
    way[rim[group]] = 0.0
    return way, rim[group]


def lids_with_a_fold(body, balls, lip=LID_LIP_M, fold=LID_FOLD_M):
    """The upper lids get a fold above them and stand a little further off the globe below it, `balls` being the middle and the
    radius of each globe (`eye_balls`): how many points of skin moved, and how far the one that moved most. The rim of the
    opening stays. Each point moves along the line from the middle of its ball. Every shape key takes the move but the ones that
    shut the lids (BLINKS), where a point they move keeps the place the kit shuts it at: the lids still meet."""
    blocks = body.data.shape_keys.key_blocks if body.data.shape_keys else None
    rest = read(blocks[0]) if blocks else read(body.data.vertices)
    world = to_world(body, rest)
    turn = np.array(body.matrix_world)[:3, :3]
    normals = np.array([vertex.normal[:] for vertex in body.data.vertices]) @ turn.T
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    _, group = np.unique(np.round(world / FAIR_SAME_M).astype(np.int64), axis=0, return_inverse=True)
    group = group.ravel()
    edges = np.zeros(len(body.data.edges) * 2, dtype=np.int64)
    body.data.edges.foreach_get("vertices", edges)
    edges = edges.reshape(-1, 2)
    move = np.zeros_like(world)
    for centre, radius in balls:
        centre = np.asarray(centre, float)
        way, rim = lid_ways(world, normals, group, edges, centre, radius)
        if rim.sum() < 8:
            continue
        away = world - centre
        far = np.maximum(np.linalg.norm(away, axis=1), 1e-12)
        across = world[rim, 0]
        low, high = float(across.min()), float(across.max())
        along = np.clip((world[:, 0] - low) / max(high - low, 1e-9), 0.0, 1.0)
        outer = along if centre[0] < 0 else 1 - along
        middle = smooth(np.minimum(outer / LID_CORNER["outer"], (1 - outer) / LID_CORNER["inner"]))
        lid = np.isfinite(way) & ~rim & (world[:, 2] > np.interp(world[:, 0], [low, high], [world[rim][across.argmin(), 2], world[rim][across.argmax(), 2]]))
        reach = way / (LID_SHORT + (1 - LID_SHORT) * middle)
        tall = np.interp(reach, [0, LID_LIP_AT_M, LID_PLATE_M, LID_FOLD_AT_M], [0, lip, lip * LID_PLATE_SHARE, 0])
        target = np.where(lid, np.maximum(far, radius + tall * middle), far)
        past = (reach - LID_FOLD_AT_M) / np.where(reach < LID_FOLD_AT_M, LID_FOLD_REACH_M[0], LID_FOLD_REACH_M[1])
        pull = np.where(lid & (np.abs(past) < 1), 0.5 + 0.5 * np.cos(np.pi * np.clip(past, -1, 1)), 0.0) * fold * middle
        target = np.maximum(target - pull, np.minimum(far, radius + LID_OFF_BALL_M))
        move += away / far[:, None] * (target - far)[:, None]
    went = np.linalg.norm(move, axis=1)
    if not (went > 1e-7).any():
        return 0, 0.0
    local = move @ np.linalg.inv(turn).T
    for block in blocks or ():
        points = read(block)
        kept = 1 - smooth(np.linalg.norm(points - rest, axis=1) / BLINK_WHOLE_M)[:, None] if block.name in BLINKS else 1.0
        block.data.foreach_set("co", (points + local * kept).ravel())
    body.data.vertices.foreach_set("co", (read(body.data.vertices) + local).ravel())
    body.data.update()
    return int((went > 1e-7).sum()), float(went.max())


def blinks_shut(meshes):
    """With each lid shut (BLINKS) the upper lid lies on the globe and comes down until nothing of the globe shows from the
    front: for each blink, how tall the slit was and how tall what is left of it, in metres. Only the shape that shuts the lid
    changes: the rest shape, the lower lid and every other shape stay. The face looks at -y."""
    body = next((obj for obj in meshes if obj.name == "Body"), None)
    eye = next((obj for obj in meshes if "low-poly" in obj.name), None)
    blocks = body.data.shape_keys.key_blocks if body is not None and body.data.shape_keys else None
    if eye is None or not blocks or not len(body.data.polygons) or not len(eye.data.polygons):
        return {}
    faces = [tuple(polygon.vertices) for polygon in body.data.polygons]
    globe = BVHTree.FromPolygons([tuple(point) for point in rest_points(eye)], [tuple(polygon.vertices) for polygon in eye.data.polygons])
    balls = eye_balls(eye)
    rest = to_world(body, read(blocks[0]))
    back = Vector((0, 1, 0))
    found = {}
    for name in BLINKS:
        if name not in blocks:
            continue
        shut = to_world(body, read(blocks[name]))
        way = np.linalg.norm(shut - rest, axis=1)
        if way.max() < BLINK_SHUT_LEAST_M:
            continue
        centre, radius = min(balls, key=lambda ball: float(np.linalg.norm(rest[way > way.max() / 2].mean(axis=0) - ball[0])))
        centre = np.asarray(centre, float)
        share = np.clip(way / (way.max() * BLINK_SHUT_WHOLE), 0.0, 1.0) * ((shut - rest)[:, 2] < 0)

        def on_the_ball(points):
            """The skin of the shut lid out of the ball, where it stands in it by no more than the inside of a lid does."""
            away = points - centre
            far = np.maximum(np.linalg.norm(away, axis=1), 1e-12)
            lift = (share > 0) & (far > radius - LID_UNDER_BALL_M) & (far < radius + LID_OFF_BALL_M) & (-away[:, 1] / far > LID_RIM_FRONT)
            points[lift] = centre + away[lift] / far[lift, None] * (radius + LID_OFF_BALL_M)
            return int(lift.sum())

        lifted = on_the_ball(shut)
        across = np.arange(-BLINK_SHUT_ACROSS * radius, BLINK_SHUT_ACROSS * radius, BLINK_SHUT_STEP_M)
        up = np.arange(-BLINK_SHUT_UP * radius, BLINK_SHUT_UP * radius, BLINK_SHUT_STEP_M)
        wide = int(round(BLINK_SHUT_REACH_M / BLINK_SHUT_STEP_M))
        gone = np.zeros(len(shut))
        first = last = 0.0
        for turn in range(BLINK_SHUT_ROUNDS + 1):
            skin = BVHTree.FromPolygons([tuple(point) for point in shut], faces)
            tall = np.zeros(len(across))
            for column, aside in enumerate(across):
                for above in up:
                    if aside * aside + above * above >= radius * radius:
                        continue
                    start = Vector((centre[0] + aside, centre[1] - radius - 0.05, centre[2] + above))
                    on_globe = globe.ray_cast(start, back, 0.2)[3]
                    if on_globe is None:
                        continue
                    on_skin = skin.ray_cast(start, back, 0.2)[3]
                    if on_skin is None or on_skin > on_globe + 1e-5:
                        tall[column] += BLINK_SHUT_STEP_M
            last = float(tall.max())
            first = last if turn == 0 else first
            if last == 0 or turn == BLINK_SHUT_ROUNDS:
                break
            asked = np.array([tall[max(0, column - wide):column + wide + 1].max() for column in range(len(tall))])
            asked = np.where(asked > 0, asked + BLINK_SHUT_OVER_M, 0.0)
            drop = np.interp(shut[:, 0] - centre[0], across, asked, left=0.0, right=0.0) * share
            if turn == 0:
                most = np.minimum(drop * BLINK_SHUT_PAST, BLINK_SHUT_MOST_M * share)
            drop = np.clip(np.minimum(drop, most - gone), 0.0, None)
            if not (drop > 1e-7).any():
                break
            shut[:, 2] -= drop
            gone += drop
            on_the_ball(shut)
        if lifted or gone.any():
            blocks[name].data.foreach_set("co", to_local(body, shut).ravel())
        found[name] = (first, last)
    body.data.update()
    return found


def fair_face(body, rig, rounds=FAIR_ROUNDS, balls=()):
    """The skin of the face evened out, in the rest shape and so in every shape key: how many points moved, and how far the one that
    moved most went.

    Owner, 2026-11-11: "o rosto da Lúcia está muito feio", with the aim given before: "avatares tipo o Rafa e Sofia, para serem mais
    estilo Disney". In clay, next to Sofia's face and to the same woman's face in the game, both smooth from the eye to the chin, the
    cast's had the folds of a real face: one from the wing of the nose to the corner of the mouth, a dent at each corner, lumps on
    the chin, a hollow under the cheekbone. Some come with the base and some from the traits of the kit laid one over another, each
    with a fall-off of its own. The ruler of `parle_beauty` sees none of them: it reads points, not the surface.

    The zone is where the soft traits of the face act (FAIR_SOFT), on the skin of the head; the lids keep their points (FAIR_KEPT),
    the nose and the lips move a little (FAIR_NOSE, FAIR_LIPS), the lips not where they meet, and what is inside the head stays.
    Points at the same place are one point. `balls` are the eye globes, middle and radius (`eye_balls`): the skin stays off them."""
    blocks = body.data.shape_keys.key_blocks if body.data.shape_keys else None
    if blocks is None or not any(name in blocks for name in FAIR_SOFT):
        return 0, 0.0
    rest = read(blocks[0])
    world = to_world(body, rest)
    share = lambda name: (lambda far: far / max(far.max(), 1e-9))(np.linalg.norm(read(blocks[name]) - rest, axis=1))
    soft = np.max([share(name) for name in FAIR_SOFT if name in blocks], axis=0)
    kept = np.max([share(name) / least for name, least in FAIR_KEPT.items() if name in blocks] or [np.zeros(len(rest))], axis=0)
    head = {"Head"} | {bone.name for bone in rig.data.bones["Head"].children_recursive}
    nose = np.clip(np.max([share(name) / least for name, least in FAIR_NOSE_BY.items() if name in blocks] or [np.zeros(len(rest))], axis=0), 0, 1)
    weight = (smooth(soft / FAIR_SOFT_FROM) * (1 - nose) + FAIR_NOSE * nose) * (1 - np.clip(kept, 0, 1)) * (group_share(body, head) >= FAIR_HEAD_FROM)
    lips = np.max([share(name) for name in FAIR_LIPS if name in blocks] or [np.zeros(len(rest))], axis=0) >= FAIR_LIP_FROM
    weight[lips] = np.minimum(weight[lips], FAIR_LIP)
    if "jawOpen" in blocks and lips.any():
        carried = np.linalg.norm(read(blocks["jawOpen"]) - rest, axis=1)
        lower = lips & (carried > FAIR_JAW_SHARE * carried[lips].max())
        for mine, other in ((lower, lips & ~lower), (lips & ~lower, lower)):
            if not mine.any() or not other.any():
                continue
            across = KDTree(int(other.sum()))
            for index in np.flatnonzero(other):
                across.insert(world[index], int(index))
            across.balance()
            for index in np.flatnonzero(mine):
                if across.find(world[index])[2] < FAIR_MEET_M:
                    weight[index] = 0.0
    tree = BVHTree.FromPolygons(world.tolist(), [tuple(polygon.vertices) for polygon in body.data.polygons])
    ahead = Vector((0, -1, 0))
    for index in np.flatnonzero(weight > 0):
        if tree.ray_cast(Vector(world[index]) + ahead * 0.0005, ahead, FAIR_INSIDE_M)[0] is not None:
            weight[index] = 0.0
    place = {}
    for index, point in enumerate(world):
        place.setdefault(tuple(np.round(point / FAIR_SAME_M).astype(int)), []).append(index)
    group = np.empty(len(world), int)
    for number, members in enumerate(place.values()):
        group[members] = number
    count = np.bincount(group, minlength=len(place)).astype(float)
    mean = lambda values: np.bincount(group, weights=values, minlength=len(place)) / count
    points = np.stack([mean(world[:, axis]) for axis in range(3)], axis=1)
    held = np.bincount(group, weights=(weight <= 0).astype(float), minlength=len(place)) > 0
    floors = []
    for centre, radius in balls:
        far = np.linalg.norm(points - centre, axis=1)
        held |= far < radius - FAIR_ON_BALL_M
        floors.append((np.asarray(centre, float), np.where(far < radius + FAIR_NEAR_BALL_M, np.minimum(far, radius + FAIR_OFF_BALL_M), 0.0)))
    free = np.where(held, 0.0, mean(weight))
    edges = np.unique(np.sort(group[np.array([tuple(edge.vertices) for edge in body.data.edges], int).reshape(-1, 2)], axis=1), axis=0)
    edges = edges[(edges[:, 0] != edges[:, 1]) & ((free[edges[:, 0]] > 0) | (free[edges[:, 1]] > 0))]
    if not len(edges):
        return 0, 0.0
    near = np.maximum(np.bincount(edges.ravel(), minlength=len(place)).astype(float), 1)

    def middle(values):
        """The mean of each point's neighbours."""
        total = np.zeros_like(values)
        np.add.at(total, edges[:, 0], values[edges[:, 1]])
        np.add.at(total, edges[:, 1], values[edges[:, 0]])
        return total / (near if values.ndim == 1 else near[:, None])

    for _ in range(FAIR_EDGE_ROUNDS):
        free = np.where(free > 0, np.minimum(free, middle(free)), 0.0)
    start = points.copy()
    for _ in range(rounds):
        for step in (FAIR_STEP, FAIR_BACK):
            points = points + step * free[:, None] * (middle(points) - points)
        for centre, least in floors:
            away = points - centre
            far = np.maximum(np.linalg.norm(away, axis=1), 1e-9)
            low = (far < least) & (free > 0)
            points[low] = centre + away[low] * (least[low] / far[low])[:, None]
    matrix = np.array(body.matrix_world.inverted())[:3, :3]
    move = ((points - start) @ matrix.T)[group]
    for block in blocks:
        block.data.foreach_set("co", (read(block) + move).ravel())
    body.data.vertices.foreach_set("co", (read(body.data.vertices) + move).ravel())
    body.data.update()
    far = np.linalg.norm(points - start, axis=1)
    return int((far > 1e-4).sum()), float(far.max())


def lining_behind_skin(meshes):
    """The inside of the mouth back behind the skin of the face where it comes through it, and how many points went.

    2026-11-12: a flake showed on each of Lúcia's cheeks beside the wing of the nose, wine on one side and lilac on the other, the
    colours of the inside of the mouth. It was the top of the lining, a face of it half a millimetre in front of the cheek: the
    traits that draw the cheek in beside the nose do not take the lining with them (a narrower nose brought out more of it), and
    `parle_face.keep_lining_inside` holds it for the shapes of speech, not against the rest shape of the skin.

    The lining is the piece of the map the inside of the mouth sits on (`parle_mouth.mouth_piece`). A place of one of its faces has
    come through when outer skin that turns its outer side to the front lies behind it, or less than LINING_BEHIND_M ahead of it; the
    corners of that face go back (+y: the face looks at -y) until it is behind. Skin of the lips does not count: there the lining and
    the skin are one."""
    import parle_mouth

    body = next((obj for obj in meshes if obj.name == "Body"), None)
    teeth = next((obj for obj in meshes if "teeth" in obj.name), None)
    blocks = body.data.shape_keys.key_blocks if body is not None and body.data.shape_keys else None
    if blocks is None or teeth is None or not body.data.uv_layers.active:
        return 0
    loops = body.data.uv_layers.active.data
    uv = np.zeros((len(body.data.vertices), 2))
    for loop in body.data.loops:
        uv[loop.vertex_index] = loops[loop.index].uv[0], 1 - loops[loop.index].uv[1]
    faces = [tuple(polygon.vertices) for polygon in body.data.polygons]
    triangles = np.array([(face[0], second, third) for face in faces for second, third in zip(face[1:], face[2:])])
    piece = parle_mouth.mouth_piece(rest_points(body), uv, rest_points(teeth), triangles)
    if not piece.any():
        return 0
    rest = read(blocks[0])
    share = lambda name: (lambda far: far / max(far.max(), 1e-9))(np.linalg.norm(read(blocks[name]) - rest, axis=1))
    lips = np.max([share(name) for name in FAIR_LIPS if name in blocks] or [np.zeros(len(rest))], axis=0) >= FAIR_LIP_FROM
    outer = [face for face in faces if not piece[list(face)].any() and not lips[list(face)].any()]
    lining = triangles[piece[triangles].all(axis=1)]
    front, back = Vector((0, -1, 0)), Vector((0, 1, 0))
    gone = np.zeros(len(rest))
    for _ in range(LINING_ROUNDS):
        world = rest_points(body)
        skin = BVHTree.FromPolygons(world.tolist(), outer)

        def through(spot):
            """How far back a place of the lining has to go to be behind the skin, or nothing."""
            at = Vector(spot)
            hit, normal, _, far = skin.ray_cast(at, back, LINING_REACH_M)
            if hit is not None and normal.dot(front) > LINING_SKIN_FACING:
                return far + LINING_BEHIND_M
            hit, normal, _, far = skin.ray_cast(at, front, LINING_BEHIND_M)
            if hit is not None and normal.dot(front) > LINING_SKIN_FACING:
                return LINING_BEHIND_M - far
            return 0.0

        move = np.zeros_like(world)
        for corners in lining:
            deep = max(through(np.asarray(place) @ world[corners]) for place in LINING_PLACES)
            if deep > 0:
                move[corners, 1] = np.maximum(move[corners, 1], deep)
        move[:, 1] = np.minimum(move[:, 1], LINING_MOST_M - gone)
        if not (move[:, 1] > 1e-6).any():
            break
        warp(body, lambda points: points + move)
        gone += move[:, 1]
    return int((gone > 1e-6).sum())


def brows_on_skin(meshes):
    """The eyebrows laid on the skin, and how far the furthest point of them stood off it. The kit's eyebrow is a strip of its own
    over the brow ridge; after the traits of the face, the eyes and the head have each moved the skin under it, its inner end sat
    1 mm inside the skin and its outer end 3 mm above it, and from the side it stood out like a visor (the judge by sight,
    2026-11-12). Every point goes to BROW_OFF_SKIN_M off the nearest skin, in the rest shape and so in every shape key."""
    body = next((obj for obj in meshes if obj.name == "Body"), None)
    brows = [obj for obj in meshes if "eyebrow" in obj.name]
    if body is None or not brows or not len(body.data.polygons):
        return 0.0
    skin = BVHTree.FromPolygons([tuple(point) for point in rest_points(body)], [tuple(polygon.vertices) for polygon in body.data.polygons])
    furthest = 0.0
    for brow in brows:
        points = rest_points(brow)
        move = np.zeros_like(points)
        for index, point in enumerate(points):
            at, normal, _, far = skin.find_nearest(Vector(point))
            if at is None:
                continue
            furthest = max(furthest, far)
            move[index] = np.array(at + normal * BROW_OFF_SKIN_M) - point
        warp(brow, lambda world, move=move: world + move)
    return furthest


def natural_body(meshes, rig, height, sex, kit):
    """The body of a finished character: the arms, then the build of the kit. The cast and the loop that finds the build (`body_match`, through
    `body_dump`) both call this and nothing else, so the body under the clothes is the body that was matched to the reference. A shoulder polish
    once ran before these steps in the cast alone: dressed, a man's shoulders came out 6 to 8 cm narrower on each side, at the levels above the
    armpit, than the body the loop had measured (owner, 2026-11-10: the dressed character was not the silhouette that had been shown). Last,
    the top goes back outside the bottom (`top_over_bottom`): that moves cloth alone, never the body. And under a top the bust of a
    woman is a bra's (`round_bust`): with no top on, as in the loop, nothing moves. The legs of the bottoms end on a level line
    (`level_hems`). Then the face: the lids lie over the eye globes (`lids_over_globes`), its skin is evened out (`fair_face`), the
    upper lids get their fold (`lids_with_a_fold`, after the evening out, which would take it away), the inside of the mouth
    stays behind the skin (`lining_behind_skin`), the eyebrows lie on it (`brows_on_skin`), the eye globes stay behind the skin
    where the lids do not show them (`globes_behind_skin`), and a lid that shuts shuts on its globe (`blinks_shut`)."""
    natural_arms(meshes, rig, height, sex)
    built = natural_build(meshes, rig, height, kit)
    over = top_over_bottom(*([obj for obj in meshes if obj.name.startswith(prefix)] for prefix in ("Outfit_Top", "Outfit_Bottom")))
    if over:
        print("BARRA", over, "pontos da peça de cima voltaram para fora da de baixo")
    cups = round_bust(meshes, rig) if sex == "female" else []
    if cups:
        print("BUSTO em taça debaixo da roupa: a ponta de cada seio voltou", [round(back * 1000, 1) for back in cups], "mm")
    hems = level_hems(meshes)
    if hems:
        print("BARRA nivelada em", hems, "aberturas de perna")
    body = next((obj for obj in meshes if obj.name == "Body"), None)
    globe = next((obj for obj in meshes if "low-poly" in obj.name), None)
    balls = eye_balls(globe) if globe is not None and body is not None else ()
    laid, far = lids_over_globes(body, balls) if balls else (0, 0.0)
    if laid:
        print("PÁLPEBRA sobre o globo:", laid, "pontos da pele, o que mais andou", round(far * 1000, 1), "mm")
    faired, far = fair_face(body, rig, balls=balls) if body is not None else (0, 0.0)
    if faired:
        print("ROSTO alisado:", faired, "pontos da pele, o que mais andou", round(far * 1000, 1), "mm")
    folded, far = lids_with_a_fold(body, balls) if balls else (0, 0.0)
    if folded:
        print("PÁLPEBRA de cima com dobra:", folded, "pontos da pele, o que mais andou", round(far * 1000, 1), "mm")
    lined = lining_behind_skin(meshes)
    if lined:
        print("FORRO DA BOCA:", lined, "pontos de volta para trás da pele do rosto")
    off = brows_on_skin(meshes)
    if off:
        print("SOBRANCELHA deitada na pele: o ponto mais longe estava a", round(off * 1000, 1), "mm")
    tucked = globes_behind_skin(meshes)
    if tucked:
        print("OLHO", tucked, "pontos do globo de volta para trás da pele, depois do corpo pronto")
    for name, (was, left) in blinks_shut(meshes).items():
        print("PISCADA", name, "fresta de", round(was * 1000, 1), "mm com a pálpebra fechada; ficou", round(left * 1000, 1), "mm")
    return built


def garment_normal(name):
    """The garment's own normal map in the MakeHuman assets, or None when its author drew none."""
    base = name.split(".")[0]
    for folder in (base, base.split("-")[0]):
        found = sorted(glob.glob(os.path.join(CLOTHES, folder, "*[nN]ormal*")))
        if found:
            return found[0]
    return None


def cloth_relief(obj, path):
    """The folds drawn by the garment's own author: its normal map, at a size the phone can pay for."""
    image = bpy.data.images.load(path, check_existing=True)
    image.colorspace_settings.name = "Non-Color"
    if image.size[0] > NORMAL_PX:
        image.scale(NORMAL_PX, NORMAL_PX)
    image.pack()
    for material in obj.data.materials:
        tree = material.node_tree
        texture = tree.nodes.new("ShaderNodeTexImage")
        texture.image = image
        normal = tree.nodes.new("ShaderNodeNormalMap")
        tree.links.new(texture.outputs["Color"], normal.inputs["Color"])
        shader = next(node for node in tree.nodes if node.type == "BSDF_PRINCIPLED")
        tree.links.new(normal.outputs["Normal"], shader.inputs["Normal"])
