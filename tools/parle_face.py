"""
    License information: data/licenses/makehuman_license.txt

    Face rules of the Parle cast, run inside Blender while a character is assembled. Each one
    removes something that looked wrong on the final avatar. The owner's examples are Rafa and the
    player: no eyelash cards, one eyebrow, no hair through the forehead, no eyeball through a lid.

      blender -b -P tools/parle_face.py -- <kit-body.glb> <out.json>   # writes the eye points the lids show
"""
import json
import os
import sys
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

sys.path.insert(0, str(Path(__file__).parent))
import parle_mouth  # noqa: E402
import parle_polish  # noqa: E402
import parle_normals  # noqa: E402
import parle_teeth  # noqa: E402
import parle_tongue  # noqa: E402

EYE_TUCK_REACH_M = 0.02
EYE_TUCK_BEHIND_M = 0.001
FOREHEAD_HAIR_M = 0.085
FOREHEAD_HALF_M = 0.06
FOREHEAD_REACH_M = 0.06
# The kit's own morphs, baked into the rest shape: eyes open all around and brows pulled down read as anger on everybody (owner's review, 2026-11-07).
# The kit rests with the lips a little apart, and the teeth showed through in a white line (owner, 2026-11-07): the jaw rests a little closed. Teeth and tongue take it too.
# A quarter closed was too far: closing carries the lower lip up and forward (29 mm up and 20 mm forward for the whole of `jawOpen`),
# and at a quarter it rode 7 mm up over the upper lip and 5 mm out in front of it. Seen from the side on 2026-11-11 (`parle_profile`),
# the lower lip stood 0.4 to 3.8 mm ahead of the upper one on the five people measured, and the mouth sat 10 to 13 mm under the nose;
# on Sofia, Rafa and the four faces of the game measured the same way the upper lip leads by 2.6 to 4 mm and the mouth sits 16 to 21 mm
# under the nose, as on real faces (the upper lip some 3 mm and the lower some 2 mm ahead of the line from under the nose to the chin:
# Burstone 1967, "Lip posture and its significance in treatment planning", American Journal of Orthodontics 53(4):262-284, DOI
# 10.1016/0002-9416(67)90022-X; Legan & Burstone 1980, Journal of Oral Surgery 38(10):744-751). From the front it read as a mouth with
# a lower lip only. The lips meet with about a tenth of the jaw, tried on the built body: at 0.05 a gap shows, at 0.10 it is closed.
CALM_LOOK = {"faceEyeOpen": -0.45, "faceBrowAngle": 0.6, "browInnerUp": 0.5, "jawOpen": -0.12}
# The reference women have about twice the height from the lips to the chin as from the nose to the lips; ours was nearly one to one (owner, 2026-11-09).
# At 0.3 it went past that: three to one on the four women measured on 2026-11-11 (chin 21% of the face, nose to lips 7%), where Sofia,
# the face the cast aims at, has 2.1 to one and a chin of 18% (`parle_beauty`). Much of the one to one was the mouth itself, ridden up
# under the nose by the jaw (`CALM_LOOK`): with the jaw where the lips meet and this at 0.1, the chin is 17% of the face.
# On 2026-11-12, with the jaw narrowed for the face to taper like a heart (`FAIR_LOOK`), 0.3 made the face a straight-sided triangle
# with a pointed chin and 0.15 still a V to the judge by sight: 0.05, a round chin.
CHIN_LONGER = {"female": 0.05, "male": 0.0}
# The look the cast aims at, on top of each person's own traits (owner, 2026-11-11: the faces were "muito feio", and the aim is avatars
# like Rafa and Sofia, "mais estilo Disney"). Measured by `parle_beauty` next to Sofia, the women had a nose 3 to 4% of the width of the
# face wider, a jaw 3 to 8% wider against the cheekbones, a chin 3 to 4% of the face longer, the eyes 2 to 4% closer together and the
# eyebrows 1.5 to 2% of the face lower. Each is a direction that goes with a woman's face being found attractive: smaller nose, shorter
# chin, cheekbones wide against a face narrow at the mouth, higher eyebrows, eyes further apart (Cunningham 1986, Journal of
# Personality and Social Psychology 50(5):925-935, DOI 10.1037/0022-3514.50.5.925: correlations, no targets; the distance between the
# eyes, Pallett, Link & Lee 2010, Vision Research 50(2):149-154, DOI 10.1016/j.visres.2009.11.003). The weights are the kit's own
# trait morphs, where 0.55 is the most a person's traits take. With nose and jaw at twice these and no more cheekbone (2026-11-11) the
# measure went on rising and the face no longer read as the same person, pinched below the cheeks: how far is by eye.
# The corners of the mouth: at rest they sat below the middle of the lips on everybody (by 2% of the width of the mouth on the women,
# 4% on Carlos; on Sofia and on the women of the game they sit 1 to 4% above it), and a face at rest that looks like an expression
# is judged as that expression: turned down, as sad or cross (Said, Sebe & Todorov 2009, Emotion 9(2):260-264, DOI 10.1037/a0014681);
# a smiling face is found more attractive (Reis et al. 1990, European Journal of Social Psychology 20(3):259-267, DOI
# 10.1002/ejsp.2420200307). The kit's own smile, a little of it, in the rest shape: the mouth still smiles from there.
# With the jaw where the lips meet (`CALM_LOOK`), the corners of the women's mouth sit 4% of its width above the middle of the lips
# with no smile at all, and with 0.12 of it 7%: past Sofia's (level) and those of the women of the game (up to 4%), a smile held at
# rest (2026-11-12, `parle_beauty.mouth_curve` on whole portraits). The men's sit 2% below with 0.12, as Rafa's.
REST_SMILE = {"female": 0.05, "male": 0.12}
# The nose stays at -0.3 in width: at -0.7, with the kit's narrow straight nose on top, the measure did not move (it reads the wings of
# the nose against the cheekbones, and the cheekbones had done it) and the cheek drawn in beside the wing let the lining of the mouth
# through (2026-11-12, `parle_polish.lining_behind_skin`).
#
# The measure is not the judge. On 2026-11-12 the women scored 9.0 to 9.3 on `parle_beauty` and the owner still found the face "bem
# feio"; a judge by sight (a vision model shown the face from the front, three quarters and the side next to Sofia's and to the same
# woman's face in the game, asked what an art director would say) gave it 3.5 in 10: "a tired, realistic adult, not a Disney-like
# girl". What it named, in the order it weighed them, and what answers each here:
#   - small startled eyes with no upper lid: the lid comes down to the top of the iris, -0.22 where the calm look has -0.45 (at
#     -0.25 with the iris of a real eye the whole iris showed, a stare; at -0.4 and at -0.3 the opening was a flat slit), and the iris itself is
#     drawn bigger (`eye_texture.IRIS_RADIUS`). Not the kit's faceEyeSize: the globe has no such shape, so the lids opened round a
#     globe that stayed, and it was dented where they met it. Nor a bigger eye still (`parle_polish.EYE_SCALE_BY_SEX`).
#     Five rounds on, at -0.22 the eye was "a small, flat, sleepy almond that the iris fills lid to lid, two dark beads" (round 14):
#     measured on the body, the upper lid stood 5 to 8 degrees above the middle of the pupil on the ball of the eye and the lower one
#     21 to 27 below, so the lid lay across the top of the pupil. Against the width of the face the opening was as wide as Sofia's
#     and less tall (0.236 by 0.087, hers 0.214 by 0.092, the same woman's in the game 0.229 by 0.097), the iris a tenth bigger than
#     hers. So the lids open as the kit has them (0.0) and the upper one is lifted by the kit's own wide-eyed shape (1.0, a third
#     of the expression it is cut from), which raises its arch and leaves the lower lid where it is; the iris is drawn a little
#     smaller and lower on the ball (`eye_texture`). Tried on the finished body: with twice the lift, or with the opening trait
#     further, white showed over the iris, a stare. The blink still closes (`blink_still_closes`). Round and open, the eye was
#     still "a symmetric lemon-shaped hole" to the judge (round 15): its outer corner goes up a little (faceEyeTilt 0.3; at 0.6 the
#     eye slants). And to the owner, who is the one who decides, the eyes of two of the five were then "gigante" and "estranhos"
#     (2026-11-12, rounds 15 and 16, the pharmacy's and the pet shop's attendants): the lift of the upper lid comes down to half
#     (0.5), and a woman's own traits no longer open or grow her eyes past the kit's (`EYES_OWN`, `fair_look`);
#   - eyebrows in a high arch, far from the eye: half the arch (`BROW_ARCH`), level over the eye as Sofia's, shorter and a little
#     thinner than the kit's, and lifted clear of the lid (`BROW_LIFT_M`). Their slope stays the calm look's: with the inner end let
#     down (`faceBrowAngle` 0.1 to 0.35, `browInnerUp` 0.2) they climbed to a far peak, a scowl, and thicker than the kit's they
#     were a frown;
#   - a square lower face with a weak chin: the jaw narrows to -0.7, which is past what was called "a bias, not another face"; with
#     the skin evened out (`parle_polish.fair_face`) it no longer pinches. The cheek keeps its volume (with -0.3 it went flat to
#     hollow) and the chin its roundness (`CHIN_LONGER`);
#   - a long, straight, sharp nose: shorter by the trait (-0.5) and by the kit's short nose (0.3), with some of its round tip (0.3;
#     alone at 0.4 the nostrils turned up to the front).
# Tried in previews on the finished body before going in; judged again by sight after each round (3.5, then 4.5 in 10). To be
# judged in the player.
# The jaw, the chin and the length of the nose of the women went back most of the way on 2026-11-12, by the owner's own eye. He
# marked, face by face, which of the game's faces and of the cast he finds beautiful (15 women, 17 men) and which ugly, and of
# the ugly women said "queixo fino e rosto largo, parece um triângulo invertido". Measured with the ruler (`parle_beauty`), the
# women he approves have a jaw 0.798 of the face wide (sd 0.022) and a chin 0.204 of its height (sd 0.013); the three he faulted
# for the triangle have jaws of 0.751 to 0.765, and with a jaw of -0.7 and a chin of -0.35 here the pharmacy's attendant and the
# florist stood at 0.763 and 0.764, with chins of 0.169 and 0.180. He approved neither; the one woman of the cast he did approve
# stood at 0.801. With the jaw at -0.1, the chin at 0.25 and the nose length at -0.2 the two stand at 0.786 and 0.779, chins
# 0.195 and 0.203, nose 0.30 against 0.302 (read off the shapes of the finished bodies). The judge by sight had asked for a face
# less square; on the shape of the face the owner's eye decides.
FAIR_LOOK = {
    "female": {"faceNoseWidth": -0.3, "faceNoseLength": -0.2, "noseShape-curto": 0.3, "noseShape-ponta-arredondada": 0.3, "faceJaw": -0.1, "faceCheek": 0.4,
               "faceEyeSpacing": 0.4, "faceBrowHeight": 0.2, "faceChin": 0.25, "faceEyeOpen": 0.0, "eyeWideLeft": 0.5, "eyeWideRight": 0.5, "faceEyeTilt": 0.3,
               "mouthSmileLeft": REST_SMILE["female"], "mouthSmileRight": REST_SMILE["female"]},
    "male": {"mouthSmileLeft": REST_SMILE["male"], "mouthSmileRight": REST_SMILE["male"]},
}
# The traits of a person's own that make her eyes bigger or more open than the kit's (`<head>.glb.face.json`, read off the game's
# face of that person by the game's repository and already in the rest shape when the look is given). The game's faces have eyes
# drawn large inside a black line; taken as they came, on top of what the look gives every woman (a globe 1.3 times its size, the
# lids opened), they gave two of the five women 0.17 and 0.20 of the trait of size and 0.12 and 0.14 of the trait of opening where
# the other two had next to none, and those were the two the owner found "com olho gigante" (2026-11-12). What a woman's own trait
# gives past none is taken back; a trait that makes her eyes smaller or less open stays hers.
EYES_OWN = ("faceEyeSize", "faceEyeOpen")
# How thick an eyebrow is left, of the kit's own, about its middle line. At three quarters the women's read as a line drawn in
# pencil once the skin of the face was smooth, next to Sofia's and Rafa's, which are thick; at 0.9 still "thin hard marker strokes"
# to the judge by sight, past the kit's own (1.2) a frown, and at the kit's own, set low, "the darkest, hardest shape on the face"
# (2026-11-12, three rounds). A little under the kit's.
BROW_THIN = {"female": 0.85, "male": 0.75}
# And how much of that is left at its outer end, the thickness falling evenly from the inner end: the women's was thickest at
# mid-length with both ends thin, a boomerang, where Sofia's is thickest by the nose and thins to the temple.
BROW_OUTER = {"female": 0.75, "male": 1.0}
# How long it is left, from its inner end: Sofia's stops well short of the temple.
BROW_LONG = {"female": 0.85, "male": 1.0}
# And how much of its arch: its middle line comes this much of the way from the straight line between its two ends. The kit's
# climbs to a peak well above the lid; Sofia's runs nearly level over the eye.
# The men's kept the whole arch until 2026-11-12: a peak two thirds of the way out with both ends low, "thick, angry brows" over
# "a scowl" to the judge by sight (Carlos, rounds 13 and 14, the lowest face of the five), where Rafa's and the same man's in the
# game run in one low sweep over the eye. Theirs keep more of it than the women's: a man's brow sits lower and is thicker.
BROW_ARCH = {"female": 0.5, "male": 0.6}
# With half the arch the whole eyebrow sat where its two ends were, 8 mm over the middle of the eye, and pressed on the lid ("the
# darkest, hardest shape on the face", the judge by sight, 2026-11-12): it goes up by this much. The kit's faceBrowHeight is not the
# way: its whole reach on the eyebrow is 3 mm.
# The men's too, once their arch came down (`BROW_ARCH`): flattened where it was, it lay on the eye, a slab, and its height over
# the eye read 12% of the face where Rafa's reads 15 (`parle_beauty.brow_high`, 2026-11-12).
BROW_LIFT_M = {"female": 0.003, "male": 0.003}
# The traits of the look that open or close the lids at rest, and the shapes that shut them: a point a blink moves this far or
# more gives back the whole of what those traits moved it (`blink_still_closes`).
LID_TRAITS = ("faceEyeOpen", "eyeWideLeft", "eyeWideRight")
BLINKS = parle_polish.BLINKS
BLINK_WHOLE_M = parle_polish.BLINK_WHOLE_M
# With the jaw a quarter closed, the upper teeth sat wholly behind the upper lip: speaking, the mouth showed the lower teeth and the
# tongue under a bare lip, and read as a snarl (owner, 2026-11-08). Measured on the pharmacy's attendant at the player's widest "a"
# (`viseme_aa` 0.55): incisor tip 1 mm above the lip's edge. Teeth go 3.5 mm down and the tongue 3 mm more, behind the lower teeth.
# The women's kit has full lips that bulged like a sausage in the player's perspective and light (owner, 2026-11-08: lower lip too
# big). The kit's own `faceLipVolume`, on the lip that follows the jaw and fading 4 mm around it: -1 on women, -0.5 on men; -2 creased.
# Next to four reference heads (owner, 2026-11-09) thinning was the wrong way: their lips are two full forms and hers a thin band. The shape comes from `full_lips`.
LIP_THIN = {"female": 0.0, "male": -0.5}
# The upper lip of the women, the lip the jaw does not carry: the kit's own `faceLipVolume` on it. Next to Sofia's, whose upper lip
# is a form of its own with an edge, theirs was "thin and flat under a smear of colour that overshoots the form" to the judge by
# sight (2026-11-12, rounds 13 to 15, the first fault of the face at rest in the last), and seen from the side the lower lip stood
# 2% of the face ahead of it where Sofia's upper lip leads (`parle_profile`). Tried on the finished body: at 1 it hardly shows, at 3
# the lip is a muzzle. The colour of the lip stays as it was (`parle_lips`): cut to where the trait stops moving the skin, 5.5 mm,
# it left a thread of colour over a bare swelling, since the colour is measured from the slit of the mesh, which lies some 3 mm
# inside the line where the lips are seen to meet. A point is of the lower lip when the open jaw carries it this share of how far
# it carries the lip it carries most.
UPPER_LIP_VOLUME = {"female": 2.0, "male": 0.0}
UPPER_LIP_JAW_SHARE = 0.3
# The lower lip read as two, a crumpled ridge over the hollow under it: three lips with the upper one (owner, 2026-11-09, pharmacy). Thinner
# only creased it and filling the hollow made a band of its own. The outer skin of its lower half and of the hollow relaxes into one form.
UNDER_LIP_ROUNDS = {"female": 0, "male": 0}
UNDER_LIP_NEAR_M = 0.01
UNDER_LIP_FAR_M = 0.018
UNDER_LIP_SHARE = 0.45
# What lies this far behind the lip is the inside of the mouth, which the relaxing would pull out of place.
UNDER_LIP_BEHIND_M = 0.012
# The mouth went 3 mm back and the chin stayed: in profile it stuck out (owner, 2026-11-09, pharmacy). It goes back as much, fading towards the jaw and under it.
# In three reference women the chin sits about 3% of the head's height (6 mm) behind the lower lip; on the women's kit it was 2 mm ahead of it.
CHIN_BACK = {"female": 0.0035, "male": 0.0}
# Each lip a form of its own, as on the reference heads: forward at the middle of its height, the lower one taller, and a soft hollow under it. All in metres; the lips' meeting line stays.
# The upper lip leads (2026-11-11, `parle_profile`): with the jaw closed no further than the lips' meeting, the lower lip still stood
# 1.5 mm ahead of the upper one on the two women measured, where it stands 2.6 to 4 mm behind on Sofia, Rafa and the faces of the game
# (and behind on real faces: Burstone 1967, DOI 10.1016/0002-9416(67)90022-X). The upper lip came 1.5 mm forward at the middle of
# its height and the lower one, which was 0.5 mm forward, went 1.2 mm back; at 2.5 and 2, tried on 2026-11-12 for the measure of the
# profile, the upper lip hung over the lower one like a beak, and at 1.6 and 1 it still did to the judge by sight, who also read the
# lower lip as thin and flat. Now 1 mm forward for the upper and none for the lower (with 0.6 and 0.4 the lower lip led again, by
# 4% of the height from the nose to the lips): the measure of the profile is not the judge, but a lower lip in front is the fault
# this started from.
# The height of the lower lip (2026-11-12): grown 2 mm down with the hollow under it 13.5 mm below the line of the lips, the lips
# stood 40% as tall as the mouth is wide where Sofia's stand 34% (`parle_beauty.lips_full`, which reads the relief); with no growth
# and the hollow at 11.5 the lower lip was a thin band under the upper one, and at 1 mm and 12.5 still thin and flat to the judge by
# sight. Back to 2 mm and 13.5: a full lower lip is what the judge asks for, not what the measure does.
LIPS_FULL = {"female": {"upper_forward": 0.001, "upper_at": 0.0045, "upper_wide": 0.0035, "upper_grow": 0.0012, "lower_forward": 0.0, "lower_at": 0.0055, "lower_wide": 0.0042, "lower_grow": 0.002, "hollow_back": 0.003, "hollow_at": 0.0135, "hollow_wide": 0.004, "roll_back": 0.0032, "roll_wide": 0.0028, "corner_up": 0.0042, "corner_from": 0.008, "corner_at": 0.027, "corner_out": 0.02, "corner_tall": 0.014}, "male": None}
LIPS_KEEP_LINE_M = 0.0012
CORNER_AT_M = 0.027
CORNER_NEAR_END_M = 0.004
CORNER_RELAX_NEAR_M = 0.004
CORNER_RELAX_FAR_M = 0.016
CORNER_RELAX_BEHIND_M = 0.012
CORNER_LIP_CLEAR_M = 0.0015
# Right at the corner the ridge sits on the lip's own points: left out, it stayed and still lit up (owner, 2026-11-09).
CORNER_LIP_FREE_M = (0.004, 0.01)
CORNER_RELAX_ROUNDS = 12
# 0.8 drew a heavy "M" whose middle hung over the lower lip, and at 0.5 the line where the lips meet still dipped in a V (the judge
# by sight, 2026-11-12): a shallow bow. And at 0.25, with the upper lip given volume (`UPPER_LIP_VOLUME`), what it left was the
# groove over the lip standing between two ridges, "a lumpy philtrum" (the same judge, round 15), and no bow that could be read:
# none of it. The bow of the lip is in its colour (`parle_lips`).
LIPS_CUPID = {"female": 0.0, "male": 0.0}
# In the wide-open vowels the lining of the mouth came out through the cheek in two cuts (owner, 2026-11-09).
MOUTH_LINING_REACH_M = 0.08
GUM_TOOTH_MOST_POINTS = 60
GUM_KEEP_M = 0.009
GUM_TORN_M = 0.005
# Held still, the cheek's skin still closed over it when the jaw dropped: it also sits further in, past 2 cm from the middle.
MOUTH_LINING_INSET_M = 0.009
MOUTH_LINING_INSET_FROM_M = 0.02
MOUTH_LINING_INSET_FADE_M = 0.02
LIP_FADE_M = 0.004
# Thinner, the women's lips still stood out in profile like a beak and the lower one hung forward (owner, 2026-11-08, pharmacy). The
# whole mouth goes 3 mm back: lips and what lies 6 mm around them fully, fading over 20 mm more, teeth and tongue with them. Only the
# lower lip back opened a gap between the lips; 4 mm flattened the face.
# Next to photographs in profile, at 3 mm the lips no longer stood out from the face (owner, 2026-11-09): 1.5 mm.
MOUTH_BACK = {"female": 0.0052, "male": 0.0}
# Speaking, the rounded vowels pushed the full lips forward into a big beak, and the player mostly shows her mid-sentence: the
# forward part (-y) of the rounded visemes keeps 40% on women. The vowel still reads as round.
POUT_FORWARD = {"female": 0.4, "male": 1.0}
POUT_VISEMES = ("viseme_U", "viseme_O", "viseme_CH", "viseme_RR")
# How far the mouth opens in speech, of what the kit's shapes open. The player drives every face with the same weights (the game's
# `LipSync.cs`: at the peak of an "a", 0.55 of viseme_aa and 0.2 of jawOpen), and with them the cast's mouth opened two to three
# times as far as those of the two faces the cast aims at: the chin of the pharmacy's attendant dropped 32 mm, Rafa's 14 and Sofia's 9.
# The point that moves most goes 41 mm in her viseme_aa and 45 in her jawOpen, against 17 and 22 on Rafa and 7 and 26 on Sofia; a
# man of the cast 31 and 35 (2026-11-12, measured on the bodies). Open that far the mouth is a dark hole with the lower teeth in a
# row of blocks, where theirs shows a row of upper teeth (owner, 2026-11-12: the face is to be looked at in every movement of the
# mouth, and was "ainda bem feio"). The same peak lands at about 12 mm on her and 11 on him, between Sofia's and Rafa's. A choice
# by measure against the two leads and by sight, with no study behind it: to be judged in the player.
# These are the shares the body has had in fact since the first night, not the ones written then. The body was handed to
# `calm_speech` twice in the same list and took each share twice over, the teeth and the tongue once: written as 0.65 of the jaw
# and 0.6 of the "a" (0.55 and 0.45 before), the body opened 0.42 and 0.36 of them (0.30 and 0.20 before, a peak of 7 mm, which is
# what the judge by sight called "the same flat slit") while the lower teeth dropped as written, further than the lip that covers
# them. Found on 2026-11-12 by measuring the shapes of a built body against what they should have been. The body keeps what was
# judged; the teeth and the tongue now go with it, and no mesh is taken twice (`calm_speech`, `mixed_speech`).
SPEECH_OPEN = {
    "female": {"jawOpen": 0.42, "viseme_aa": 0.36, "viseme_E": 0.36, "viseme_O": 0.81, "viseme_I": 0.64},
    "male": {"jawOpen": 0.49, "viseme_aa": 0.42, "viseme_E": 0.49, "viseme_I": 0.81},
}
# The rounded shapes of speech. With the weights the player gives them (`LipSync.cs`: about half of each shape), the kit's "o" was
# "a wide letterbox gape where Sofia makes a small round pucker", its "ch" "the same slack half-open slit as RR, E and TH, where
# Sofia pushes the lips forward", and "a" and the open jaw could not be told apart (the judge by sight, 2026-11-12, round 14;
# speaking had stood at 3.5 in 10 for three rounds). The kit cuts them from the same few moves: its "o" is 0.6 of the open mouth
# with 0.6 of the pursed one, its "u" 0.9 of the pursed one, its "ch" the lips forward with a little of the open mouth. Each is
# made again from the shapes the body has, after `calm_speech`: what it takes of itself and of the others. Chosen by sight on the
# finished body next to Sofia's at the player's weights (2026-11-12), with no study behind them, to be judged in the player:
#   - "o": the pucker of the "u" with under a third of its own opening, and some of the jaw given back (the player adds the jaw
#     itself): a small round mouth, hardly open;
#   - "u": a little more of itself with the jaw closing, so the lips meet in a pucker;
#   - "ch": two thirds of itself with most of the pucker of the "u" and a little of the jaw given back: a small pout;
#   - "a": itself with some of the pucker of the "u", which draws the corners in: an oval, where it was a letterbox.
# A first try went further (the "o" with 1.18 of the "u" and half of itself, the "u" at 1.3, the "ch" with the whole "u", the "a"
# with 0.45 of the smile): looked at on the whole face, the "o" and the "ch" were a tall oval with both rows of teeth and the chin
# creased under it, the lower lip of the "u" hung like a tongue, and the "a" was a grin. In round 15 the "o" and the "u" "finally
# read as rounded" to the judge by sight; the "ch", then the whole of itself with two thirds of the "u", was "an oversized ring, a
# gasp", and the "a" with the corners of the mouth lifted (0.22 of the smile) "a grimace": the two as they are above.
# The open jaw itself is left alone: with the pucker in it, the lips of both jaws would move with it, and the jaw is how the lower
# lip is told from the upper one (`parle_polish.fair_face`).
SPEECH_MIX = {
    "viseme_O": {"viseme_O": 0.29, "viseme_U": 1.0, "jawOpen": -0.29},
    "viseme_U": {"viseme_U": 1.11, "jawOpen": -0.23},
    "viseme_CH": {"viseme_CH": 0.67, "viseme_U": 0.82, "jawOpen": -0.18},
    "viseme_aa": {"viseme_aa": 1.0, "viseme_U": 0.4},
}
# Seen as the player frames her (front, eye level), the women's mouth still read as a pout: it was short and the lips round. The
# kit's own faceMouthWidth +0.8 lengthens it; thinning the lips further opened a gap between them (owner, 2026-11-08, pharmacy).
# At +0.8 it read as a long flat band next to photographs (owner, 2026-11-09): +0.4.
# Measured (2026-11-12, `parle_beauty.mouth_wide`): at +0.4 the women's mouth was 41% of the width of the face, where Sofia's is 34%
# and the same women's in the game 30 to 38%; the lips have since been given a form of their own (`LIPS_FULL`) and the mouth sent
# back (`MOUTH_BACK`), which is what the pout was. +0.1 left it at 38%, to the judge by sight still twice the width of the nose
# where Sofia's is one and a half: -0.25, about 35%.
MOUTH_WIDE = {"female": -0.25, "male": 0.0}
MOUTH_CORE_M = 0.006
MOUTH_FADE_M = 0.02
TEETH_DOWN_M = 0.0035
TONGUE_DOWN_M = 0.009
# Halves of a point split on a UV seam drift apart in the head fit and the background shows through, in a line along the jaw.
SEAM_REACH_M = 0.008
SEAM_SAME_MOVE_M = 0.001
SEAM_TOP_M = 0.05
SEAM_LOW_M = 0.4
# Brows were one flat dark colour with a hard edge and read as painted on (owner, 2026-11-08): a little way to the skin, and the
# strands' edge fades. A third of the way made the restaurant attendant's brows almost vanish.
BROW_TOWARD_SKIN = 0.15
BROW_ALPHA = 0.95
# A matte eye (roughness 0.7) had no catchlight and looked dead; the cornea is wet.
EYE_ROUGHNESS = 0.15
# Earrings: one pair, smaller, and not the same game pair on every woman (owner, 2026-11-07).
EARRING_SCALE = 0.6
GAME_EARRINGS_KEPT_EVERY = 3


def skin_tree(body):
    return BVHTree.FromPolygons([tuple(body.matrix_world @ vertex.co) for vertex in body.data.vertices], [tuple(polygon.vertices) for polygon in body.data.polygons])


def wears_from_game(name, earrings, glasses=True):
    """Men of the cast wear no earrings (owner, 2026-11-06). Glasses of the game body stay unless the look says otherwise (Renata, owner, 2026-11-07)."""
    return (earrings or "Earring" not in name) and (glasses or not name.startswith(("Glasses", "Aviator")))


def trim_forehead_hair(obj, body, mine, band_z=None):
    """O crânio novo é menor: o afro que no jogo fica debaixo do turbante saía pela testa, numa mancha preta acima das sobrancelhas. Com lenço, o corte para no meio dele: acima, a testa aparecia entre o lenço e o cabelo."""
    skin = skin_tree(body)
    back = Vector((0, 1, 0))
    mesh = bmesh.new()
    mesh.from_mesh(obj.data)
    out = [vert for vert in mesh.verts if vert.co.z < (band_z if band_z is not None else mine["eye"] + FOREHEAD_HAIR_M) and abs(vert.co.x - mine["x"]) < FOREHEAD_HALF_M and skin.ray_cast(vert.co, back, FOREHEAD_REACH_M)[0] is not None]
    bmesh.ops.delete(mesh, geom=out, context="VERTS")
    mesh.to_mesh(obj.data)
    mesh.free()
    obj.data.update()
    print("TESTA", len(out), "pontos de cabelo")


def own_traits(job):
    """The traits of the person of a job, as the game's repository wrote them beside the head: none when it wrote none."""
    file = Path(f"{job.get('new', '')}.face.json")
    return json.loads(file.read_text()) if job.get("new") and file.exists() else {}


def fair_look(sex, own):
    """The look of that sex for a person with those traits of her own: FAIR_LOOK, and for a woman what her own traits give her
    eyes past the kit's taken back (EYES_OWN)."""
    look = dict(FAIR_LOOK[sex])
    if sex == "female":
        for name in EYES_OWN:
            past = max(0.0, float(own.get(name, 0.0)))
            if past:
                look[name] = look.get(name, CALM_LOOK.get(name, 0.0)) - past
    return look


def tuck_eyes(body, added):
    """O globo anda inteiro com o rosto e a pele anda ponto a ponto: onde o rosto do jogo é mais estreito, o branco furava a pálpebra em duas marcas.
    A point of the globe with no skin in front of it and skin lying over the globe somewhere between it and the middle of the eye
    (`parle_polish.under_the_lid`) has come out through a lid: it goes back behind the skin behind it. A point with nothing over the
    globe between it and the middle of the eye is part of what the lids open on, and stays.
    Until 2026-11-12 what counted as seen was a list read on the kit's own face, and any point off it that
    stood clear of the skin went back: with the women's lids opened past the kit's (`FAIR_LOOK`), points of the globe that the kit
    hides stood in the opening and were pushed into the socket, bites out of the iris along the upper lid; with the lids below the
    kit's it had already left a jag at the outer corner (the judge by sight, round 14). The face itself says what shows.
    What moves the face after this (the eyes growing, the build of the head) is seen to at the end, by `parle_polish.globes_behind_skin`."""
    skin = skin_tree(body)
    front, back = Vector((0, -1, 0)), Vector((0, 1, 0))
    tucked = 0
    for eye in (obj for obj in added if obj.type == "MESH" and "low-poly" in obj.name):
        keys = eye.data.shape_keys.key_blocks if eye.data.shape_keys else []
        inverse = eye.matrix_world.inverted()
        centres, radius = [Vector(centre) for centre in parle_polish.eye_centres(eye)], parle_polish.eye_radius(eye)
        for vertex in eye.data.vertices:
            at = eye.matrix_world @ vertex.co
            centre = min(centres, key=lambda spot: (spot - at).length)
            if skin.ray_cast(at + front * 1e-4, front, 0.05)[0] is not None or not parle_polish.under_the_lid(skin, at, centre, radius, front):
                continue
            hit = skin.ray_cast(at, back, EYE_TUCK_REACH_M)[0]
            if hit is None:
                continue
            step = inverse.to_3x3() @ (back * ((hit - at).length + EYE_TUCK_BEHIND_M))
            vertex.co += step
            for key in keys:
                key.data[vertex.index].co += step
            tucked += 1
        eye.data.update()
    print("OLHO", tucked, "pontos furando a pele")


def key_points(block):
    points = np.zeros(len(block.data) * 3)
    block.data.foreach_get("co", points)
    return points.reshape(-1, 3)


def move_rest(obj, move):
    """Every key stores absolute points: the rest shape only moves if all of them move with it."""
    for block in obj.data.shape_keys.key_blocks:
        block.data.foreach_set("co", (key_points(block) + move).ravel())
    for vertex, shift in zip(obj.data.vertices, move):
        vertex.co += Vector(shift)
    obj.data.update()


def brow_line_move(points, thin, outer=1.0, long=1.0, arch=1.0, lift=0.0):
    """Thinner around its middle line, side by side: `thin` of its thickness is left at the inner end, and `outer` of that at the
    outer end, falling evenly along the eyebrow; `long` of its length is left, from the inner end; `arch` of the rise of its
    middle line over the straight line between its two ends; and the whole of it goes up by `lift`."""
    middle = points[:, 0].mean()
    move = np.zeros_like(points)
    for side in (points[:, 0] < middle, points[:, 0] >= middle):
        curve = np.polyfit(points[side, 0], points[side, 2], 3)
        line = np.polyval(curve, points[side, 0])
        away = np.abs(points[side, 0] - middle)
        along = (away - away.min()) / max(np.ptp(away), 1e-9)
        ends = np.polyval(curve, (points[side, 0][away.argmin()], points[side, 0][away.argmax()]))
        level = ends[0] + (ends[1] - ends[0]) * along
        move[side, 2] = level + (line - level) * arch + (points[side, 2] - line) * thin * (1 - (1 - outer) * along) - points[side, 2] + lift
        move[side, 0] = np.sign(points[side, 0] - middle) * (away - away.min()) * (long - 1)
    return move


def upper_lip_volume(keys, rest, amount):
    """The move of the kit's lip volume, that many times over, on the lip the jaw does not carry."""
    if "faceLipVolume" not in keys or "jawOpen" not in keys:
        return np.zeros_like(rest)
    volume = key_points(keys["faceLipVolume"]) - rest
    lips = np.linalg.norm(volume, axis=1) > 1e-6
    carried = np.linalg.norm(key_points(keys["jawOpen"]) - rest, axis=1)
    lower = parle_polish.smooth(carried / max(carried[lips].max() if lips.any() else 0.0, 1e-9) / UPPER_LIP_JAW_SHARE)
    return volume * amount * (1 - lower)[:, None]


def lip_thinning(keys, rest, amount):
    """The lip is what moves in the kiss and the closed-lips visemes; only the part the jaw carries thins."""
    lip_moves = [np.linalg.norm(key_points(keys[name]) - rest, axis=1) for name in ("viseme_U", "viseme_PP") if name in keys]
    if not lip_moves or "faceLipVolume" not in keys or "jawOpen" not in keys:
        return np.zeros_like(rest)
    moved = np.maximum.reduce(lip_moves)
    lip = np.where((moved >= moved.max() * 0.45) & ((key_points(keys["jawOpen"]) - rest)[:, 2] < -0.001))[0]
    tree = KDTree(len(lip))
    for index in lip:
        tree.insert(rest[index], int(index))
    tree.balance()
    weight = np.zeros(len(rest))
    near = np.where(np.linalg.norm(rest - rest[lip].mean(axis=0), axis=1) < 0.05)[0]
    for index in near:
        weight[index] = max(0.0, 1 - tree.find(rest[index])[2] / LIP_FADE_M)
    return (key_points(keys["faceLipVolume"]) - rest) * (weight * amount)[:, None]


def full_lips(keys, rest, shape):
    """Sculpts both lips and the hollow under the lower one on the outer skin."""
    move = np.zeros_like(rest)
    if not shape:
        return move
    moves = [np.linalg.norm(key_points(keys[name]) - rest, axis=1) for name in ("viseme_U", "viseme_PP") if name in keys]
    moved = np.maximum.reduce(moves)
    lips = np.where(moved >= moved.max() * 0.45)[0]
    jaw = (key_points(keys["jawOpen"]) - rest)[:, 2] < -0.001
    lower, upper = lips[jaw[lips]], lips[~jaw[lips]]
    middle = rest[lips, 0].mean()
    half = np.ptp(rest[lips, 0]) / 2
    x_lower, x_upper = rest[lower, 0] - middle, rest[upper, 0] - middle
    edges = np.linspace(-half, half, 15)
    centres, line = [], []
    for start, stop in zip(edges[:-1], edges[1:]):
        low, up = rest[lower[(x_lower >= start) & (x_lower < stop)], 2], rest[upper[(x_upper >= start) & (x_upper < stop)], 2]
        if len(low) and len(up):
            centres.append((start + stop) / 2)
            line.append((low.max() + up.min()) / 2)
    curve = np.polyfit(centres, line, 4)
    x = rest[:, 0] - middle
    height = rest[:, 2] - np.polyval(curve, np.clip(x, -half, half))
    across = np.sqrt(np.clip(1 - (x / (half * 1.05)) ** 2, 0, 1))
    outer = rest[:, 1] < rest[lips, 1].mean() + 0.012
    gauss = lambda value, at, wide: np.exp(-(((value - at) / wide) ** 2))
    keep = np.clip(np.abs(height) / LIPS_KEEP_LINE_M, 0, 1)
    up = height > 0
    move[:, 1] -= (shape["upper_forward"] * gauss(height, shape["upper_at"] * across, shape["upper_wide"]) * across * keep * up + shape["lower_forward"] * gauss(height, -shape["lower_at"] * across, shape["lower_wide"]) * across * keep * ~up)
    move[:, 1] += shape["hollow_back"] * gauss(height, -shape["hollow_at"], shape["hollow_wide"]) * np.clip(across * 1.6, 0, 1) * ~up
    # The kit's lips are furthest forward right at the line where they meet, a beak in profile; real ones roll in towards it. Both sides move alike, so they still touch.
    move[:, 1] += shape["roll_back"] * gauss(height, 0.0, shape["roll_wide"]) * across ** 2
    move[:, 2] += shape["upper_grow"] * np.clip(height / 0.008, 0, 1) * gauss(height, 0.008, 0.008) * across * up
    move[:, 2] -= shape["lower_grow"] * np.clip(-height / 0.01, 0, 1) * gauss(height, -0.01, 0.01) * across * ~up
    # The kit's meeting line falls about 6 mm from the middle to each corner, and the face reads as sulking (owner, 2026-11-09). Both lips and the skin beside them go up together.
    side = np.abs(x)
    rise = np.clip((side - shape["corner_from"]) / (shape["corner_at"] - shape["corner_from"]), 0, 1)
    past = np.clip((side - shape["corner_at"]) / shape["corner_out"], 0, 1)
    move[:, 2] += shape["corner_up"] * rise * rise * (3 - 2 * rise) * (1 - past * past * (3 - 2 * past)) * gauss(height, 0.0, shape["corner_tall"])
    move[~outer] = 0
    return move


def chin_back(rest, lip, amount):
    """The move, backwards (+y), of the chin: starts under the lip, fades out to the sides, behind the jawbone's front and on the neck."""
    move = np.zeros_like(rest)
    if not amount or not len(lip):
        return move
    low, middle = rest[lip, 2].min(), rest[lip, 0].mean()
    below = rest[:, 2] < low
    front = rest[below & (np.abs(rest[:, 0] - middle) < 0.01), 1].min()
    start = np.clip((low - 0.004 - rest[:, 2]) / 0.012, 0, 1)
    across = np.clip((0.06 - np.abs(rest[:, 0] - middle)) / 0.03, 0, 1)
    deep = np.clip((front + 0.075 - rest[:, 1]) / 0.03, 0, 1)
    floor = np.clip((rest[:, 2] - (low - 0.075)) / 0.02, 0, 1)
    move[:, 1] = amount * start * start * (3 - 2 * start) * across * deep * floor
    return move


def relaxed_under_lip(obj, rest, move, lip, rounds):
    """`move` after relaxing the outer skin from the middle of the lower lip down through the hollow under it. The lips' meeting line and the inside of the mouth stay."""
    if not rounds or not len(lip):
        return move
    tree = KDTree(len(lip))
    for index in lip:
        tree.insert(rest[index], int(index))
    tree.balance()
    share = np.zeros(len(rest))
    for index in np.where(np.linalg.norm(rest - rest[lip].mean(axis=0), axis=1) < 0.07)[0]:
        share[index] = np.clip((UNDER_LIP_FAR_M - tree.find(rest[index])[2]) / (UNDER_LIP_FAR_M - UNDER_LIP_NEAR_M), 0, 1)
    low, high = rest[lip, 2].min(), rest[lip, 2].max()
    share *= np.clip((low + (high - low) * UNDER_LIP_SHARE - rest[:, 2]) / 0.002, 0, 1)
    share *= np.clip((rest[lip, 1].mean() + UNDER_LIP_BEHIND_M - rest[:, 1]) / 0.004, 0, 1)
    return relaxed(obj, rest + move, share, rounds) - rest


def relaxed(obj, placed, share, rounds):
    """`placed` after each point moved `rounds` times towards the middle of its neighbours, as far as `share` lets it."""
    edges = np.empty(len(obj.data.edges) * 2, int)
    obj.data.edges.foreach_get("vertices", edges)
    edges = edges.reshape(-1, 2)
    count = np.maximum(np.bincount(edges.ravel(), minlength=len(placed)), 1)[:, None]
    for _ in range(rounds):
        total = np.zeros_like(placed)
        np.add.at(total, edges[:, 0], placed[edges[:, 1]])
        np.add.at(total, edges[:, 1], placed[edges[:, 0]])
        placed = placed + (total / count - placed) * (share * 0.5)[:, None]
    return placed


def relaxed_corners(obj, rest, move, lip):
    """`move` after smoothing the skin beside the mouth corners: a ridge there faces the ceiling and lit up as a white patch in the player (owner, 2026-11-09). The lips stay."""
    if not len(lip):
        return move
    placed = rest + move
    middle = placed[lip, 0].mean()
    tree = KDTree(len(lip))
    for index in lip:
        tree.insert(placed[index], int(index))
    tree.balance()
    share = np.zeros(len(rest))
    free = np.zeros(len(rest))
    for sign in (-1, 1):
        ends = lip[np.abs((placed[lip, 0] - middle) * sign - CORNER_AT_M) < CORNER_NEAR_END_M]
        if not len(ends):
            continue
        corner = np.median(placed[ends], axis=0)
        away = np.linalg.norm(placed - corner, axis=1)
        close = np.clip((CORNER_RELAX_FAR_M - away) / (CORNER_RELAX_FAR_M - CORNER_RELAX_NEAR_M), 0, 1)
        share = np.maximum(share, close * (placed[:, 1] < corner[1] + CORNER_RELAX_BEHIND_M))
        free = np.maximum(free, np.clip((CORNER_LIP_FREE_M[1] - away) / (CORNER_LIP_FREE_M[1] - CORNER_LIP_FREE_M[0]), 0, 1))
    for index in np.where(share > 0)[0]:
        share[index] *= max(np.clip((tree.find(placed[index])[2] - CORNER_LIP_CLEAR_M) / CORNER_LIP_CLEAR_M, 0, 1), free[index])
    return relaxed(obj, placed, share, CORNER_RELAX_ROUNDS) - rest


def lining_points(rest, polygons, lip):
    """The points that line the inside of the mouth: there is skin ahead of them and skin beside them. Only ahead also caught the jaw and the neck, under the chin."""
    tree = BVHTree.FromPolygons(rest.tolist(), polygons)
    middle = rest[lip, 0].mean()
    near = np.where((np.linalg.norm(rest - rest[lip].mean(axis=0), axis=1) < MOUTH_LINING_REACH_M) & (rest[:, 1] > rest[lip, 1].mean()))[0]

    def hidden(index, direction):
        return tree.ray_cast(Vector(rest[index]) + direction * 0.0005, direction, MOUTH_LINING_REACH_M * 2)[0] is not None

    return np.array([index for index in near if hidden(index, Vector((0, -1, 0))) and hidden(index, Vector((1 if rest[index, 0] >= middle else -1, 0, 0)))], int)


def held_in(rest, delta, lining, middle):
    """`delta` with the lining never going forward or out to the side, the two ways it can come through the cheek."""
    delta = delta.copy()
    delta[lining, 1] = np.maximum(delta[lining, 1], 0.0)
    outward = np.sign(rest[lining, 0] - middle) * delta[lining, 0] > 0
    delta[lining[outward], 0] = 0.0
    return delta


def keep_lining_inside(body):
    keys = body.data.shape_keys.key_blocks
    rest = key_points(keys[0])
    lip = carried_lip(keys, rest)
    if not len(lip):
        return
    lining = lining_points(rest, [list(polygon.vertices) for polygon in body.data.polygons], lip)
    for block in keys[1:]:
        if parle_tongue.is_mouth_key(block.name):
            block.data.foreach_set("co", (rest + held_in(rest, key_points(block) - rest, lining, rest[lip, 0].mean())).ravel())
    body.data.update()
    side = rest[lining, 0] - rest[lip, 0].mean()
    move = np.zeros_like(rest)
    move[lining, 0] = -np.sign(side) * MOUTH_LINING_INSET_M * np.clip((np.abs(side) - MOUTH_LINING_INSET_FROM_M) / MOUTH_LINING_INSET_FADE_M, 0, 1)
    move_rest(body, move)
    print("FORRO DA BOCA", len(lining), "pontos")


def trim_gums(teeth):
    """Drops the parts of the denture nobody should see: gum far from any tooth (the broken palate and floor) and the faces stretched between the two arches."""
    keys = teeth.data.shape_keys.key_blocks
    rest = key_points(keys[0])
    jaw = np.linalg.norm(key_points(keys["jawOpen"]) - rest, axis=1)
    mesh = bmesh.new()
    mesh.from_mesh(teeth.data)
    mesh.verts.ensure_lookup_table()
    seen, crowns = set(), []
    for start in mesh.verts:
        if start.index in seen:
            continue
        part, stack = [], [start]
        seen.add(start.index)
        while stack:
            vertex = stack.pop()
            part.append(vertex.index)
            for edge in vertex.link_edges:
                other = edge.other_vert(vertex)
                if other.index not in seen:
                    seen.add(other.index)
                    stack.append(other)
        if len(part) <= GUM_TOOTH_MOST_POINTS:
            crowns += part
    tree = KDTree(len(crowns))
    for index in crowns:
        tree.insert(rest[index], index)
    tree.balance()
    crown = set(crowns)
    def useless(face):
        points = [vertex.index for vertex in face.verts]
        if any(index in crown for index in points):
            return False
        moves = jaw[points]
        return moves.max() - moves.min() > GUM_TORN_M or all(tree.find(rest[index])[2] > GUM_KEEP_M for index in points)
    doomed = [face for face in mesh.faces if useless(face)]
    bmesh.ops.delete(mesh, geom=doomed, context="FACES_ONLY")
    bmesh.ops.delete(mesh, geom=[vertex for vertex in mesh.verts if not vertex.link_faces], context="VERTS")
    mesh.to_mesh(teeth.data)
    mesh.free()
    teeth.data.update()
    print("GENGIVA aparada:", len(doomed), "faces")


def carried_lip(keys, rest):
    moves = [np.linalg.norm(key_points(keys[name]) - rest, axis=1) for name in ("viseme_U", "viseme_PP") if name in keys]
    if not moves or "jawOpen" not in keys:
        return np.array([], int)
    moved = np.maximum.reduce(moves)
    return np.where((moved >= moved.max() * 0.45) & ((key_points(keys["jawOpen"]) - rest)[:, 2] < -0.001))[0]


def mouth_back(keys, rest, amount):
    """The lips and the skin around them, back (+y, the face looks to -y), with a smooth edge."""
    lip_moves = [np.linalg.norm(key_points(keys[name]) - rest, axis=1) for name in ("viseme_U", "viseme_PP") if name in keys]
    if not lip_moves or not amount:
        return np.zeros_like(rest)
    moved = np.maximum.reduce(lip_moves)
    lip = np.where(moved >= moved.max() * 0.45)[0]
    tree = KDTree(len(lip))
    for index in lip:
        tree.insert(rest[index], int(index))
    tree.balance()
    move = np.zeros_like(rest)
    reach = MOUTH_CORE_M + MOUTH_FADE_M
    for index in np.where(np.linalg.norm(rest - rest[lip].mean(axis=0), axis=1) < 0.08)[0]:
        t = min(1.0, max(0.0, 1 - (tree.find(rest[index])[2] - MOUTH_CORE_M) / MOUTH_FADE_M)) if tree.find(rest[index])[2] < reach else 0.0
        move[index, 1] = amount * t * t * (3 - 2 * t)
    return move


def soften_pout(body, forward):
    keys = body.data.shape_keys.key_blocks if body.data.shape_keys else None
    if not keys or forward == 1.0:
        return
    rest = key_points(keys[0])
    for name in (name for name in POUT_VISEMES if name in keys):
        delta = key_points(keys[name]) - rest
        delta[:, 1] = np.where(delta[:, 1] < 0, delta[:, 1] * forward, delta[:, 1])
        keys[name].data.foreach_set("co", (rest + delta).ravel())
    body.data.update()


def mixed_speech(meshes, mix):
    """Each of those shapes made again from the shapes each mesh has: the shares of the move of each, from the rest shape, read
    before any of them is written. A share of a shape a mesh does not have counts for nothing there (the teeth have no smile).
    A mesh that comes twice in the list is taken once."""
    for obj in dict.fromkeys(meshes):
        keys = obj.data.shape_keys.key_blocks if obj.data.shape_keys else None
        if not keys:
            continue
        rest = key_points(keys[0])
        moves = {name: key_points(keys[name]) - rest for name in {part for parts in mix.values() for part in parts} if name in keys}
        for name, parts in mix.items():
            if name in keys:
                keys[name].data.foreach_set("co", (rest + sum((moves[part] * share for part, share in parts.items() if part in moves), np.zeros_like(rest))).ravel())
        obj.data.update()


def lid_traits_moved(look, own):
    """How much of each trait of the lids is in the rest shape on top of the kit's: what the look gives, and what the person's own
    traits gave before it (the game's repository moves them into every shape key, the blink too). Lúcia's own lids are 0.12 of the
    trait more open than the kit's: with the look alone given back her blink stopped that much short of closing, a dark slit
    with a thread of white in it under each shut lid (2026-11-12, seen with the eyes shut)."""
    return {name: float(look.get(name, 0.0)) + float(own.get(name, 0.0)) for name in LID_TRAITS}


def blink_still_closes(keys, look):
    """What is moved to the rest shape goes into every shape key, the blink too: with the lids opened at rest the blink stopped
    short of closing by as much, a slit of eye left showing. Where a blink moves a point, that point's blink gives back what the
    traits of the lids (LID_TRAITS) moved it, `look` saying how much of each (`lid_traits_moved`): the lids close where the kit
    closes them. How many points it took."""
    rest = key_points(keys[0])
    lids = sum(((key_points(keys[name]) - rest) * look[name] for name in LID_TRAITS if name in keys and look.get(name)), np.zeros_like(rest))
    taken = 0
    for name in BLINKS:
        if name not in keys or not lids.any():
            continue
        shut = key_points(keys[name])
        share = parle_polish.smooth(np.linalg.norm(shut - rest, axis=1) / BLINK_WHOLE_M)
        keys[name].data.foreach_set("co", (shut - lids * share[:, None]).ravel())
        taken += int((share > 0).sum())
    return taken


def calm_speech(meshes, shares):
    """Each of those shapes keeps that share of its move from the rest shape, on the body, the teeth and the tongue alike: the jaw
    carries all three, and a shape cut on one alone would part them. A mesh that comes twice in the list is taken once: the body
    did, and its mouth opened by the share of the share while the teeth opened by the share (2026-11-12)."""
    for obj in dict.fromkeys(meshes):
        keys = obj.data.shape_keys.key_blocks if obj.data.shape_keys else None
        if not keys:
            continue
        rest = key_points(keys[0])
        for name, share in shares.items():
            if name in keys:
                keys[name].data.foreach_set("co", (rest + (key_points(keys[name]) - rest) * share).ravel())
        obj.data.update()


def calm_look(meshes, lip_thin=0.0, back=0.0, wide=0.0, chin=0.0, rounds=0, lips=None, bow=0.0, longer=0.0, fair=None, brow=BROW_THIN["male"], brow_outer=1.0, brow_long=1.0, brow_arch=1.0, brow_lift=0.0, upper_lip=0.0, own=None):
    look = {**CALM_LOOK, **({"faceMouthWidth": wide} if wide else {}), **({"faceCupidBow": bow} if bow else {}), **({"faceVChin": longer} if longer else {}), **(fair or {})}
    for obj in meshes:
        keys = obj.data.shape_keys.key_blocks if obj.data.shape_keys else None
        if not keys:
            continue
        rest = key_points(keys[0])
        move = sum(((key_points(keys[name]) - rest) * value for name, value in look.items() if name in keys), np.zeros_like(rest))
        if obj.name == "Body" and lip_thin:
            move = move + lip_thinning(keys, rest, lip_thin)
        if obj.name == "Body" and upper_lip:
            move = move + upper_lip_volume(keys, rest, upper_lip)
        if obj.name == "Body" and chin:
            move = move + chin_back(rest, carried_lip(keys, rest), chin)
        if obj.name == "Body" and lips:
            move = move + full_lips(keys, rest, lips)
        if obj.name == "Body" and back:
            move = move + mouth_back(keys, rest, back)
        if obj.name == "Body":
            move = relaxed_under_lip(obj, rest, move, carried_lip(keys, rest), rounds)
            if lips:
                move = relaxed_corners(obj, rest, move, carried_lip(keys, rest))
        if back and ("teeth" in obj.name or "tongue" in obj.name):
            move[:, 1] += back
        if "eyebrow" in obj.name:
            move = move + brow_line_move(rest + move, brow, brow_outer, brow_long, brow_arch, brow_lift)
        if "teeth" in obj.name or "tongue" in obj.name:
            move[:, 2] -= TEETH_DOWN_M if "teeth" in obj.name else TONGUE_DOWN_M
        if move.any():
            move_rest(obj, move)
        if obj.name == "Body":
            blink_still_closes(keys, lid_traits_moved(look, own or {}))


def tongue_follows_jaw(meshes):
    """Runs after the rest-shape rules, so the moves it reads are the ones the game plays."""
    teeth = next((obj for obj in meshes if "teeth" in obj.name), None)
    tongue = next((obj for obj in meshes if "tongue" in obj.name), None)
    if teeth is None or tongue is None or "jawOpen" not in teeth.data.shape_keys.key_blocks:
        return
    teeth_keys, tongue_keys = teeth.data.shape_keys.key_blocks, tongue.data.shape_keys.key_blocks
    teeth_rest, tongue_rest = key_points(teeth_keys[0]), key_points(tongue_keys[0])
    lower = parle_tongue.lower_teeth(key_points(teeth_keys["jawOpen"]) - teeth_rest)
    carried = 0
    for block in tongue_keys[1:]:
        if not parle_tongue.is_mouth_key(block.name) or block.name not in teeth_keys:
            continue
        move = parle_tongue.carried(tongue_rest, key_points(teeth_keys[block.name]) - teeth_rest, lower)
        if move is not None:
            block.data.foreach_set("co", (key_points(block) + move).ravel())
            carried += 1
    print("LINGUA segue a mandibula em", carried, "formas")


def teeth_atlas(teeth):
    """Gives the teeth the atlas `parle_teeth.tooth_map` paints: every polygon corner, from the rest shape."""
    keys = teeth.data.shape_keys.key_blocks
    rest = key_points(keys[0])
    jaw = np.linalg.norm(key_points(keys["jawOpen"]) - rest, axis=1) if "jawOpen" in keys else None
    loops = [(polygon.loop_indices[0], second, third) for polygon in teeth.data.polygons for second, third in zip(polygon.loop_indices[1:], polygon.loop_indices[2:])]
    corners = np.array([[teeth.data.loops[index].vertex_index for index in triangle] for triangle in loops])
    atlas = parle_teeth.atlas(rest, corners, jaw)
    target = teeth.data.uv_layers.active.data
    for triangle, uv in zip(loops, atlas):
        for index, value in zip(triangle, uv):
            target[index].uv = value.tolist()


def shade_map_of(obj, shade, gums=False, only=None, wine=None, inside=False):
    """Multiplies the object's colour map by `shade`, one value per point, through its UV islands. `only`: a map the mesh shares; see parle_mouth."""
    image = next(node.image for node in obj.active_material.node_tree.nodes if node.type == "TEX_IMAGE" and node.image)
    width, height = image.size
    loops = obj.data.uv_layers.active.data
    triangles = [(polygon.loop_indices[0], second, third) for polygon in obj.data.polygons for second, third in zip(polygon.loop_indices[1:], polygon.loop_indices[2:])]
    uv = np.array([[loops[index].uv[:] for index in triangle] for triangle in triangles], np.float64)
    points = np.array([[obj.data.loops[index].vertex_index for index in triangle] for triangle in triangles])
    if only is not None:
        vertex_uv = np.zeros((len(obj.data.vertices), 2))
        for corner, point in zip(uv.reshape(-1, 2), points.ravel()):
            vertex_uv[point] = corner[0], 1 - corner[1]
        kept = parle_mouth.body_triangles(points, only, vertex_uv)
        uv, points = uv[kept], points[kept]
    canvas = parle_mouth.grow(parle_mouth.shade_map(width, height, uv, shade[points]), fill=None if only is None else 1.0)
    pixels = np.empty(width * height * 4, np.float32)
    image.pixels.foreach_get(pixels)
    image.pixels.foreach_set(parle_mouth.darken(pixels.reshape(height, width, 4), canvas, gums, wine, inside).ravel())
    image.pack()


def body_wine(body):
    """The wet floor of this body, from the lip in its own skin map."""
    keys = body.data.shape_keys.key_blocks
    rest = key_points(keys[0])
    moves = [key_points(keys[name]) - rest for name in ("viseme_U", "viseme_PP") if name in keys]
    image = next(node.image for node in body.active_material.node_tree.nodes if node.type == "TEX_IMAGE" and node.image and node.image.colorspace_settings.name != "Non-Color")
    width, height = image.size
    pixels = np.empty(width * height * 4, np.float32)
    image.pixels.foreach_get(pixels)
    loops = body.data.uv_layers.active.data
    uv = np.zeros((len(body.data.vertices), 2))
    for loop in body.data.loops:
        uv[loop.vertex_index] = loops[loop.index].uv[0], 1 - loops[loop.index].uv[1]
    return parle_mouth.wine_for(parle_mouth.lip_luma(pixels.reshape(height, width, 4)[::-1, :, :3], uv, moves))


def shade_mouth(meshes, wine):
    """Runs on the maps the export takes, after the teeth map went in, and on the rest shape the game plays."""
    teeth = next((obj for obj in meshes if "teeth" in obj.name), None)
    tongue = next((obj for obj in meshes if "tongue" in obj.name), None)
    if teeth is not None and teeth.data.shape_keys:
        rest = key_points(teeth.data.shape_keys.key_blocks[0])
        shade_map_of(teeth, parle_mouth.teeth_shade(rest), gums=True, wine=wine)
    if tongue is not None:
        shade_map_of(tongue, parle_mouth.tongue_shade(key_points(tongue.data.shape_keys.key_blocks[0])))
    print("BOCA escurecida", bool(teeth), bool(tongue))


def shade_inside_of_mouth(body, teeth_rest, wine):
    """The body's own inside of the mouth, on the skin map."""
    rest = key_points(body.data.shape_keys.key_blocks[0])
    loops = body.data.uv_layers.active.data
    uv = np.zeros((len(body.data.vertices), 2))
    for loop in body.data.loops:
        uv[loop.vertex_index] = loops[loop.index].uv[0], 1 - loops[loop.index].uv[1]
    triangles = np.array([(polygon.vertices[0], second, third) for polygon in body.data.polygons for second, third in zip(polygon.vertices[1:], polygon.vertices[2:])])
    piece = parle_mouth.mouth_piece(rest, uv, teeth_rest, triangles)
    shade_map_of(body, parle_mouth.mucosa_shade(rest, teeth_rest, piece), only=piece, wine=wine, inside=True)


def weld_seams(body):
    """Mutual nearest open points on the head and neck that move together when the jaw and the lids move: those are one point, and lips and eyelids never qualify."""
    keys = body.data.shape_keys.key_blocks
    rest = key_points(keys[0])
    top = rest[:, 2].max()
    mesh = bmesh.new()
    mesh.from_mesh(body.data)
    mesh.verts.ensure_lookup_table()
    open_points = [vert.index for vert in mesh.verts if any(len(edge.link_faces) == 1 for edge in vert.link_edges) and top - SEAM_LOW_M < rest[vert.index, 2] < top - SEAM_TOP_M]
    mesh.free()
    tree = KDTree(len(open_points))
    for index in open_points:
        tree.insert(rest[index], index)
    tree.balance()

    def nearest(index):
        return next((found for found in tree.find_n(rest[index], 2) if found[1] != index), None)

    movers = [key_points(keys[name]) - rest for name in ("jawOpen", "eyeBlinkLeft", "eyeBlinkRight") if name in keys]
    pairs = []
    for index in open_points:
        found = nearest(index)
        if found and found[2] < SEAM_REACH_M and index < found[1] and nearest(found[1])[1] == index and all(np.linalg.norm(mover[index] - mover[found[1]]) < SEAM_SAME_MOVE_M for mover in movers):
            pairs.append((index, found[1]))
    if not pairs:
        print("EMENDA 0")
        return
    first, second = np.array(pairs).T
    for block in keys:
        points = key_points(block)
        middle = (points[first] + points[second]) / 2
        points[first], points[second] = middle, middle
        block.data.foreach_set("co", points.ravel())
    for one, other in pairs:
        middle = (body.data.vertices[one].co + body.data.vertices[other].co) / 2
        body.data.vertices[one].co = middle
        body.data.vertices[other].co = middle
    body.data.update()
    print("EMENDA", len(pairs), "pares, maior", round(float(np.linalg.norm(rest[first] - rest[second], axis=1).max()) * 1000, 1), "mm")


def shrink_earring(obj):
    """Each side shrinks around the point that hangs from the lobe."""
    points = np.array([tuple(obj.matrix_world @ vertex.co) for vertex in obj.data.vertices])
    middle = points[:, 0].mean()
    moved = points.copy()
    for side in (points[:, 0] < middle, points[:, 0] >= middle):
        if side.any():
            top = points[side][points[side][:, 2].argmax()]
            moved[side] = top + (points[side] - top) * EARRING_SCALE
    inverse = obj.matrix_world.inverted()
    for vertex, point in zip(obj.data.vertices, moved):
        vertex.co = inverse @ Vector(point)
    obj.data.update()


def tidy_earrings(meshes, seed):
    """The look's own earrings win over the game body's; the game pair stays on one woman in three, chosen by name, so it is not on everybody."""
    chosen = [obj for obj in meshes if not obj.name.startswith(("Acc_", "Earring")) and any(word in obj.name.lower() for word in ("earring", "hoops"))]
    game = [obj for obj in meshes if obj.name.startswith(("Acc_Earring", "Earring"))]
    if chosen or sum(map(ord, seed)) % GAME_EARRINGS_KEPT_EVERY:
        for obj in game:
            bpy.data.objects.remove(obj, do_unlink=True)
        game = []
    for obj in chosen + game:
        shrink_earring(obj)
    print("BRINCOS", [obj.name for obj in chosen + game])


def soften_brows(obj, skin):
    """`skin` is the skin's median colour in the map's own (sRGB) values."""
    for material in obj.data.materials:
        for node in material.node_tree.nodes:
            if node.type != "TEX_IMAGE" or not node.image:
                continue
            width, height = node.image.size
            pixels = np.array(node.image.pixels[:], dtype=np.float32).reshape(height, width, 4)
            pixels[:, :, :3] = pixels[:, :, :3] * (1 - BROW_TOWARD_SKIN) + np.asarray(skin[:3], np.float32) * BROW_TOWARD_SKIN
            alpha = pixels[:, :, 3]
            soft = sum(np.roll(np.roll(alpha, dy, 0), dx, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)) / 9
            pixels[:, :, 3] = np.minimum(alpha, soft) * BROW_ALPHA
            node.image.pixels[:] = pixels.ravel()
            node.image.pack()


def set_map(obj, image):
    for node in obj.active_material.node_tree.nodes:
        if node.type == "TEX_IMAGE":
            node.image = image


def clean_face(body, added, job, env):
    for obj in [item for item in added if item.type == "MESH" and "eyelash" in item.name.lower()]:
        added.remove(obj)
        bpy.data.objects.remove(obj, do_unlink=True)
    weld_seams(body)
    sex = "female" if str(job.get("kit", "")).startswith("female") else "male"
    for obj in [item for item in added if item.type == "MESH" and "low-poly" in item.name]:
        parle_polish.round_globes(obj)
    own = own_traits(job)
    calm_look([body, *(obj for obj in added if obj.type == "MESH" and any(word in obj.name for word in ("eyebrow", "teeth", "tongue", "low-poly")))], LIP_THIN[sex], MOUTH_BACK[sex], MOUTH_WIDE[sex], CHIN_BACK[sex], UNDER_LIP_ROUNDS[sex], LIPS_FULL[sex], LIPS_CUPID[sex], CHIN_LONGER[sex], fair_look(sex, own), BROW_THIN[sex], BROW_OUTER[sex], BROW_LONG[sex], BROW_ARCH[sex], BROW_LIFT_M[sex], UPPER_LIP_VOLUME[sex], own)
    soften_pout(body, POUT_FORWARD[sex])
    keep_lining_inside(body)
    tongue_follows_jaw([obj for obj in added if obj.type == "MESH" and obj.data.shape_keys])
    for obj in [item for item in added if item.type == "MESH" and "teeth" in item.name]:
        trim_gums(obj)
    tuck_eyes(body, added)
    if env.get("CAST_TEETH"):
        teeth_map = bpy.data.images.load(parle_mouth.clean_teeth(env["CAST_TEETH"]))
        teeth_map.pack()
        for obj in added:
            if obj.type == "MESH" and "teeth" in obj.name:
                material = obj.active_material
                material.blend_method = "OPAQUE"
                for link in [link for link in material.node_tree.links if link.to_socket.name == "Alpha"]:
                    material.node_tree.links.remove(link)
                set_map(obj, teeth_map)
                teeth_atlas(obj)
    wine = body_wine(body)
    shade_mouth([obj for obj in added if obj.type == "MESH" and obj.data.shape_keys], wine)
    teeth_for_inside = next((obj for obj in added if obj.type == "MESH" and "teeth" in obj.name), None)
    if teeth_for_inside is not None and body.data.shape_keys:
        shade_inside_of_mouth(body, key_points(teeth_for_inside.data.shape_keys.key_blocks[0]), wine)
    if env.get("CAST_EYE"):
        eye_map = bpy.data.images.load(env["CAST_EYE"])
        eye_map.pack()
        for obj in added:
            if obj.type == "MESH" and "low-poly" in obj.name:
                set_map(obj, eye_map)
                shader = obj.active_material.node_tree.nodes.get("Principled BSDF")
                if shader:
                    shader.inputs["Roughness"].default_value = EYE_ROUGHNESS
                for polygon in obj.data.polygons:
                    polygon.use_smooth = True
    # Last: the steps above tell the lips, the lining and the gums apart by how far the open jaw carries them.
    speaking = list(dict.fromkeys([body, *(obj for obj in added if obj.type == "MESH" and obj.data.shape_keys)]))
    calm_speech(speaking, SPEECH_OPEN[sex])
    mixed_speech(speaking, SPEECH_MIX)


def soften_under_chin(body):
    """Custom normals for the export: last thing before it, on the rest shape the game plays. See `parle_normals`."""
    mesh = body.data
    points = np.array([vertex.co[:] for vertex in mesh.vertices])
    point_of = np.array([loop.vertex_index for loop in mesh.loops])
    corners = np.array([item.vector[:] for item in mesh.corner_normals])
    average = np.zeros_like(points)
    np.add.at(average, point_of, corners)
    average /= np.maximum(np.linalg.norm(average, axis=1, keepdims=True), 1e-9)
    edges = np.array([edge.vertices[:] for edge in mesh.edges])
    share = parle_normals.smoothed_share(parle_normals.under_chin_share(points, average), edges)[point_of][:, None] * parle_normals.TURN
    turned = corners * (1 - share) + parle_normals.AHEAD[None] * share
    turned /= np.maximum(np.linalg.norm(turned, axis=1, keepdims=True), 1e-9)
    mesh.normals_split_custom_set(turned.tolist())
    mesh.update()
    print("QUEIXO normais", int((share > 0.01).sum()))
