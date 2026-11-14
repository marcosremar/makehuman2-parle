"""Regression for the drawn head (2026-11-14): run with `python tests/test_parle_head.py` (no Blender).

The owner asked for the cast to be made by the faces he finds pretty, "os de estilo cartoon disney tipo o rafa", and for "a cabeça de
desenho". Moving the kit's own trait morphs until the ruler read the standard's numbers broke the face (2026-11-13): the ears pushed out,
the white of the eye loose from the lids, the chin crushed. The head is now moved as one thing by a chart of the face (`parle_head`).
This checks the chart and its field: both sides alike, each eye whole (a globe stays a ball and what lies on it stays on it), the
back of the skull wider and no more, no chart moves nothing, and the cast's step applies it to every shape key and to the bones.
"""
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import parle_head  # noqa: E402

os.environ.pop(parle_head.TRIAL, None)
women, men = parle_head.load_chart("female-slim"), parle_head.load_chart("male-normal")
assert women is not None and men is not None and parle_head.load_chart("dog") is None, "a chart per sex, and none for what is neither"
os.environ[parle_head.TRIAL] = "0"
assert parle_head.load_chart("female-slim") is None, "the trial door does not turn the chart off"
os.environ.pop(parle_head.TRIAL)

# The chart: the two sides of the face alike, nothing across on the middle line, and the grid read back at its own nodes.
rng = np.random.default_rng(3)
spots = rng.uniform((-1.8, -2.5), (1.8, 2.5), (300, 2))
for chart in (women, men):
    here, there = parle_head.chart_move(chart, spots), parle_head.chart_move(chart, spots * (-1, 1))
    assert np.allclose(here[:, 0], -there[:, 0], atol=1e-9) and np.allclose(here[:, 1], there[:, 1], atol=1e-9), "the two sides of the face do not move alike"
    middle = parle_head.chart_move(chart, np.column_stack([np.zeros(20), np.linspace(-2.5, 2.5, 20)]))
    assert np.abs(middle[:, 0]).max() < 1e-9, "the middle line of the face goes to one side"
    nodes = np.array([(x, y) for y in chart["y"][::7] for x in chart["x"][::5]])
    read = parle_head.chart_move(chart, nodes)
    rows, cols = np.searchsorted(chart["y"], nodes[:, 1]), np.searchsorted(chart["x"], nodes[:, 0])
    assert np.allclose(read[:, 0], chart["grid_across"][rows, cols], atol=1e-9) and np.allclose(read[:, 1], chart["grid_up"][rows, cols], atol=1e-9), "the grid is not read back at its nodes"
    far = parle_head.chart_move(chart, np.array([[9.0, 0.0], [0.0, 9.0], [0.0, -9.0]]))
    assert np.isfinite(far).all() and np.abs(far).max() < 1.0, "outside the grid the chart is not held at its edge"

# What the women's chart says (read on 2026-11-14): the face wider, the eyes lower, the nose and the mouth lower still, so the lower
# third is shorter; the men's asks for less, the standard of the men being four faces with Rafa's counted twice.
eye, nose, mouth = (parle_head.chart_move(women, np.array([spot]))[0] for spot in ((0.5, 0.0), (0.0, -0.6), (0.0, -1.0)))
assert 1.05 < women["wide"] < 1.15, f"the women's face is not a tenth wider: {women['wide']}"
assert -0.16 < eye[1] < -0.05 and nose[1] < eye[1] - 0.03 and mouth[1] < -0.08, f"eyes, nose and mouth do not come down as read: {eye[1]:.3f}, {nose[1]:.3f}, {mouth[1]:.3f}"
assert 1.05 < women["eye"]["grows"] < 1.15, f"the women's eye grows as its width asks, canthus to canthus: {women['eye']['grows']}"
assert abs(parle_head.chart_move(men, np.array([[0.5, 0.0]]))[0, 1]) < abs(eye[1]) and men["wide"] < women["wide"], "the men's chart asks for more than the women's"

# The field, on a head 0.07 m between the eyes. Each eye whole: points on a ball about the middle of a globe, inside EYE_WHOLE radii,
# stay on a ball about where that middle goes, grown by the chart's number, and keep their directions.
unit, radius = 0.07, 0.016
eyes = [(np.array([-unit / 2, -0.05, 1.5]), radius), (np.array([unit / 2, -0.05, 1.5]), radius)]
field = parle_head.head_field(women, eyes)
ways = rng.normal(size=(200, 3))
ways /= np.linalg.norm(ways, axis=1)[:, None]
for centre, _ in eyes:
    for share in (0.5, 1.0, parle_head.EYE_WHOLE - 0.01):
        ball = centre + ways * radius * share
        moved, middle = field(ball), field(centre[None])[0]
        assert np.allclose(moved - middle, women["eye"]["grows"] * (ball - centre), atol=1e-9), "an eye does not go as one body: a globe would not stay a ball, nor the lids on it"
far_out = eyes[0][0] + ways * radius * (parle_head.EYE_REACH + 0.5)
assert np.allclose(parle_head.eye_share(far_out, *eyes[0]), 0), "an eye still pulls past its reach"
middles = field(np.array([eyes[0][0], eyes[1][0]]))
apart = np.linalg.norm(middles[0] - middles[1]) / unit
assert abs(apart - (1 + 2 * women["eye"]["across"])) < 0.02 and abs((middles[0][2] - 1.5) / unit - women["eye"]["up"]) < 0.02, "the eyes do not go where the chart sends them"
assert abs(middles[0][1] - eyes[0][0][1]) < 1e-9, "the field moves the face front to back: the chart is of the face seen from the front"

# Behind the ears the skull is wider and no more: the nape does not slide down the neck with the nose.
back = np.array([[0.05, -0.05 + unit * (parle_head.FACE_BACK[1] + 0.2), 1.45], [-0.06, -0.05 + unit * (parle_head.FACE_BACK[1] + 0.5), 1.52]])
moved = field(back) - back
assert np.allclose(moved[:, 2], 0, atol=1e-9) and np.allclose(moved[:, 0], (women["wide"] - 1) * back[:, 0], atol=1e-9), f"the back of the skull does more than widen: {moved}"
front = np.array([[0.0, -0.11, 1.5 - 0.6 * unit]])
assert (field(front) - front)[0, 2] < -0.1 * unit, "the nose does not come down on the face itself"

# None of the chart, nothing moves; half of it, half the way (outside the eyes, where the move is the chart's alone).
cloud = rng.uniform((-0.1, -0.12, 1.3), (0.1, 0.1, 1.62), (300, 3))
assert np.abs(parle_head.head_field(women, eyes, share=0.0)(cloud) - cloud).max() < 1e-12, "no share of the chart still moves the head"
clear = (parle_head.eye_share(cloud, *eyes[0]) == 0) & (parle_head.eye_share(cloud, *eyes[1]) == 0)
assert np.allclose(parle_head.head_field(women, eyes, share=0.5)(cloud)[clear] - cloud[clear], (field(cloud)[clear] - cloud[clear]) / 2, atol=1e-12), "half the chart is not half the way"

# The data is measures only: a grid of moves and a few numbers, nothing of any one face.
for path in sorted((ROOT / "data" / "parle_head").glob("*.json")):
    text = path.read_text()
    assert set(__import__("json").loads(text)) == {"sex", "about", "unit", "size", "wide", "across", "up", "move_across", "move_up", "eye"}, f"{path.name} keeps something else than the chart"
    for word in ("sofia", "rafa", "anna", "jasmine", "mirabel", "aladdin", "ryder", "position", "marcos"):
        assert word not in text.lower(), f"{path.name} names a face or keeps its points: {word}"

# The cast's step: the chart goes to every shape key (`warp`), by how much of each point follows the head, a point of the jaw being
# the head's from HEAD_WHOLE of its weight on, and the bones of the head go with it; it runs before the build of the body.
polish = (ROOT / "tools" / "parle_polish.py").read_text()
step = polish.split("def drawn_head(meshes, rig, kit):")[1].split("\n\n\ndef ")[0]
assert "parle_head.head_field(chart, eye_balls(globe))" in step and "warp(obj, move)" in step, "the head is not moved by the chart, or not in every shape key"
assert "smooth(group_share(obj, head) / HEAD_WHOLE)" in step and "edit_bones" in step, "the jaw is taken by its weight alone, or the bones of the head stay behind"
body = polish.split("def natural_body(meshes, rig, height, sex, kit):")[1].split("\n\n\ndef ")[0]
assert body.index("drawn_head(meshes, rig, kit)") < body.index("natural_build(meshes, rig, height, kit)"), "the head takes its shape after the body is built"
assert "eyes_no_rounder(meshes, eye_round_most(sex))" in body, "the eyes of a drawn head open no rounder than the kit's own limit"
print("ok")
