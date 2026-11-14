"""
    License information: data/licenses/makehuman_license.txt

    The drawn head (owner, 2026-11-13 and 14). He chose the faces he finds pretty, "os de estilo cartoon disney tipo o rafa", had them
    kept as the standard the cast is to be made by (`data/parle_beauty/pretty_faces.json`), and then asked for "a cabeça de desenho":
    the head follows the shape of the drawn ones, "o rosto dos desenhos está bonito".
    The first try moved the kit's own trait morphs until the ruler read the standard's numbers (`parle_beauty.PRETTY`): the numbers
    came and the face broke, the ears pushed out, the white of the eye loose from the lids, the chin crushed (2026-11-13). A trait
    morph moves the skin alone and each its own way. Here the head is moved as one thing instead: a chart says where each place of
    the face goes, seen from the front, and everything that is on the head goes with it, skin, eye globes, teeth, tongue, eyebrows,
    hair, glasses. Every shape key is moved the same way (`parle_polish.warp`), so the mouth still speaks and the lids still shut.

    The chart (`data/parle_head/<sex>.json`) is measures only. It comes from the 478 points the ruler reads on a portrait
    (`parle_beauty`), in distances between the pupils: the mean face of the standard of that sex, at the size where its face is as
    tall as the cast's, less the mean face of the cast, with the chin held still, as it is the chin that ties the head to the neck.
    A smooth surface through those differences is kept on a grid. Read on the women (2026-11-14, five faces of the standard
    against four of the cast): the face a tenth wider, the eyes 0.11 of the distance between them lower (8 mm), the nose 0.19
    lower and the mouth 0.15, so the lower third of the face, from the nose to the chin, a quarter shorter; forehead and chin
    where they were. On the men (Rafa, counted twice as the face of the game's own style, and two more, against three of the
    cast) it asks for less: the face 3% wider, the eyes 0.05 lower and a sixth bigger. A face of the standard with its mouth
    open or smiling is left out of the mean (one of the women's, one of the men's): its lips are not at rest, and with the smiling
    man in it the chart pulled the corners of every man's mouth up. No mesh, texture or point of any one face is kept: the film
    characters among the standard are a reference of style and no more, and the aim is the middle of the group.

    The surface is a thin-plate spline with slack, and the eyebrows are left out of it with the eyes: the ruler reads the hair of
    an eyebrow, which has its own height (`parle_face.BROW_LIFT_M`). How it was found: a loop that built three people, measured
    them with the ruler and added to the chart what was still missing drove the skin about the eyes to stretch by 89% in four
    rounds, with creases down the cheeks, since a point of the ruler that lies in an eye's own zone does not answer to the chart
    and the loop asked more of it every round. The chart is the surface through the first differences, tighter slack having bought
    little (summed over 26 measures, the ruler read the three women 26 to 31 deviations from the standard with it against 24 to 25
    after four rounds, and 42 to 47 before any chart), plus half of what one round still found missing, each point held to 0.05
    of the distance between the eyes, and nothing along the outline of the face, where the ruler's points slide: 27 to 28 as it is
    kept, and the three men 20 to 21 where they read 26 to 28. It stretches the skin by no more than 27% anywhere on the face.
    The tools that make it stay with the game's repository, next to the registry of the faces.

    Each eye goes as one body: within EYE_WHOLE radii of the middle of its globe everything moves alike, grown about that middle
    by what the chart stretches the face there, so a globe stays a ball and the lids stay on it; from there to EYE_REACH radii it
    gives way to the chart. What the chart says of the eyes themselves is left out of it for that reason, and how round they open
    is the lids' business (`parle_face.FAIR_LOOK`, `parle_polish.eyes_no_rounder`).
    Behind the ears the chart gives way to the skull's own move, wider and no more: the nape does not slide down the neck with the nose.
    This file has no bpy: `parle_polish.natural_build` applies the field.
"""
import json
import os

import numpy as np

FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "parle_head")
# In radii of the eye globe: everything this near the middle of a globe goes with it whole, and its share is gone this far out.
# Seen from the front the corners of the lids stand 1.1 radii from the pupil and the eyebrow 1.5 (the ruler's points on the cast,
# a globe being 0.22 of the distance between the eyes); with the depth taken by guess, 1.3 and 1.8 from the middle of the globe.
# So the lids are within the whole and the eyebrow keeps most of the chart's move.
EYE_WHOLE, EYE_REACH = 1.35, 2.0
# In distances between the eyes, behind their middles: the chart is whole to the first, past the ears, and gone by the second.
FACE_BACK = (1.25, 1.9)
# A trial of another chart, or of none ("0"), without touching the data: a QA door, not a mode of the cast.
TRIAL = "PARLE_HEAD"


def smooth(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def load_chart(kit):
    """The chart of the kit's sex, as arrays; None for a kit with no chart. The grid is kept for one side of the face and is the
    same on the other."""
    path = os.environ.get(TRIAL, "")
    if path in ("0", "off"):
        return None
    if not path.endswith(".json"):
        sex = next((name for name in ("female", "male") if str(kit).startswith(name)), None)
        path = os.path.join(FOLDER, f"{sex}.json")
    if not os.path.exists(path):
        return None
    chart = json.load(open(path))
    across, up = np.asarray(chart["move_across"], float), np.asarray(chart["move_up"], float)
    # Both sides: the move across turns over with the side, the move up does not.
    chart["grid_across"] = np.concatenate([-across[:, :0:-1], across], axis=1)
    chart["grid_up"] = np.concatenate([up[:, :0:-1], up], axis=1)
    chart["x"] = np.concatenate([-np.asarray(chart["across"], float)[:0:-1], np.asarray(chart["across"], float)])
    chart["y"] = np.asarray(chart["up"], float)
    return chart


def shares(t):
    """Catmull-Rom: how much each of the four nodes about a place counts, `t` being how far the place is past the second."""
    return np.stack([(-t + 2 * t * t - t ** 3) / 2, (2 - 5 * t * t + 3 * t ** 3) / 2, (t + 4 * t * t - 3 * t ** 3) / 2, (-t * t + t ** 3) / 2], axis=1)


def read(grid, xs, ys, spots):
    """The grid's value at each spot (across, up): a curve with no corner between its nodes, held at its edge outside."""
    spots = np.asarray(spots, float)
    rows, cols = grid.shape
    col = np.clip((spots[:, 0] - xs[0]) / (xs[1] - xs[0]), 0, cols - 1)
    row = np.clip((spots[:, 1] - ys[0]) / (ys[1] - ys[0]), 0, rows - 1)
    first_col, first_row = np.clip(np.floor(col).astype(int), 0, cols - 2), np.clip(np.floor(row).astype(int), 0, rows - 2)
    by_col, by_row = shares(col - first_col), shares(row - first_row)
    value = np.zeros(len(spots))
    for i in range(4):
        line = np.clip(first_row + i - 1, 0, rows - 1)
        for j in range(4):
            value += by_row[:, i] * by_col[:, j] * grid[line, np.clip(first_col + j - 1, 0, cols - 1)]
    return value


def chart_move(chart, spots):
    """How far each spot of the face goes, across and up, all in distances between the eyes."""
    return np.stack([read(chart["grid_across"], chart["x"], chart["y"], spots), read(chart["grid_up"], chart["x"], chart["y"], spots)], axis=1)


def eye_share(points, centre, radius):
    """How much of each point goes with that eye as one body."""
    away = np.linalg.norm(np.asarray(points, float) - centre, axis=1) / radius
    return smooth((EYE_REACH - away) / (EYE_REACH - EYE_WHOLE))


def head_field(chart, eyes, share=1.0):
    """`field(points)` gives the new place of each point of the head, in the world: x across, y from the face to the back, z up.
    `eyes` are the middle and the radius of each eye globe (`parle_polish.eye_balls`); `share` is how much of the chart is taken."""
    centres = [np.asarray(centre, float) for centre, _ in eyes]
    middle, unit = (centres[0] + centres[1]) / 2, float(np.linalg.norm(centres[0] - centres[1]))
    wide, grows = float(chart.get("wide", 1.0)), float(chart["eye"]["grows"])

    def face(points):
        spots = (points[:, [0, 2]] - middle[[0, 2]]) / unit
        front = chart_move(chart, spots)
        # Behind the ears the skull is wider and no more: the nape does not slide down with the nose.
        back = np.stack([(wide - 1) * spots[:, 0], np.zeros(len(spots))], axis=1)
        ahead = 1 - smooth(((points[:, 1] - middle[1]) / unit - FACE_BACK[0]) / (FACE_BACK[1] - FACE_BACK[0]))
        move = np.zeros_like(points)
        move[:, [0, 2]] = (back + (front - back) * ahead[:, None]) * unit * share
        return move

    def field(points):
        points = np.asarray(points, float)
        move, left = face(points), np.ones(len(points))
        whole = np.zeros_like(points)
        for centre, radius in eyes:
            centre = np.asarray(centre, float)
            mine = eye_share(points, centre, radius)
            whole += mine[:, None] * (face(centre[None])[0] + (grows - 1) * share * (points - centre))
            left = left - mine
        return points + np.clip(left, 0, 1)[:, None] * move + whole

    return field
