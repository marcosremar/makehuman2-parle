"""
    License information: data/licenses/makehuman_license.txt

    Cartoon head targets of the MakeHuman community (CC0, `data/parle_toon/README.md`), mixed into the cast's heads by kind of person
    (owner, 2026-11-15: "cabeça base esculpida", then yes to fetching these). Sculpted by hand by their authors on the same base mesh
    (hm08), they make the jump the fields of `parle_style` could not: those stopped at 5 in 10 against the series. Each target is read
    onto the body by the walk of `parle_mouth_arkit` (base head onto body head), eased like its shapes, and added to the rest shape
    and to every shape key alike, so every move keeps its own size. TOON has the share of each target by kind.
"""
from pathlib import Path

import numpy as np

import parle_mouth_arkit as arkit

FOLDER = Path(__file__).resolve().parents[1] / "data" / "parle_toon"
# At 0.6 (round 17) the heads read as one round baby-faced cartoon, further from the series' long faces and big noses (3.6 against
# 5.0 without); at 0.3 (round 19), 4.25, with glasses off the nose and the gaze astray. Off for every kind until a target nearer the
# series is found; the door (TRIAL) stays for trials.
TOON = {
    "male": {},
    "male-old": {},
    "female": {},
    "female-old": {},
}
# The ring of skin read about each eye globe, in its radii, and how much a globe may grow with it.
GLOBE_RING = 1.8
GLOBE_GROW_MOST = 1.15
# A trial of other shares, as JSON {target: share}, without touching this file: a QA door, not a mode of the cast.
TRIAL = "PARLE_TOON"


def shares(kind):
    """The targets of a kind and their shares: the hand-sculpted head of that kind (`parle_toon_sculpt`, `<kind>.target`) whole when
    it exists, TOON otherwise, and the trial door over both."""
    import json
    import os

    own = {kind: 1.0} if (FOLDER / f"{kind}.target").exists() else {}
    return {**TOON[kind], **own, **json.loads(os.environ.get(TRIAL) or "{}")}


def toon_move(rest, jaw, edges, mix, base=None, base_edges=None):
    """The move of every body point for the targets of `mix` ({name: share}), read onto the body and eased."""
    if base is None:
        base, base_edges = arkit.read_base()
    ids, scale = arkit.base_ids(rest + jaw, base + arkit.dense(arkit.PACK / "jawOpen.target"), base_edges)
    total = sum(share * arkit.dense(FOLDER / f"{name}.target") for name, share in mix.items())
    arkit.BASE_SHIFT = total
    return arkit.clean(arkit.on_body(total, ids, scale), rest, edges)


def points_of(block):
    """The points of a shape key, or of a mesh's own vertices."""
    items = block if hasattr(block, "foreach_get") else block.data
    flat = np.empty(len(items) * 3)
    items.foreach_get("co", flat)
    return flat.reshape(-1, 3)


def follow(obj, skin_rest, skin_move):
    """Every point of another piece (eyebrow, teeth, tongue) moved as the skin point nearest to it moved, in every shape key."""
    keys = obj.data.shape_keys.key_blocks if obj.data.shape_keys else []
    rest = points_of(obj.data.vertices)
    hit, _ = arkit.nearest(skin_rest, rest)
    move = skin_move[hit]
    for block in keys:
        block.data.foreach_set("co", (points_of(block) + move).ravel())
    obj.data.vertices.foreach_set("co", (rest + move).ravel())
    obj.data.update()


def globes_follow(obj, skin_rest, skin_move):
    """Each eye globe goes with its orbit as one body: moved by the mean move of the skin about it and grown by how much that ring
    of skin grew about its middle (round 16: the cartoon head opened the orbits wider and the globes stayed, leaving a hole)."""
    rest = points_of(obj.data.vertices)
    out = rest.copy()
    for half in (rest[:, 0] < rest[:, 0].mean(), rest[:, 0] >= rest[:, 0].mean()):
        centre = rest[half].mean(axis=0)
        radius = float(np.linalg.norm(rest[half] - centre, axis=1).max())
        away = np.linalg.norm(skin_rest - centre, axis=1)
        ring = (away > radius) & (away < GLOBE_RING * radius) & (skin_rest[:, 1] < centre[1])
        if ring.sum() < 8:
            continue
        before, after = skin_rest[ring], skin_rest[ring] + skin_move[ring]
        grow = float(np.linalg.norm(after - after.mean(axis=0), axis=1).mean() / np.linalg.norm(before - before.mean(axis=0), axis=1).mean())
        moved = centre + (after.mean(axis=0) - before.mean(axis=0))
        out[half] = moved + (rest[half] - centre) * min(grow, GLOBE_GROW_MOST)
    keys = obj.data.shape_keys.key_blocks if obj.data.shape_keys else []
    for block in keys:
        block.data.foreach_set("co", (points_of(block) + out - rest).ravel())
    obj.data.vertices.foreach_set("co", out.ravel())
    obj.data.update()


def toon_head(body, kind, others=()):
    """Adds the kind's cartoon targets to the rest shape and to every shape key of the body, and takes the other pieces of the head
    with it (`others`: eye globes as one body each, the rest point by point). How far the point that moved most went."""
    mix = {name: share for name, share in shares(kind).items() if share}
    if not mix:
        return 0.0
    keys = body.data.shape_keys.key_blocks if body.data.shape_keys else None
    if keys is None or "jawOpen" not in keys:
        raise ValueError("the body has no jawOpen: the cartoon head cannot be read onto it")

    def points(block):
        flat = np.empty(len(block.data) * 3)
        block.data.foreach_get("co", flat)
        return flat.reshape(-1, 3)

    rest = points(keys[0])
    edges = np.empty(len(body.data.edges) * 2, int)
    body.data.edges.foreach_get("vertices", edges)
    move = toon_move(rest, points(keys["jawOpen"]) - rest, edges.reshape(-1, 2), mix)
    for block in keys:
        block.data.foreach_set("co", (points(block) + move).ravel())
    body.data.vertices.foreach_set("co", (rest + move).ravel())
    body.data.update()
    for obj in others:
        (globes_follow if "low-poly" in obj.name else follow)(obj, rest, move)
    return float(np.linalg.norm(move, axis=1).max())

