"""Regression for the hand-sculpted head (2026-11-15): run with `python tests/test_parle_toon_sculpt.py` (no Blender).

A sculpted "estilo" key becomes a MakeHuman target of the hm08 base (index dx dy dz, decimetres, y up) holding only head points
that moved, and `parle_toon` takes a kind's own sculpture whole when its file exists."""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import parle_mouth_arkit as arkit  # noqa: E402
import parle_toon  # noqa: E402
import parle_toon_sculpt as sculpt  # noqa: E402

rng = np.random.default_rng(5)
points = rng.normal(size=(50, 3))
assert np.allclose(sculpt.blender_to_mh(arkit.mh_to_blender(points)), points), "the axes do not go back to MakeHuman's"
rest = rng.normal(size=(6, 3))
moved = rest.copy()
moved[1] += [0.001, -0.002, 0.003]
moved[4] += [0.002, 0.0, 0.0]
head = np.array([True, True, True, True, False, True])
lines = sculpt.target_lines(rest, moved, head)
assert [line.split()[0] for line in lines] == ["1"], f"a point that did not move, or out of the head, was written: {lines}"
index, *move = lines[0].split()
assert np.allclose([float(value) for value in move], sculpt.blender_to_mh(np.array([[0.001, -0.002, 0.003]]))[0], atol=1e-5)
assert len(sculpt.base_faces()) > 13000, "the base mesh's faces were not read"
# A kind with its own sculpture takes it whole; with none, TOON; the trial door goes over both.
os.environ.pop(parle_toon.TRIAL, None)
for kind in parle_toon.TOON:
    own = (parle_toon.FOLDER / f"{kind}.target").exists()
    assert parle_toon.shares(kind).get(kind, 0.0) == (1.0 if own else 0.0), f"the sculpture of {kind} is not taken whole"
print("ok")
