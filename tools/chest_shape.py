#!/usr/bin/env python3
"""Gives a rigged body dump the torso muscles of a man, and the shoulders of a muscled one, keeping the originals so it can run twice.

    python tools/chest_shape.py <body.npz> <base.npy> <shift.npy> [kit=<male-muscled|male-normal|male-slim>] [--plain]

Every bare-chested man takes the chest of the sculpture in the place of the base mesh's own, which is two round mounds: on the normal man
of the beach they hung like breasts (owner, 2026-11-11). The muscled man takes all of its relief and the deltoid targets; the others take
a share of it and keep their shoulders.

The dump is the one core/garment_weights reads. The base mesh moves too: clothes and necklaces fit it through
their .mhclo, so they follow the new torso. The shift is one displacement per vertex of the rigged body.
"""
import os
import shutil
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, ".")
from core.body_shape import moulded, muscles  # noqa: E402


RELIEF = "data/sculpt/male-torso-relief.npz"
# How much of the sculpture's relief each kit takes: the plates of a man who trains are not those of a slim one.
STRENGTH = {"male-muscled": 1.0, "male-normal": 0.9, "male-slim": 0.4}
# How many times the relief is smoothed over the mesh. Not smoothed, on the normal man the lower edge of the chest came out as a hard
# straight ledge with a square notch at the breastbone; six times over it is the soft edge of a chest that does not train (2026-11-11).
# On the slim man 0.7 read as a band strapped across the chest (2026-11-11).
SOFT = {"male-muscled": 1, "male-normal": 6, "male-slim": 8}


def main():
    path, base_path, target = sys.argv[1:4]
    kit = next((word.split("=", 1)[1] for word in sys.argv[4:] if word.startswith("kit=")), "male-muscled")
    data = dict(np.load(path, allow_pickle=False))
    original = data.setdefault("points_before", data["points"]).astype(np.float64)
    kept = base_path.replace(".npy", ".antes.npy")
    if not os.path.exists(kept):
        shutil.copy(base_path, kept)
    base = np.load(kept).astype(np.float64)
    move = muscles(len(base)) if kit == "male-muscled" else np.zeros_like(base)
    shift = move[cKDTree(base).query(original)[1]]
    names = [str(name) for name in data["names"]]
    torso = np.array([name.startswith(("Spine", "Hips")) for name in names])
    trunk = data["weights"][:, torso].sum(axis=1)
    moved = base + move
    # Under a shirt the relief does not show and only tears the cloth: --plain leaves it out.
    if "--plain" not in sys.argv:
        chart = np.load(RELIEF)
        relief = chart["relief"].astype(np.float64), chart["known"].astype(np.float64)
        tuned = dict(word.split("=", 1) for word in sys.argv[4:] if word.startswith(("strength=", "soft=")))  # to try a value out
        fine = moulded(original + shift, data["faces"].astype(np.int64), relief, trunk, float(tuned.get("strength", STRENGTH[kit])), int(tuned.get("soft", SOFT[kit])))
        shift = shift + fine
        gaps, near = cKDTree(original).query(base)
        moved += fine[near] * (gaps < 0.02)[:, None]
    np.save(base_path, moved.astype(np.float32))
    data["points"] = (original + shift).astype(np.float32)
    np.savez_compressed(path, **data)
    np.save(target, shift.astype(np.float32))
    print("PEITO", target, "passo", round(float(np.linalg.norm(shift, axis=1).max()) * 1000, 1), "mm")


main()
