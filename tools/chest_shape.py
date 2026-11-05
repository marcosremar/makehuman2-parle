#!/usr/bin/env python3
"""Gives a rigged body dump the torso relief of a sculpture, keeping the originals so it can run twice.

    python tools/chest_shape.py <body.npz> <base.npy> <delta.npy> <sculpture.npz> <shift.npy>

The dump is the one core/garment_weights reads. The base mesh moves too: clothes and necklaces fit it through
their .mhclo, so they follow the new torso. The shift is one displacement per vertex of the rigged body.
"""
import os
import shutil
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, ".")
from core.body_shape import moulded, pectoral  # noqa: E402


def main():
    path, base_path, delta_path, sculpt_path, target = sys.argv[1:6]
    data = dict(np.load(path, allow_pickle=False))
    original = data.setdefault("points_before", data["points"]).astype(np.float64)
    joints = dict(zip([str(name) for name in data["joint_names"]], data["joints"].astype(np.float64)))
    kept = base_path.replace(".npy", ".antes.npy")
    if not os.path.exists(kept):
        shutil.copy(base_path, kept)
    base = np.load(kept).astype(np.float64)
    delta = np.load(delta_path).astype(np.float64)
    move = pectoral(base[:len(delta)], delta, joints)
    tree = cKDTree(base[:len(delta)])
    shift = move[tree.query(original)[1]]
    names = [str(name) for name in data["names"]]
    torso = np.array([name.startswith(("Spine", "Hips")) for name in names])
    trunk = data["weights"][:, torso].sum(axis=1)
    # Under a shirt the relief does not show and only tears the cloth: "-" leaves it out.
    if sculpt_path != "-":
        sculpt = np.load(sculpt_path)["points"].astype(np.float64)
        shift = shift + moulded(original + shift, data["faces"].astype(np.int64), sculpt, trunk)
    gaps, near = cKDTree(original).query(base)
    np.save(base_path, (base + shift[near] * (gaps < 0.02)[:, None]).astype(np.float32))
    data["points"] = (original + shift).astype(np.float32)
    np.savez_compressed(path, **data)
    np.save(target, shift.astype(np.float32))
    print("PEITO", target, "passo", round(float(np.linalg.norm(shift, axis=1).max()) * 1000, 1), "mm")


main()
