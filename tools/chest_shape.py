#!/usr/bin/env python3
"""Writes the defined chest into a rigged body dump, keeping the original so it can run twice.

    python tools/chest_shape.py <body.npz> <base.npy> <sculpt.delta.npy> <shift.npy> [strength]

The dump is the one core/garment_weights reads. The shift is one displacement per vertex of that body.
"""
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, ".")
from core.body_shape import chest_shift  # noqa: E402


def main():
    path, base_path, sculpt_path, target = sys.argv[1:5]
    strength = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
    data = dict(np.load(path, allow_pickle=False))
    original = data.setdefault("points_before", data["points"]).astype(np.float64)
    joints = dict(zip([str(name) for name in data["joint_names"]], data["joints"].astype(np.float64)))
    sculpt = np.load(sculpt_path).astype(np.float64)
    # The base mesh moves too: clothes and necklaces fit it through their .mhclo, so they follow the new chest.
    kept = base_path.replace(".npy", ".antes.npy")
    if not os.path.exists(kept):
        shutil.copy(base_path, kept)
    base = np.load(kept).astype(np.float64)
    shift, move = chest_shift(original, base[:len(sculpt)], sculpt, joints, strength)
    moved = base.copy()
    moved[:len(sculpt)] += move
    np.save(base_path, moved.astype(np.float32))
    data["points"] = (original + shift).astype(np.float32)
    np.savez_compressed(path, **data)
    np.save(target, shift.astype(np.float32))
    print("PEITO", target, "passo", round(float(np.linalg.norm(shift, axis=1).max()) * 1000, 1), "mm")


main()
