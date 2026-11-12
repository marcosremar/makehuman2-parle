#!/usr/bin/env python3
"""
    License information: data/licenses/makehuman_license.txt

    The normals of `mouthClose` in a finished cast GLB, made the jaw's own, backwards (2026-11-12).

    Blender's glTF export takes a shape key's normals from the body with that key alone at full weight. `mouthClose` alone is the
    lower lip ridden 33 to 45 mm up into the nose, so its normal deltas were those of a crumpled chin (350 of 507 moved points off
    by more than 0.3, up to 3.3, on Ivone). The player only ever shows it under `jawOpen` (`mouthClose = jawOpen x seal`), and the
    normals blend linearly: at the seal the skin round the mouth caught the light in pale smears at both corners, out to the cheeks
    (docs/reports/2026-11-12-boca-formas-arkit/ in babylon-cinema). `mouthClose` moves each point by minus the jaw's move times a
    weight (`parle_mouth_close`), so its normal delta is minus the jaw's normal delta times the same weight: with the two at the
    same weight the lips meet with the normals of the rest.

      python tools/parle_close_normals.py <cast.glb>        (in place, after each export: the old normal bytes stay in the file, unread)
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np

CLOSE, JAW = "mouthClose", "jawOpen"
TYPES = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}
MOVED_M = 1e-6


def close_normals(jaw_moves, close_moves, jaw_normals):
    """Normal delta of `mouthClose`: minus the jaw's, times the share of the jaw's move that `mouthClose` takes back."""
    jaw_length = np.linalg.norm(jaw_moves, axis=1)
    share = np.where(jaw_length > MOVED_M, np.linalg.norm(close_moves, axis=1) / np.maximum(jaw_length, MOVED_M), 0.0)
    return -jaw_normals * np.clip(share, 0.0, 1.0)[:, None]


def read_glb(path):
    data = Path(path).read_bytes()
    size = struct.unpack_from("<I", data, 12)[0]
    doc = json.loads(data[20:20 + size])
    start = 20 + size + 8
    return doc, bytearray(data[start:start + struct.unpack_from("<I", data, 20 + size)[0]])


def write_glb(path, doc, binary):
    text = json.dumps(doc, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    binary = bytes(binary) + b"\0" * (-len(binary) % 4)
    total = 12 + 8 + len(text) + 8 + len(binary)
    Path(path).write_bytes(struct.pack("<III", 0x46546C67, 2, total) + struct.pack("<II", len(text), 0x4E4F534A) + text
                           + struct.pack("<II", len(binary), 0x004E4942) + binary)


def view_array(doc, binary, view, offset, dtype, count):
    start = doc["bufferViews"][view].get("byteOffset", 0) + offset
    return np.frombuffer(bytes(binary[start:start + count * np.dtype(dtype).itemsize]), dtype=dtype).copy()


def read_vec3(doc, binary, index):
    """A VEC3 float accessor, dense or sparse, as a dense array."""
    accessor = doc["accessors"][index]
    count = accessor["count"]
    out = np.zeros((count, 3), np.float32)
    if "bufferView" in accessor:
        out[:] = view_array(doc, binary, accessor["bufferView"], accessor.get("byteOffset", 0), np.float32, count * 3).reshape(count, 3)
    sparse = accessor.get("sparse")
    if sparse:
        ids = view_array(doc, binary, sparse["indices"]["bufferView"], sparse["indices"].get("byteOffset", 0),
                         TYPES[sparse["indices"]["componentType"]], sparse["count"]).astype(np.int64)
        values = view_array(doc, binary, sparse["values"]["bufferView"], sparse["values"].get("byteOffset", 0), np.float32, sparse["count"] * 3)
        out[ids] = values.reshape(-1, 3)
    return out


def append_view(doc, binary, payload):
    binary.extend(b"\0" * (-len(binary) % 4))
    doc["bufferViews"].append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(payload)})
    binary.extend(payload)
    return len(doc["bufferViews"]) - 1


def write_sparse_vec3(doc, binary, index, values):
    """Replaces a VEC3 accessor by a sparse one holding the non-zero rows of `values` (the old bytes stay, unread)."""
    rows = np.flatnonzero(np.any(values != 0, axis=1)).astype(np.uint32)
    accessor = doc["accessors"][index]
    for key in ("bufferView", "byteOffset", "min", "max"):
        accessor.pop(key, None)
    if len(rows) == 0:
        accessor.pop("sparse", None)
        return
    accessor["sparse"] = {
        "count": int(len(rows)),
        "indices": {"bufferView": append_view(doc, binary, rows.tobytes()), "componentType": 5125},
        "values": {"bufferView": append_view(doc, binary, values[rows].astype(np.float32).tobytes())},
    }


def patch(doc, binary):
    """Rewrites the `mouthClose` normals of every mesh that has it with `jawOpen`; returns how many points they move."""
    moved = 0
    for mesh in doc["meshes"]:
        names = mesh.get("extras", {}).get("targetNames", [])
        if CLOSE not in names or JAW not in names:
            continue
        for primitive in mesh["primitives"]:
            jaw, close = primitive["targets"][names.index(JAW)], primitive["targets"][names.index(CLOSE)]
            if "NORMAL" not in close or "NORMAL" not in jaw:
                continue
            normals = close_normals(read_vec3(doc, binary, jaw["POSITION"]), read_vec3(doc, binary, close["POSITION"]),
                                    read_vec3(doc, binary, jaw["NORMAL"]))
            write_sparse_vec3(doc, binary, close["NORMAL"], normals.astype(np.float32))
            moved += int(np.count_nonzero(np.any(normals != 0, axis=1)))
    doc["buffers"][0]["byteLength"] = len(binary) + (-len(binary) % 4)
    return moved


def main(path):
    doc, binary = read_glb(path)
    moved = patch(doc, binary)
    write_glb(path, doc, binary)
    print("BOCA normais do mouthClose:", moved, "pontos")


if __name__ == "__main__":
    main(sys.argv[1])
