#!/usr/bin/env python3
"""
    License information: data/licenses/makehuman_license.txt

    Puts the current Parle eye map into a body that is already built, without rebuilding it. The 42 cast bodies get the map from
    tools/npc-editor/cast-build.ts in the game repo; Rafa is not one of its jobs, so he stays on the old map unless this is run.
    The body keeps the iris it has: the map is chosen by the name of the texture on its eye material (parle_eye_<colour>).

      python tools/apply_eye_map.py <materials-dir> <body.glb> [<out.glb>]
"""
import json
import os
import struct
import sys

import cv2

MAGIC = 0x46546C67
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN = 0x004E4942
JPEG_QUALITY = 92


def read_glb(path):
    with open(path, "rb") as handle:
        data = handle.read()
    magic, version, _ = struct.unpack_from("<III", data, 0)
    if magic != MAGIC or version != 2:
        raise ValueError(f"{path}: not a glTF 2 binary")
    size, kind = struct.unpack_from("<II", data, 12)
    if kind != CHUNK_JSON:
        raise ValueError(f"{path}: first chunk is not JSON")
    doc = json.loads(data[20:20 + size])
    at = 20 + size
    bin_size, kind = struct.unpack_from("<II", data, at)
    if kind != CHUNK_BIN:
        raise ValueError(f"{path}: second chunk is not BIN")
    return doc, data[at + 8:at + 8 + bin_size]


def write_glb(path, doc, binary):
    text = json.dumps(doc, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    binary += b"\0" * (-len(binary) % 4)
    total = 12 + 8 + len(text) + 8 + len(binary)
    with open(path, "wb") as handle:
        handle.write(struct.pack("<III", MAGIC, 2, total))
        handle.write(struct.pack("<II", len(text), CHUNK_JSON) + text)
        handle.write(struct.pack("<II", len(binary), CHUNK_BIN) + binary)


def eye_image_index(doc):
    for material in doc["materials"]:
        if "low-poly" not in material.get("name", ""):
            continue
        texture = doc["textures"][material["pbrMetallicRoughness"]["baseColorTexture"]["index"]]
        return texture["source"]
    raise ValueError("no low-poly eye material in this body")


def swap_view(doc, binary, view_index, payload):
    """Replaces the bytes of one buffer view and moves every view that comes after it."""
    view = doc["bufferViews"][view_index]
    start, old = view.get("byteOffset", 0), view["byteLength"]
    old += -old % 4  # the padding after the old bytes goes with them, or the views after it lose their alignment
    padded = payload + b"\0" * (-len(payload) % 4)
    delta = len(padded) - old
    for other in doc["bufferViews"]:
        if other is not view and other.get("byteOffset", 0) >= start + old:
            other["byteOffset"] = other.get("byteOffset", 0) + delta
    view["byteLength"] = len(payload)
    doc["buffers"][0]["byteLength"] += delta
    return binary[:start] + padded + binary[start + old:]


def apply_map(materials, source, target):
    doc, binary = read_glb(source)
    image = doc["images"][eye_image_index(doc)]
    name = image.get("name", "")
    if not name.startswith("parle_eye_"):
        raise ValueError(f"{source}: the eye map is {name!r}, not one of the Parle maps")
    painted = cv2.imread(os.path.join(materials, f"{name}.png"))
    if painted is None:
        raise ValueError(f"no painted map for {name}")
    ok, encoded = cv2.imencode(".jpg", painted, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ValueError("could not encode the map")
    image["mimeType"] = "image/jpeg"
    binary = swap_view(doc, binary, image["bufferView"], encoded.tobytes())
    write_glb(target, doc, binary)
    return name


if __name__ == "__main__":
    print("EYE", apply_map(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else sys.argv[2]))
