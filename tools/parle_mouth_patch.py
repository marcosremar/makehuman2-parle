#!/usr/bin/env python3
"""
    License information: data/licenses/makehuman_license.txt

    Applies `parle_mouth` to a finished cast GLB, changing nothing else: the colour maps of teeth and tongue and, where the tooth does
    not show, the teeth's `viseme_aa`. A rebuild also picks up whatever the skin inputs hold at that moment; this one can run on the cast
    as it was published.

      python tools/parle_mouth_patch.py <in.glb> <out.glb>
"""
import json
import struct
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import parle_mouth  # noqa: E402
import parle_tongue  # noqa: E402

JPEG_QUALITY = 85
COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
TYPES = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}


def read_glb(path):
    data = Path(path).read_bytes()
    size = struct.unpack_from("<I", data, 12)[0]
    doc = json.loads(data[20:20 + size])
    start = 20 + size + 8
    return doc, data[start:start + struct.unpack_from("<I", data, 20 + size)[0]]


def view_of(doc, binary, index):
    view = doc["bufferViews"][index]
    start = view.get("byteOffset", 0)
    return binary[start:start + view["byteLength"]]


def accessor(doc, binary, index):
    item = doc["accessors"][index]
    kind, width = TYPES[item["componentType"]], COMPONENTS[item["type"]]
    if "bufferView" in item:
        raw = view_of(doc, binary, item["bufferView"])
        out = np.frombuffer(raw, kind, item["count"] * width, item.get("byteOffset", 0)).reshape(item["count"], width).copy()
    else:
        out = np.zeros((item["count"], width), kind)
    sparse = item.get("sparse")
    if sparse:
        at = np.frombuffer(view_of(doc, binary, sparse["indices"]["bufferView"]), TYPES[sparse["indices"]["componentType"]], sparse["count"],
                           sparse["indices"].get("byteOffset", 0))
        out[at] = np.frombuffer(view_of(doc, binary, sparse["values"]["bufferView"]), kind, sparse["count"] * width,
                                sparse["values"].get("byteOffset", 0)).reshape(sparse["count"], width)
    return out


def to_mouth_space(points):
    """glTF is +y up, +z front; `parle_mouth` reads the Blender frame, +z up, -y front."""
    return np.stack((points[:, 0], -points[:, 2], points[:, 1]), axis=1)


def mesh_with(doc, material_name):
    for mesh in doc["meshes"]:
        for primitive in mesh["primitives"]:
            if doc["materials"][primitive["material"]]["name"] == material_name:
                return mesh, primitive
    return None, None


def target_of(mesh, primitive, name):
    return primitive["targets"][mesh["extras"]["targetNames"].index(name)]["POSITION"]


def shaded_image(doc, binary, primitive, shade):
    source = doc["textures"][doc["materials"][primitive["material"]]["pbrMetallicRoughness"]["baseColorTexture"]["index"]]["source"]
    view = doc["images"][source]["bufferView"]
    image = cv2.imdecode(np.frombuffer(view_of(doc, binary, view), np.uint8), cv2.IMREAD_COLOR)
    height, width = image.shape[:2]
    uv = accessor(doc, binary, primitive["attributes"]["TEXCOORD_0"]).astype(np.float64)
    triangles = accessor(doc, binary, primitive["indices"]).reshape(-1, 3)
    canvas = parle_mouth.grow(parle_mouth.shade_map(width, height, uv[triangles], shade[triangles]))
    pixels = np.dstack((image.astype(np.float32), np.zeros(image.shape[:2], np.float32)))
    out = np.clip(parle_mouth.darken(pixels, canvas)[:, :, :3], 0, 255).round().astype(np.uint8)
    return view, cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])[1].tobytes()


def body_mesh(doc):
    speaking = [mesh for mesh in doc["meshes"] if {"viseme_aa", "jawOpen"} <= set(mesh.get("extras", {}).get("targetNames", []))]
    return max(speaking, key=lambda mesh: doc["accessors"][mesh["primitives"][0]["attributes"]["POSITION"]]["count"])


def write_glb(path, doc, binary, swapped, extra):
    out = bytearray()
    old = binary
    for index, view in enumerate(doc["bufferViews"]):
        start = view.get("byteOffset", 0)
        chunk = swapped.get(index, old[start:start + view["byteLength"]])
        out += b"\0" * (-len(out) % 4)
        view["byteOffset"] = len(out)
        view["byteLength"] = len(chunk)
        out += chunk
    for view, chunk in extra:
        out += b"\0" * (-len(out) % 4)
        view["byteOffset"] = len(out)
        view["byteLength"] = len(chunk)
        doc["bufferViews"].append(view)
        out += chunk
    out += b"\0" * (-len(out) % 4)
    doc["buffers"][0]["byteLength"] = len(out)
    text = json.dumps(doc, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    Path(path).write_bytes(b"".join((struct.pack("<III", 0x46546C67, 2, 28 + len(text) + len(out)), struct.pack("<I", len(text)), b"JSON", text,
                                     struct.pack("<I", len(out)), b"BIN\0", bytes(out))))


def lowered_aa(doc, binary, mesh, primitive, upper, drop):
    """A dense copy of the teeth's `viseme_aa` with the upper arch lower: the target may be sparse and the points it needs absent."""
    moves = accessor(doc, binary, target_of(mesh, primitive, "viseme_aa")).copy()
    moves[upper, 1] -= drop / parle_mouth.AA_WEIGHT
    index = len(doc["accessors"])
    doc["accessors"].append({"bufferView": len(doc["bufferViews"]), "componentType": 5126, "count": len(moves), "type": "VEC3",
                             "min": moves.min(axis=0).tolist(), "max": moves.max(axis=0).tolist()})
    primitive["targets"][mesh["extras"]["targetNames"].index("viseme_aa")]["POSITION"] = index
    return {"buffer": 0}, moves.astype(np.float32).tobytes()


def patch(source, out):
    doc, binary = read_glb(source)
    teeth_mesh, teeth = mesh_with(doc, "Human.teeth_base")
    tongue_mesh, tongue = mesh_with(doc, "Human.tongue01")
    body = body_mesh(doc)
    rest = to_mouth_space(accessor(doc, binary, teeth["attributes"]["POSITION"]).astype(np.float64))
    jaw = to_mouth_space(accessor(doc, binary, target_of(teeth_mesh, teeth, "jawOpen")).astype(np.float64))
    aa = to_mouth_space(accessor(doc, binary, target_of(teeth_mesh, teeth, "viseme_aa")).astype(np.float64))
    lower = parle_tongue.lower_teeth(jaw)
    skin = body["primitives"][0]
    body_rest = to_mouth_space(accessor(doc, binary, skin["attributes"]["POSITION"]).astype(np.float64))
    body_aa = to_mouth_space(accessor(doc, binary, target_of(body, skin, "viseme_aa")).astype(np.float64))
    body_jaw = to_mouth_space(accessor(doc, binary, target_of(body, skin, "jawOpen")).astype(np.float64))
    show = parle_mouth.incisor_show(rest, aa, jaw, lower, body_rest, body_aa, body_jaw)
    drop = parle_mouth.incisor_drop(show)
    swapped = dict((shaded_image(doc, binary, teeth, parle_mouth.teeth_shade(rest, lower)),))
    tongue_rest = to_mouth_space(accessor(doc, binary, tongue["attributes"]["POSITION"]).astype(np.float64))
    swapped.update((shaded_image(doc, binary, tongue, parle_mouth.tongue_shade(tongue_rest)),))
    extra = [lowered_aa(doc, binary, teeth_mesh, teeth, ~lower, drop)] if drop else []
    write_glb(out, doc, binary, swapped, extra)
    print(Path(source).name, "incisor", None if show is None else round(show * 1000, 1), "mm, dropped", round(drop * 1000, 1), "bytes", Path(source).stat().st_size, "->", Path(out).stat().st_size)


if __name__ == "__main__":
    patch(*sys.argv[1:3])
