#!/usr/bin/env python3
"""
    License information: data/licenses/makehuman_license.txt

    Applies `parle_mouth` to a finished cast GLB, changing nothing else: the colour maps of teeth, tongue and the skin's inside of the
    mouth. A rebuild also picks up whatever the skin inputs hold at that moment; this one can run on the cast
    as it was published.

      python tools/parle_mouth_patch.py <in.glb> <out.glb>

    Not idempotent: the shade multiplies, so <in.glb> must be the body as published before any patch. A body already patched is refused, by
    the mark this script leaves in `asset.extras`, or by a tongue already dark (the #2896 bodies carry no mark).
"""
import json
import struct
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import parle_mouth  # noqa: E402

JPEG_QUALITY = 85
MARK = "parleMouthShade"
# The teeth map starts from the fork's clean map, not from the body: a body rebuilt with an earlier shade in its map would be shaded twice.
TEETH_MAP = Path(__file__).parent.parent / "data/eyes/hm08/materials/parle_teeth_clean.png"
TREATED_TONGUE_LUMA = 120.0
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


def base_image(doc, binary, primitive):
    source = doc["textures"][doc["materials"][primitive["material"]]["pbrMetallicRoughness"]["baseColorTexture"]["index"]]["source"]
    view = doc["images"][source]["bufferView"]
    return view, cv2.imdecode(np.frombuffer(view_of(doc, binary, view), np.uint8), cv2.IMREAD_COLOR)


def shaded_image(doc, binary, primitive, shade, gums=False, only=None, base=None, wine=None, inside=False):
    """`only`: the points whose triangles take the shade, for a map the mesh shares with the rest of the body; elsewhere the map stays.
    `base`: a clean map to start from instead of the one in the body."""
    view, image = base_image(doc, binary, primitive)
    if base is not None:
        image = cv2.resize(cv2.imread(str(base), cv2.IMREAD_COLOR), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)
    height, width = image.shape[:2]
    uv = accessor(doc, binary, primitive["attributes"]["TEXCOORD_0"]).astype(np.float64)
    triangles = accessor(doc, binary, primitive["indices"]).reshape(-1, 3)
    if only is not None:
        triangles = triangles[parle_mouth.body_triangles(triangles, only, uv)]
    canvas = parle_mouth.grow(parle_mouth.shade_map(width, height, uv[triangles], shade[triangles]), fill=None if only is None else 1.0)
    rgba = np.dstack((image[:, :, ::-1].astype(np.float32) / 255, np.ones(image.shape[:2], np.float32)))
    out = (parle_mouth.darken(rgba, canvas, gums, wine, inside)[:, :, 2::-1] * 255).clip(0, 255).round().astype(np.uint8)
    return view, cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])[1].tobytes()


def tongue_luma(doc, binary, primitive):
    _, image = base_image(doc, binary, primitive)
    uv = accessor(doc, binary, primitive["attributes"]["TEXCOORD_0"])
    rows = np.clip((uv[:, 1] * image.shape[0]).astype(int), 0, image.shape[0] - 1)
    columns = np.clip((uv[:, 0] * image.shape[1]).astype(int), 0, image.shape[1] - 1)
    return float((image[rows, columns, ::-1].astype(np.float32) @ parle_mouth.LUMA).mean())


def body_mesh(doc):
    speaking = [mesh for mesh in doc["meshes"] if {"viseme_aa", "jawOpen"} <= set(mesh.get("extras", {}).get("targetNames", []))]
    return max(speaking, key=lambda mesh: doc["accessors"][mesh["primitives"][0]["attributes"]["POSITION"]]["count"])


def write_glb(path, doc, binary, swapped):
    out = bytearray()
    old = binary
    for index, view in enumerate(doc["bufferViews"]):
        start = view.get("byteOffset", 0)
        chunk = swapped.get(index, old[start:start + view["byteLength"]])
        out += b"\0" * (-len(out) % 4)
        view["byteOffset"] = len(out)
        view["byteLength"] = len(chunk)
        out += chunk
    out += b"\0" * (-len(out) % 4)
    doc["buffers"][0]["byteLength"] = len(out)
    text = json.dumps(doc, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    Path(path).write_bytes(b"".join((struct.pack("<III", 0x46546C67, 2, 28 + len(text) + len(out)), struct.pack("<I", len(text)), b"JSON", text,
                                     struct.pack("<I", len(out)), b"BIN\0", bytes(out))))


def patch(source, out):
    doc, binary = read_glb(source)
    _, teeth = mesh_with(doc, "Human.teeth_base")
    _, tongue = mesh_with(doc, "Human.tongue01")
    if doc["asset"].get("extras", {}).get(MARK) or tongue_luma(doc, binary, tongue) < TREATED_TONGUE_LUMA:
        sys.exit(f"{source}: already treated; patch the body as published")
    body = body_mesh(doc)
    rest = to_mouth_space(accessor(doc, binary, teeth["attributes"]["POSITION"]).astype(np.float64))
    skin = body["primitives"][0]
    body_rest = to_mouth_space(accessor(doc, binary, skin["attributes"]["POSITION"]).astype(np.float64))
    _, skin_image = base_image(doc, binary, skin)
    skin_uv = accessor(doc, binary, skin["attributes"]["TEXCOORD_0"]).astype(np.float64)
    moves = [accessor(doc, binary, target_of(body, skin, name)).astype(np.float64) for name in ("viseme_U", "viseme_PP")]
    wine = parle_mouth.wine_for(parle_mouth.lip_luma(skin_image[:, :, ::-1].astype(np.float32) / 255, skin_uv, moves))
    swapped = dict((shaded_image(doc, binary, teeth, parle_mouth.teeth_shade(rest), gums=True, base=TEETH_MAP, wine=wine),))
    tongue_rest = to_mouth_space(accessor(doc, binary, tongue["attributes"]["POSITION"]).astype(np.float64))
    swapped.update((shaded_image(doc, binary, tongue, parle_mouth.tongue_shade(tongue_rest)),))
    inside = parle_mouth.body_inside(body_rest, skin_uv, rest, accessor(doc, binary, skin["indices"]).reshape(-1, 3))
    swapped.update((shaded_image(doc, binary, skin, np.full(len(body_rest), parle_mouth.MARK_INSIDE, np.float64), only=inside, wine=wine, inside=True),))
    doc["asset"].setdefault("extras", {})[MARK] = 2
    write_glb(out, doc, binary, swapped)
    print(Path(source).name, "bytes", Path(source).stat().st_size, "->", Path(out).stat().st_size)


if __name__ == "__main__":
    patch(*sys.argv[1:3])
