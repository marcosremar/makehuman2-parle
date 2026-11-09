"""Regression for re-embedding the eye map (2026-11-09): run with `.local/venv/bin/python tests/test_parle_eye_apply.py`."""
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import apply_eye_map as tool  # noqa: E402

work = tempfile.mkdtemp()
old = cv2.imencode(".jpg", np.zeros((8, 8, 3), np.uint8))[1].tobytes()
old_padded = old + b"\0" * (-len(old) % 4)
after = bytes(range(16))
doc = {
    "buffers": [{"byteLength": 8 + len(old_padded) + 16}],
    "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 8}, {"buffer": 0, "byteOffset": 8, "byteLength": len(old)},
                    {"buffer": 0, "byteOffset": 8 + len(old_padded), "byteLength": 16}],
    "images": [{"name": "parle_eye_mel", "bufferView": 1, "mimeType": "image/jpeg"}],
    "textures": [{"source": 0}],
    "materials": [{"name": "Human.low-poly", "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
}
source = os.path.join(work, "body.glb")
tool.write_glb(source, doc, b"\1" * 8 + old_padded + after)
painted = np.full((64, 64, 3), (10, 200, 90), np.uint8)
cv2.imwrite(os.path.join(work, "parle_eye_mel.png"), painted)

target = os.path.join(work, "out.glb")
assert tool.apply_map(work, source, target) == "parle_eye_mel"
new_doc, binary = tool.read_glb(target)
view = new_doc["bufferViews"][1]
image = cv2.imdecode(np.frombuffer(binary[view["byteOffset"]:view["byteOffset"] + view["byteLength"]], np.uint8), cv2.IMREAD_COLOR)
assert image.shape[:2] == (64, 64), image.shape
assert abs(image.astype(int) - painted.astype(int)).max() < 12, "the eye is not the painted map"
last = new_doc["bufferViews"][2]
assert binary[last["byteOffset"]:last["byteOffset"] + 16] == after, "the data after the eye moved wrong"
assert binary[:8] == b"\1" * 8, "the data before the eye changed"
assert new_doc["buffers"][0]["byteLength"] == len(binary), "buffer length does not match"

print("ok")
