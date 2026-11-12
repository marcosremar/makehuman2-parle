"""Regression for the pale smears at the corners of the sealed mouth (2026-11-12): run with `.local/venv/bin/python tests/test_parle_close_normals.py`.

Blender exported `mouthClose`'s normals from the body with the key alone, a crumpled chin; under `jawOpen` they lit the corners in
pale smears. Its normal delta must be minus the jaw's, by the share of the jaw's move it takes back, and the GLB keeps the rest.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import parle_close_normals as normals  # noqa: E402


def test_the_normal_delta_is_the_jaws_backwards():
    jaw = np.array([[0.0, 0.0, -0.02], [0.0, 0.0, -0.01], [0.0, 0.0, 0.0]])
    close = np.array([[0.0, 0.0, 0.02], [0.0, 0.0, 0.005], [0.0, 0.0, 0.0]])
    jaw_normals = np.array([[0.0, 0.4, 0.1], [0.2, 0.0, 0.0], [0.0, 0.3, 0.0]])
    out = normals.close_normals(jaw, close, jaw_normals)
    assert np.allclose(out[0], [0.0, -0.4, -0.1]), "where the lips seal whole, the jaw's normals undone whole"
    assert np.allclose(out[1], [-0.1, 0.0, 0.0]), "half the jaw's move taken back, half its normal"
    assert np.allclose(out[2], 0.0), "where the jaw does not move, nothing"


def test_a_glb_round_trip_rewrites_only_the_close_normals():
    count = 4
    jaw_p = np.array([[0, 0, -0.02], [0, 0, -0.01], [0, 0, 0], [0, 0, 0]], np.float32)
    close_p = -jaw_p
    jaw_n = np.array([[0, 0.5, 0], [0.1, 0, 0], [0, 0, 0], [0, 0, 0]], np.float32)
    crumpled = np.full((count, 3), 0.9, np.float32)
    binary = bytearray()
    doc = {"asset": {"version": "2.0"}, "buffers": [{"byteLength": 0}], "bufferViews": [], "accessors": [], "meshes": []}
    for array in (jaw_p, jaw_n, close_p, crumpled):
        view = normals.append_view(doc, binary, array.tobytes())
        doc["accessors"].append({"bufferView": view, "componentType": 5126, "count": count, "type": "VEC3"})
    doc["meshes"].append({"extras": {"targetNames": ["jawOpen", "mouthClose"]}, "primitives": [
        {"attributes": {}, "targets": [{"POSITION": 0, "NORMAL": 1}, {"POSITION": 2, "NORMAL": 3}]}]})
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "a.glb"
        normals.write_glb(path, doc, binary)
        normals.main(path)
        doc2, binary2 = normals.read_glb(path)
    assert np.allclose(normals.read_vec3(doc2, binary2, 3), -jaw_n), "mouthClose normals are the jaw's backwards"
    assert np.allclose(normals.read_vec3(doc2, binary2, 1), jaw_n), "jawOpen's normals untouched"
    assert np.allclose(normals.read_vec3(doc2, binary2, 2), close_p), "the positions untouched"


if __name__ == "__main__":
    test_the_normal_delta_is_the_jaws_backwards()
    test_a_glb_round_trip_rewrites_only_the_close_normals()
    print("ok")
