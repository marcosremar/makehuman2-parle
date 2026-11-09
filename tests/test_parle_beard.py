"""Regressions for the drawn beard (2026-11-09): run with `.local/venv/bin/python tests/test_parle_beard.py`.

The bake's beard stopped in a round end in the middle of the cheek. The outline runs from the sideburn to the moustache, with no bare line down the middle.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
from parle_beard import FRONT_OUTLINE, SIDE_OUTLINE, signed_mm, strands  # noqa: E402


def side(depth, height):
    return float(signed_mm(SIDE_OUTLINE, np.array([depth]), np.array([height]))[0])


def front(across, height):
    return float(signed_mm(FRONT_OUTLINE, np.array([across]), np.array([height]))[0])


assert side(10.7, 2.0) > 3, "the sideburn runs up in front of the ear"
assert side(8.0, 3.0) < 0, "the cheek above the beard stays bare"
assert side(13.0, 0.0) < 0, "the ear stays bare"
assert front(0.0, -6.0) > 5, "the middle of the chin is inside the beard, far from any border"
assert front(0.5, -2.1) > 2, "the moustache sits under the nose"
hairs = strands(256)
along_rows = np.abs(np.diff(hairs, axis=1)).mean()
across_rows = np.abs(np.diff(hairs, axis=0)).mean()
assert along_rows < across_rows, "the texture's rows run down the face"
print("ok")
