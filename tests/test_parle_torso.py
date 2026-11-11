"""Regression for the torso of the muscled man (2026-11-10): run with `.local/venv/bin/python tests/test_parle_torso.py`.

The back and the chest of the muscled man were wrong next to the reference body (owner, 2026-11-10). The step took the sculpture out of a band of
the trunk, front and back, and put the MakeHuman pectoral in its place: a crease across the back, flat back muscles, lumps beside the armpits and a
chest that dropped 34 mm under the pectoral (9 mm on the body as it arrived). Now the shoulders grow with the artist-made deltoid targets and,
on a bare trunk, the muscles of the sculpture come as depth from a front and a back chart (core/body_shape.py). With --plain (a top covers the
trunk) only the shoulders move, and this test runs that way; the chart is checked on its own at the end.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from core import body_shape  # noqa: E402

COUNT = 19158
TARGETS = ROOT / "data" / "target" / "hm08"


def target(name):
    return body_shape.read_target(str(TARGETS / f"{name}.target"), COUNT)


def touched(shift):
    return np.linalg.norm(shift, axis=1) > 1e-9


added = sum(body_shape.read_target(str(ROOT / path), COUNT) for path in body_shape.MUSCLES)
for part in ("armslegs/l-upperarm-shoulder-muscle-incr", "armslegs/r-upperarm-shoulder-muscle-incr"):
    own = touched(target(part))
    assert own.sum() > 100 and touched(added[own]).all(), f"{part} does not reach the body"
left, right = target("armslegs/l-upperarm-shoulder-muscle-incr"), target("armslegs/r-upperarm-shoulder-muscle-incr")
assert abs(np.abs(left).max() - np.abs(right).max()) < 1e-4, "one shoulder grows more than the other"

rng = np.random.default_rng(11)
base = rng.uniform((-0.5, -0.2, 0.0), (0.5, 0.2, 1.75), (COUNT, 3)).astype(np.float32)
skin = np.arange(0, COUNT, 3)
dump = dict(points=base[skin], faces=np.zeros((1, 3), np.int64), weights=np.ones((len(skin), 1), np.float32), names=np.array(["Spine"]))

with tempfile.TemporaryDirectory() as folder:
    body, mesh, shift = (str(Path(folder) / name) for name in ("w.npz", "p.npy", "shift.npy"))
    np.savez_compressed(body, **dump)
    np.save(mesh, base)
    for _ in range(2):  # twice: the step keeps the originals, so the second run lands on the first
        done = subprocess.run([sys.executable, "tools/chest_shape.py", body, mesh, shift, "--plain"], cwd=ROOT, capture_output=True, text=True)
        assert done.returncode == 0, done.stderr[-600:]
        moved, step, after = np.load(mesh) - base, np.load(shift), np.load(body)["points"]
        assert np.abs(moved - added).max() < 1e-5, "the step moved the base by something other than the shoulder targets"
        assert np.abs(step - added[skin]).max() < 1e-5, "the rigged body does not follow the base"
        assert np.abs(after - base[skin] - step).max() < 1e-5

    # Nothing is taken away: where no shoulder target acts, the body stays as it arrived, the chest and the back included.
    still = ~touched(added)
    assert still.sum() > COUNT * 0.5 and np.abs(moved[still]).max() == 0, "the step moved a part of the body no target owns"
    for part in ("torso/torso-muscle-pectoral-incr", "torso/torso-muscle-dorsi-incr"):
        assert (touched(target(part)) & still).sum() > 300, f"the step moves the trunk again: {part}"

# A man who is not muscled keeps his shoulders: under a shirt the step leaves him as he came (2026-11-11, the chest of every bare man).
with tempfile.TemporaryDirectory() as folder:
    body, mesh, shift = (str(Path(folder) / name) for name in ("w.npz", "p.npy", "shift.npy"))
    np.savez_compressed(body, **dump)
    np.save(mesh, base)
    done = subprocess.run([sys.executable, "tools/chest_shape.py", body, mesh, shift, "kit=male-slim", "--plain"], cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-600:]
    assert np.abs(np.load(mesh) - base).max() == 0 and np.abs(np.load(shift)).max() == 0, "the slim man's shoulders grew"

source = (ROOT / "tools" / "chest_shape.py").read_text()
assert "delta" not in source and "pectoral" not in source, "the step takes the sculpture out of the trunk again"
print("torso: the step adds", len(body_shape.MUSCLES), "targets and", int(touched(added).sum()), "vertices move")

# The chart of the sculpture: front and back, how much of each is known, and relief the size of a muscle, not of the proportions.
chart = np.load(ROOT / "data" / "sculpt" / "male-torso-relief.npz")
relief, known = chart["relief"], chart["known"]
assert relief.shape == known.shape == (2, body_shape.HEIGHTS, body_shape.COLUMNS), f"old chart: {relief.shape}, run tools/sculpt_relief.py"
middle = (known > 0.5)[:, body_shape.HEIGHTS // 4: body_shape.HEIGHTS * 3 // 4, body_shape.COLUMNS // 4: body_shape.COLUMNS * 3 // 4]
assert middle.mean() > 0.9, "the chart does not know the middle of the trunk"
inner = relief[:, body_shape.HEIGHTS // 4: body_shape.HEIGHTS * 3 // 4, body_shape.COLUMNS // 4: body_shape.COLUMNS * 3 // 4][middle]
assert 0.005 < np.abs(inner).max() < 0.08, f"relief in the middle of the trunk out of a muscle's size: {np.abs(inner).max():.3f}"
print("torso: chart", relief.shape, "relief in the middle up to", round(float(np.abs(inner).max()), 3), "of crotch to neck pit")
