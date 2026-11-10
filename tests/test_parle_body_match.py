"""Regression for the loop of the build (2026-11-10): run with `.local/venv/bin/python tests/test_parle_body_match.py`.

The body the loop measures has to be the body the cast builds. Twice it was not. Measured with a shoulder polish that fitted the hanging arm, the
loop pushed the arms out to undo it (male-slim: from 1.6 cm in to 1.5 cm out on each side) and the dressed arms stood away from the trunk. Measured
without it while the cast still ran it, the dressed shoulders came out 6 to 8 cm narrower on each side than the silhouette shown as matched (owner,
2026-11-10). Now one call, `parle_polish.natural_body`, is the whole step, in the cast and in the dump of the loop, and the polish is gone.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import body_match  # noqa: E402

options = body_match.dump_options(dict(height="1.74", sex="male"))
assert options == ["height=1.74", "sex=male"], f"the loop runs a step of its own, not the cast's: {options}"
assert "chest=shift.npy" in body_match.dump_options(dict(height="1.74", sex="male", chest="shift.npy"))

# Blender is not here: the sources are read as text.
polish = (ROOT / "tools" / "parle_polish.py").read_text()
whole = re.search(r"def natural_body\(meshes, rig, height, sex, kit\):.*?(?=\n\n\ndef )", polish, re.S)
assert whole, "the cast and the loop have no step in common"
steps = re.findall(r"\b(natural_\w+)\(", whole.group(0).split('"""')[-1])
assert steps == ["natural_arms", "natural_build"], f"the body is the arms, then the build: {steps}"
dump = (ROOT / "tools" / "body_dump.py").read_text()
assert re.findall(r"parle_polish\.(natural_\w+)\(", dump) == ["natural_body"], "the dump measures a body the cast does not build"
assert "body=" in (ROOT / "tools" / "body_match.py").read_text(), "the loop does not ask the dump for the cast's step"
for path in sorted((ROOT / "tools").glob("*.py")):
    assert "def natural_shoulders" not in path.read_text() and "natural_shoulders(" not in path.read_text(), f"{path.name} still polishes the shoulders"

profiles = sorted(path for path in (ROOT / "data" / "parle_build").glob("*.json") if not path.name.startswith("reference"))
assert len(profiles) == 6, f"a profile per kit: {[path.name for path in profiles]}"
for path in profiles:
    arm_out = json.load(open(path))["arm_out"]
    assert abs(arm_out) < body_match.ARM_OUT_LIMIT, f"{path.name} holds the arms at the limit: {arm_out}"
print("ok")
