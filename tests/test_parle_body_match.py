"""Regression for the loop of the build (2026-11-10): run with `.local/venv/bin/python tests/test_parle_body_match.py`.

The loop once measured the body with the shoulders polished. The polish narrows the shoulder for the arm that hangs, the measure reads the body
with the arm out, and the loop answered by pushing the arms out (male-slim: from 1.6 cm in to 1.5 cm out on each side): dressed, the arms
stood away from the trunk. The loop measures without the polish, and no kit's profile holds the arms at the limit of the loop.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import body_match  # noqa: E402

options = body_match.dump_options(dict(height="1.74", sex="male"))
assert "arms=1" in options, f"the loop measures the arms the cast builds: {options}"
assert not [item for item in options if item.startswith("shoulders")], f"the loop measured with the shoulder polish: {options}"
assert "chest=shift.npy" in body_match.dump_options(dict(height="1.74", sex="male", chest="shift.npy"))
profiles = sorted(path for path in (ROOT / "data" / "parle_build").glob("*.json") if not path.name.startswith("reference"))
assert len(profiles) == 6, f"a profile per kit: {[path.name for path in profiles]}"
for path in profiles:
    arm_out = json.load(open(path))["arm_out"]
    assert abs(arm_out) < body_match.ARM_OUT_LIMIT, f"{path.name} holds the arms at the limit: {arm_out}"
print("ok")
