"""Regression for the stains seen on the cast (2026-11-07): run with `python tests/test_parle_stamp.py`, it needs nothing but Python."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
from parle_stamp import GAME_BOTTOM as BOTTOM, GAME_TOP as TOP, stamp_sources  # noqa: E402

# How the game names the pieces of each garment is said here, once, for the bake and for the print that repeats.
assert "lctshirt" in TOP and "Shirt_" in TOP and "Dress" in TOP and "Dress" in BOTTOM and "bjjeans" in BOTTOM

# Lia: the necklace baked into the shirt read as a torn collar.
assert stamp_sources(["Rolled_sleeves_shirt", "Placket", "Necklace", "Base_Body"], TOP) == ["Rolled_sleeves_shirt", "Placket"]
# Seu Nelson: same with the chain.
assert stamp_sources(["lctshirt_4587_Shape", "Chain"], TOP) == ["lctshirt_4587_Shape"]
# Lúcia and m15: the bikini under the tank top came out as a red stain.
assert stamp_sources(["Tank", "Bikini"], TOP) == ["Tank"]
assert stamp_sources(["bjjeans_8905_Shape", "Fray", "Bikini"], BOTTOM) == ["bjjeans_8905_Shape", "Fray"]
# Surfista and h24: the cord of the shorts came out as a grey patch.
assert stamp_sources(["Shorts", "Shorts_Cord"], BOTTOM) == ["Shorts"]
# A bikini worn alone is the garment, and its print stays.
assert stamp_sources(["Bikini", "Canga"], TOP) == ["Bikini"]
print("ok")
