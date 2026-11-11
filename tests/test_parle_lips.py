"""Regression for the lip layer (2026-11-07): run with `.local/venv/bin/python tests/test_parle_lips.py`."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
from parle_lips import painted  # noqa: E402

skin = np.full((64, 64, 3), (120, 150, 200), np.uint8)
# The game's wine lipstick: very dark, very red.
skin[28:36, 20:44] = (40, 30, 110)
share = np.zeros((64, 64), np.float32)
share[28:36, 20:44] = 1

natural = painted(skin, share)
lab = cv2.cvtColor(natural, cv2.COLOR_BGR2LAB).astype(int)
skin_lab = cv2.cvtColor(skin, cv2.COLOR_BGR2LAB).astype(int)
# Darker than the skin, but nowhere near the black-wine of the game.
assert skin_lab[0, 0, 0] * 0.7 < lab[32, 32, 0] < skin_lab[0, 0, 0], (lab[32, 32], skin_lab[0, 0])
# A woman's lips stand out more from her skin than a man's (2026-11-11: facial contrast), and still nowhere near the wine of the game.
man_lab = cv2.cvtColor(painted(skin, share, man=True), cv2.COLOR_BGR2LAB).astype(int)
assert lab[32, 32, 0] < man_lab[32, 32, 0] < skin_lab[0, 0, 0], (lab[32, 32], man_lab[32, 32])
# Rosier than the skin.
assert lab[32, 32, 1] > skin_lab[0, 0, 1]
# Outside the map nothing changes.
assert (natural[0, 0] == skin[0, 0]).all()
# The step starts from the base each time: run twice, same lips.
assert (painted(skin, share) == natural).all()
# A colour from the look replaces the hue.
coral = painted(skin, share, "#e06060")
assert cv2.cvtColor(coral, cv2.COLOR_BGR2LAB)[32, 32, 1] > lab[32, 32, 1]

# A linha onde os lábios se encontram (11/11/2026): com a boca aberta esses texels esticam sobre a face de dentro do lábio, e a 0,68 de
# escuro neutro viravam cunhas pretas nos cantos e um contorno de batom borrado. Ela escurece pouco e puxa para o vermelho da mucosa.
line = np.zeros((64, 64), np.float32)
line[32, 20:44] = 1
lined = painted(skin, share, line=line)
plain = painted(skin, share)
lined_lab = cv2.cvtColor(lined, cv2.COLOR_BGR2LAB).astype(int)
plain_lab = cv2.cvtColor(plain, cv2.COLOR_BGR2LAB).astype(int)
assert lined_lab[32, 32, 0] < plain_lab[32, 32, 0], "the line is still a line"
assert lined_lab[32, 32, 0] > 0.6 * plain_lab[32, 32, 0], f"the line must not go near black: {lined_lab[32, 32]} against {plain_lab[32, 32]}"
assert lined_lab[32, 32, 1] >= plain_lab[32, 32, 1], f"the line leans red, like wet mucosa: {lined_lab[32, 32]} against {plain_lab[32, 32]}"
print("ok")
