"""Regression for prints that repeat (2026-11-11): Carlos's plaid came out in waves, with flat sleeves and a torn armhole, and Dona Rosa's
flowers as streaks (2026-11-05). Run with `.local/venv/bin/python tests/test_parle_print.py`: it needs numpy alone."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
from parle_print import as_in_game, cloth_frame, laid, mirror_line, period, repeated, shared, sheet, tile_of  # noqa: E402

# A piece that lays its picture again and again runs past one picture (Dona Rosa's dress); a picture drawn for the piece does not.
assert repeated(np.array([[-2.5, 0.5], [2.6, 6.4]])) and not repeated(np.array([[0.0, 0.0], [1.0, 0.42]]))

# The two sleeves of the plaid shirt lie on the same part of the picture: it is a sheet. Front and back of a shirt lie side by side.
strip = lambda left, right, low, high: np.array([[[left, low], [right, low], [right, high]], [[left, low], [right, high], [left, high]]])
assert shared([strip(0, 1, 0, 0.42), strip(0, 1, 0, 0.40)]), "two sleeves on the same plaid did not count as a sheet"
assert not shared([strip(0.02, 0.49, 0.28, 0.92), strip(0.53, 0.99, 0.28, 0.94)]), "front and back side by side passed for a sheet"
assert not shared([strip(0, 1, 0, 0.42)])

# A plaid of 64 pixels in a picture of 500: it meets itself at 448 (seven checks), the longest shift that keeps a whole number of them.
grid = np.indices((500, 500))
plaid = 0.8 - ((grid[0] % 64 < 12) | (grid[1] % 64 < 12)).astype(float) * 0.6
assert period(plaid, 0) == 448 and period(plaid, 1) == 448, (period(plaid, 0), period(plaid, 1))
assert sheet(np.dstack([plaid] * 3)) == (448, 448)
# A picture that already closes on itself is kept whole.
whole = ((grid[0] % 50 < 10) | (grid[1] % 50 < 10)).astype(float)
assert sheet(np.dstack([whole] * 3)) == (500, 500)
# Stripes: the same all along one way, and they close on themselves.
assert sheet(np.dstack([(grid[1] % 50 < 10).astype(float)] * 3)) == (500, 500)
# One colour is nothing to lay, and a picture drawn for one garment (a shirt on its ground) is not a sheet.
assert sheet(np.full((64, 64, 3), 0.4)) is None
drawn = np.full((500, 500), 0.9)
drawn[60:300, 40:220] = 0.3
drawn[330:470, 260:480] = np.linspace(0.1, 0.8, 220)
assert sheet(np.dstack([drawn] * 3)) is None, "a picture drawn for one garment passed for a sheet of print"

# Which piece of the game wears a print that repeats. Dona Rosa's dress lays its flowers again and again: all of the picture. The two
# sleeves of Carlos's shirt share a sheet of plaid: the part of it that closes on itself. The body of his shirt has a picture drawn for
# it, each island on its own part: none, and it takes the print of the sleeves. A plain picture is nothing to lay, however it is used.
picture = np.dstack([plaid] * 3)
sleeves = [strip(0, 1, 0, 0.42), strip(0, 1, 0, 0.40)]
body = [strip(0.02, 0.49, 0.28, 0.92), strip(0.53, 0.99, 0.28, 0.94)]
assert tile_of(picture, np.array([[-2.5, 0.5], [2.6, 6.4]]), body) == (500, 500)
assert tile_of(picture, np.concatenate(sleeves).reshape(-1, 2), sleeves) == (448, 448)
assert tile_of(picture, np.concatenate(body).reshape(-1, 2), body) is None, "a picture drawn for the shirt was laid as a print"
assert tile_of(np.full((64, 64, 3), 0.4), np.array([[0.0, 0.4], [9.4, 12.1]]), body) is None, "a plain picture was laid as a print"

# The plaid is its own mirror about the middle of its dark lines: 12 pixels of line from 0, so about the 6th of 448, across and along.
# The buttons of the shirt sit on one. Flowers, the mirror of nothing, go as they are.
across_line, along_line = mirror_line(np.dstack([plaid[:448, :448]] * 3))
assert abs(across_line * 448 - 6) % 64 < 0.6 and abs(along_line * 448 - 6) % 64 < 0.6, (across_line * 448, along_line * 448)
flowers = np.random.default_rng(7).random((96, 96))
assert mirror_line(np.dstack([flowers] * 3)) == (0.0, 0.0)

# A panel of 0.4 by 0.6 m, upright in front of the body, with its lay turned by 30 degrees, shrunk and moved: the frame sets it upright
# again, in metres.
across, up = np.meshgrid(np.linspace(-0.2, 0.2, 5), np.linspace(1.0, 1.6, 7))
points = np.stack([across.ravel(), np.full(across.size, -0.1), up.ravel()], axis=1)
faces = []
for row in range(6):
    for column in range(4):
        first = row * 5 + column
        # Seen from the front (the viewer at -y), the corners run anticlockwise: the face looks out.
        faces += [(first, first + 1, first + 6), (first, first + 6, first + 5)]
faces = faces_panel = np.array(faces)
angle = np.radians(30)
spin = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
flat = (points[:, [0, 2]] @ spin.T) * 0.37 + np.array([0.3, 0.2])
corners, lay = points[faces], flat[faces]
turn, metres = cloth_frame(corners, lay, (0, 0, 1))
assert abs(metres - 1 / 0.37) < 1e-6, metres
cloth = laid(corners, lay, (0, 0, 1), middle=0.0, level=1.0)
assert np.allclose(cloth[:, :, 0], corners[:, :, 0], atol=1e-6), "across the cloth is not across the body, in metres"
assert np.allclose(cloth[:, :, 1], corners[:, :, 2] - 1.0, atol=1e-6), "along the grain is not the height over the level"
# The same panel with its lay mirrored: the print still reads from the left to the right of whoever looks at it.
mirrored = lay * np.array([-1.0, 1.0])
assert np.allclose(laid(corners, mirrored, (0, 0, 1), middle=0.0, level=1.0), cloth, atol=1e-6), "a mirrored lay mirrored the print"
# A piece away from the middle of the body (a sleeve) is centred on itself.
aside = corners + np.array([0.5, 0.0, 0.0])
assert abs(float(laid(aside, lay, (0, 0, 1), middle=0.0)[:, :, 0].mean())) < 1e-6

# A sleeve: a tube of 5 cm of radius along an arm that hangs at 45 degrees, unrolled across the whole picture. The grain is the arm.
axis = np.array([1.0, 0.0, -1.0]) / np.sqrt(2)
side, front = np.array([0.0, 1.0, 0.0]), np.cross(np.array([1.0, 0.0, -1.0]) / np.sqrt(2), np.array([0.0, 1.0, 0.0]))
around, along = np.meshgrid(np.linspace(0, 2 * np.pi, 13), np.linspace(0, 0.2, 5))
tube = np.array([0.2, 0.0, 1.4]) + along.ravel()[:, None] * axis + 0.05 * (np.cos(around.ravel())[:, None] * side + np.sin(around.ravel())[:, None] * front)
faces = []
for row in range(4):
    for column in range(12):
        first = row * 13 + column
        faces += [(first, first + 1, first + 14), (first, first + 14, first + 13)]
faces = np.array(faces)
# Unrolled as cloth, at any turn and size (an unwrap): 31 cm around and 20 cm along come back in metres, the length along the grain.
unwrap = (np.stack([around.ravel() * 0.05, along.ravel()], axis=1) @ spin.T) * 2.1
sleeve = laid(tube[faces], unwrap[faces], -axis)
length = sleeve[:, :, 1].max() - sleeve[:, :, 1].min()
girth = sleeve[:, :, 0].max() - sleeve[:, :, 0].min()
assert abs(length - 0.2) < 0.005 and abs(girth - 2 * np.pi * 0.05) < 0.01, (length, girth)

# On the game's sleeve the picture runs once around the arm (31 cm) and 0.4 of it along 20 cm: one picture is 0.39 m of cloth, the
# geometric mean, and it stands upright along the arm, turned by no quarter. A step of that much cloth is a step of one picture.
unrolled = np.stack([around.ravel() / (2 * np.pi), along.ravel() / 0.2 * 0.4], axis=1)
game = as_in_game(tube[faces], unrolled[faces], -axis)
size = np.sqrt(2 * np.pi * 0.05 / 1.0 * 0.2 / 0.4)
assert np.allclose(np.abs(game), np.eye(2) / size, atol=0.02), game
# The same picture laid on its side on the game's cloth comes out on its side on ours: a quarter turn, never a few degrees.
sideways = unrolled[:, ::-1] * np.array([1.0, 1.0])
turned = as_in_game(tube[faces], sideways[faces], -axis) * size
assert np.allclose(np.abs(turned), np.array([[0.0, 1.0], [1.0, 0.0]]), atol=0.05), turned
# The sign, on the panel in front of the body: a picture that stands upright on the game's cloth is laid upright on ours, one that hangs
# upside down hangs upside down (flowers), and a mirrored one stays mirrored.
upright = points[:, [0, 2]] * 2.0
assert np.allclose(as_in_game(corners, upright[faces_panel], (0, 0, 1)), np.eye(2) * 2.0, atol=1e-6)
assert np.allclose(as_in_game(corners, -upright[faces_panel], (0, 0, 1)), -np.eye(2) * 2.0, atol=1e-6)
assert np.allclose(as_in_game(corners, (upright * np.array([-1.0, 1.0]))[faces_panel], (0, 0, 1)), np.diag([-2.0, 2.0]), atol=1e-6)
# Tilted by 10 degrees on the game's cloth (a curved piece read on average): still upright, never a few degrees off.
tilt = np.radians(10)
tilted = upright @ np.array([[np.cos(tilt), -np.sin(tilt)], [np.sin(tilt), np.cos(tilt)]]).T
assert np.allclose(as_in_game(corners, tilted[faces_panel], (0, 0, 1)), np.eye(2) * 2.0, atol=1e-6)
# The script that paints asks the rules here: it reads the bodies and writes the picture, and decides nothing.
painter = (Path(__file__).parent.parent / "tools" / "print_cloth.py").read_text()
for rule in ("parle_print.tile_of(", "parle_print.as_in_game(", "parle_print.laid(", "parle_print.mirror_line(", "stamp_sources("):
    assert rule in painter, f"print_cloth.py no longer asks: {rule}"
print("ok")
