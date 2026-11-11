"""Regression for the teeth that read as a grid or braces (scoreboard of 2026-11-10): run with `.local/venv/bin/python tests/test_parle_teeth.py`."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import parle_mouth as mouth  # noqa: E402
import parle_teeth as teeth  # noqa: E402


def box(centre, half):
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * half + centre
    faces = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    return corners, np.array(faces)


def mouth_of():
    """Two teeth per arch and a gum per arch, the gums big enough to count as gums."""
    points, triangles, offset = [], [], 0
    for z in (0.01, -0.01):
        for x in (-0.012, 0.012):
            corners, faces = box(np.array((x, -0.05, z)), np.array((0.004, 0.003, 0.005)))
            points.append(corners)
            triangles.append(faces + offset)
            offset += 8
    for z in (0.02, -0.02):
        for step in range(teeth.GUM_LEAST // 8 + 1):
            corners, faces = box(np.array((-0.03 + 0.06 * step / (teeth.GUM_LEAST // 8), -0.045, z)), np.array((0.002, 0.004, 0.004)))
            points.append(corners)
            triangles.append(faces + offset)
            offset += 8
    return np.concatenate(points), np.concatenate(triangles)


def bridged(points, triangles):
    """The gum boxes joined into one part by a thin strip of triangles between neighbours."""
    extra = []
    total = len(points)
    for start in range(32, total - 8, 8):
        if abs(points[start, 2] - points[start + 8, 2]) < 1e-6:
            extra.append((start, start + 1, start + 8))
    return np.concatenate([triangles, np.array(extra)])


def raster(uv, owner):
    """Which texels get ink from more than one part."""
    seen = np.full((teeth.SIZE, teeth.SIZE), -1, np.int32)
    shared = np.zeros_like(seen, bool)
    for corners, part in zip(uv, owner):
        mask = np.zeros_like(seen, np.uint8)
        cv2.fillConvexPoly(mask, np.stack([corners[:, 0] * teeth.SIZE, (1 - corners[:, 1]) * teeth.SIZE], 1).round().astype(np.int32), 1)
        hit = mask > 0
        shared |= hit & (seen >= 0) & (seen != part)
        seen[hit] = part
    return shared, seen


points, triangles = mouth_of()
triangles = bridged(points, triangles)
owner = teeth.parts(points, triangles)[triangles[:, 0]]
uv = teeth.atlas(points, triangles)
assert 0 <= uv.min() and uv.max() <= 1, "the atlas must stay inside the map"
shared, seen = raster(uv, owner)
assert not shared.any(), "no texel may be shared by two parts: the darkest shade would land on the wrong tooth"
assert len(np.unique(owner)) == 6 and len(np.unique(seen[seen >= 0])) == 6, "four teeth and two gums, each with its own ink"

# The two faces of a tooth lie apart, so the shade of its inner side does not land on the outer one.
out = teeth.facing_out(points, triangles, teeth.arch_centre(points))
first = owner == owner[0]
assert (uv[first & out, :, 1].min() > uv[first & ~out, :, 1].max()), "outer and inner faces share cells"

image = teeth.tooth_map()
assert image.shape == (teeth.SIZE, teeth.SIZE, 3)
left, top = teeth.tooth_cell(0, 3)
tooth = image[top + 16, left + teeth.TOOTH_W // 2]
assert tooth[2] > 0.7 and abs(tooth[2] - tooth[0]) < 0.3, f"a tooth is ivory: {tooth}"
gum = image[teeth.gum_strip(1) + 10, 100]
assert mouth.gum_of(gum[None, None, ::-1].astype(np.float32).repeat(32, 0).repeat(32, 1)).min() > 0.5, f"the gum strip must read as gum to the shade: {gum}"
edge = image[top + 16, left]
centre = image[top + 16, left + teeth.TOOTH_W // 2]
assert edge.mean() < centre.mean(), "a thin darker line separates a tooth from the next"
print("ok")

# Dentes de aparelho no placar de 11/11/2026 (Ivone, Lúcia): cada dente ocupava ~10 % da célula, porque a escala vinha de todos os pontos da
# boca e não do dente, e o filtro do player misturava o dente com o fundo; e o arco saía da altura mediana, que numa arcada com curva (molar mais
# alto que o incisivo) punha incisivos de cima e de baixo na mesma fileira. Os dentes precisam encher a célula e cair no arco da gengiva deles.
def curved_mouth():
    """Four teeth per arch on a curve: the molars sit 11 mm above the incisors, so a lower molar is higher than an upper incisor."""
    points, triangles, offset, arch_of = [], [], 0, []
    for arch, z0 in enumerate((0.005, -0.005)):
        for x in (-0.02, -0.006, 0.006, 0.02):
            lift = 0.011 if abs(x) > 0.015 else 0.0
            corners, faces = box(np.array((x, -0.05 + abs(x) * 0.8, z0 + lift)), np.array((0.003, 0.003, 0.003)))
            points.append(corners); triangles.append(faces + offset); offset += 8; arch_of.append(arch)
    for arch, z0 in enumerate((0.0115, -0.0115)):
        for step in range(teeth.GUM_LEAST // 8 + 1):
            x = -0.022 + 0.044 * step / (teeth.GUM_LEAST // 8)
            corners, faces = box(np.array((x, -0.048 + abs(x) * 0.8, z0 + (0.011 if abs(x) > 0.015 else 0.0))), np.array((0.002, 0.004, 0.002)))
            points.append(corners); triangles.append(faces + offset); offset += 8
    triangles = np.concatenate(triangles)
    per_gum = teeth.GUM_LEAST // 8 + 1
    links = [(start, start + 1, start + 8) for gum in range(2) for start in range(64 + gum * per_gum * 8, 64 + (gum + 1) * per_gum * 8 - 8, 8)]
    return np.concatenate(points), np.concatenate([triangles, np.array(links)]), arch_of


cpoints, ctriangles, carch = curved_mouth()
cowner = teeth.parts(cpoints, ctriangles)[ctriangles[:, 0]]
cuv = teeth.atlas(cpoints, ctriangles)
for index, arch in enumerate(carch):
    mine = cowner == teeth.parts(cpoints, ctriangles)[index * 8]
    rows = (1 - cuv[mine, :, 1]) * teeth.SIZE
    assert rows.max() <= (arch + 1) * teeth.TOOTH_H + 1 and rows.min() >= arch * teeth.TOOTH_H - 1, f"tooth {index} left its arch row {arch}: {rows.min():.0f}..{rows.max():.0f}"
    cols = cuv[mine, :, 0] * teeth.SIZE
    assert cols.max() - cols.min() > 0.6 * teeth.TOOTH_W, f"tooth {index} must fill its cell across: {cols.max() - cols.min():.1f} px of {teeth.TOOTH_W}"
    assert rows.max() - rows.min() > 0.6 * teeth.TOOTH_H, f"tooth {index} must fill its cell up and down: {rows.max() - rows.min():.1f} px of {teeth.TOOTH_H}"
print("ok celula cheia")

# No elenco, a gengiva mais perto ainda punha os incisivos de baixo na fileira de cima (os dois arcos se encostam na frente): quem diz o arco é a
# mandíbula. Os pontos que o jawOpen leva são o arco de baixo.
jaw = np.zeros(len(cpoints))
for index, arch in enumerate(carch):
    jaw[index * 8:index * 8 + 8] = 0.02 * arch
lower_gum = cpoints[:, 2] < 0
jaw[64:][lower_gum[64:]] = 0.02
juv = teeth.atlas(cpoints, ctriangles, jaw)
for index, arch in enumerate(carch):
    mine = cowner == teeth.parts(cpoints, ctriangles)[index * 8]
    rows = (1 - juv[mine, :, 1]) * teeth.SIZE
    assert arch * teeth.TOOTH_H - 1 <= rows.min() and rows.max() <= (arch + 1) * teeth.TOOTH_H + 1, f"with the jaw, tooth {index} left arch {arch}"
print("ok mandibula")
