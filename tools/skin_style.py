#!/usr/bin/env python3
"""Pele do elenco no estilo do rosto do Rafa: corpo limpo num tom só e blush suave em bochecha, nariz, orelha e queixo.

O rosto vem do NPC do jogo (`cast-heads.py`), com barba e marcas de cada pessoa, e fica: é identidade. O corpo só recebe o tom, e a emenda virava colar no pescoço.
Aqui o corpo limpa, o pescoço faz a transição, e a barba cinza do rosto escuro ganha o matiz da pele. A malha é a mesma em todo o elenco, então as zonas saem de um mapa só.

    skin_style.py <pele.png> <uvpos.npz> <corpo.npz> <saída.png> [força-do-blush] [--shave]
"""
import sys

import cv2
import numpy as np

SIZE = 1024
WIDE_PX = 61
# Abaixo disso é mancha e emenda; acima é sobrancelha, barba, lábio e mamilo.
QUIET = 0.1
LOUD = 0.22
# Rafa: bochecha e nariz puxam para o vermelho quente, sem escurecer.
BEARD_DARK = 0.4
RELIEF_NEAR = 2
RELIEF_FAR = 30
RELIEF_M = 0.006
SHADOW = 0.3
LIGHT = 0.04
SHADOW_TINT = np.array([1.25, 1.1, 0.8])  # BGR
BEARD_LEAST = 0.25
HUE_SHARE = 0.8
CHIN_M = 0.1
NECK_FADE_M = 0.07
WARM = np.array([0.82, 0.86, 1.04])  # BGR
BROW_NEAR_M = 0.004
BROW_FAR_M = 0.085
BROW_LOW_M = 0.064
BROW_HIGH_M = 0.1
BROW_DARK = 0.82
# Canto da boca no kit: onde `mouthSmile` mais anda.
LIP_CORNER_M = 0.033
LIP_REACH_M = 0.03
LIP_HIGH_M = 0.012
LIP_LOW_M = 0.058
LIP_RED = 4.0
LIP_DARK = 0.86
LIP_BEARD_SHARE = 0.2
SHAVE_HALF_M = 0.05
SHAVE_LIP_HALF_M = 0.04
SHAVE_LIP_WARM = (10.0, 3.0)
SHAVE_DEPTH_M = 0.05
SHAVE_LIP_TOP_M = 0.041
SHAVE_LIP_BOTTOM_M = 0.053
SHAVE_CHIN_M = 0.08
PHILTRUM_HALF_M = 0.012
PHILTRUM_TOP_M = 0.006
PHILTRUM_BOTTOM_M = 0.032
PHILTRUM_DEPTH_M = 0.03
PHILTRUM_DARK = 0.62


def painter(data):
    """Pinta na UV um valor por vértice: o mesmo vértice é o mesmo lugar do rosto em todo o elenco."""
    corners = np.stack([data["uv"][:, :, 0] * SIZE, (1 - data["uv"][:, :, 1]) * SIZE], axis=2).round().astype(np.int32)
    triangles = data["tri"]

    def paint(value, soft):
        canvas = np.zeros((SIZE, SIZE), np.float32)
        for corner, triangle in zip(corners, triangles):
            cv2.fillConvexPoly(canvas, corner, float(value[triangle].mean()))
        return cv2.GaussianBlur(canvas, (0, 0), soft)

    return paint


def zones(data):
    points = data["pos"]
    top = points[:, 2].max()
    head = points[points[:, 2] > top - 0.26]
    middle = float(np.median(head[:, 0]))
    wide = float(np.percentile(np.abs(head[:, 0] - middle), 99))
    tip = head[np.abs(head[:, 0] - middle) < 0.01]
    tip = tip[tip[:, 1].argmin()]
    near = lambda spot, reach: np.exp(-(((points - np.array(spot)) ** 2).sum(axis=1)) / (2 * reach * reach))
    blush = np.zeros(len(points))
    for side in (-1, 1):
        blush = np.maximum(blush, near((middle + side * wide * 0.55, tip[1] + 0.035, tip[2] - 0.005), 0.032))
        blush = np.maximum(blush, 0.7 * near((middle + side * wide * 0.98, tip[1] + 0.11, tip[2] + 0.015), 0.03))
    blush = np.maximum(blush, 0.7 * near(tip, 0.022))
    blush = np.maximum(blush, 0.35 * near((tip[0], tip[1] + 0.012, tip[2] - 0.085), 0.022))
    nose = np.zeros(len(points))
    # O queixo fica uns 10 cm abaixo da ponta do nariz; a transição desce 7 cm pelo pescoço.
    chin = tip[2] - CHIN_M
    face = np.clip((points[:, 2] - (chin - NECK_FADE_M)) / NECK_FADE_M, 0, 1)
    beard = ((points[:, 2] > chin) & (points[:, 2] < tip[2] - 0.012)).astype(float)
    lips = np.exp(-((points[:, 0] - tip[0]) ** 2 / (2 * 0.024 ** 2) + (points[:, 2] - tip[2] + 0.046) ** 2 / (2 * 0.012 ** 2))) * (points[:, 1] < tip[1] + 0.05)
    paint = painter(data)
    return paint(blush, 3), paint(face, 6), paint(nose, 2), paint(beard, 3), np.clip(paint(lips, 1.5) * 2, 0, 1)


def clear_philtrum(skin, data):
    """O bake pega o lado de dentro do nariz no sulco acima do lábio: no bigode do Seu Geraldo o meio saía escuro e azulado.

    Só o que é bem mais escuro que o resto do sulco sai, e a cor vem dos lados.
    """
    points = data["pos"]
    top = points[:, 2].max()
    head = points[points[:, 2] > top - 0.26]
    middle = float(np.median(head[:, 0]))
    tip = head[np.abs(head[:, 0] - middle) < 0.01]
    tip = tip[tip[:, 1].argmin()]
    inside = (np.abs(points[:, 0] - tip[0]) < PHILTRUM_HALF_M) & (points[:, 2] < tip[2] - PHILTRUM_TOP_M) & (points[:, 2] > tip[2] - PHILTRUM_BOTTOM_M)
    inside = inside & (points[:, 1] < tip[1] + PHILTRUM_DEPTH_M)
    area = painter(data)(inside.astype(float), 1) > 0.5
    if not area.any():
        return skin, 0
    gray = skin.mean(axis=2)
    dark = area & (gray < PHILTRUM_DARK * np.percentile(gray[area], 80))
    if not dark.any():
        return skin, 0
    mask = cv2.dilate(dark.astype(np.uint8) * 255, np.ones((5, 5), np.uint8))
    return cv2.inpaint(np.clip(skin, 0, 255).astype(np.uint8), mask, 6, cv2.INPAINT_TELEA).astype(np.float32), int(dark.sum())


def clear_painted_brows(skin, data):
    """O rosto do jogo traz a sobrancelha fina pintada: com a peça por cima, saíam duas, e sozinha ela fica serrilhada. Fica só a peça, como no Rafa."""
    points = data["pos"]
    top = points[:, 2].max()
    head = points[points[:, 2] > top - 0.26]
    middle = float(np.median(head[:, 0]))
    tip = head[np.abs(head[:, 0] - middle) < 0.01]
    tip = tip[tip[:, 1].argmin()]
    side = np.abs(points[:, 0] - tip[0])
    zone = (side > BROW_NEAR_M) & (side < BROW_FAR_M) & (points[:, 2] > tip[2] + BROW_LOW_M) & (points[:, 2] < tip[2] + BROW_HIGH_M) & (points[:, 1] < tip[1] + 0.06)
    area = painter(data)(zone.astype(float), 1) > 0.5
    gray = skin.mean(axis=2)
    dark = area & (gray < BROW_DARK * cv2.GaussianBlur(gray, (WIDE_PX, WIDE_PX), 0))
    if not dark.any():
        return skin, 0
    mask = cv2.dilate(dark.astype(np.uint8) * 255, np.ones((5, 5), np.uint8))
    return cv2.inpaint(np.clip(skin, 0, 255).astype(np.uint8), mask, 6, cv2.INPAINT_TELEA).astype(np.float32), int(dark.sum())


def trim_lip_corners(skin, data):
    """A boca do jogo é mais larga que a nova: o batom assado passava dos cantos, em dois riscos na bochecha. Fora do canto, a cor volta à da pele."""
    points = data["pos"]
    top = points[:, 2].max()
    head = points[points[:, 2] > top - 0.26]
    middle = float(np.median(head[:, 0]))
    tip = head[np.abs(head[:, 0] - middle) < 0.01]
    tip = tip[tip[:, 1].argmin()]
    side = np.abs(points[:, 0] - tip[0])
    depth = tip[2] - points[:, 2]
    zone = (side > LIP_CORNER_M) & (side < LIP_CORNER_M + LIP_REACH_M) & (depth > LIP_HIGH_M) & (depth < LIP_LOW_M) & (points[:, 1] < tip[1] + 0.05)
    area = painter(data)(zone.astype(float), 1) > 0.5
    lab = cv2.cvtColor(np.clip(skin, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    red = lab[:, :, 1] - cv2.GaussianBlur(lab[:, :, 1], (WIDE_PX, WIDE_PX), 0)
    gray = skin.mean(axis=2)
    dark = gray / np.maximum(cv2.GaussianBlur(gray, (WIDE_PX, WIDE_PX), 0), 1)
    # Região toda escura é barba, que fica: ali só o vermelho conta como batom.
    bearded = (dark[area] < LIP_DARK).mean() > LIP_BEARD_SHARE
    paint = area & (red > LIP_RED) if bearded else area & ((red > LIP_RED) | (dark < LIP_DARK))
    if not paint.any():
        return skin, 0
    mask = cv2.dilate(paint.astype(np.uint8) * 255, np.ones((5, 5), np.uint8))
    return cv2.inpaint(np.clip(skin, 0, 255).astype(np.uint8), mask, 6, cv2.INPAINT_TELEA).astype(np.float32), int(paint.sum())


def shave_face(skin, data):
    """Tira o bigode e a barbicha assados: o que difere da pele em volta, no sulco e abaixo do lábio, com a cor vinda de fora deles."""
    points = data["pos"]
    top = points[:, 2].max()
    head = points[points[:, 2] > top - 0.26]
    middle = float(np.median(head[:, 0]))
    tip = head[np.abs(head[:, 0] - middle) < 0.01]
    tip = tip[tip[:, 1].argmin()]
    side = np.abs(points[:, 0] - tip[0]) < SHAVE_HALF_M
    front = points[:, 1] < tip[1] + SHAVE_DEPTH_M
    above = (points[:, 2] < tip[2] - 0.004) & (points[:, 2] > tip[2] - SHAVE_LIP_TOP_M)
    below = (points[:, 2] < tip[2] - SHAVE_LIP_BOTTOM_M) & (points[:, 2] > tip[2] - SHAVE_CHIN_M)
    area = painter(data)((side & front & (above | below)).astype(float), 1) > 0.5
    mask = cv2.dilate(area.astype(np.uint8) * 255, np.ones((3, 3), np.uint8))
    healed = cv2.inpaint(np.clip(skin, 0, 255).astype(np.uint8), mask, 9, cv2.INPAINT_TELEA).astype(np.float32)
    soft = cv2.GaussianBlur(mask.astype(np.float32) / 255, (0, 0), 2)[:, :, None]
    shaved = skin * (1 - soft) + cv2.GaussianBlur(healed, (0, 0), 3) * soft
    # O pelo do bigode descia sobre o lábio, que ficava cinza: mantém a luz dele e volta à cor da pele, mais quente.
    lip = (np.abs(points[:, 0] - tip[0]) < SHAVE_LIP_HALF_M) & (points[:, 2] <= tip[2] - SHAVE_LIP_TOP_M + 0.002) & (points[:, 2] >= tip[2] - SHAVE_LIP_BOTTOM_M - 0.002) & front
    share = cv2.GaussianBlur((painter(data)(lip.astype(float), 1) > 0.5).astype(np.float32), (0, 0), 2)[:, :, None]
    lab = cv2.cvtColor(np.clip(shaved, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    tone = cv2.cvtColor(np.median(skin.reshape(-1, 3), axis=0)[None, None, :].astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
    lab[:, :, 1:] = lab[:, :, 1:] * (1 - share) + (tone[1:] + np.array(SHAVE_LIP_WARM)) * share
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)


def relief(data, own):
    """A «maquiagem» do Rafa é volume pintado: sombra nas cavidades e luz nas saliências. Aqui ela sai da forma de cada rosto, sem custo no player."""
    points = np.load(own)["points"].astype(np.float64)
    faces = data["tri"]
    normals = np.zeros_like(points)
    cross = np.cross(points[faces[:, 1]] - points[faces[:, 0]], points[faces[:, 2]] - points[faces[:, 0]])
    for corner in range(3):
        np.add.at(normals, faces[:, corner], cross)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    _, twin = np.unique(np.round(points, 4), axis=0, return_inverse=True)
    twin = twin.ravel()
    edges = np.unique(np.sort(twin[np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])], axis=1), axis=0)
    edges = edges[edges[:, 0] != edges[:, 1]]
    count = twin.max() + 1
    welded = np.zeros((count, 3))
    welded[twin] = points
    degree = np.maximum(np.bincount(edges.ravel(), minlength=count), 1)[:, None]

    def smooth(values, rounds):
        for _ in range(rounds):
            total = np.zeros_like(values)
            np.add.at(total, edges[:, 0], values[edges[:, 1]])
            np.add.at(total, edges[:, 1], values[edges[:, 0]])
            values = (values + total / degree) / 2
        return values

    near = smooth(welded, RELIEF_NEAR)
    far = smooth(near, RELIEF_FAR)
    height = ((near - far)[twin] * normals).sum(axis=1)
    return painter(data)(np.clip(height / RELIEF_M, -1, 1), 2.5)


def chroma(image):
    lab = cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    return np.hypot(lab[:, :, 1] - 128, lab[:, :, 2] - 128)


def main():
    words = [word for word in sys.argv[1:] if word != "--shave"]
    source, map_path, shape_path, target = words[:4]
    strength = float(words[4]) if len(words) > 4 else 0.8
    skin, cleared = clear_philtrum(cv2.imread(source).astype(np.float32), np.load(map_path))
    skin, brows = clear_painted_brows(skin, np.load(map_path))
    skin, corners = trim_lip_corners(skin, np.load(map_path))
    if "--shave" in sys.argv:
        skin = shave_face(skin, np.load(map_path))
    gray = skin.mean(axis=2)
    tone = np.median(skin.reshape(-1, 3), axis=0)
    level = float(np.median(gray))
    blush, face, nose, beard_zone, lips = zones(np.load(map_path))
    ratio = gray / np.maximum(cv2.GaussianBlur(gray, (WIDE_PX, WIDE_PX), 0), 1)
    keep = np.clip((np.abs(ratio - 1) - QUIET) / (LOUD - QUIET), 0, 1)
    keep = cv2.GaussianBlur(keep, (0, 0), 1.2)[:, :, None]
    shaded = tone * np.clip(gray / level, 0.12, 1.4)[:, :, None]
    body = tone * (1 - keep) + shaded * keep
    # Barba é pelo: no rosto, onde a pele perde cor, o cinza vira a própria pele mais escura.
    own = float(chroma(tone[None, None, :])[0, 0])
    beard = cv2.GaussianBlur(np.clip((own - chroma(skin)) / (own * 0.4), 0, 1), (0, 0), 1) * beard_zone
    # Sem barba, o que perde cor no rosto é o lábio: a boca da Dona Rosa ficava da cor da pele.
    covered = float(beard.sum() / max(beard_zone.sum(), 1))
    if covered < BEARD_LEAST:
        beard[:] = 0
    hair = tone * (np.minimum(gray / level, 1) * (1 - BEARD_DARK))[:, :, None]
    head = skin * (1 - beard[:, :, None]) + hair * beard[:, :, None]
    # O rosto cinza ao lado do corpo quente: matiz da pele do corpo, com a luz do rosto.
    cheek = blush > 0.5
    lab = cv2.cvtColor(np.clip(head, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    wanted = cv2.cvtColor(tone[None, None, :].astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
    lab[:, :, 1:] += (wanted[1:] - np.median(lab[cheek][:, 1:], axis=0)) * HUE_SHARE
    head = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
    # O lábio tem cor própria e fica fora do acerto de matiz.
    lips = lips * (1 - beard)
    head = head * (1 - lips[:, :, None]) + skin * lips[:, :, None]
    clean = body * (1 - face[:, :, None]) + head * face[:, :, None]
    # Em pele escura o mesmo vermelho pesa mais: o rosto do Fábio saiu vermelho com a força cheia.
    share = (blush * strength * min(1.0, (level / 190) ** 2))[:, :, None]
    clean = clean * (1 - share) + clean * WARM * share
    shape = relief(np.load(map_path), shape_path)[:, :, None]
    # A sombra da pele é quente: o azul cai mais que o vermelho.
    clean = clean * np.where(shape < 0, 1 + SHADOW * shape * SHADOW_TINT, 1 + LIGHT * shape)
    cv2.imwrite(target, np.clip(clean, 0, 255).astype(np.uint8))
    print("PELE", target, "tom", tone.round().tolist(), "barba", round(covered, 2), "sulco", cleared, "sobrancelha", brows, "batom", corners)


main()
