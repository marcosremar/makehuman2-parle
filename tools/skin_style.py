#!/usr/bin/env python3
"""Pele do elenco no estilo do rosto do Rafa: corpo limpo num tom só e blush suave em bochecha, nariz, orelha e queixo.

O rosto vem do NPC do jogo (`cast-heads.py`), com barba e marcas de cada pessoa, e fica: é identidade. O corpo só recebe o tom, e a emenda virava colar no pescoço.
Aqui o corpo limpa, o pescoço faz a transição, e a barba cinza do rosto escuro ganha o matiz da pele. A malha é a mesma em todo o elenco, então as zonas saem de um mapa só.

    skin_style.py <pele.png> <uvpos.npz> <corpo.npz> <saída.png> [força-do-blush] [--shave] [--no-beard]
"""
import sys

import cv2
import numpy as np

from parle_bald import bare_crown
from parle_beard import beard_hair, beard_shape, inside_mouth
from parle_lips import lip_maps, lip_points, lip_zone

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
# A beard too light to have a border stayed as a smudge on the jaw once the skin around it was clean (carioca-h22).
BEARD_LEAST = 0.3
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
SHAVE_LIP_WARM = (10.0, 3.0)
SHAVE_DEPTH_M = 0.05
SHAVE_CHIN_M = 0.08
SHAVE_KEEP_LIGHT = 0.85
SHAVE_FILL_SOFT = 12
# The warm lip of a shaved face was measured on light skin; on dark skin the same chroma reads as orange stains.
SHAVE_WARM_LEVEL = 190
SCALP_BEHIND_TIP_M = 0.045
SCALP_ABOVE_TIP_M = 0.02
SCALP_DARK = 0.45
SCALP_LEAST_PX = 400
SCALP_REACH_PX = 41
SCALP_CLOSE_PX = 15
SCALP_GREY = 0.55
SCALP_PALE = 0.7
SCALP_TONE = 20
PHILTRUM_HALF_M = 0.012
PHILTRUM_TOP_M = 0.006
PHILTRUM_BOTTOM_M = 0.032
PHILTRUM_DEPTH_M = 0.03
PHILTRUM_DARK = 0.62
# The game head leaves a thread of dark pixels along its hairline: a thin line around the face.
THREAD_PX = 13
# On the forehead a wider gap also closes: the patches of painted hair that show under a cap's brim (Tião).
FOREHEAD_PATCH_PX = 31
THREAD_DARK = 0.85
THREAD_ABOVE_TIP_M = 0.1
THREAD_SIDE_M = 0.06
THREAD_DEPTH_M = 0.1
# Same scratches on the cheeks, beside the mouth (h21).
CHEEK_SIDE_M = 0.035
CHEEK_LOW_M = 0.06
# The game face changes light in a step across the cheeks, at the height of the nose: one face over another (André). Measured per height on the cheeks, away from nose and mouth.
EVEN_LOW_M = 0.09
EVEN_HIGH_M = 0.04
EVEN_SIDE = (0.035, 0.07)
EVEN_BIN_M = 0.005
EVEN_SOFT_BINS = 1.5
EVEN_LEAST, EVEN_MOST = 0.8, 1.25
# The blush came out as a pink ball on each cheek, like make-up (owner, 2026-11-08): weaker and wider. The player's own flush (`NpcFaceShading`) adds to it on the cheeks and
# the nose, and at 0.5 the two together read as a face rosier than its neck (Lúcia, scoreboard of 2026-11-10). 0.2 keeps a trace; no source, a design choice to pilot.
BLUSH_STRENGTH = 0.2
BLUSH_REACH_M = 0.042
# The eyelash cards are gone (parle_face): a thin darker line on the upper lid's rim stands in for them, as on the Rafa.
LASH_DARK = 0.5
LASH_SOFT = 1.2
# The head and neck are two islands of the map, and the game face and the body tone met in a coloured line under the jaw. Each side
# of a seam moves halfway to the other, fading out over SEAM_SOFT_PX.
SEAM_SOFT_PX = 10
SEAM_SAMPLE_PX = 3
SEAM_FADE = 3
SEAM_REACH_M = 0.0015
PAD_PX = 6
RIM_PX = 1
# Painted scalp that is not attached to the big one is a speck the bake left on the temple, and it heals to skin.
SCALP_KEEP_PX = 2500
SCALP_EDGE_PX = 15
SCALP_EDGE_SOFT = 3
# The beard starts on the lip line, not in a straight cut across the cheeks.
BEARD_TOP_M = 0.008
BEARD_FADE_M = 0.022
# At 1024 the face is about 1.2 px per mm: the border is rounded over 12 mm and the halo reaches 8 mm out on each side of it.
BEARD_ROUND_PX = 15
BEARD_KEEP_PX = 400
BEARD_SOFT = 3
# A light beard has no border of its own: tidied, carioca-h22 kept only a dark patch on the jaw.
BEARD_TIDY = 0.3
BEARD_RING_PX = 21
HALO_LEAST = 0.75
HALO_SAMPLE_SOFT = 18
# The scalp keeps its painted hair a little above the cut of the hair mesh: a razor-straight mesh border reads as a sticker on bare skin.
CROWN_SOFT = 6
CROWN_ABOVE_M = 0.008
# The game face has grey-green bruises on the jaw: below the eyes the skin never goes greener or bluer than this.
JAW_ABOVE_TIP_M = 0.02
# The game face changes colour in a step across the cheeks: two faces, one over the other (André). Light keeps the features; colour alone may melt.
FACE_COLOUR_SOFT = 15
GREEN_MOST = 2
BLUE_MOST = 3
# Rafa's skin is one clean tone (owner, 2026-11-08). The game face, enlarged, brought blotches and a red band across the cheeks (Carlos): only its broad light stays, beard and painted hair kept.
PLAIN_SOFT = 35
PAINTED_HAIR_ABOVE_TIP_M = 0.04
PAINTED_HAIR_DARK = 0.62
PLAIN_LEAST = 0.75
# The player has no occlusion inside the mouth: lit like the face, its lining read orange (owner, 2026-11-09). Darker as it goes in.
MOUTH_RIM = 0.6
MOUTH_DEEP = 0.2


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
        blush = np.maximum(blush, near((middle + side * wide * 0.55, tip[1] + 0.035, tip[2] - 0.005), BLUSH_REACH_M))
        blush = np.maximum(blush, 0.7 * near((middle + side * wide * 0.98, tip[1] + 0.11, tip[2] + 0.015), 0.03))
    blush = np.maximum(blush, 0.7 * near(tip, 0.022))
    blush = np.maximum(blush, 0.35 * near((tip[0], tip[1] + 0.012, tip[2] - 0.085), 0.022))
    nose = np.zeros(len(points))
    # O queixo fica uns 10 cm abaixo da ponta do nariz; a transição desce 7 cm pelo pescoço.
    chin = tip[2] - CHIN_M
    face = np.clip((points[:, 2] - (chin - NECK_FADE_M)) / NECK_FADE_M, 0, 1)
    beard = (points[:, 2] > chin) * np.clip((tip[2] - BEARD_TOP_M - points[:, 2]) / BEARD_FADE_M, 0, 1)
    lower = (points[:, 2] < tip[2] + JAW_ABOVE_TIP_M).astype(float)
    paint = painter(data)
    return paint(blush, 3), paint(face, 6), paint(nose, 2), paint(beard, 3), paint(lower, 4)


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


def clear_hair_thread(skin, data):
    """Only what is thin goes: closing keeps the big painted scalp and the eyebrows' zone is out of reach."""
    points = data["pos"]
    top = points[:, 2].max()
    head = points[points[:, 2] > top - 0.26]
    middle = float(np.median(head[:, 0]))
    tip = head[np.abs(head[:, 0] - middle) < 0.01]
    tip = tip[tip[:, 1].argmin()]
    forehead = points[:, 2] > tip[2] + THREAD_ABOVE_TIP_M
    side = np.abs(points[:, 0] - tip[0])
    temple = (side > THREAD_SIDE_M) & (points[:, 2] > tip[2] - 0.01) & (points[:, 1] < tip[1] + THREAD_DEPTH_M)
    temple = temple | ((side > CHEEK_SIDE_M) & (points[:, 2] <= tip[2]) & (points[:, 2] > tip[2] - CHEEK_LOW_M) & (points[:, 1] < tip[1] + THREAD_DEPTH_M))
    high = painter(data)(forehead.astype(float), 1) > 0.5
    area = painter(data)((forehead | temple).astype(float), 1) > 0.5
    gray = skin.mean(axis=2).astype(np.float32)
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((THREAD_PX, THREAD_PX), np.uint8))
    wider = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((FOREHEAD_PATCH_PX, FOREHEAD_PATCH_PX), np.uint8))
    thread = area & (gray < THREAD_DARK * np.where(high, wider, closed))
    if not thread.any():
        return skin, 0
    mask = cv2.dilate(thread.astype(np.uint8) * 255, np.ones((3, 3), np.uint8))
    return cv2.inpaint(np.clip(skin, 0, 255).astype(np.uint8), mask, 5, cv2.INPAINT_TELEA).astype(np.float32), int(thread.sum())


def even_face(skin, data):
    points = data["pos"]
    top = points[:, 2].max()
    head = points[points[:, 2] > top - 0.26]
    middle = float(np.median(head[:, 0]))
    tip = head[np.abs(head[:, 0] - middle) < 0.01]
    tip = tip[tip[:, 1].argmin()]
    height = points[:, 2] - tip[2]
    side = np.abs(points[:, 0] - tip[0])
    cheek = (side > EVEN_SIDE[0]) & (side < EVEN_SIDE[1]) & (height > -EVEN_LOW_M) & (height < EVEN_HIGH_M) & (points[:, 1] < tip[1] + THREAD_DEPTH_M)
    paint = painter(data)
    edges = np.arange(-EVEN_LOW_M, EVEN_HIGH_M + EVEN_BIN_M, EVEN_BIN_M)
    centres = (edges[:-1] + edges[1:]) / 2
    gray = skin.mean(axis=2)
    levels = []
    for low, high in zip(edges[:-1], edges[1:]):
        band = paint((cheek & (height >= low) & (height < high)).astype(float), 0.5) > 0.5
        levels.append(float(np.median(gray[band])) if band.any() else np.nan)
    levels = np.array(levels)
    if np.isnan(levels).all():
        return skin
    known = ~np.isnan(levels)
    levels = np.interp(centres, centres[known], levels[known])
    weights = np.exp(-(((np.arange(len(levels))[:, None] - np.arange(len(levels))) / EVEN_SOFT_BINS) ** 2) / 2)
    smooth = (weights * levels).sum(axis=1) / weights.sum(axis=1)
    target = float(np.median(smooth))
    gain = np.clip(target / np.maximum(np.interp(height, centres, smooth), 1), EVEN_LEAST, EVEN_MOST)
    # Fades out on the jaw and the forehead, where nothing was measured.
    fade = np.clip(1 - np.maximum(-EVEN_LOW_M - height, height - EVEN_HIGH_M) / 0.02, 0, 1)
    return skin * paint(1 + (gain - 1) * fade, 2)[:, :, None]


def clear_painted_scalp(skin, data):
    """The bake of the scalp leaves grey plates and specks beside the temple, where the game hair ended (h19, Marcelo). They take the colour of the painted hair next to them."""
    points = data["pos"]
    top = points[:, 2].max()
    head = points[points[:, 2] > top - 0.26]
    middle = float(np.median(head[:, 0]))
    tip = head[np.abs(head[:, 0] - middle) < 0.01]
    tip = tip[tip[:, 1].argmin()]
    zone = (points[:, 1] > tip[1] + SCALP_BEHIND_TIP_M) & (points[:, 2] > tip[2] + SCALP_ABOVE_TIP_M)
    area = painter(data)(zone.astype(float), 1) > 0.5
    gray = skin.mean(axis=2)
    dark = area & (gray < SCALP_DARK * float(np.median(gray)))
    if dark.sum() < SCALP_LEAST_PX:
        return skin, 0
    closed = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((SCALP_CLOSE_PX, SCALP_CLOSE_PX), np.uint8)) > 0
    near = cv2.dilate(dark.astype(np.uint8), np.ones((SCALP_REACH_PX, SCALP_REACH_PX), np.uint8)) > 0
    own = float(chroma(np.median(skin.reshape(-1, 3), axis=0)[None, None, :])[0, 0])
    grey = area & near & (chroma(skin) < SCALP_GREY * own) & ~dark
    # Painted hair in two greys reads as a plate next to the hair mesh: one colour, the dark one.
    pale = area & near & (gray < SCALP_PALE * float(np.median(gray)))
    plate = (closed | grey | pale) & area
    if not plate.any():
        return skin, 0
    _, labels, stats, _ = cv2.connectedComponentsWithStats(plate.astype(np.uint8))
    big = plate & (stats[labels, cv2.CC_STAT_AREA] >= SCALP_KEEP_PX)
    specks = plate & ~big
    tone = np.percentile(skin[dark], SCALP_TONE, axis=0)
    if specks.any():
        mask = cv2.dilate(specks.astype(np.uint8) * 255, np.ones((7, 7), np.uint8))
        skin = cv2.inpaint(np.clip(skin, 0, 255).astype(np.uint8), mask, 6, cv2.INPAINT_TELEA).astype(np.float32)
    # The bake's noise leaves the plate with a serrated edge: a round, soft one reads as a hairline.
    edge = cv2.medianBlur(big.astype(np.uint8) * 255, SCALP_EDGE_PX)
    soft = cv2.GaussianBlur(edge.astype(np.float32) / 255, (0, 0), SCALP_EDGE_SOFT)[:, :, None]
    return skin * (1 - soft) + tone * soft, int(plate.sum())


def shave_face(skin, data, own):
    """Tira o bigode e a barbicha assados: o que difere da pele em volta, no sulco e abaixo do lábio, com a cor vinda de fora deles.

    O lábio é o medido pela fenda da boca (`parle_lips.lip_points`). A faixa fixa de 41 a 53 mm abaixo do nariz caía na metade de baixo
    do lábio inferior e no queixo logo abaixo, e o tom quente dela virou uma mancha escura embaixo da boca (Seu Nelson, dono, 08/11/2026).
    """
    points = own
    mouth = lip_points(data, points)
    top = points[:, 2].max()
    head = points[points[:, 2] > top - 0.26]
    middle = float(np.median(head[:, 0]))
    tip = head[np.abs(head[:, 0] - middle) < 0.01]
    tip = tip[tip[:, 1].argmin()]
    side = np.abs(points[:, 0] - tip[0]) < SHAVE_HALF_M
    front = points[:, 1] < tip[1] + SHAVE_DEPTH_M
    around = (points[:, 2] < tip[2] - 0.004) & (points[:, 2] > tip[2] - SHAVE_CHIN_M) & ~mouth
    paint = painter(data)
    area = paint((side & front & around).astype(float), 1) > 0.5
    mask = cv2.dilate(area.astype(np.uint8) * 255, np.ones((3, 3), np.uint8))
    # Filled from the skin around it, never from the lip or the hair at its edge: pulled from the border, the chin came out with a dark stain (Seu Nelson).
    lip_zone = paint(mouth.astype(float), 1) > 0.5
    gray = skin.mean(axis=2)
    keep = (~(mask > 0) & ~lip_zone & (gray > SHAVE_KEEP_LIGHT * np.median(gray))).astype(np.float32)
    weight = cv2.GaussianBlur(keep, (0, 0), SHAVE_FILL_SOFT)[:, :, None]
    healed = cv2.GaussianBlur(skin * keep[:, :, None], (0, 0), SHAVE_FILL_SOFT) / np.maximum(weight, 1e-3)
    soft = cv2.GaussianBlur(mask.astype(np.float32) / 255, (0, 0), 2)[:, :, None]
    shaved = skin * (1 - soft) + healed * soft
    # O pelo do bigode descia sobre o lábio, que ficava cinza: mantém a luz dele e volta à cor da pele, mais quente.
    share = cv2.GaussianBlur(lip_zone.astype(np.float32), (0, 0), 2)[:, :, None]
    lab = cv2.cvtColor(np.clip(shaved, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    tone = cv2.cvtColor(np.median(skin.reshape(-1, 3), axis=0)[None, None, :].astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
    share = share * min(1.0, (float(np.median(skin)) / SHAVE_WARM_LEVEL) ** 2)
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


def open_points(data):
    triangles = data["tri"]
    edges = np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1)
    unique, count = np.unique(edges, axis=0, return_counts=True)
    return np.unique(unique[count == 1])


def lash_line(skin, data, points):
    """Darkens the triangles on the upper rim of each eye's hole."""
    top = points[:, 2].max()
    head = points[points[:, 2] > top - 0.26]
    middle = float(np.median(head[:, 0]))
    tip = head[np.abs(head[:, 0] - middle) < 0.01]
    tip = tip[tip[:, 1].argmin()]
    rim = open_points(data)
    side = np.abs(points[rim, 0] - middle)
    rim = rim[(side > 0.01) & (side < 0.06) & (points[rim, 2] > tip[2] + 0.01) & (points[rim, 2] < tip[2] + 0.07) & (points[rim, 1] < tip[1] + 0.05)]
    upper = np.zeros(len(points), bool)
    for half in (points[rim, 0] < middle, points[rim, 0] >= middle):
        ring = rim[half]
        if len(ring):
            upper[ring[points[ring, 2] > points[ring, 2].mean()]] = True
    value = upper.astype(np.float32)
    shade = np.clip(painter(data)(value, LASH_SOFT) * 3, 0, 1)[:, :, None]
    return skin * (1 - shade * (1 - LASH_DARK)), int(upper.sum())


def even_seams(image, data, points):
    """Each side of a head or neck seam moves halfway to the colour of the other side."""
    top = points[:, 2].max()
    corners = np.stack([data["uv"][:, :, 0] * SIZE, (1 - data["uv"][:, :, 1]) * SIZE], axis=2)
    triangles = data["tri"]
    flat = corners.reshape(-1, 2)
    owner = triangles.ravel()
    # Sampled from inside the islands only: at the border the blur took in the padding around them.
    inside = np.zeros(image.shape[:2], np.float32)
    for corner in corners.round().astype(np.int32):
        cv2.fillConvexPoly(inside, corner, 1.0)
    inside = cv2.erode(inside, np.ones((3, 3), np.uint8))
    sample = cv2.GaussianBlur(image * inside[:, :, None], (0, 0), SEAM_SAMPLE_PX) / np.maximum(cv2.GaussianBlur(inside, (0, 0), SEAM_SAMPLE_PX), 1e-6)[:, :, None]
    # The glTF splits a point on a seam into one index per side: the sides are the indices at the same place.
    spot = {}
    for vertex, (u, v) in zip(owner, flat):
        spot[int(vertex)] = (int(round(u)), int(round(v)))
    # Seam halves sit on the open edge of the mesh and need not land on exactly the same point: the nearest edge point within reach.
    edge = open_points(data)
    edge = edge[points[edge, 2] > top - 0.35]
    places = {}
    for index, vertex in enumerate(edge):
        gap = np.linalg.norm(points[edge] - points[vertex], axis=1)
        gap[index] = np.inf
        other = edge[int(gap.argmin())]
        if gap.min() < SEAM_REACH_M:
            places.setdefault(tuple(sorted((int(vertex), int(other)))), set()).update({spot[int(vertex)], spot[int(other)]})
    push = np.zeros_like(image)
    weight = np.zeros(image.shape[:2], np.float32)
    seams = 0
    for spots in places.values():
        spots = [(min(max(u, 0), SIZE - 1), min(max(v, 0), SIZE - 1)) for u, v in spots]
        if len(spots) < 2 or max(abs(a[0] - b[0]) + abs(a[1] - b[1]) for a in spots for b in spots) < 4:
            continue
        seams += 1
        colours = np.array([sample[v, u] for u, v in spots])
        middle = colours.mean(axis=0)
        for (u, v), colour in zip(spots, colours):
            cv2.circle(push, (u, v), 2, (middle - colour).tolist(), -1)
            cv2.circle(weight, (u, v), 2, 1.0, -1)
    spread = cv2.GaussianBlur(weight, (0, 0), SEAM_SOFT_PX)
    field = cv2.GaussianBlur(push, (0, 0), SEAM_SOFT_PX) / np.maximum(spread, 1e-6)[:, :, None]
    fade = np.clip(spread / max(float(spread.max()), 1e-6) * SEAM_FADE, 0, 1)[:, :, None]
    return image + field * fade, seams


def pad_islands(image, data, rim=0):
    """The texture filter reads a few texels past an island's border, and the relief's shadow there drew a thin dark line on the seam.

    With `rim`, the island's own outer texels go too: the bake left them darker, a line along the jaw seam (Léo, rooftop).
    """
    corners = np.stack([data["uv"][:, :, 0] * SIZE, (1 - data["uv"][:, :, 1]) * SIZE], axis=2).round().astype(np.int32)
    covered = np.zeros(image.shape[:2], np.uint8)
    for corner in corners:
        cv2.fillConvexPoly(covered, corner, 1)
    covered = cv2.erode(covered, np.ones((3 + 2 * rim, 3 + 2 * rim), np.uint8))
    band = ((cv2.dilate(covered, np.ones((PAD_PX * 2 + 1 + 2 * rim, PAD_PX * 2 + 1 + 2 * rim), np.uint8)) > 0) & (covered == 0)).astype(np.uint8)
    return cv2.inpaint(np.clip(image, 0, 255).astype(np.uint8), band, PAD_PX, cv2.INPAINT_TELEA).astype(np.float32)


def clear_crown(skin, data):
    """A balding man's scalp is skin up to the line where the hair mesh is cut (`parle_bald`): the bake had painted hair there."""
    bare = painter(data)(bare_crown(data["pos"] - np.array([0, 0, CROWN_ABOVE_M]), data["pos"]).astype(float), CROWN_SOFT)[:, :, None]
    tone = np.median(skin.reshape(-1, 3), axis=0)
    return skin * (1 - bare) + tone * bare


def dark_mouth(image, data):
    area, deep = inside_mouth(data, SIZE)
    factor = (MOUTH_RIM + (MOUTH_DEEP - MOUTH_RIM) * deep).astype(np.float32)
    image = image.copy()
    image[area] *= factor[area][:, None]
    return image, int(area.sum())


def plain_face(head, face, beard, lips, data):
    """One clean tone on the bare skin of the head: its broad light stays, blotches go. Beard, lip and the painted hair stay."""
    points = data["pos"]
    top = points[:, 2].max()
    around = points[points[:, 2] > top - 0.26]
    middle = float(np.median(around[:, 0]))
    tip = around[np.abs(around[:, 0] - middle) < 0.01]
    tip = tip[tip[:, 1].argmin()]
    gray = head.mean(axis=2)
    level = float(np.median(gray[face > 0.5]))
    high = painter(data)((points[:, 2] > tip[2] + PAINTED_HAIR_ABOVE_TIP_M).astype(float), 1) > 0.5
    hair = cv2.dilate(cv2.morphologyEx((high & (gray < PAINTED_HAIR_DARK * level)).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)), np.ones((5, 5), np.uint8))
    # A sideburn the hair mesh does not reach is a loose dark patch beside the ear (Carlos, Marcelo): only what joins the big scalp stays.
    _, labels, stats, _ = cv2.connectedComponentsWithStats(hair)
    hair = ((stats[labels, cv2.CC_STAT_AREA] >= SCALP_KEEP_PX) & (labels > 0)).astype(np.float32)
    plain = ((face > 0.5) & (beard < 0.05) & (lips < 0.05) & (hair == 0) & (gray > PLAIN_LEAST * level)).astype(np.float32)
    weight = np.maximum(cv2.GaussianBlur(plain, (0, 0), PLAIN_SOFT), 1e-3)[:, :, None]
    smooth = cv2.GaussianBlur(head * plain[:, :, None], (0, 0), PLAIN_SOFT) / weight
    share = (face * (1 - beard) * (1 - lips) * (1 - cv2.GaussianBlur(hair, (0, 0), 3)))[:, :, None]
    return head * (1 - share) + smooth * share


def tidy_beard(skin, beard, lips, shape=None):
    """The mask comes from the bake's noise: rounds its border, drops the loose specks and heals the dark halo left on the skin outside it.

    With `shape`, the beard drawn on the head (`parle_beard`) takes over, and all of the baked one outside it heals.
    """
    solid = cv2.medianBlur((beard > 0.5).astype(np.uint8) * 255, BEARD_ROUND_PX)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(solid)
    solid = ((stats[labels, cv2.CC_STAT_AREA] >= BEARD_KEEP_PX) & (labels > 0)).astype(np.uint8) * 255
    soft = cv2.GaussianBlur(solid.astype(np.float32) / 255, (0, 0), BEARD_SOFT) if shape is None else shape
    grown = cv2.dilate(solid, np.ones((BEARD_RING_PX, BEARD_RING_PX), np.uint8))
    gray = skin.mean(axis=2)
    # What the healed skin is made of: nothing from the halo, the lip or the painted hair.
    plain = ((grown == 0) & (lips < 0.05) & (gray > HALO_LEAST * float(np.median(gray)))).astype(np.float32)
    weight = np.maximum(cv2.GaussianBlur(plain, (0, 0), HALO_SAMPLE_SOFT), 1e-3)[:, :, None]
    healed = cv2.GaussianBlur(skin * plain[:, :, None], (0, 0), HALO_SAMPLE_SOFT) / weight
    halo = (cv2.GaussianBlur(grown.astype(np.float32) / 255, (0, 0), BEARD_SOFT) * (1 - soft) * (1 - np.clip(lips, 0, 1)))[:, :, None]
    return skin * (1 - halo) + healed * halo, soft


def main():
    words =[word for word in sys.argv[1:] if word not in ("--shave", "--no-beard", "--bald")]
    source, map_path, shape_path, target = words[:4]
    strength = float(words[4]) if len(words) > 4 else BLUSH_STRENGTH
    skin, cleared = clear_philtrum(cv2.imread(source).astype(np.float32), np.load(map_path))
    skin, brows = clear_painted_brows(skin, np.load(map_path))
    skin, corners = trim_lip_corners(skin, np.load(map_path))
    skin, thread = clear_hair_thread(skin, np.load(map_path))
    skin, scalp = clear_painted_scalp(skin, np.load(map_path))
    if "--shave" in sys.argv:
        skin = shave_face(skin, np.load(map_path), np.load(shape_path)["points"])
    if "--bald" in sys.argv:
        skin = clear_crown(skin, np.load(map_path))
    gray = skin.mean(axis=2)
    tone = np.median(skin.reshape(-1, 3), axis=0)
    level = float(np.median(gray))
    blush, face, nose, beard_zone, lower = zones(np.load(map_path))
    lips = lip_zone(np.load(map_path), np.load(shape_path)["points"])
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
    # Face less saturated than the body is not a beard: Ivone came out with a dark jaw.
    if covered < BEARD_LEAST or "--no-beard" in sys.argv:
        beard[:] = 0
        skin = even_face(skin, np.load(map_path))
    elif covered >= BEARD_TIDY:
        baked = beard
        skin, beard = tidy_beard(skin, beard, lips, beard_shape(np.load(map_path), lips, SIZE))
        gray = skin.mean(axis=2)
    hair = tone * (np.minimum(gray / level, 1) * (1 - BEARD_DARK))[:, :, None]
    if beard.any():
        hair = beard_hair(hair, baked, lips, SIZE)
    head = skin * (1 - beard[:, :, None]) + hair * beard[:, :, None]
    # O rosto cinza ao lado do corpo quente: matiz da pele do corpo, com a luz do rosto.
    cheek = blush > 0.5
    lab = cv2.cvtColor(np.clip(head, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    wanted = cv2.cvtColor(tone[None, None, :].astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
    lab[:, :, 1:] += (wanted[1:] - np.median(lab[cheek][:, 1:], axis=0)) * HUE_SHARE
    # A beard is hair and keeps its colour: pulled to the skin's, it came out orange.
    bare = lower * (1 - beard)
    lab[:, :, 1] += np.maximum(0, wanted[1] - GREEN_MOST - lab[:, :, 1]) * bare
    lab[:, :, 2] += np.maximum(0, wanted[2] - BLUE_MOST - lab[:, :, 2]) * bare
    melt = (face * (1 - beard) * (1 - lips))[:, :, None]
    lab[:, :, 1:] = lab[:, :, 1:] * (1 - melt) + cv2.GaussianBlur(lab[:, :, 1:], (0, 0), FACE_COLOUR_SOFT) * melt
    head = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
    # O lábio tem cor própria e fica fora do acerto de matiz.
    lips = lips * (1 - beard)
    head = plain_face(head, face, beard, lips, np.load(map_path))
    # The lip is a layer of its own, measured on the mesh (`parle_lips.lip_zone`): its map goes beside the skin and `parle_lips.py` paints it, as many times as wanted, without redoing the skin.
    cv2.imwrite(target.replace(".png", ".lips.png"), np.clip(lips * 255, 0, 255).astype(np.uint8))
    _, line, upper = lip_maps(np.load(map_path), np.load(shape_path)["points"])
    cv2.imwrite(target.replace(".png", ".line.png"), np.clip(np.dstack([line, upper, np.zeros_like(line)]) * 255, 0, 255).astype(np.uint8))
    clean = body * (1 - face[:, :, None]) + head * face[:, :, None]
    own_points = np.load(shape_path)["points"]
    clean, seams = even_seams(pad_islands(clean, np.load(map_path), RIM_PX), np.load(map_path), own_points)
    # Em pele escura o mesmo vermelho pesa mais: o rosto do Fábio saiu vermelho com a força cheia.
    share = (blush * strength * min(1.0, (level / 190) ** 2))[:, :, None]
    clean = clean * (1 - share) + clean * WARM * share
    # On the body the same shadow came out as a brown stain on the small of the back and around the neck: the relief is the face's.
    shape = (relief(np.load(map_path), shape_path) * face)[:, :, None]
    # A sombra da pele é quente: o azul cai mais que o vermelho.
    clean = clean * np.where(shape < 0, 1 + SHADOW * shape * SHADOW_TINT, 1 + LIGHT * shape)
    clean, lashes = lash_line(clean, np.load(map_path), own_points)
    clean, inside = dark_mouth(clean, np.load(map_path))
    clean = pad_islands(clean, np.load(map_path))
    cv2.imwrite(target, np.clip(clean, 0, 255).astype(np.uint8))
    print("PELE", target, "tom", tone.round().tolist(), "barba", round(covered, 2), "sulco", cleared, "fio", thread, "sobrancelha", brows, "batom", corners, "couro", scalp, "costura", seams, "cílio", lashes, "boca", inside)


if __name__ == "__main__":
    main()
