#!/usr/bin/env python3
"""Veste a estampa repetível do jogo direto na UV da peça, sem passar pelo bake: assada de um vestido a outro, a flor virava risco.

O bake segue dizendo onde há estampa: pedaço liso, como o corpete preto, fica como está.

    stamp_tile.py <estampa-assada.png> <estampa-do-jogo.png> <repetições-por-UV>
"""
import sys

import cv2
import numpy as np

BUSY_PX = 33
BUSY_SPREAD = 6.0
EDGE_PX = 25


def main():
    baked_path, print_path, repeats = sys.argv[1], sys.argv[2], float(sys.argv[3])
    baked = cv2.imread(baked_path).astype(np.float32)
    pattern = cv2.imread(print_path).astype(np.float32)
    size = baked.shape[0]
    gray = cv2.cvtColor(baked.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.blur(gray, (BUSY_PX, BUSY_PX))
    spread = np.sqrt(np.maximum(cv2.blur(gray * gray, (BUSY_PX, BUSY_PX)) - mean * mean, 0))
    busy = cv2.GaussianBlur(cv2.morphologyEx((spread > BUSY_SPREAD).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((EDGE_PX, EDGE_PX), np.uint8)).astype(np.float32), (EDGE_PX, EDGE_PX), 0)
    # What the bake did not reach is part of the same cloth: the skirt, longer than the game's.
    unreached = cv2.imread(baked_path.replace(".png", ".hole.png"), cv2.IMREAD_GRAYSCALE)
    if unreached is not None:
        busy = np.maximum(busy, cv2.GaussianBlur(unreached.astype(np.float32) / 255, (EDGE_PX, EDGE_PX), 0))
    side = max(8, round(size / repeats))
    tile = cv2.resize(pattern, (side, side), interpolation=cv2.INTER_AREA)
    tiled = np.tile(tile, (size // side + 2, size // side + 2, 1))[:size, :size]
    cv2.imwrite(baked_path, baked * (1 - busy[:, :, None]) + tiled * busy[:, :, None])
    print("TILE", baked_path, "repetições", repeats, "área", round(float(busy.mean()), 2))


main()
