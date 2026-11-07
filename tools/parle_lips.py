#!/usr/bin/env python3
"""
    License information: data/licenses/makehuman_license.txt

    Paints the lips of a Parle cast skin, as a step of its own after `skin_style.py` (owner, 2026-11-07:
    the lip colour is set later, not only when the skin is made). The skin step leaves the skin without
    painted lips and a map of where the lip is; this step always starts from those two, so it can run
    again with another colour and nothing drifts.

    Natural lips are the person's own skin, 20% darker, rosier and a little less yellow in Lab, with half
    of the lip's own light and shade kept as its drawing: the game's wine lipstick and grey lips read as
    make-up or as cold. A colour given in the look (`lipColor`) replaces the hue, and the drawing stays.

        parle_lips.py <skin-base.png> <lips-map.png> <out.png> [#rrggbb] [--man]
"""
import sys

import cv2
import numpy as np

LIP_LIGHT = 0.8
LIP_ROSE = 11
LIP_WARM = -4
LIP_DRAWING = 0.5
INSIDE = 0.5
# The same rose reads as lipstick on dark skin and on men (Seu Nelson): it fades with the skin's lightness, and a man's lip is less rosy.
ROSE_FULL_LIGHT = 170
MAN_ROSE = 0.55


def lab_of(bgr):
    return cv2.cvtColor(np.array(bgr, np.uint8)[None, None, :], cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]


def painted(skin, share, colour=None, man=False):
    lab = cv2.cvtColor(skin, cv2.COLOR_BGR2LAB).astype(np.float32)
    own = lab_of(np.median(skin.reshape(-1, 3), axis=0))
    inside = share > INSIDE
    light = float(np.median(lab[inside, 0])) if inside.any() else own[0]
    if colour:
        wanted = lab_of([int(colour[index:index + 2], 16) for index in (5, 3, 1)])
        base, hue = wanted[0], wanted[1:]
    else:
        rose = LIP_ROSE * min(1.0, (own[0] / ROSE_FULL_LIGHT) ** 2) * (MAN_ROSE if man else 1.0)
        base, hue = own[0] * LIP_LIGHT, own[1:] + np.array([rose, LIP_WARM])
    lip = lab.copy()
    lip[:, :, 0] = base + (lab[:, :, 0] - light) * LIP_DRAWING
    lip[:, :, 1:] = hue
    lip = cv2.cvtColor(np.clip(lip, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
    return (skin * (1 - share[:, :, None]) + lip * share[:, :, None]).round().astype(np.uint8)


def main():
    words = [word for word in sys.argv[1:] if word != "--man"]
    base, mask, out = words[:3]
    colour = words[3] if len(words) > 3 else None
    share = cv2.imread(mask, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
    cv2.imwrite(out, painted(cv2.imread(base), share, colour, "--man" in sys.argv))
    print("LABIOS", out.split("/")[-1], colour or "natural", "área", round(float((share > INSIDE).mean()), 4))


if __name__ == "__main__":
    main()
