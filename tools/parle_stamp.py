"""
    License information: data/licenses/makehuman_license.txt

    Which pieces of a game body paint the print of a new garment. Whatever sits on top of the cloth or
    under it in the old body is not print: a necklace baked into a shirt reads as a torn collar, a
    bikini under a tank top as a red stain, and the cord of the shorts as a grey patch.
"""
# How the game names the pieces of each garment, by the start of the name. A dress and a bikini are both halves.
GAME_TOP = ("lctshirt", "Shirt_", "Rolled_sleeves_shirt", "Tank", "Dress", "Placket", "Pen", "Necklace", "Chain", "Crucifix", "Bikini")
GAME_BOTTOM = ("lcpants", "bjjeans", "Shorts", "Dress", "Belt", "Fray", "Bikini")
NOT_PRINT = ("Necklace", "Chain", "Crucifix", "Shorts_Cord")
UNDERWEAR = "Bikini"
# The last centimetres of a game garment are its hem: a dark cuff, a white fray. On a longer new garment they would end up in the middle of the leg.
HEM_PX = 28
HOLE_SOFT = 5
# Rays miss here and there all over the cloth: a pinhole is not a stretch of garment beyond the old hem.
HOLE_THICK_PX = 9
HOLE_LEAST_PX = 1500
# A garment is plain when one colour group covers this much of it. A striped one goes on striped, from the nearest print.
MAIN_GROUPS = 3
MAIN_PLAIN = 0.6
MAIN_SAMPLE = 20000
# Light and shadow of one cloth come out as separate groups: nearby ones are the same colour.
MAIN_SAME = 45
# When the bake reached little of the garment what it found is a trim, and the colour measured on the game body is the garment's (Fábio's red shorts came out white).
REACHED_LEAST = 0.35
# On a plain garment the shadow of the old hem stays baked at the edge of the cloth, in a dark band at the waist (Cláudia). A striped one has dark there on purpose.
EDGE_BAND_PX = 40
EDGE_DARK = 0.65
PLAIN_LOW_PERCENTILE = 5
PLAIN_LOW = 0.55
# A shoe of nearly the skin's colour reads as a bare foot (Helena).
SHOE_LIKE_SKIN = 70
SHOE_DARKER = 0.45


def stamp_sources(names, prefixes):
    wanted = [name for name in names if name.startswith(prefixes) and not name.startswith(NOT_PRINT)]
    covering = [name for name in wanted if not name.startswith(UNDERWEAR)]
    return covering or wanted


def main_colour(pixels):
    """The biggest group of colours and the share it covers, never a plain median: the median of green, wine and white stripes is a brown that is not on the shirt. Groups, not exact colours: the shaded red of Fábio's shorts is thousands of reds, and the one white of its trim would win."""
    import cv2
    import numpy as np

    step = max(1, len(pixels) // MAIN_SAMPLE)
    sample = pixels[::step].astype(np.float32)
    groups = min(MAIN_GROUPS, len(sample))
    _, labels, centres = cv2.kmeans(sample, groups, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0), 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.ravel(), minlength=groups)
    biggest = counts.argmax()
    same = np.linalg.norm(centres - centres[biggest], axis=1) < MAIN_SAME
    return np.median(sample[same[labels.ravel()]], axis=0), counts[same].sum() / len(sample)


def close_stamp(image, ground, triangles):
    """Fills the part of the new garment the bake did not reach from the print next to it, leaving out the old hem. `triangles` are its UV triangles in pixels; the rest of the image is not cloth. Runs outside Blender, whose Python has no OpenCV."""
    import cv2
    import numpy as np

    cloth = np.zeros(image.shape[:2], np.uint8)
    for triangle in triangles:
        cv2.fillConvexPoly(cloth, triangle, 1)
    empty = np.abs(image.astype(int) - ground).max(axis=2) <= 2
    hole = (empty & (cloth > 0)).astype(np.uint8)
    wide = cv2.morphologyEx(hole, cv2.MORPH_OPEN, np.ones((HOLE_THICK_PX, HOLE_THICK_PX), np.uint8))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(wide)
    wide = wide * (stats[labels, cv2.CC_STAT_AREA] >= HOLE_LEAST_PX)
    hem = cv2.dilate(wide.astype(np.uint8), np.ones((HEM_PX * 2 + 1, HEM_PX * 2 + 1), np.uint8)) > 0
    broad = hem | (wide > 0)
    gray = image.mean(axis=2)
    known = (cloth > 0) & ~empty & ~broad
    if known.any():
        middle = float(np.median(gray[known]))
        if np.percentile(gray[known], PLAIN_LOW_PERCENTILE) >= PLAIN_LOW * middle:
            border = (cloth > 0) & (cv2.erode(cloth, np.ones((EDGE_BAND_PX * 2 + 1, EDGE_BAND_PX * 2 + 1), np.uint8), borderValue=0) == 0)
            broad = broad | (border & (gray < EDGE_DARK * middle))
    missing = empty | broad
    if missing.all() or not missing.any():
        return image, 0.0, np.zeros(image.shape[:2], bool)
    _, labels = cv2.distanceTransformWithLabels(missing.astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    rows, columns = np.nonzero(~missing)
    by_label = np.zeros((labels.max() + 1, 3), dtype=np.uint8)
    by_label[labels[rows, columns]] = image[rows, columns]
    closed = image.copy()
    closed[missing] = by_label[labels[missing]]
    # A pinhole takes its neighbour's colour. A big part takes the garment's main colour: the nearest one would drag a flower or a white trim over the whole shoulder.
    share = cv2.GaussianBlur(broad.astype(np.float32), (0, 0), HOLE_SOFT)[:, :, None]
    reached = (cloth > 0) & ~missing
    if reached.sum() < REACHED_LEAST * (cloth > 0).sum():
        main, plain = np.array(ground), 1.0
    else:
        main, plain = main_colour(image[reached])
    # A striped garment goes on as it is: a blur or a single colour over the shoulder read as a stain (Léo, Seu Nelson).
    if plain < MAIN_PLAIN:
        return closed, float(wide.mean()), broad
    return (closed * (1 - share) + main * share).round().astype(np.uint8), float(wide.mean()), broad


def shoe_colour(shoes, skin):
    """Hex colours in, hex colour out."""
    rgb = [int(shoes[index:index + 2], 16) for index in (1, 3, 5)]
    near = sum((one - other) ** 2 for one, other in zip(rgb, skin)) ** 0.5 < SHOE_LIKE_SKIN
    return "#" + "".join(f"{round(value * SHOE_DARKER) if near else value:02x}" for value in rgb)
