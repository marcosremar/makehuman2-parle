"""
    License information: data/licenses/makehuman_license.txt

    How alike the two halves of a face are, by measure, from its portrait from the front.

    Owner, 2026-11-12: "deve adicionar cálculos de simetria de beleza. busca na internet teorias sobre isso e implementa".

    What was read (2026-11-12; each DOI checked against its record, and the method on the page of the paper where it says "read"):
      - Symmetry counts for little, and less than was thought. Corrected for the studies that were never published the effect falls
        by 30 to 50%, and in the studies of more than 100 faces it is next to nothing: Van Dongen (2011), "Associations between
        asymmetry and human attractiveness: possible direct effects of asymmetry and signatures of publication bias", Annals of Human
        Biology 38(3):317-323, DOI 10.3109/03014460.2010.544676 (abstract). On 1550 faces of ten peoples it had no effect, and how far
        a face stands from the average had: Kleisner et al. (2024), "Distinctiveness and femininity, rather than symmetry and
        masculinity, affect facial attractiveness across the world", Evolution and Human Behavior 45(1):82-90, DOI
        10.1016/j.evolhumbehav.2023.10.001 (abstract).
      - A face made of one half and its mirror image is found less attractive than the face it came from: Swaddle & Cuthill (1995),
        "Asymmetry and human facial attractiveness: symmetry may not always be beautiful", Proceedings of the Royal Society B
        261(1360):111-116, DOI 10.1098/rspb.1995.0124; Zaidel & Deblieck (2007), "Attractiveness of natural faces compared to computer
        constructed perfectly symmetrical faces", International Journal of Neuroscience 117(4):423-431, DOI 10.1080/00207450600581928
        (abstracts). So no measure here asks for less unevenness than a face can have; they are read, and marked only by
        `parle_beauty` where it chooses to.
      - `sym_sides`, the index of Little, Jones, Waitt et al. (2008), "Symmetry is related to sexual dimorphism in faces: data across
        culture and species", PLoS ONE 3(5):e2106, DOI 10.1371/journal.pone.0002106 (read): "symmetry was calculated by taking left
        and right deviation from the midline, calculated from inter-pupillary distance, for points D1-D6 and then summing the absolute
        value of individual scores", on images "normalised on inter-pupillary distance". For each of six pairs of points, how far the
        right one stands from the line down the middle less how far the left one does; the absolute values summed. Their six pairs are
        in a figure that was not read: ours are the corners of the eyes, outside and inside, the cheekbones, the wings of the nose,
        the corners of the mouth and the jaw. On their photographs it averaged 50 with a deviation of 29, in the units of their
        normalised picture; here it is a share of the distance between the pupils.
      - `sym_size`, after Simmons, Rhodes, Peters & Koehler (2004), "Are human preferences for facial symmetry focused on signals of
        developmental instability?", Behavioral Ecology 15(5):864-871, DOI 10.1093/beheco/arh099 (read): each trait's unsigned
        difference between the sides "scaled for trait size by dividing the unsigned value of FA by mean trait size", then "summed
        across traits, and divided by the total number of traits". They had 35 distances a side; whether the mean size is the face's
        own or the sample's the text read does not say. Here six traits, each over the mean of its two sides. On their faces what
        was measured so told 17 to 22% of how symmetric a face looked.
      - `sym_midline` and `sym_level` are ours, for what the studies found to weigh most where unevenness weighs at all: the middle of
        the face (Springer et al. 2007, "Facial attractiveness: visual impact of symmetry increases significantly towards the
        midline", Annals of Plastic Surgery 59(2):156-162, DOI 10.1097/01.sap.0000252041.66540.ec, abstract) and the tilt of the
        corners of the mouth and of the base of the nose (Hatch, Wehby, Nidey & Moreno Uribe 2017, "Effects of objective
        3-dimensional measures of facial shape and symmetry on perceptions of facial attractiveness", Journal of Oral and
        Maxillofacial Surgery 75(9):1958-1970, DOI 10.1016/j.joms.2017.04.042, read).
    Not read, so not used: the two indices of Grammer & Thornhill (1994, Journal of Comparative Psychology 108(3):233-242, DOI
    10.1037/0735-7036.108.3.233), whose method is only in the paper itself.

    What the faces of the game said (2026-11-12): on the 32 the owner finds beautiful and the 12 he faults, none of these measures
    tells one from the other (the faulted ones are, if anything, a little more even). All of them are faces built by mirroring, 0.02
    to 0.03 mm from their own mirror image on the mesh (`parle_mirror`), where a real face stands 0.8 to 1 mm; what these measures
    read on a portrait is mostly how much the points of the Face Mesh tremble from one portrait to the next. They are kept as a
    watch on a body that comes out crooked, not as a mark of beauty.

    Read on the points of the Face Mesh (MediaPipe Face Landmarker, 478 points, the irises among them) of a portrait from the front
    with the head upright: the portraits of the cast are taken that way, so no turn is taken out first (taking the line of the eyes
    for level would hide an eye that sits higher than the other). Numpy alone.
"""
import numpy as np

# Face Mesh points. Left and right are the picture's.
CHEEKS = (234, 454)
PUPILS = (468, 473)
# The six pairs across the face, top to bottom, each a point and its twin on the other side: outer corners of the eyes, inner
# corners, cheekbones, wings of the nose, corners of the mouth, jaw.
PAIRS = ((33, 263), (133, 362), (234, 454), (48, 278), (61, 291), (172, 397))
# Twins whose heights are compared: corners of the eyes, lids, eyebrows (inner, middle and outer), wings of the nose, corners of the
# mouth, cheekbones, jaw.
LEVEL = ((33, 263), (133, 362), (159, 386), (145, 374), (107, 336), (105, 334), (70, 300), (48, 278), (61, 291), (234, 454), (172, 397))
# The points the Face Mesh has down the middle of the face, from the top of the forehead to the chin.
MIDDLE = (10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 2, 164, 0, 17, 18, 200, 199, 175, 152)
# One eye: outer corner, inner corner, upper lid, lower lid, eyebrow above it.
EYES = ((33, 133, 159, 145, 105), (263, 362, 386, 374, 334))
NOSE, MOUTH, JAW = (48, 278), (61, 291), (172, 397)
NAMES = {
    "sym_sides": "assimetria dos seis pares contra a linha do meio (Little et al. 2008)",
    "sym_level": "um lado mais alto que o outro",
    "sym_midline": "linha do meio torta (nariz, lábios, queixo)",
    "sym_size": "um lado maior que o outro (olho, sobrancelha, meia boca, meio nariz, meia mandíbula)",
}


def measures(points):
    """The measures of how uneven a face is, from its Face Mesh points in the pixels of a portrait from the front. All are nothing
    on a face whose halves mirror each other, and grow with the difference. `sym_sides` is a share of the distance between the
    pupils, `sym_level` and `sym_midline` of the width of the face between the cheekbones, `sym_size` of the size of each trait."""
    spot = lambda index: np.asarray(points[index][:2], float)
    wide = float(np.linalg.norm(spot(CHEEKS[0]) - spot(CHEEKS[1])))
    between = float(np.linalg.norm(spot(PUPILS[0]) - spot(PUPILS[1])))
    middle = (spot(PUPILS[0])[0] + spot(PUPILS[1])[0]) / 2
    sides = float(sum(abs((spot(b)[0] - middle) - (middle - spot(a)[0])) for a, b in PAIRS))
    level = float(np.mean([abs(spot(a)[1] - spot(b)[1]) for a, b in LEVEL]))
    # Down the middle: how far the points of the nose, the lips and the chin stand from the line through the middle of the pupils.
    crooked = float(np.mean([abs(spot(index)[0] - middle) for index in MIDDLE]))
    sizes = []
    for (outer, inner, up, low, brow), (other_outer, other_inner, other_up, other_low, other_brow) in (EYES,):
        sizes += [(np.linalg.norm(spot(outer) - spot(inner)), np.linalg.norm(spot(other_outer) - spot(other_inner))),
                  (np.linalg.norm(spot(up) - spot(low)), np.linalg.norm(spot(other_up) - spot(other_low))),
                  (spot(up)[1] - spot(brow)[1], spot(other_up)[1] - spot(other_brow)[1])]
    for a, b in (NOSE, MOUTH, JAW):
        sizes.append((middle - spot(a)[0], spot(b)[0] - middle))
    size = float(np.mean([abs(one - other) / max((abs(one) + abs(other)) / 2, 1e-9) for one, other in sizes]))
    return {"sym_sides": sides / max(between, 1e-9), "sym_level": level / wide, "sym_midline": crooked / wide, "sym_size": size}
