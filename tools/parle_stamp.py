"""
    License information: data/licenses/makehuman_license.txt

    Which pieces of a game body paint the print of a new garment. Whatever sits on top of the cloth or
    under it in the old body is not print: a necklace baked into a shirt reads as a torn collar, a
    bikini under a tank top as a red stain, and the cord of the shorts as a grey patch.
"""
NOT_PRINT = ("Necklace", "Chain", "Crucifix", "Shorts_Cord")
UNDERWEAR = "Bikini"


def stamp_sources(names, prefixes):
    wanted = [name for name in names if name.startswith(prefixes) and not name.startswith(NOT_PRINT)]
    covering = [name for name in wanted if not name.startswith(UNDERWEAR)]
    return covering or wanted
