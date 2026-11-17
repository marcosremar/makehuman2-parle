#!/usr/bin/env python3
"""
    License information: data/licenses/makehuman_license.txt

    Makes a new garment out of an existing one, as a real .mhclo asset: trousers cut into
    shorts, a shirt cut into a tank top. Being an asset, it fits any body like every other
    piece, instead of being cut again on each character after export.

    Every point of the new mesh keeps a binding to the base mesh. A point born on a cut takes
    the three base vertices of its nearest end, and the offset makes up the difference.

      python tools/derive_garment.py <clothes-dir> <source> <name> --shorts male|female [--tank]
      python tools/derive_garment.py <clothes-dir> toigo_shift_dress parle_apron --apron
      python tools/derive_garment.py <clothes-dir> toigo_female_suit parle_jacket --jacket
      python tools/derive_garment.py <clothes-dir> toigo_shift_dress parle_badge --badge
"""
import argparse
import json
import os
import shutil
import uuid

import numpy as np

# Heights in metres on the base mesh; the shares are the ones of the game cast (knee 0, hip 1).
HEM_SHARE = {"male": 0.18, "female": 0.3}
HEM_RADIUS_M = {"male": 0.098, "female": 0.088}
HIP_EASE_M = 0.008
HEM_FOLD_M = 0.014
HEM_ZONE_M = 0.012
SLEEVE_INSET_M = 0.004
ARMHOLE_TILT = 0.15
SLEEVE_ZONE_M = 0.16
SMALLEST_ISLAND = 8
ON_PLANE = 1e-5
DECIMETRES = 10.0
# The work apron of who serves at a counter, cut from the front of a straight dress (gauntlet of the shop interiors,
# 2026-11-17: the vision critic saw the baker, the waitress and the barista in a plain T-shirt, «no apron»). Heights come
# from the joints of the base mesh: the waist band sits at this share from spine03 to spine02 (the navel line).
APRON_WAIST_SHARE = 0.3
APRON_BAND_M = 0.018
# A bib of 26 cm and a skirt that wraps a little less than the front half: an apron is tied over the sides, not round them.
APRON_BIB_HALF_M = 0.13
APRON_SIDE_M = 0.012
# Above the band the back keeps two straps, from the shoulder down to the band, between these distances from the spine.
APRON_STRAP_IN_M = 0.07
APRON_STRAP_OUT_M = 0.11
# Front and back of the torso split at this depth (+z is the front of the base mesh).
APRON_FRONT_M = 0.02
# Over the shirt: the dress was cut to sit on the skin. 8 mm let the baker's T-shirt through at the belly in the idle clip.
APRON_OVER_M = 0.012
# From the hips up the apron lies this far off the body and forgets the dress's own shape: cut from a woman's dress, the
# baker's apron had two mounds on his chest and the barista's a lumpy belly where the T-shirt came through (2026-11-17).
# The change fades in over the first centimetres above the hip joints; below them the skirt hangs like the dress's.
APRON_BIB_GAP_M = 0.006
APRON_BIB_FADE_M = 0.03
# An offset shorter than this (in decimetres, 1 mm) lies on the body and does not tell which side the cloth is on.
ON_BODY = 0.01
# The jacket of a suit, without its skirt or trousers, worn open over the shirt: the white coat of the pharmacist and the
# cardigan of the bookseller (the same critic: «jaleco branco», «cardigã ou blusa»).
JACKET_OVER_M = 0.004
# A name badge on the left of the chest, 7 by 4 cm, over whatever the person wears on top (the critic: «crachá»). The
# centre sits at this share from the shoulder joint down to spine01, and this far out from the middle of the chest.
BADGE_HALF_M = (0.035, 0.02)
BADGE_DOWN_SHARE = 0.75
BADGE_OUT_M = 0.085
BADGE_OVER_M = 0.016


class Garment:
    def __init__(self, folder, name):
        self.head, self.points, self.uvs, self.faces = [], [], [], []
        with open(os.path.join(folder, f"{name}.mhclo"), encoding="utf-8") as lines:
            found = [line.split()[1] for line in lines if line.startswith("obj_file")]
        with open(os.path.join(folder, found[0] if found else f"{name}.obj"), encoding="utf-8") as lines:
            for line in lines:
                word = line.split(" ", 1)[0]
                if word == "v":
                    self.points.append([float(item) for item in line.split()[1:4]])
                elif word == "vt":
                    self.uvs.append([float(item) for item in line.split()[1:3]])
                elif word == "f":
                    corners = [item.split("/") for item in line.split()[1:]]
                    self.faces.append([(int(item[0]) - 1, int(item[1]) - 1) for item in corners])
                else:
                    self.head.append(line)
        self.points = np.array(self.points)
        self.uvs = np.array(self.uvs)
        self.front, rows = [], []
        with open(os.path.join(folder, f"{name}.mhclo"), encoding="utf-8") as lines:
            reading = False
            for line in lines:
                if line.startswith("verts"):
                    reading = True
                elif line.startswith("delete_verts"):
                    break
                elif reading and line.strip():
                    rows.append(line.split())
                elif not reading:
                    self.front.append(line)
        self.triple = np.array([row[:3] for row in rows], dtype=np.int64)
        self.weights = np.array([row[3:6] for row in rows], dtype=np.float64)
        self.offsets = np.array([row[6:9] for row in rows], dtype=np.float64)

    def bind_like(self, source, position):
        """A new point: the binding of `source`, moved to `position` by the offset alone."""
        index = len(self.points)
        self.triple = np.vstack([self.triple, self.triple[source]])
        self.weights = np.vstack([self.weights, self.weights[source]])
        self.offsets = np.vstack([self.offsets, self.offsets[source] + position - self.points[source]])
        self.points = np.vstack([self.points, position])
        return index

    def cut(self, point, normal, chosen=None):
        """Keeps what is under the plane, on the faces `chosen` lets through."""
        side = (self.points - point) @ normal
        side[np.abs(side) < ON_PLANE] = 0
        points, uvs = {}, {}

        def crossing(first, second):
            key = tuple(sorted((first[0], second[0])))
            share = side[first[0]] / (side[first[0]] - side[second[0]])
            if key not in points:
                near = first[0] if abs(side[first[0]]) < abs(side[second[0]]) else second[0]
                points[key] = self.bind_like(near, self.points[first[0]] * (1 - share) + self.points[second[0]] * share)
            mark = tuple(sorted((first[1], second[1])))
            if mark not in uvs:
                uvs[mark] = len(self.uvs)
                self.uvs = np.vstack([self.uvs, self.uvs[first[1]] * (1 - share) + self.uvs[second[1]] * share])
            return points[key], uvs[mark]

        kept = []
        for face in self.faces:
            sides = side[[corner[0] for corner in face]]
            if (chosen is not None and not chosen(face)) or (sides <= 0).all():
                kept.append(face)
            elif (sides < 0).any():
                clipped = []
                for place, corner in enumerate(face):
                    after = face[(place + 1) % len(face)]
                    if side[corner[0]] <= 0:
                        clipped.append(corner)
                    if side[corner[0]] * side[after[0]] < 0:
                        clipped.append(crossing(corner, after))
                kept.append(clipped)
        self.faces = kept

    def drop_islands(self):
        owner = list(range(len(self.points)))

        def find(item):
            while owner[item] != item:
                owner[item] = owner[owner[item]]
                item = owner[item]
            return item

        for face in self.faces:
            for corner in face[1:]:
                owner[find(corner[0])] = find(face[0][0])
        roots = [find(face[0][0]) for face in self.faces]
        sizes = {root: roots.count(root) for root in set(roots)}
        self.faces = [face for face, root in zip(self.faces, roots) if sizes[root] >= SMALLEST_ISLAND]

    def fold_hem(self, height):
        """Turns the hem inwards, so that the inside of the leg is not seen from below."""
        uses = {}
        for face in self.faces:
            for place, corner in enumerate(face):
                after = face[(place + 1) % len(face)]
                uses.setdefault(tuple(sorted((corner[0], after[0]))), []).append((corner, after))
        twins = {}

        def twin(corner):
            if corner[0] not in twins:
                twins[corner[0]] = self.bind_like(corner[0], self.points[corner[0]] + np.array([0, HEM_FOLD_M * DECIMETRES, 0]))
            return twins[corner[0]], corner[1]

        for owners in list(uses.values()):
            corner, after = owners[0]
            if len(owners) == 1 and max(self.points[corner[0], 1], self.points[after[0], 1]) < height + HEM_ZONE_M * DECIMETRES:
                self.faces.append([after, corner, twin(corner), twin(after)])

    def write(self, folder, name):
        os.makedirs(folder, exist_ok=True)
        # A binary of an earlier cut would stand for this one while it is the newer file.
        stale = os.path.join(folder, f"{name}.mhbin")
        if os.path.exists(stale):
            os.remove(stale)
        used = sorted({corner[0] for face in self.faces for corner in face})
        marks = sorted({corner[1] for face in self.faces for corner in face})
        new_point = {old: new for new, old in enumerate(used)}
        new_mark = {old: new for new, old in enumerate(marks)}
        with open(os.path.join(folder, f"{name}.obj"), "w", encoding="utf-8") as out:
            out.write("".join(self.head))
            out.writelines(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in self.points[used])
            out.writelines(f"vt {u:.6f} {v:.6f}\n" for u, v in self.uvs[marks])
            for face in self.faces:
                out.write("f " + " ".join(f"{new_point[item[0]] + 1}/{new_mark[item[1]] + 1}" for item in face) + "\n")
        front = []
        for line in self.front:
            word = line.split(" ", 1)[0]
            if word in ("uuid", "name", "obj_file", "material"):
                value = {"uuid": uuid.uuid5(uuid.NAMESPACE_DNS, name), "name": name, "obj_file": f"{name}.obj", "material": f"{name}.mhmat"}[word]
                line = f"{word} {value}\n"
            front.append(line)
        with open(os.path.join(folder, f"{name}.mhclo"), "w", encoding="utf-8") as out:
            out.write("".join(front) + "verts 0\n")
            for index in used:
                triple, weights, offsets = self.triple[index], self.weights[index], self.offsets[index]
                out.write(" ".join(str(item) for item in triple) + " " + " ".join(f"{item:.5f}" for item in [*weights, *offsets]) + "\n")


class Bones:
    """Joints of the base mesh, in its own decimetres and Y up."""

    def __init__(self, data):
        points = []
        with open(os.path.join(data, "base", "hm08", "base.obj"), encoding="utf-8") as lines:
            points = [[float(item) for item in line.split()[1:4]] for line in lines if line.startswith("v ")]
        self.points = np.array(points)
        with open(os.path.join(data, "rigs", "hm08", "default.mhskel"), encoding="utf-8") as skeleton:
            self.rig = json.load(skeleton)
        self.data = data

    def normals(self):
        """Vertex normals of the base mesh, its helpers included: a dress is bound to the tights helper round the hips."""
        faces = []
        with open(os.path.join(self.data, "base", "hm08", "base.obj"), encoding="utf-8") as lines:
            for line in lines:
                if line.startswith("f "):
                    faces.append([int(item.split("/")[0]) - 1 for item in line.split()[1:]])
        normals = np.zeros_like(self.points)
        for face in faces:
            for place in range(1, len(face) - 1):
                first, second, third = self.points[[face[0], face[place], face[place + 1]]]
                normal = np.cross(second - first, third - first)
                for corner in (face[0], face[place], face[place + 1]):
                    normals[corner] += normal
        return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)

    def head(self, bone):
        return self.points[self.rig["joints"][self.rig["bones"][bone]["head"]]].mean(axis=0)


def shorts(garment, bones, sex):
    hip, knee = bones.head("upperleg01.L"), bones.head("lowerleg01.L")
    hem = knee[1] + (hip[1] - knee[1]) * HEM_SHARE[sex]
    garment.cut(np.array([0, hem, 0]), np.array([0.0, -1.0, 0.0]))
    garment.drop_islands()
    radius = HEM_RADIUS_M[sex] * DECIMETRES
    for index, point in enumerate(garment.points):
        if point[1] > hip[1]:
            continue
        depth = min(1.0, (hip[1] - point[1]) / max(hip[1] - hem, 1e-3))
        way = np.array([point[0] - np.sign(point[0] or 1) * abs(hip[0]), 0, point[2]])
        now = np.linalg.norm(way)
        wanted = (now + HIP_EASE_M * DECIMETRES) * (1 - depth) + radius * depth
        if now > 1e-3 and wanted > now:
            shift = way / now * (wanted - now)
            garment.points[index] += shift
            garment.offsets[index] += shift
    garment.fold_hem(hem)


def tank(garment, bones):
    joint = bones.head("upperarm01.L")
    floor = joint[1] - SLEEVE_ZONE_M * DECIMETRES
    for side in (1.0, -1.0):
        at = np.array([side * (abs(joint[0]) - SLEEVE_INSET_M * DECIMETRES), joint[1], joint[2]])
        outward = np.array([side, ARMHOLE_TILT, 0.0])
        points = garment.points

        def zone(face, points=points):
            return (points[[corner[0] for corner in face], 1] > floor).all()

        garment.cut(at, outward / np.linalg.norm(outward), zone)
    garment.drop_islands()


def apron(garment, bones):
    """Bib, straps, waist band and the front of the skirt: the rest of the dress goes."""
    low, high = bones.head("spine03")[1], bones.head("spine02")[1]
    waist = low + (high - low) * APRON_WAIST_SHARE
    band = APRON_BAND_M * DECIMETRES

    def region(test):
        points = garment.points

        def chosen(face, points=points):
            corners = points[[corner[0] for corner in face]]
            return test(corners)

        return chosen

    below = region(lambda corners: (corners[:, 1] < waist - band).all())
    garment.cut(np.array([0.0, 0.0, APRON_SIDE_M * DECIMETRES]), np.array([0.0, 0.0, -1.0]), below)
    front = APRON_FRONT_M * DECIMETRES
    bib = APRON_BIB_HALF_M * DECIMETRES
    for side in (1.0, -1.0):
        chest = region(lambda corners: (corners[:, 1] > waist + band).all() and corners[:, 2].mean() >= front)
        garment.cut(np.array([side * bib, 0.0, 0.0]), np.array([side, 0.0, 0.0]), chest)
    inner, outer = APRON_STRAP_IN_M * DECIMETRES, APRON_STRAP_OUT_M * DECIMETRES
    for side in (1.0, -1.0):
        back = region(lambda corners: (corners[:, 1] > waist + band).all() and corners[:, 2].mean() < front)
        garment.cut(np.array([side * outer, 0.0, 0.0]), np.array([side, 0.0, 0.0]), back)
        half = region(lambda corners, side=side: (corners[:, 1] > waist + band).all() and corners[:, 2].mean() < front
                      and side * corners[:, 0].mean() > 0)
        garment.cut(np.array([side * inner, 0.0, 0.0]), np.array([-side, 0.0, 0.0]), half)
    garment.drop_islands()
    flat_bib(garment, bones, bones.head("upperleg01.L")[1])
    over(garment, APRON_OVER_M * DECIMETRES)


def flat_bib(garment, bones, bottom):
    """Above `bottom` every point sits at a fixed gap over the body it is bound to, not where the dress's cups put it."""
    scale = np.ones(3)
    for line in garment.front:
        word = line.split()[0] if line.split() else ""
        if word in ("x_scale", "y_scale", "z_scale"):
            first, second, length = line.split()[1:4]
            axis = "xyz".index(word[0])
            scale[axis] = abs(bones.points[int(first), axis] - bones.points[int(second), axis]) / float(length)
    normal = outward(garment, bones)
    wanted = normal * APRON_BIB_GAP_M * DECIMETRES / scale
    share = np.clip((garment.points[:, 1] - bottom) / (APRON_BIB_FADE_M * DECIMETRES), 0, 1)[:, None]
    shift = (wanted - garment.offsets) * share
    garment.offsets += shift
    garment.points += shift * scale


def jacket(garment):
    """Keeps the biggest piece, the jacket of a suit: the skirt, the blouse and each button are other pieces of the mesh."""
    owner = list(range(len(garment.points)))

    def find(item):
        while owner[item] != item:
            owner[item] = owner[owner[item]]
            item = owner[item]
        return item

    for face in garment.faces:
        for corner in face[1:]:
            owner[find(corner[0])] = find(face[0][0])
    roots = [find(face[0][0]) for face in garment.faces]
    biggest = max(set(roots), key=roots.count)
    garment.faces = [face for face, root in zip(garment.faces, roots) if root == biggest]
    over(garment, JACKET_OVER_M * DECIMETRES)


def badge(garment, bones):
    """A small patch of the front of the chest, on the wearer's left (+x), lifted over the clothes."""
    shoulder, chest = bones.head("upperarm01.L")[1], bones.head("spine01")[1]
    middle = np.array([BADGE_OUT_M * DECIMETRES, shoulder + (chest - shoulder) * BADGE_DOWN_SHARE])
    half_x, half_y = (item * DECIMETRES for item in BADGE_HALF_M)
    for normal, edge in (((1.0, 0.0), middle[0] + half_x), ((-1.0, 0.0), -(middle[0] - half_x)),
                         ((0.0, 1.0), middle[1] + half_y), ((0.0, -1.0), -(middle[1] - half_y))):
        points = garment.points

        def front(face, points=points):
            return points[[corner[0] for corner in face], 2].mean() > 0

        way = np.array([normal[0], normal[1], 0.0])
        garment.cut(way * edge, way, front)
        garment.faces = [face for face in garment.faces if front(face, garment.points)]
    over(garment, BADGE_OVER_M * DECIMETRES)


def outward(garment, bones):
    """The normal of the base mesh under every point, turned to the side the cloth is on: round the hips a dress is bound to the
    tights helper, whose faces run the other way, and the bib went into the body there. The cloth sits outside the body, so its
    offset tells the side; where the offset lies flat on the body, the side the cloth faces does (the turned-in hem faces in)."""
    normals = bones.normals()
    normal = (normals[garment.triple] * garment.weights[:, :, None]).sum(axis=1)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    away = (normal * garment.offsets).sum(axis=1)
    facing = np.where(np.abs(away) > ON_BODY, away, (normal * cloth_normals(garment)).sum(axis=1))
    return normal * np.where(facing < 0, -1.0, 1.0)[:, None]


def cloth_normals(garment):
    """Normal of the cloth at every point it uses (zero elsewhere)."""
    normals = np.zeros_like(garment.points)
    for face in garment.faces:
        corners = [corner[0] for corner in face]
        for place in range(1, len(corners) - 1):
            first, second, third = garment.points[[corners[0], corners[place], corners[place + 1]]]
            normal = np.cross(second - first, third - first)
            for corner in (corners[0], corners[place], corners[place + 1]):
                normals[corner] += normal
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    return np.where(length > 1e-12, normals / np.maximum(length, 1e-12), 0)


def over(garment, gap):
    """Moves every point out along the normal of the cloth, the binding with it."""
    used = sorted({corner[0] for face in garment.faces for corner in face})
    shift = cloth_normals(garment) * gap
    garment.points[used] += shift[used]
    garment.offsets[used] += shift[used]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("clothes")
    parser.add_argument("source")
    parser.add_argument("name")
    parser.add_argument("--shorts", choices=tuple(HEM_SHARE))
    parser.add_argument("--tank", action="store_true")
    parser.add_argument("--apron", action="store_true")
    parser.add_argument("--jacket", action="store_true")
    parser.add_argument("--badge", action="store_true")
    parser.add_argument("--data", default="data")
    args = parser.parse_args()
    home = os.path.join(args.clothes, args.source)
    garment = Garment(home, args.source)
    bones = Bones(args.data)
    if args.tank:
        tank(garment, bones)
    if args.shorts:
        shorts(garment, bones, args.shorts)
    if args.apron:
        apron(garment, bones)
    if args.jacket:
        jacket(garment)
    if args.badge:
        badge(garment, bones)
    target = os.path.join(args.clothes, args.name)
    garment.write(target, args.name)
    with open(os.path.join(home, f"{args.source}.mhclo"), encoding="utf-8") as lines:
        named = [line.split()[1] for line in lines if line.startswith("material")]
    with open(os.path.join(home, named[0] if named else f"{args.source}.mhmat"), encoding="utf-8") as lines:
        material = lines.read().replace(f"name {args.source}", f"name {args.name}")
    with open(os.path.join(target, f"{args.name}.mhmat"), "w", encoding="utf-8") as out:
        out.write(material)
    for item in os.listdir(home):
        if item.endswith(".png") and not os.path.exists(os.path.join(target, item)):
            shutil.copy(os.path.join(home, item), target)
    print("DERIVED", args.name, len(garment.faces), "faces")


if __name__ == "__main__":
    main()
