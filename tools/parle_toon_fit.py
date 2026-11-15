"""
    License information: data/licenses/makehuman_license.txt

    A sculpted head of a kind taken from a generated one (owner, 2026-11-15: "você deve esculpir", then a quick test of a stylized
    head generated from text, its shape only). The hm08 base head is walked onto the generated head, coarse to fine, as the ARKit pack
    is walked onto the body (`parle_mouth_arkit.walk`), and what it moved becomes the kind's target (`parle_toon_sculpt`), which every
    body of that kind takes. Only the shape crosses: the mesh, the skeleton, the shapes of speech and the blinks stay the cast's.

    The eyes and the mouth are not laid onto the generated surface: there it has its eyes and lips as bumps on one skin, and the
    cast's lids and lips laid onto them would close the eyes and seal the mouth. Those zones (EYE_ZONE, MOUTH_ZONE about the base's
    own eye joints and mouth) only follow the move of the skin about them. Below the jaw the move fades out (NECK_FADE) so the neck
    joins the body as before.

      Blender -b -P tools/parle_toon_fit.py -- <generated.glb> <kind> [<marks stem of the base> <marks stem of the generated head>]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import parle_mouth_arkit as arkit  # noqa: E402
import parle_toon_sculpt as sculpt  # noqa: E402

# In metres on the base mesh: the reach of each eye's zone about its joint, the mouth's about its middle.
EYE_ZONE = 0.022
MOUTH_ZONE = 0.026
# The move is whole above the first height under the nose tip and gone at the second (metres): the neck stays.
NECK_FADE = (0.07, 0.12)
ROUNDS = (400, 200, 100, 50, 25, 12, 6, 3, 1)


def joint_centres(path=arkit.BASE_MESH):
    """The middles of the base's eye joints (helper cubes), in Blender axes."""
    points, group, groups = [], None, {}
    for line in Path(path).read_text().splitlines():
        if line.startswith("v "):
            points.append([float(value) for value in line.split()[1:4]])
        elif line.startswith("g "):
            group = line.split()[1]
        elif line.startswith("f ") and group in ("joint-l-eye", "joint-r-eye"):
            groups.setdefault(group, set()).update(int(item.split("/")[0]) - 1 for item in line.split()[1:])
    points = arkit.mh_to_blender(np.array(points))
    return {name: points[sorted(ids)].mean(axis=0) for name, ids in groups.items()}


def landmarks(points):
    """Crown (highest point), nose tip (furthest to the front in the upper middle of the face) and the head's width at the nose."""
    top = points[:, 2].max()
    tall = np.ptp(points[:, 2])
    middle = (np.abs(points[:, 0] - np.median(points[:, 0])) < 0.05 * tall) & (points[:, 2] > top - 0.75 * tall) & (points[:, 2] < top - 0.2 * tall)
    tip = points[middle][np.argmin(points[middle][:, 1])]
    level = np.abs(points[:, 2] - tip[2]) < 0.03 * tall
    return top, tip, float(np.ptp(points[level][:, 0]))


def rbf_warp(sources, targets):
    """`warp(points)`: a smooth 3D field through the marks (biharmonic kernel, r, with an affine part) taking each source to its target."""
    sources, targets = np.asarray(sources, float), np.asarray(targets, float)
    count = len(sources)
    kernel = np.linalg.norm(sources[:, None] - sources[None], axis=2)
    flat = np.hstack([np.ones((count, 1)), sources])
    system = np.zeros((count + 4, count + 4))
    system[:count, :count] = kernel
    system[:count, count:] = flat
    system[count:, :count] = flat.T
    solved = np.linalg.solve(system + 1e-9 * np.eye(count + 4), np.vstack([targets - sources, np.zeros((4, 3))]))

    def warp(points):
        points = np.asarray(points, float)
        far = np.linalg.norm(points[:, None] - sources[None], axis=2)
        return points + far @ solved[:count] + np.hstack([np.ones((len(points), 1)), points]) @ solved[count:]

    return warp


def fit(base, edges, head, surface_points, closest, eyes, mouth, eye_zone=EYE_ZONE, mouth_zone=MOUTH_ZONE):
    """The base head points walked onto the generated surface. `closest(points)` gives the nearest point of the surface for each."""
    keep = np.flatnonzero(head)
    renumber = np.full(len(base), -1)
    renumber[keep] = np.arange(len(keep))
    inner = renumber[edges]
    inner = inner[(inner >= 0).all(axis=1)]
    start = base[keep]
    weight = np.ones(len(keep))
    for centre in eyes:
        weight *= np.clip(np.linalg.norm(start - centre, axis=1) / eye_zone, 0, 1) ** 2
    weight *= np.clip(np.linalg.norm(start - mouth, axis=1) / mouth_zone, 0, 1) ** 2
    walked = start.copy()
    for rounds in ROUNDS:
        pull = (closest(walked) - walked) * weight[:, None]
        walked = walked + arkit.smooth(pull, inner, rounds) / np.maximum(arkit.smooth(weight[:, None], inner, rounds), 1e-3)
    out = base.copy()
    out[keep] = walked
    return out


def keep_eyes_and_mouth(base, sculpted, edges, eyes, mouth):
    """The cast's own lids and lips kept: inside each eye's and the mouth's zone (KEEP_EYE, KEEP_MOUTH, about the base's eye joints and
    mouth) the move is the smoothed move of the skin about it, not the generated head's, so they go with the face as one piece and keep
    the shape the blinks and the shapes of speech were made for (round of 2026-11-15: laid on the generated head, the lids covered
    the eyes and the lips came apart)."""
    move = sculpted - base
    keep = np.ones(len(base))
    for centre, reach in [(eye, KEEP_EYE) for eye in eyes] + [(mouth, KEEP_MOUTH)]:
        keep *= np.clip((np.linalg.norm(base - centre, axis=1) - reach) / (0.5 * reach), 0, 1)
    around = arkit.smooth(move * keep[:, None], edges, KEEP_ROUNDS) / np.maximum(arkit.smooth(keep[:, None], edges, KEEP_ROUNDS), 1e-4)
    return base + move * keep[:, None] + around * (1 - keep[:, None])


KEEP_EYE = 0.024
KEEP_MOUTH = 0.042
KEEP_MOUTH_BACK = 0.015
KEEP_ROUNDS = 120


def main(glb, kind, marks_base=None, marks_gen=None):
    """`marks_base` and `marks_gen` are the stems of `parle_toon_marks` shots of the base and of the generated head: with them the
    base is first bent onto the generated head's marks (eyes, nose, mouth, chin, brows, sides), and the eyes and mouth kept out of
    the walk are measured on those marks."""
    import bpy
    from mathutils.bvhtree import BVHTree

    import parle_toon_marks as marks

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=glb)
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    obj = max(meshes, key=lambda item: len(item.data.vertices))
    world = np.array(obj.matrix_world)
    raw = np.array([vertex.co for vertex in obj.data.vertices]) @ world[:3, :3].T + world[:3, 3]
    faces = [tuple(polygon.vertices) for polygon in obj.data.polygons]
    base, edges = arkit.read_base()
    head = arkit.head_of(base)
    joints = joint_centres()
    eyes = [joints["joint-l-eye"], joints["joint-r-eye"]]
    top_b, tip_b, wide_b = landmarks(base[head])
    top_g, tip_g, wide_g = landmarks(raw)
    # Placed by the crown and the nose: the generated head scaled so its crown-to-nose height is the base's, its nose tip on the base's.
    scale = (top_b - tip_b[2]) / max(top_g - tip_g[2], 1e-9)
    place = lambda points: (np.asarray(points, float) - tip_g) * scale + tip_b
    placed = place(raw)
    tree = BVHTree.FromPolygons([tuple(point) for point in placed], faces)
    closest = lambda points: np.array([tree.find_nearest(tuple(point))[0] or point for point in points])
    mouth = tip_b + np.array([0.0, 0.012, -0.034])
    start, eye_zone, mouth_zone = base, EYE_ZONE, MOUTH_ZONE
    if marks_base and marks_gen:
        base_tree = BVHTree.FromPolygons([tuple(point) for point in base], sculpt.base_faces())
        raw_tree = BVHTree.FromPolygons([tuple(point) for point in raw], faces)
        on_base, on_gen = marks.marks_3d(marks_base, base_tree, base), marks.marks_3d(marks_gen, raw_tree, raw)
        names = sorted(set(on_base) & set(on_gen))
        warp = rbf_warp([on_base[name] for name in names], [place(on_gen[name][None])[0] for name in names])
        bent = base.copy()
        bent[head] = warp(base[head])
        fade = np.clip((tip_b[2] - NECK_FADE[0] - base[:, 2]) / (NECK_FADE[1] - NECK_FADE[0]), 0, 1)
        start = bent + (base - bent) * fade[:, None]
        target = {name: place(on_gen[name][None])[0] for name in names}
        eyes = [(target["eye_l_out"] + target["eye_l_in"]) / 2, (target["eye_r_out"] + target["eye_r_in"]) / 2]
        eyes = [centre + np.array([0.0, 0.012, 0.0]) for centre in eyes]
        eye_zone = 0.85 * float(np.linalg.norm(target["eye_l_out"] - target["eye_l_in"]))
        mouth = (target["mouth_l"] + target["mouth_r"]) / 2
        mouth_zone = 0.8 * float(np.linalg.norm(target["mouth_l"] - target["mouth_r"]))
        print("MARCOS usados:", len(names), "zona do olho", round(eye_zone * 1000, 1), "mm, da boca", round(mouth_zone * 1000, 1), "mm")
    walked = fit(start, edges, head, placed, closest, eyes, mouth, eye_zone, mouth_zone)
    fade = np.clip((tip_b[2] - NECK_FADE[0] - base[:, 2]) / (NECK_FADE[1] - NECK_FADE[0]), 0, 1)
    sculpted = walked + (base - walked) * fade[:, None]
    lips = tip_b + np.array([0.0, 0.012, -0.034])
    if marks_base and marks_gen and "mouth_l" in on_base and "mouth_r" in on_base:
        # The middle of the base's own mouth, a little behind the lips so the zone takes the lining inside it too.
        lips = (on_base["mouth_l"] + on_base["mouth_r"]) / 2 + np.array([0.0, KEEP_MOUTH_BACK, 0.0])
    sculpted = keep_eyes_and_mouth(base, sculpted, edges, [joints["joint-l-eye"], joints["joint-r-eye"]], lips)
    lines = sculpt.target_lines(base, sculpted, head)
    out = sculpt.FOLDER / f"{kind}.target"
    out.write_text(f"# The head of the kind {kind}, taken from a generated head (tools/parle_toon_fit.py), base mesh hm08.\n" + "\n".join(lines) + "\n")
    print("ENCAIXE", kind, out, len(lines), "pontos; escala", round(scale, 3), "largura gerada/base", round(wide_g * scale / wide_b, 3))


if __name__ == "__main__" and "--" in sys.argv:
    main(*sys.argv[sys.argv.index("--") + 1:][:4])
