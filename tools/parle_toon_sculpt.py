"""
    License information: data/licenses/makehuman_license.txt

    The hand-sculpted head of each kind of person (owner, 2026-11-15: "cabeça base à mão"). Nothing a field could do came near the
    series he chose (5 in 10 at best); a person sculpts instead, once per kind, and the fork carries the sculpture to every body of that
    kind. Two steps, both inside Blender:

      Blender -b -P tools/parle_toon_sculpt.py -- template <kind> <out.blend> [<reference.glb> ...] [<picture.png|jpg> ...]
          A file to sculpt in: the hm08 base mesh (body points only, in their own order) as "Cabeca", with a "estilo" shape key
          ready to sculpt and the rest of the body hidden from the brush by a mask; beside it, for sight only, a body of the cast
          of that kind (a GLB) and pictures of the style (image planes). Neither reference goes anywhere.
      Blender -b -P tools/parle_toon_sculpt.py -- export <sculpted.blend> <kind>
          Writes data/parle_toon/<kind>.target: what "estilo" moves against the base, head points only, as MakeHuman writes a
          target (index dx dy dz, decimetres, y up). `parle_toon` mixes it into every body of that kind, whole.

    The kinds are those of `parle_style.kind_of`: male, male-old, female, female-old.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import parle_mouth_arkit as arkit  # noqa: E402
import parle_style  # noqa: E402

FOLDER = Path(__file__).resolve().parents[1] / "data" / "parle_toon"
KEY = "estilo"
# Points that move less than this are not written: a target lists only what moves.
LEAST_M = 1e-5


def base_faces(path=arkit.BASE_MESH):
    faces = []
    for line in Path(path).read_text().splitlines():
        if line.startswith("f "):
            ring = [int(item.split("/")[0]) - 1 for item in line.split()[1:]]
            if max(ring) < arkit.BODY_POINTS:
                faces.append(ring)
    return faces


def blender_to_mh(points):
    """The inverse of `parle_mouth_arkit.mh_to_blender`: Blender z-up metres facing -y to MakeHuman y-up decimetres."""
    return np.stack([points[:, 0], points[:, 2], -points[:, 1]], axis=1) * 10.0


def target_lines(rest, sculpted, head):
    """The lines of a target: each head point that moved, by its base index, in MakeHuman's axes and units."""
    move = blender_to_mh(sculpted - rest)
    keep = np.flatnonzero(head & (np.linalg.norm(sculpted - rest, axis=1) > LEAST_M))
    return [f"{index} {move[index, 0]:.5f} {move[index, 1]:.5f} {move[index, 2]:.5f}" for index in keep]


def template(kind, out, refs):
    import bpy

    parle_style.kind_of(kind.split("-")[0], {"age": "old"} if kind.endswith("-old") else {})
    bpy.ops.wm.read_factory_settings(use_empty=True)
    points, _ = arkit.read_base()
    mesh = bpy.data.meshes.new("Cabeca")
    mesh.from_pydata(points.tolist(), [], base_faces())
    obj = bpy.data.objects.new("Cabeca", mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.shape_key_add(name="Basis")
    obj.shape_key_add(name=KEY, from_mix=False).value = 1.0
    obj.active_shape_key_index = 1
    obj["parle_kind"] = kind
    # The brush reaches the head alone: everything below it is masked.
    group = obj.vertex_groups.new(name="cabeca")
    group.add([int(index) for index in np.flatnonzero(arkit.head_of(points))], 1.0, "REPLACE")
    for path in refs:
        if path.lower().endswith(".glb"):
            before = set(bpy.data.objects)
            bpy.ops.import_scene.gltf(filepath=path)
            for item in set(bpy.data.objects) - before:
                item.location.x += 0.45
                item.hide_select = True
        else:
            image = bpy.data.images.load(path)
            empty = bpy.data.objects.new(Path(path).stem, None)
            empty.empty_display_type = "IMAGE"
            empty.data = image
            empty.empty_display_size = 0.3
            empty.location = (-0.45 - 0.32 * sum(1 for item in bpy.data.objects if item.type == "EMPTY"), 0.0, float(points[:, 2].max()) - 0.15)
            empty.rotation_euler = (1.5708, 0.0, 0.0)
            bpy.context.scene.collection.objects.link(empty)
    bpy.ops.wm.save_as_mainfile(filepath=out)
    print("MOLDE", kind, out)


def export(path, kind):
    import bpy

    bpy.ops.wm.open_mainfile(filepath=path)
    obj = bpy.data.objects["Cabeca"]
    keys = obj.data.shape_keys.key_blocks

    def points(block):
        flat = np.empty(len(block.data) * 3)
        block.data.foreach_get("co", flat)
        return flat.reshape(-1, 3)

    rest = points(keys["Basis"])
    lines = target_lines(rest, points(keys[KEY]), arkit.head_of(rest))
    FOLDER.mkdir(exist_ok=True)
    out = FOLDER / f"{kind}.target"
    out.write_text(f"# The hand-sculpted head of the kind {kind} (tools/parle_toon_sculpt.py), base mesh hm08.\n" + "\n".join(lines) + "\n")
    print("ALVO", kind, out, len(lines), "pontos")


if __name__ == "__main__" and "--" in sys.argv:
    args = sys.argv[sys.argv.index("--") + 1:]
    if args[0] == "template":
        template(args[1], args[2], args[3:])
    elif args[0] == "export":
        export(args[1], args[2])
