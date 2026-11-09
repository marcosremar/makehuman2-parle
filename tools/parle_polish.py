"""
    License information: data/licenses/makehuman_license.txt

    Finishing touches of the Parle cast's look, run inside Blender on a mounted character, what was judged an improvement
    next to the player and Rafa (owner, 2026-11-08). Every shape change moves all shape keys, so blinking and speaking keep working.
"""
import glob
import os

import bpy
import numpy as np

CLOTHES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".local/mhd/data/clothes/hm08")

# 12% with a staring look all round; 10% reads as awake next to the player.
EYE_SCALE = 1.1
EYE_CORE_M = 0.017
EYE_REACH_M = 0.032
LASH_LENGTH_M = {"female": 0.0075, "male": 0.005}
LASH_FORWARD = 0.65
LASH_OUT = 0.75
# The hole's rim folds inward behind the lid: rooted there, the lash hid behind the skin and only darkened it. Out by this much it reaches the visible edge.
LASH_OUT_M = 0.0003
LASH_UP_M = 0.0
LASH_OVER_GLOBE_M = 0.0004
LASH_HALF_WIDTH = 1.08
LASH_ARC_M = 0.0012
LASH_SLICES = 14
LASH_POINTS = 14
LASH_HIGHEST = 1.1
LASH_LOWEST = 0.15
LASH_OVER_LID_M = 0.0014
LASH_SHIFT_M = 0.002
LASH_BAND_M = 0.0014
LASH_COLOUR = (0.018, 0.013, 0.011, 1)
LASH_MOVES_M = 1e-5
LASH_FIT_DEGREE = 2
NORMAL_PX = 512


def smooth(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def read(block):
    """Points of a shape key, or of `mesh.vertices`."""
    items = block.data if hasattr(block, "relative_key") else block
    points = np.empty(len(items) * 3)
    items.foreach_get("co", points)
    return points.reshape(-1, 3)


def to_world(obj, points):
    matrix = np.array(obj.matrix_world)
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def to_local(obj, points):
    matrix = np.array(obj.matrix_world.inverted())
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def rest_points(obj):
    return to_world(obj, read(obj.data.vertices))


def warp(obj, field):
    """`field` takes world points; every key is absolute, so a stretched lid still closes all the way."""
    blocks = obj.data.shape_keys.key_blocks if obj.data.shape_keys else []
    for block in blocks:
        block.data.foreach_set("co", to_local(obj, field(to_world(obj, read(block)))).ravel())
    obj.data.vertices.foreach_set("co", to_local(obj, field(rest_points(obj))).ravel())
    obj.data.update()


def eye_radius(eye):
    points = rest_points(eye)
    return float(np.ptp(points[points[:, 0] < points[:, 0].mean(), 0]) / 2)


def eye_centres(eye):
    points = rest_points(eye)
    middle = points[:, 0].mean()
    centres = []
    for side in (points[:, 0] < middle, points[:, 0] >= middle):
        part = points[side]
        radius = np.ptp(part[:, 0]) / 2
        centres.append(np.array([(part[:, 0].min() + part[:, 0].max()) / 2, part[:, 1].min() + radius, (part[:, 2].min() + part[:, 2].max()) / 2]))
    return centres


def bigger_eyes(meshes, eye, scale=EYE_SCALE):
    """Globe and lids grow together around each eye, fading out before the nose and the temple."""
    centres = eye_centres(eye)

    def field(points):
        out = points.copy()
        for centre in centres:
            away = points - centre
            weight = smooth((EYE_REACH_M - np.linalg.norm(away, axis=1)) / (EYE_REACH_M - EYE_CORE_M))
            out += away * (scale - 1) * weight[:, None]
        return out

    for obj in meshes:
        warp(obj, field)


def upper_rim(body, centre, radius):
    """The lid's edge as the eye is seen from the front, nose to temple: the lowest skin point over the globe in each slice. The hole's own rim folds behind the lid and never shows."""
    world = rest_points(body)
    away = world - centre
    over = np.where((np.linalg.norm(away, axis=1) < EYE_REACH_M) & (np.linalg.norm(away, axis=1) > radius + LASH_OVER_GLOBE_M) & (away[:, 1] < 0) & (away[:, 2] > 0) & (away[:, 2] < radius * LASH_HIGHEST))[0]
    width = radius * LASH_HALF_WIDTH
    edges = np.linspace(centre[0] - width, centre[0] + width, LASH_SLICES + 1)
    chosen = []
    for low, high in zip(edges[:-1], edges[1:]):
        here = over[(world[over, 0] >= low) & (world[over, 0] < high)]
        if len(here):
            chosen.append(int(here[world[here, 2].argmin()]))
    # One slice can pick a point of the crease above the lid: that was a spike. The lid is a smooth arc, so what is far from the arc goes.
    chosen = np.array(chosen)
    for _ in range(3):
        arc = np.polyval(np.polyfit(world[chosen, 0], world[chosen, 2], 2), world[chosen, 0])
        chosen = chosen[np.abs(world[chosen, 2] - arc) < LASH_ARC_M]
    return chosen


def axis_slack(axis):
    """The lid curves more front to back than it does up and down."""
    return 3 if axis == 1 else 1


def lash_mesh(body, centres, eye_radius, sex, name="Lid_Lash"):
    """A solid dark crescent on each upper rim, like Rafa's. Each point rides the skin point it grows from, in the blink and in the bones.

    The name avoids "eyelash": the player hides any renderer called that (`NpcFaceShading.HideEyelashes`)."""
    world = rest_points(body)
    middle = (centres[0][0] + centres[1][0]) / 2
    places, faces, sources, eyes = [], [], [], []
    # Both eyes feed one mirrored arc: on some faces one eye's lid has too few good points, and its own arc flew up the forehead (Helena).
    rims = [upper_rim(body, centre, eye_radius) for centre in centres]
    sides = [1 if centre[0] > middle else -1 for centre in centres]
    pooled = np.concatenate([(world[rim] - centre) * np.array([side, 1, 1]) for rim, centre, side in zip(rims, centres, sides)])
    for _ in range(3):
        for axis in (1, 2):
            pooled = pooled[np.abs(pooled[:, axis] - np.polyval(np.polyfit(pooled[:, 0], pooled[:, axis], 2), pooled[:, 0])) < LASH_ARC_M * axis_slack(axis)]
    height_fit = np.polyfit(pooled[:, 0], pooled[:, 2], 2)
    half = min(eye_radius * LASH_HALF_WIDTH, float(np.abs(pooled[:, 0]).max()))
    for centre, rim, side in zip(centres, rims, sides):
        out = np.linspace(-half, half, LASH_POINTS)
        # The lid rests on the globe, so depth comes from the sphere: fitted from a handful of points it flew 18 cm off (Cláudia, Juliana). The height stays between the eye's middle and its top.
        height = np.clip(np.polyval(height_fit, out), -eye_radius * LASH_LOWEST, eye_radius * LASH_HIGHEST)
        depth = -np.sqrt(np.maximum((eye_radius + LASH_OVER_LID_M) ** 2 - out ** 2 - height ** 2, 0))
        spots = centre + np.stack([out * side, depth, height], axis=1)
        # The two lids are not mirror images: the arc's shape is shared, its height is each eye's own.
        own = (world[rim] - centre) * np.array([side, 1, 1])
        spots[:, 2] += float(np.clip(np.median(own[:, 2] - np.polyval(height_fit, own[:, 0])), -LASH_SHIFT_M, LASH_SHIFT_M))
        across = spots[:, 0]
        eyes.append((rim, across, world[rim, 0], centre))
        rim = rim[np.abs(world[rim, 0][:, None] - across).argmin(axis=0)]
        span = np.ptp(spots[:, 0])
        outward = np.abs(spots[:, 0] - middle)
        along = (outward - outward.min()) / max(span, 1e-6)
        radial = spots - centre
        radial[:, 1] = 0
        radial /= np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-6)
        way = radial * LASH_OUT + np.array([0, -LASH_FORWARD, 0])
        way /= np.linalg.norm(way, axis=1, keepdims=True)
        length = LASH_LENGTH_M[sex] * (0.35 + 0.65 * smooth(along)) * np.sin(np.pi * np.clip(along, 0.02, 0.98)) ** 0.5
        start = len(places)
        for spot, step, reach, band in zip(spots, way, length, radial):
            root = spot + np.array([0, -LASH_OUT_M, 0]) + band * LASH_UP_M
            places += [root, root + step * reach, root + band * LASH_BAND_M]
        for at in range(len(rim) - 1):
            a, b = start + at * 3, start + (at + 1) * 3
            faces += [(a, a + 1, b + 1, b), (a + 1, a + 2, b + 2, b + 1)]
        sources += [index for index in rim for _ in range(3)]
    data = bpy.data.meshes.new(name)
    data.from_pydata([tuple(point) for point in to_local(body, np.array(places))], [], faces)
    for polygon in data.polygons:
        polygon.use_smooth = True
    material = bpy.data.materials.new("Cilios")
    material.use_nodes = True
    shader = material.node_tree.nodes["Principled BSDF"]
    shader.inputs["Base Color"].default_value = LASH_COLOUR
    shader.inputs["Roughness"].default_value = 0.6
    data.materials.append(material)
    lash = bpy.data.objects.new(name, data)
    body.users_collection[0].objects.link(lash)
    lash.parent = body.parent
    lash.matrix_world = body.matrix_world.copy()
    wear_body_motion(lash, body, np.array(sources), eyes)
    return lash


def wear_body_motion(lash, body, sources, eyes):
    """Bones from the nearest skin point. The lid's move is read as a smooth curve along the arc: riding one point each, the blink broke the line into steps."""
    names = {group.index: group.name for group in body.vertex_groups}
    groups = {}
    for index, source in enumerate(sources):
        for item in body.data.vertices[int(source)].groups:
            name = names[item.group]
            group = groups.get(name) or lash.vertex_groups.new(name=name)
            groups[name] = group
            group.add([index], item.weight, "REPLACE")
    for modifier in body.modifiers:
        if modifier.type == "ARMATURE":
            lash.modifiers.new("Armature", "ARMATURE").object = modifier.object
    keys = body.data.shape_keys.key_blocks
    basis = read(keys[0])
    rest = read(lash.data.vertices)
    half = len(rest) // 2
    world_rest = to_world(lash, rest)
    lash.shape_key_add(name=keys[0].name)
    for block in keys[1:]:
        moved = read(block) - basis
        if block.name.startswith("eyeBlink"):
            closed = to_world(body, basis + moved)
            start = to_world(body, basis)
            placed = world_rest.copy()
            for side, (rim, across, x, centre) in enumerate(eyes):
                part = np.arange(half) + side * half
                placed[part] = turned_about_eye(world_rest[part], centre, lid_turn(start[rim], closed[rim], centre, x, across).repeat(3))
            delta = to_local(lash, placed) - rest
        else:
            delta = np.concatenate([lid_curve(moved[rim], x, across) for rim, across, x, _ in eyes]).repeat(3, axis=0)
        if np.linalg.norm(delta, axis=1).max() > LASH_MOVES_M:
            lash.shape_key_add(name=block.name).data.foreach_set("co", (rest + delta).ravel())


def lid_curve(moved, x, across):
    """The lid's move along the arc as a smooth curve, read only inside the part of the arc the lid has: past it, the fitted curve flew out and a tip of the strip left the eye."""
    inside = np.clip(across, x.min(), x.max())
    return np.stack([np.polyval(np.polyfit(x, moved[:, axis], LASH_FIT_DEGREE), inside) for axis in range(3)], axis=1)


def elevation(points, centre):
    """Angle of each point seen from the side of the eye: front is -Y, up is +Z."""
    away = points - centre
    return np.arctan2(away[:, 2], -away[:, 1])


def lid_turn(start, closed, centre, x, across):
    """How far the lid turns around the eye at each point of the strip: a smooth curve along the arc, read only inside the part the lid has."""
    turn = elevation(closed, centre) - elevation(start, centre)
    # Even in the distance from the eye's middle: with a slanted term the strip closed tilted and crossed the lid.
    along = x - centre[0]
    fitted = np.linalg.lstsq(np.stack([np.ones_like(along), along**2], axis=1), turn, rcond=None)[0]
    reach = np.minimum(np.abs(across - centre[0]), np.abs(along).max())
    return fitted[0] + fitted[1] * reach**2


def turned_about_eye(points, centre, turn):
    """The strip turns around the eye as the lid does, so it stays on the globe and in one piece. Moved point by point it cut through the closing lid and showed in black pieces (owner, 2026-11-09)."""
    away = points - centre
    cos, sin = np.cos(turn), np.sin(turn)
    out = points.copy()
    out[:, 1] = centre[1] + away[:, 1] * cos + away[:, 2] * sin
    out[:, 2] = centre[2] + away[:, 2] * cos - away[:, 1] * sin
    return out


def garment_normal(name):
    """The garment's own normal map in the MakeHuman assets, or None when its author drew none."""
    base = name.split(".")[0]
    for folder in (base, base.split("-")[0]):
        found = sorted(glob.glob(os.path.join(CLOTHES, folder, "*[nN]ormal*")))
        if found:
            return found[0]
    return None


def cloth_relief(obj, path):
    """The folds drawn by the garment's own author: its normal map, at a size the phone can pay for."""
    image = bpy.data.images.load(path, check_existing=True)
    image.colorspace_settings.name = "Non-Color"
    if image.size[0] > NORMAL_PX:
        image.scale(NORMAL_PX, NORMAL_PX)
    image.pack()
    for material in obj.data.materials:
        tree = material.node_tree
        texture = tree.nodes.new("ShaderNodeTexImage")
        texture.image = image
        normal = tree.nodes.new("ShaderNodeNormalMap")
        tree.links.new(texture.outputs["Color"], normal.inputs["Color"])
        shader = next(node for node in tree.nodes if node.type == "BSDF_PRINCIPLED")
        tree.links.new(normal.outputs["Normal"], shader.inputs["Normal"])
