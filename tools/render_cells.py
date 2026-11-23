"""Cycles stills of each cellhippo pattern: glass volume and a close cell.

Adapted from Annotile's scripts/render_cells.py.  The scene construction, light
energies, material parameters and framing are kept as they are there, because
that combination is what produces the target look; only the data plumbing and
the output paths differ.

1 Blender unit = 1 µm.  Depth positions are multiplied by z_scale so the 10 µm
section reads as a cube; radii stay in micrometres.
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import bpy
from mathutils import Vector

ROOT = Path(sys.argv[sys.argv.index("--") + 1]) if "--" in sys.argv else \
    Path(r"D:\coda_matlab\cellhippo\figures\volume")
SCENE_PATH = ROOT / "scene.json"
OUT = ROOT / "blender"
Z_FALLBACK = 4.0


def args_after_double_dash() -> list:
    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1:][1:]


def hex_rgb(value: str):
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) / 255 for i in (0, 2, 4))


def grape_lift(hex_color: str, toward: float = 0.45, lift: float = 1.30):
    """Pull toward the render's magenta, then raise the value.

    grape() alone produced a dark plum: the cell-type hexes are already muted, so
    mixing them with (0.72,0.28,0.62) at 0.62 lands well below the reference's
    light rose.  The lift closes that gap without touching the hue.
    """
    source = hex_rgb(hex_color)
    target = (0.72, 0.28, 0.62)
    mixed = [(1.0 - toward) * source[i] + toward * target[i] for i in range(3)]
    return tuple(min(1.0, channel * lift) for channel in mixed)


def grape(hex_color: str, toward: float = 0.55):
    """Keep a hint of the cell-type hue inside the purple render."""
    source = hex_rgb(hex_color)
    target = (0.72, 0.28, 0.62)
    return tuple((1.0 - toward) * source[i] + toward * target[i] for i in range(3))


def set_input(node, names, value) -> None:
    for name in names:
        if name in node.inputs:
            node.inputs[name].default_value = value
            return


def principled(name, color, *, roughness, transmission, subsurface, scale, ior,
               coat, emission=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    rgba = (color[0], color[1], color[2], 1.0)
    set_input(bsdf, ("Base Color",), rgba)
    set_input(bsdf, ("Roughness",), roughness)
    set_input(bsdf, ("IOR",), ior)
    set_input(bsdf, ("Transmission Weight", "Transmission"), transmission)
    set_input(bsdf, ("Subsurface Weight", "Subsurface"), subsurface)
    set_input(bsdf, ("Subsurface Radius",), (1.0, 0.28, 0.22))
    set_input(bsdf, ("Subsurface Scale",), scale)
    set_input(bsdf, ("Coat Weight", "Coat"), coat)
    set_input(bsdf, ("Coat Roughness",), 0.18)
    set_input(bsdf, ("Specular IOR Level", "Specular"), 0.45)
    set_input(bsdf, ("Emission Color",), rgba)
    set_input(bsdf, ("Emission Strength",), emission)
    mat.use_transparent_shadow = True
    return mat


def new_material_bump(name, color, *, roughness, transmission, subsurface, scale,
                      ior, coat, bump, emission=0.0, thin=False):
    mat = principled(name, color, roughness=roughness, transmission=transmission,
                     subsurface=subsurface, scale=scale, ior=ior, coat=coat,
                     emission=emission)
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if thin:
        set_input(bsdf, ("Thin Wall",), True)
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 2.4
    noise.inputs["Detail"].default_value = 6.0
    noise.inputs["Roughness"].default_value = 0.55
    mix = nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.inputs["Factor"].default_value = 0.0
    links.new(noise.outputs["Fac"], mix.inputs["Factor"])
    base = bsdf.inputs["Base Color"].default_value
    mix.inputs["A"].default_value = base
    mix.inputs["B"].default_value = (base[0] * 0.35, base[1] * 0.22,
                                     base[2] * 0.4, 1.0)
    links.new(mix.outputs["Result"], bsdf.inputs["Base Color"])
    bump_node = nodes.new("ShaderNodeBump")
    bump_node.inputs["Strength"].default_value = bump
    bump_node.inputs["Distance"].default_value = 0.55
    links.new(noise.outputs["Fac"], bump_node.inputs["Height"])
    links.new(bump_node.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def clear_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def configure_render(samples: int, threshold: float, resolution: int) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.cycles.samples = samples
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = threshold
    scene.cycles.max_bounces = 8
    scene.cycles.diffuse_bounces = 3
    scene.cycles.glossy_bounces = 4
    scene.cycles.transmission_bounces = 8
    scene.cycles.transparent_max_bounces = 8
    scene.cycles.sample_clamp_indirect = 6.0
    scene.cycles.use_denoising = True
    scene.render.use_persistent_data = True
    try:
        scene.cycles.denoiser = "OPENIMAGEDENOISE"
        scene.cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    except TypeError:
        pass
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "CUDA"
        prefs.get_devices()
        for device in prefs.devices:
            device.use = device.type != "CPU"
        scene.cycles.device = "GPU"
    except Exception:
        scene.cycles.device = "CPU"
    scene.view_settings.view_transform = "AgX"
    try:
        scene.view_settings.look = "AgX - Punchy"
    except TypeError:
        pass
    scene.view_settings.exposure = 0.35
    scene.view_settings.gamma = 1.0


def world_color(color, strength: float) -> None:
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    bg.inputs["Color"].default_value = (color[0], color[1], color[2], 1.0)
    bg.inputs["Strength"].default_value = strength


def add_area(name, location, energy, color, size) -> None:
    light_data = bpy.data.lights.new(name, "AREA")
    light_data.energy = energy
    light_data.color = color
    light_data.shape = "SQUARE"
    light_data.size = size
    light = bpy.data.objects.new(name, light_data)
    bpy.context.collection.objects.link(light)
    light.location = location


def look_at(obj, target: Vector) -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def add_camera(location: Vector, target: Vector, lens: float, fstop: float,
               focus: Vector):
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = lens
    cam_data.dof.use_dof = True
    cam_data.dof.focus_distance = (focus - location).length
    cam_data.dof.aperture_fstop = fstop
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    cam.location = location
    look_at(cam, target)
    bpy.context.scene.camera = cam
    return cam


def ico(name, radius, location: Vector, material, subdiv: int, scale, displace,
        seed: int, smooth_subdiv: int = 0):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=subdiv, radius=radius,
                                          location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = scale
    obj.data.materials.append(material)
    if displace > 0:
        tex = bpy.data.textures.new(name + "_tex", "CLOUDS")
        tex.noise_scale = 0.18 + (seed % 4) * 0.04
        tex.noise_depth = 3
        modifier = obj.modifiers.new("disp", "DISPLACE")
        modifier.texture = tex
        modifier.strength = displace
        modifier.mid_level = 0.45
    if smooth_subdiv:
        subsurf = obj.modifiers.new("sub", "SUBSURF")
        subsurf.levels = smooth_subdiv
        subsurf.render_levels = smooth_subdiv
    bpy.ops.object.shade_smooth()
    return obj


def glass_box(span: float, height: float, pad: float) -> None:
    center = Vector((span / 2, span / 2, height / 2))
    size = Vector((span + pad * 2, span + pad * 2, height + pad * 2))
    bpy.ops.mesh.primitive_cube_add(size=1, location=center)
    cube = bpy.context.active_object
    cube.scale = (size.x, size.y, size.z)
    glass = principled("glass", (0.97, 0.95, 0.99), roughness=0.02,
                       transmission=1.0, subsurface=0.0, scale=0.0, ior=1.04,
                       coat=0.02)
    cube.data.materials.append(glass)
    cube.visible_shadow = False
    bpy.ops.mesh.primitive_cube_add(size=1, location=center)
    wire = bpy.context.active_object
    wire.scale = (size.x, size.y, size.z)
    modifier = wire.modifiers.new("wire", "WIREFRAME")
    # 0.0016 was invisible, 0.004 too harsh; the reference reads as a faint line
    modifier.thickness = 0.0026
    modifier.use_replace = True
    edge = principled("edge", (0.62, 0.50, 0.66), roughness=0.30,
                      transmission=0.05, subsurface=0.0, scale=0.0, ior=1.4,
                      coat=0.2, emission=0.5)
    wire.data.materials.append(edge)
    wire.visible_shadow = False


def backdrop_plane(path: str, camera, extra: float) -> None:
    bpy.context.view_layer.update()
    forward = camera.matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0))
    distance = camera.data.dof.focus_distance + extra
    bpy.ops.mesh.primitive_plane_add(size=1,
                                     location=camera.location + forward * distance)
    plane = bpy.context.active_object
    facing = camera.location - plane.location
    plane.rotation_euler = facing.to_track_quat("Z", "Y").to_euler()
    half = distance * math.tan(camera.data.angle / 2)
    plane.scale = (half * 2.4, half * 2.4, 1.0)
    mat = bpy.data.materials.new("backdrop")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    image = bpy.data.images.load(path)
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = image
    tex.projection = "FLAT"
    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    set_input(bsdf, ("Roughness",), 1.0)
    set_input(bsdf, ("Specular IOR Level", "Specular"), 0.0)
    set_input(bsdf, ("Emission Strength",), 0.85)
    links.new(tex.outputs["Color"], bsdf.inputs["Emission Color"])
    plane.data.materials.append(mat)


def render_to(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    print("WROTE", path, flush=True)


def overview(model: dict, zscale: float, samples: int, resolution: int) -> None:
    clear_scene()
    configure_render(samples, 0.05 if samples < 80 else 0.02, resolution)
    world_color((0.93, 0.88, 0.91), 0.55)
    span = float(model["patch_um"])
    height = 10.0 * zscale
    cells = model["cells"]
    for index, cell in enumerate(cells):
        rng = random.Random(index + 3)
        # per-cell hue and value jitter: the reference runs from salmon through
        # lavender to pale blue-grey, and a single flat colour is what made the
        # first version read as plastic beads rather than cells
        base = grape_lift(cell["color"], 0.45, 1.30)
        hue = rng.uniform(-0.10, 0.10)
        val = rng.uniform(0.86, 1.16)
        color = (min(1.0, max(0.0, base[0] * val + hue)),
                 min(1.0, max(0.0, base[1] * val + hue * 0.35)),
                 min(1.0, max(0.0, base[2] * val - hue * 0.5)))
        radius = float(cell["r"])
        # transmission carries the look: at 0.44 the rear cells were still
        # hidden.  Roughness is raised at the same time so the highlight spreads
        # into the wide soft sheen the reference has instead of a hard dot.
        material = principled(f"body_{index}", color, roughness=0.42,
                              transmission=0.86, subsurface=0.35,
                              scale=radius * 0.30, ior=1.32, coat=0.25)
        location = Vector((float(cell["x"]), float(cell["y"]),
                           float(cell["z"]) * zscale))
        wobble = tuple(rng.uniform(0.94, 1.06) for _ in range(3))
        ico(f"cell_{index}", radius, location, material, 3, wobble,
            radius * 0.05, index)
        nucleus = principled(f"nuc_{index}",
                             tuple(channel * 0.28 for channel in color),
                             roughness=0.42, transmission=0.02, subsurface=0.35,
                             scale=radius * 0.2, ior=1.4, coat=0.05)
        ico(f"nuc_{index}", radius * 0.42, location, nucleus, 1, wobble, 0.0,
            index)
    glass_box(span, height, pad=1.5)
    center = Vector((span / 2, span / 2, height / 2))
    direction = Vector((0.95, -1.25, 0.72)).normalized()
    # 3.35 filled half the frame; 4.3 puts the block at roughly a third, which
    # is how much of the plate the reference leaves visible around it
    distance = max(span, height) * 4.3
    add_area("key", center + Vector((-span * 0.8, -span * 1.5, height * 1.8)),
             4500, (1.0, 0.95, 0.92), span * 1.3)
    add_area("fill", center + Vector((span * 1.4, -span * 0.2, height * 0.4)),
             900, (0.82, 0.62, 0.95), span * 1.4)
    add_area("rim", center + Vector((span * 0.3, span * 1.6, height * 1.2)),
             1600, (0.95, 0.85, 1.0), span)
    camera = add_camera(center + direction * distance, center, 42, 5.6, center)
    backdrop = ROOT / model.get("backdrop", "")
    if backdrop.is_file():
        # 0.85 left the plate too sharp relative to the reference; pushing it
        # further behind the focal plane lets the camera's own defocus soften it
        backdrop_plane(str(backdrop), camera, max(span, height) * 1.7)


def closeup(model: dict, zscale: float, samples: int, resolution: int) -> None:
    clear_scene()
    configure_render(samples, 0.04 if samples < 80 else 0.015, resolution)
    world_color((0.16, 0.04, 0.20), 1.15)
    bpy.context.scene.view_settings.exposure = 0.55
    cells = sorted(model["cells"], key=lambda item: item["r"], reverse=True)
    hero = cells[0]
    hero_at = Vector((float(hero["x"]), float(hero["y"]),
                      float(hero["z"]) * zscale))
    near = sorted(cells[1:], key=lambda item: (
        Vector((item["x"], item["y"], item["z"] * zscale)) - hero_at).length)[:2]
    chosen = [hero] + near
    for index, cell in enumerate(chosen):
        rng = random.Random(index + 11)
        color = grape(cell["color"], 0.48)
        radius = float(cell["r"])
        location = Vector((float(cell["x"]), float(cell["y"]),
                           float(cell["z"]) * zscale))
        scale = (rng.uniform(0.88, 1.12), rng.uniform(0.9, 1.14),
                 rng.uniform(0.86, 1.1))
        membrane = new_material_bump(
            f"mem_{index}", (0.92, 0.62, 0.84), roughness=0.32,
            transmission=0.96, subsurface=0.05, scale=radius * 0.15, ior=1.28,
            coat=0.25, bump=0.45, thin=True)
        body = new_material_bump(
            f"cyto_{index}", (0.9, 0.48, 0.78), roughness=0.42,
            transmission=0.78, subsurface=1.0, scale=radius * 0.8, ior=1.34,
            coat=0.02, bump=0.25, emission=0.18)
        nucleus_shift = Vector((rng.uniform(-0.06, 0.08), -0.22 * radius,
                                rng.uniform(0.0, 0.12)))
        nucleus = new_material_bump(
            f"core_{index}", (0.16, 0.02, 0.24), roughness=0.62,
            transmission=0.0, subsurface=0.4, scale=radius * 0.2, ior=1.4,
            coat=0.0, bump=0.4, emission=0.04)
        ico(f"mem_{index}", radius * 1.03, location, membrane, 3, scale,
            radius * 0.22, index, smooth_subdiv=2)
        ico(f"cyto_{index}", radius * 0.88, location, body, 3, scale,
            radius * 0.1, index + 5, smooth_subdiv=1)
        ico(f"core_{index}", radius * 0.52, location + nucleus_shift, nucleus, 3,
            scale, radius * 0.1, index + 9, smooth_subdiv=1)
    reach = radius_of(hero)
    add_area("key", hero_at + Vector((-reach * 2.4, -reach * 3.4, reach * 1.8)),
             2800, (1.0, 0.94, 0.97), reach * 3.2)
    add_area("rim", hero_at + Vector((reach * 2.2, reach * 3.6, reach * 1.4)),
             2200, (0.85, 0.62, 1.0), reach * 4.5)
    add_area("back", hero_at + Vector((0, reach * 4.5, reach * 0.4)),
             1800, (0.95, 0.75, 1.0), reach * 6)
    camera_at = hero_at + Vector((reach * 0.8, -reach * 6.2, reach * 1.7))
    add_camera(camera_at, hero_at, 52, 2.8, hero_at)


def radius_of(cell: dict) -> float:
    return float(cell["r"])


def main() -> None:
    options = args_after_double_dash()
    mode = options[0] if options else "preview"
    which = options[1] if len(options) > 1 else "all"
    view = options[2] if len(options) > 2 else "both"
    scene = json.loads(SCENE_PATH.read_text(encoding="utf-8"))
    zscale = float(scene.get("z_scale", Z_FALLBACK))
    patterns = scene["patterns"]
    if which != "all":
        patterns = [item for item in patterns
                    if f"{item['pattern']}_{item['slice']}" == which
                    or item["pattern"] == which]
    samples = 28 if mode == "preview" else 48
    resolution = 640 if mode == "preview" else 800
    for model in patterns:
        stem = f"{model['pattern']}_{model['slice']}"
        if view in ("both", "cube"):
            overview(model, zscale, samples, resolution)
            render_to(OUT / f"{stem}_cube.png")
        if view in ("both", "close"):
            closeup(model, zscale, samples, resolution)
            render_to(OUT / f"{stem}_close.png")
    print("RENDER_DONE", mode, which, len(patterns), flush=True)


if __name__ == "__main__":
    main()
