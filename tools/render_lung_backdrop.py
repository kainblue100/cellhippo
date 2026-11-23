"""Render the anatomical lung once, as a backdrop for the evolution animation.

The model is BodyParts3D's human lung.  It is reference anatomy: it is *not* a
reconstruction of the specimen these patterns came from, and the page must say so.
The coda-mouse-lung-volume project treats exactly this asset the same way — it is
listed as deliberately excluded from that reconstruction because it is the wrong
species.  Here the species is right (human lung cancer Xenium), but the provenance
point still holds: the lung silhouette is an atlas, the cells inside it are
measurements, and a figure that shows both has to label which is which.

Two renders:
  lung_silhouette.png   pale outline on transparent, to sit behind the 3D panel
  lung_wash.png         very low-contrast filled form, for use at low opacity
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import bpy
from mathutils import Vector

ROOT = Path(sys.argv[sys.argv.index("--") + 1]) if "--" in sys.argv else \
    Path(r"D:\coda_matlab\lungs_original_model")
OUT = ROOT / "cellhippo_backdrop"
GLB = ROOT / "LUNG_anatomical.glb"


def clear():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def bounds(objs):
    mn = Vector((1e18, 1e18, 1e18))
    mx = Vector((-1e18, -1e18, -1e18))
    for o in objs:
        for c in o.bound_box:
            w = o.matrix_world @ Vector(c)
            for i in range(3):
                mn[i] = min(mn[i], w[i])
                mx[i] = max(mx[i], w[i])
    return mn, mx


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"importing {GLB}")
    clear()
    bpy.ops.import_scene.gltf(filepath=str(GLB))
    objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    print(f"  {len(objs)} mesh objects")
    mn, mx = bounds(objs)
    ctr = (mn + mx) / 2
    size = max((mx - mn)[i] for i in range(3))
    print(f"  bounds {tuple(round(v, 1) for v in mn)} .. "
          f"{tuple(round(v, 1) for v in mx)}, size {size:.1f}")

    scene = bpy.context.scene
    # EEVEE is enough for a flat silhouette and much quicker than Cycles.  The
    # engine id moved between releases, so try the names in order rather than
    # probing the enum through bpy.types (that probe raised on this build).
    for cand in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES"):
        try:
            scene.render.engine = cand
            print(f"  engine {cand}")
            break
        except TypeError:
            continue
    scene.render.resolution_x = 1400
    scene.render.resolution_y = 1400
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "Standard"
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "CUDA"
        prefs.get_devices()
        for d in prefs.devices:
            d.use = d.type != "CPU"
        scene.cycles.device = "GPU"
    except Exception:
        pass

    world = bpy.data.worlds.new("W")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = \
        (0.55, 0.55, 0.60, 1.0)
    world.node_tree.nodes["Background"].inputs[1].default_value = 0.9

    # a pale, almost flat material: this is a backdrop, not the subject
    mat = bpy.data.materials.new("lung")
    mat.use_nodes = True
    b = mat.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (0.62, 0.60, 0.72, 1.0)
    b.inputs["Roughness"].default_value = 0.85
    if "Emission Strength" in b.inputs:
        b.inputs["Emission Strength"].default_value = 0.22
        if "Emission Color" in b.inputs:
            b.inputs["Emission Color"].default_value = (0.62, 0.60, 0.74, 1.0)
    for o in objs:
        o.data.materials.clear()
        o.data.materials.append(mat)

    cam_data = bpy.data.cameras.new("cam")
    cam = bpy.data.objects.new("cam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    cam_data.lens = 60.0
    half = math.atan(0.5 * cam_data.sensor_width / cam_data.lens)
    dist = (size * 0.5) / math.tan(half) * 1.35
    eye = ctr + Vector((0.55, -1.0, 0.28)).normalized() * dist
    cam.location = eye
    cam.rotation_euler = (ctr - eye).to_track_quat("-Z", "Y").to_euler()
    print(f"  camera at {dist:.1f} (size {size:.1f}, lens {cam_data.lens})")

    key = bpy.data.lights.new("key", "AREA")
    key.energy = 40 * size
    key.size = size
    ko = bpy.data.objects.new("key", key)
    scene.collection.objects.link(ko)
    ko.location = ctr + Vector((-0.7, -0.9, 0.9)) * size
    ko.rotation_euler = (ctr - ko.location).to_track_quat("-Z", "Y").to_euler()

    for name, energy in (("lung_silhouette", 0.22), ("lung_wash", 0.55)):
        b.inputs["Emission Strength"].default_value = energy
        p = OUT / f"{name}.png"
        scene.render.filepath = str(p)
        bpy.ops.render.render(write_still=True)
        print(f"  wrote {p}")

    bpy.ops.wm.save_as_mainfile(filepath=str(OUT / "lung_backdrop.blend"))
    print("done")


main()
