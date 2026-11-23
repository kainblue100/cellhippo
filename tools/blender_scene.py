"""Build and render the 3D cell model in Blender, headless.

Blender is used for the same reason Alon et al. used it: the target look is a
shaded, translucent, depth-of-field rendering, and a real renderer with real
optics and real volumetric shading is the shortest path to it.  Falling back to a
hand-written rasteriser would have meant faking the parts that matter -- the
membrane's Fresnel sheen, the way cells read through each other, and the
out-of-focus falloff.

Scene layout follows the assembly conventions: explicit collections, stable object
IDs, one delivery camera, no hidden import dependencies.

Two shots are produced per slide:
  wide   the cluster inside its translucent bounding slab, oblique camera
  macro  a close-up of two cells, shallow focus

Output: PNG sequences of stills, one file per shot.
"""
from __future__ import annotations

import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
MODEL = ARGS[0] if ARGS else r"D:\coda_matlab\cellhippo\docs\model3d_5k.json"
OUTDIR = ARGS[1] if len(ARGS) > 1 else r"D:\coda_matlab\cellhippo\figures\blender"
TAG = ARGS[2] if len(ARGS) > 2 else "5k"
SAMPLES = int(ARGS[3]) if len(ARGS) > 3 else 96
RES = int(ARGS[4]) if len(ARGS) > 4 else 1100
N_CELLS = int(ARGS[5]) if len(ARGS) > 5 else 56

os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------------------------------------------- palette
# Kept from the clustering so the figure still says which phenotype is which,
# but pulled toward the violet end of each hue so the palette reads as one tissue
# rather than as a categorical chart.
PH_BASE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]


def srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def hex_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


def blend_to_violet(rgb, amount=0.45, lift=0.22):
    """Nudge a hue toward violet and lift its value.

    Pulling toward violet without lifting made the first render dark navy, and
    against a grey background dark navy has almost no separation.  The palette has
    to sit *above* the background in luminance for the spheres to read as
    translucent glass rather than as holes.
    """
    violet = np.array([0.68, 0.50, 0.96])
    out = rgb * (1 - amount) + violet * amount
    return np.clip(out + lift * (1.0 - out), 0, 1)


# ---------------------------------------------------------------- scene reset
def fresh_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.unit_settings.system = "METRIC"
    sc.unit_settings.scale_length = 1e-6      # one Blender unit = one micron
    return sc


def make_collections():
    root = bpy.data.collections.new("cellhippo")
    bpy.context.scene.collection.children.link(root)
    cols = {}
    for name in ("Cells", "Guides", "Lighting", "Cameras"):
        c = bpy.data.collections.new(name)
        root.children.link(c)
        cols[name] = c
    return cols


def link_to(obj, coll):
    for c in obj.users_collection:
        c.objects.unlink(obj)
    coll.objects.link(obj)


# ---------------------------------------------------------------- materials
def membrane_material(name, base_rgb, ior=1.42, roughness=0.16, coat=0.6):
    """Glossy translucent membrane over a denser interior.

    Principled BSDF with transmission and a coat reproduces the wet, glassy look
    in the reference: colour comes through from behind, a bright narrow specular
    sits on top, and the rim brightens at grazing angles through the coat's
    Fresnel.
    """
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]

    def setv(key, val):
        if key in bsdf.inputs:
            bsdf.inputs[key].default_value = val
            return True
        return False

    lin = srgb_to_linear(base_rgb)
    setv("Base Color", (lin[0], lin[1], lin[2], 1.0))
    setv("Roughness", roughness)
    setv("Metallic", 0.0)
    setv("IOR", ior)
    # Transmission was pushed to 0.85 to escape an opaque-plastic look, and
    # overshot: at that level the surface is glass, so the render shows the
    # back-facing shells and internal seams of the merged body through its own
    # skin.  The reference reads as dense translucent tissue, not glass -- light
    # scatters under the surface rather than passing straight through -- so
    # subsurface now carries the translucency and transmission is only a hint.
    setv("Transmission Weight", 0.18)
    setv("Transmission", 0.18)
    setv("Coat Weight", coat)
    setv("Coat", coat)
    setv("Coat Roughness", 0.06)
    setv("Coat IOR", 1.5)
    setv("Subsurface Weight", 0.55)
    setv("Subsurface", 0.55)
    if "Subsurface Radius" in bsdf.inputs:
        bsdf.inputs["Subsurface Radius"].default_value = (0.55, 0.22, 0.70)
    if "Subsurface Scale" in bsdf.inputs:
        bsdf.inputs["Subsurface Scale"].default_value = 0.12
    setv("Sheen Weight", 0.30)
    setv("Sheen", 0.30)
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.68
    elif "Specular" in bsdf.inputs:
        bsdf.inputs["Specular"].default_value = 0.68
    # a touch of emission keeps the interior from going muddy in shadow
    if "Emission Strength" in bsdf.inputs:
        bsdf.inputs["Emission Strength"].default_value = 0.05
        if "Emission Color" in bsdf.inputs:
            bsdf.inputs["Emission Color"].default_value = (
                lin[0], lin[1], lin[2], 1.0)
    return mat


def slab_material(name):
    """The faint bounding box: almost invisible faces, slightly brighter edges."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.62, 0.60, 0.66, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.45
    if "Transmission Weight" in bsdf.inputs:
        bsdf.inputs["Transmission Weight"].default_value = 0.0
    if "Alpha" in bsdf.inputs:
        bsdf.inputs["Alpha"].default_value = 0.055
    mat.blend_method = "BLEND" if hasattr(mat, "blend_method") else mat.blend_method
    return mat


# ---------------------------------------------------------------- geometry
def add_metaball_cluster(name, centres, radii, phenos, mats, coll,
                         resolution=0.06, stiffness=1.35):
    """One metaball object per phenotype: the cells fuse into a continuous body.

    Separate UV spheres were the obvious construction and the wrong one.  A
    loose cloud of discrete balls reads as a scatter plot; the reference shows a
    contiguous lump of tissue with organic necks between cells.  Metaballs merge
    where the fields overlap, which is exactly what real packed cells look like,
    and it also removes the alpha-sorting artefacts that translucent overlapping
    spheres produce.

    One object per phenotype so the surfaces stay colour-separable: balls in the
    same metaball family merge regardless of material, so a single family would
    blend everything into one surface.
    """
    made = []
    for g in sorted(set(int(v) for v in phenos)):
        sel = np.where(phenos == g)[0]
        mb = bpy.data.metaballs.new(f"MB_P{g}")
        mb.resolution = resolution
        mb.render_resolution = resolution * 0.7
        mb.threshold = 0.6
        ob = bpy.data.objects.new(name=f"{name}_P{g}", object_data=mb)
        coll.objects.link(ob)
        for i in sel:
            el = mb.elements.new()
            el.co = Vector([float(v) for v in centres[i]])
            el.radius = float(radii[i])
            el.stiffness = stiffness
        ob.data.materials.append(mats[g])
        made.append(ob)
    return made


def add_sphere(name, centre, radius, mat, segments=48, rings=32, coll=None):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segments, ring_count=rings,
                                         radius=radius, location=centre)
    ob = bpy.context.object
    ob.name = name
    ob.data.name = name + "_mesh"
    bpy.ops.object.shade_smooth()
    ob.data.materials.append(mat)
    if coll is not None:
        link_to(ob, coll)
    return ob


def add_box(name, mn, mx, mat, coll=None):
    ctr = (mn + mx) / 2.0
    size = (mx - mn)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=ctr)
    ob = bpy.context.object
    ob.name = name
    ob.data.name = name + "_mesh"
    ob.scale = (size[0], size[1], size[2])
    ob.data.materials.append(mat)
    if coll is not None:
        link_to(ob, coll)
    return ob


def slab_edges(mn, mx, coll, radius=0.35, emissive=0.9):
    """Thin tubes along the slab's twelve edges.

    A translucent box alone reads as a smudge; the reference figure shows the
    bounding volume as a *drawn* box, with the faces merely tinting what is
    behind them.  The edges are what make the volume legible.
    """
    mn = np.asarray(mn, dtype=np.float64)
    mx = np.asarray(mx, dtype=np.float64)
    corn = np.array([[x, y, z] for x in (mn[0], mx[0])
                     for y in (mn[1], mx[1]) for z in (mn[2], mx[2])])
    pairs = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
             (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    mat = bpy.data.materials.new("M_SlabEdge")
    mat.use_nodes = True
    b = mat.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (0.78, 0.76, 0.82, 1.0)
    b.inputs["Roughness"].default_value = 0.35
    if "Emission Strength" in b.inputs:
        b.inputs["Emission Strength"].default_value = emissive
        if "Emission Color" in b.inputs:
            b.inputs["Emission Color"].default_value = (0.80, 0.78, 0.86, 1.0)
    for k, (i, j) in enumerate(pairs):
        p0, p1 = corn[i], corn[j]
        mid = (p0 + p1) / 2.0
        length = float(np.linalg.norm(p1 - p0))
        bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=radius,
                                            depth=length, location=mid)
        ob = bpy.context.object
        ob.name = f"SlabEdge_{k:02d}"
        d = Vector([float(v) for v in (p1 - p0)])
        ob.rotation_euler = d.to_track_quat("Z", "Y").to_euler()
        ob.data.materials.append(mat)
        link_to(ob, coll)


# ---------------------------------------------------------------- model input
def load_cells(path, n_cells):
    """Take a compact cluster of cells around the densest point.

    The first version boxed a 90 um span, subsampled to 24 cells and got a cloud
    475 um wide made of 6 um spheres -- the subject then filled a tenth of the
    frame.  What the reference panel shows is a *local* lump of nuclei that touch:
    so the densest point is found, and the n nearest cells to it are taken.  The
    cluster's own physical size then follows from the local density instead of
    being imposed.
    """
    d = json.load(open(path))
    x = np.asarray(d["x"], dtype=np.float64)
    y = np.asarray(d["y"], dtype=np.float64)
    z = np.asarray(d["z"], dtype=np.float64)
    r = np.asarray(d["r_true"], dtype=np.float64)
    p = np.asarray(d["phenotype"], dtype=np.int64)

    # Radius-limited, not count-limited: taking "the N nearest" from a 126 um field
    # gave a pancake, because our inferred z spans only +-6 um.  The centre is
    # chosen inside the radius, so the search scores phenotype diversity and
    # density *within the radius* rather than in a much larger window.
    z_thick = float(np.percentile(np.abs(z), 95)) * 2.0
    radius = max(20.0, z_thick * 2.4)
    best, bc = -1, (float(x.mean()), float(y.mean()))
    step = max(1.0, radius / 3)
    for cx in np.arange(x.min() + radius, x.max() - radius + 1, step):
        for cy in np.arange(y.min() + radius, y.max() - radius + 1, step):
            d2 = (x - cx) ** 2 + (y - cy) ** 2
            m = d2 < radius ** 2
            n = int(m.sum())
            if n < 8:
                continue
            n_ph = int(len(np.unique(p[m])))
            score = n_ph ** 2 * 10000 + n
            if score > best:
                best, bc = score, (float(cx), float(cy))

    d2 = (x - bc[0]) ** 2 + (y - bc[1]) ** 2
    inside = np.where(d2 < radius ** 2)[0]
    # grow until the field actually holds the requested number of cells: local
    # density varies by an order of magnitude across a slide, so a fixed radius
    # gave 18 cells in one place and hundreds in another
    grow = radius
    while len(inside) < n_cells and grow < 6 * radius:
        grow *= 1.25
        inside = np.where(d2 < grow ** 2)[0]
    radius = grow
    order = inside[np.argsort(d2[inside])]
    print(f"  radius {radius:.1f} um around ({bc[0]:.0f},{bc[1]:.0f}) holds "
          f"{len(order)} cells, {len(np.unique(p[order]))} phenotypes; "
          f"took up to {n_cells}")

    chosen = list(order[:n_cells])
    near = order
    have = set(p[chosen].tolist())
    rng = np.random.default_rng(0)
    for ph in np.unique(p[near]):
        if ph in have:
            continue
        cand = near[p[near] == ph]
        if len(cand):
            drop = rng.choice([c for c in chosen if p[c] != ph], 1)
            chosen.remove(int(drop[0]))
            chosen.append(int(cand[0]))
    chosen = np.array(sorted(set(int(c) for c in chosen)))
    print(f"  densest point ({bc[0]:.0f},{bc[1]:.0f}) with {best} cells nearby; "
          f"took the {len(chosen)} nearest")
    print(f"  phenotypes {sorted(set(p[chosen].tolist()))}")

    pts = np.column_stack([x[chosen], y[chosen], z[chosen]])
    pts -= pts.mean(axis=0)                   # centre on the cluster
    return pts, r[chosen], p[chosen]


# ---------------------------------------------------------------- camera
def add_camera(coll, name="CAM_delivery"):
    cam_data = bpy.data.cameras.new(name)
    cam = bpy.data.objects.new(name, cam_data)
    coll.objects.link(cam)
    bpy.context.scene.camera = cam
    return cam


def aim(cam, eye, target):
    """Point the camera.  Accepts numpy arrays as well as Vectors -- mixing them
    raised "Vector + numpy.ndarray invalid type" on the first run."""
    eye = Vector([float(v) for v in np.asarray(eye).ravel()])
    target = Vector([float(v) for v in np.asarray(target).ravel()])
    cam.location = eye
    d = target - eye
    cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def setup_lighting(scale):
    """Key from the upper left, cool fill, and a rim from behind.

    Positions are multiplied by `scale`, the cluster's size in microns: the first
    version hard-coded them at a few units, which put the lights *inside* a
    100 um cluster and lit nothing.  The dim world plus these lights is what
    replaced the flat, washed-out first pass.
    """
    base = [
        ("KEY", (-3.0, -2.6, 3.6), 900.0, 2.4, (1.0, 0.965, 0.94)),
        ("FILL", (3.4, -1.8, 1.0), 260.0, 3.4, (0.90, 0.94, 1.0)),
        ("RIM", (0.6, 3.4, 2.8), 520.0, 2.0, (0.82, 0.86, 1.0)),
    ]
    out = []
    for name, loc, power, size, col in base:
        d = bpy.data.lights.new(name, type="AREA")
        # power scales with distance squared so the exposure is size-independent
        d.energy = power * (scale / 100.0) ** 2
        d.size = size * scale / 100.0
        d.color = col
        ob = bpy.data.objects.new(name, d)
        ob.location = Vector([float(v) * scale / 100.0 * 4.0 for v in loc])
        ob.rotation_euler = (Vector((0, 0, 0)) - ob.location) \
            .to_track_quat("-Z", "Y").to_euler()
        out.append(ob)
    return out


def world_grey(strength=0.42, bright=(0.95, 0.945, 0.94),
               dark=(0.74, 0.735, 0.75)):
    """Gradient environment: bright behind the subject, darker at the edges.

    A flat grey world reads as a void.  A backdrop *plane* was tried next and left
    a visible wall seam where the plane ended, so the gradient is built into the
    world instead -- no geometry, no seam, and it surrounds the subject.
    """
    w = bpy.data.worlds.new("W")
    w.use_nodes = True
    nt = w.node_tree
    for n in list(nt.nodes):
        if n.type == "BACKGROUND":
            nt.nodes.remove(n)
    out = nt.nodes["World Output"]
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = strength
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    # Z of the view direction runs -1..1; a ColorRamp clamps outside 0..1, which
    # flattened the whole lower hemisphere to one colour and made the field look
    # uniform.  Map it into range first.
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value = -1.0
    mr.inputs["From Max"].default_value = 1.0
    mr.inputs["To Min"].default_value = 0.0
    mr.inputs["To Max"].default_value = 1.0
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = (dark[0], dark[1], dark[2], 1.0)
    ramp.color_ramp.elements[1].position = 1.0
    ramp.color_ramp.elements[1].color = (bright[0], bright[1], bright[2], 1.0)
    nt.links.new(tc.outputs["Generated"], sep.inputs["Vector"])
    nt.links.new(sep.outputs["Z"], mr.inputs["Value"])
    nt.links.new(mr.outputs["Result"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    bpy.context.scene.world = w
    return w


def configure_render(sc):
    sc.render.engine = "CYCLES"
    sc.cycles.samples = SAMPLES
    sc.cycles.use_denoising = True
    sc.cycles.max_bounces = 8
    sc.cycles.transmission_bounces = 8
    sc.cycles.transparent_max_bounces = 8
    sc.cycles.caustics_reflective = False
    sc.cycles.caustics_refractive = False
    sc.render.resolution_x = RES
    sc.render.resolution_y = RES
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = False
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    sc.view_settings.view_transform = "AgX"
    # this build's AgX looks are named "AgX - Base Contrast" etc.; "Medium
    # Contrast" does not exist in 5.4 alpha
    for look in ("AgX - Punchy", "AgX - Medium High Contrast",
                 "AgX - Base Contrast", "None"):
        try:
            sc.view_settings.look = look
            break
        except TypeError:
            continue
    sc.view_settings.exposure = 0.95
    # probe the real device rather than assuming
    dev = "CPU"
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "CUDA"
        prefs.get_devices()
        gpus = [d for d in prefs.devices if d.type == "CUDA"]
        for d in prefs.devices:
            d.use = True
        if gpus:
            sc.cycles.device = "GPU"
            dev = f"GPU/CUDA x{len(gpus)} ({gpus[0].name})"
        else:
            sc.cycles.device = "CPU"
    except Exception as e:
        sc.cycles.device = "CPU"
        dev = f"CPU (probe failed: {type(e).__name__})"
    return dev


def render_to(path):
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    print(f"  wrote {path}")


# ---------------------------------------------------------------- main
def main():
    print(f"=== blender scene: {TAG} ===")
    sc = fresh_scene()
    cols = make_collections()
    dev = configure_render(sc)
    print(f"  device: {dev}")
    world_grey()

    pts, radii, ph = load_cells(MODEL, N_CELLS)
    # everything downstream -- light distance, light power, edge thickness, camera
    # distance -- is expressed relative to the cluster's own size in microns, so
    # the scene works whether the field is 30 um or 300
    subject = float(np.max(pts.max(axis=0) - pts.min(axis=0)))
    print(f"  cluster extent {np.round(np.ptp(pts, axis=0), 1)} um, "
          f"subject {subject:.1f} um")

    # one material per phenotype, violet-leaning
    mats = {}
    for g in sorted(set(ph.tolist())):
        c = blend_to_violet(hex_rgb(PH_BASE[g % len(PH_BASE)]), 0.42)
        mats[g] = membrane_material(f"M_P{g}", c)
    inner_mats = {}
    for g in mats:
        c = blend_to_violet(hex_rgb(PH_BASE[g % len(PH_BASE)]), 0.62)
        inner_mats[g] = membrane_material(f"M_P{g}_core", c,
                                          roughness=0.35, coat=0.15)

    mean_r = float(np.median(radii))
    print(f"  median cell radius {mean_r:.2f} um, "
          f"cluster extent {np.ptp(pts, axis=0)}")

    # The wide shot uses metaballs: cells that touch fuse into one continuous
    # body, which is what packed tissue looks like and what a cloud of discrete
    # spheres can never look like.  The visible radius of a metaball element is
    # r*sqrt(1-sqrt(threshold/stiffness)), so the element radius is scaled up to
    # put the iso-surface at the measured cell radius.
    mb_stiff, mb_thresh = 1.35, 0.60
    scale_r = 1.0 / math.sqrt(1.0 - math.sqrt(mb_thresh / mb_stiff))
    mb_col = bpy.data.collections.new("CellsMetaball")
    cols["Cells"].children.link(mb_col)
    add_metaball_cluster("CellBody", pts, radii * scale_r, ph, mats, mb_col,
                         resolution=mean_r * 0.10, stiffness=mb_stiff)
    print(f"  metaball cluster: {len(pts)} elements, "
          f"element scale x{scale_r:.2f}")

    # separate meshes for the macro shot only; hidden in the wide render so they
    # do not duplicate the metaball surface
    dmat = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    np.fill_diagonal(dmat, 1e9)
    mi, mj = np.unravel_index(np.argmin(dmat), dmat.shape)
    macro_col = bpy.data.collections.new("MacroCell")
    cols["Cells"].children.link(macro_col)
    for k, (i, tag2) in enumerate(((mi, "A"), (mj, "B"))):
        g = int(ph[i])
        add_sphere(f"macro_{tag2}_membrane", pts[i], radii[i], mats[g],
                   segments=128, rings=80, coll=macro_col)
        add_sphere(f"macro_{tag2}_nucleus", pts[i], radii[i] * 0.45,
                   inner_mats[g], segments=64, rings=44, coll=macro_col)
    macro_objs = [o for o in macro_col.objects]

    def set_macro_visible(val):
        for o in macro_objs:
            o.hide_render = not val
            o.hide_viewport = not val
        for o in mb_col.objects:
            o.hide_render = val
            o.hide_viewport = val

    # translucent bounding slab, sized to the cluster, with drawn edges
    lo = pts.min(axis=0) - mean_r * 1.6
    hi = pts.max(axis=0) + mean_r * 1.6
    add_box("Slab", lo, hi, slab_material("M_Slab"), coll=cols["Guides"])
    slab_edges(lo, hi, cols["Guides"], radius=mean_r * 0.055)
    print(f"  slab {np.round(hi - lo, 1)} um")

    for ob in setup_lighting(subject):
        link_to(ob, cols["Lighting"])

    cam = add_camera(cols["Cameras"])
    cam.data.lens = 55.0
    cam.data.sensor_width = 36.0
    cam.data.clip_start = 0.5
    cam.data.clip_end = 10000.0

    ctr = np.zeros(3)
    cam.data.lens = 50.0
    sensor = cam.data.sensor_width
    # frame the slab, not the cluster: the slab is the visible extent of the
    # volume, and framing the cells cropped the box
    box_extent = float(np.max(hi - lo))
    fill = 0.86
    dist = box_extent * cam.data.lens / (sensor * fill)

    # -- wide: oblique, the fused cell body inside its slab
    set_macro_visible(False)
    cam.data.dof.use_dof = True
    cam.data.dof.aperture_fstop = 4.0
    dirv = np.array([1.15, -1.55, 0.62])
    dirv = dirv / np.linalg.norm(dirv)
    eye = ctr + dirv * dist
    tgt = ctr.copy()
    cam.data.dof.focus_distance = dist
    aim(cam, eye, tgt)
    print(f"  wide: box {box_extent:.0f} um, camera at {dist:.0f} um")
    render_to(os.path.join(OUTDIR, f"{TAG}_wide.png"))

    # -- macro: a close-up of one cell, shallow focus
    set_macro_visible(True)
    mid = pts[mi]
    cam.data.lens = 85.0
    cam.data.dof.aperture_fstop = 1.2
    span_macro = radii[mi] * 3.2
    dclose = span_macro * cam.data.lens / (cam.data.sensor_width * 0.72)
    dirv = np.array([0.55, -1.30, 0.34])
    dirv = dirv / np.linalg.norm(dirv)
    eye = mid + dirv * dclose
    cam.data.dof.focus_distance = dclose
    print(f"  macro: cell {mi} r={radii[mi]:.2f} um, camera at {dclose:.0f} um")
    aim(cam, eye, mid + np.array([radii[mi] * 0.10, 0, 0]))
    render_to(os.path.join(OUTDIR, f"{TAG}_macro.png"))

    bpy.ops.wm.save_as_mainfile(
        filepath=os.path.join(OUTDIR, f"{TAG}_scene.blend"))
    print(f"  saved {TAG}_scene.blend")
    n_cells = len(bpy.data.objects)
    print(f"  objects {n_cells}, collections "
          f"{[c.name for c in bpy.data.collections['cellhippo'].children]}")


main()
