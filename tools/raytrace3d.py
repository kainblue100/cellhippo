"""Analytic ray tracer for the 3D cell model.

Written rather than borrowed because neither Blender nor bpy is available here
(Blender ships bpy only for Python 3.10/3.11 wheels and this environment is 3.12),
and pyvista would need VTK.  Only numpy and PIL are required, and every visual
term is then under our control -- which matters, because the target look is
specific: translucent glossy membranes, specular sheen, internal mottling, a
translucent bounding slab, and depth-of-field blur.

Rendering strategy.  Rays are not traced per pixel against every sphere; each
sphere is instead rasterised into its own screen-space bounding box and resolved
analytically there.  That turns an O(pixels x spheres) problem into roughly
O(pixels_covered), which is what makes a few hundred translucent spheres at
1200x1200 tractable in numpy.

Compositing is back-to-front over up to `layers` spheres per pixel, so overlapping
cells really do show through one another instead of the nearest one winning.

Depth of field is true aperture sampling: the camera origin is jittered within a
disc and the focal distance is fixed, so out-of-focus geometry blurs by exactly
the amount the optics imply.
"""
from __future__ import annotations

import numpy as np

# ------------------------------------------------------------------ utilities


def _norm(v, axis=-1):
    return v / np.maximum(np.linalg.norm(v, axis=axis, keepdims=True), 1e-12)


def _valnoise(seed, n=512):
    rng = np.random.default_rng(seed)
    return rng.random((n, n)).astype(np.float32)


def sample_noise(noise, u, v):
    """Bilinear sample of a tiling noise field; u,v in [0, inf)."""
    n = noise.shape[0]
    x = (u * n) % n
    y = (v * n) % n
    x0 = np.floor(x).astype(np.int64) % n
    y0 = np.floor(y).astype(np.int64) % n
    x1 = (x0 + 1) % n
    y1 = (y0 + 1) % n
    fx = (x - np.floor(x)).astype(np.float32)
    fy = (y - np.floor(y)).astype(np.float32)
    a = noise[y0, x0] * (1 - fx) + noise[y0, x1] * fx
    b = noise[y1, x0] * (1 - fx) + noise[y1, x1] * fx
    return a * (1 - fy) + b * fy


class Camera:
    """Pinhole camera with an aperture, looking at a target."""

    def __init__(self, eye, target, up, fov_deg, width, height, focal=None,
                 aperture=0.0):
        self.eye = np.asarray(eye, dtype=np.float64)
        self.target = np.asarray(target, dtype=np.float64)
        self.up = _norm(np.asarray(up, dtype=np.float64))
        self.width = width
        self.height = height
        self.aperture = aperture
        fwd = _norm(self.target - self.eye)
        right = _norm(np.cross(fwd, self.up))
        true_up = np.cross(right, fwd)
        self.fwd, self.right, self.upv = fwd, right, true_up
        self.focal = float(np.linalg.norm(self.target - self.eye)) \
            if focal is None else float(focal)
        self.tan_half = np.tan(np.deg2rad(fov_deg) / 2)

    def ray_dirs(self, px, py, jitter=None):
        """Direction cosines for pixel coordinates (float arrays, subpixel)."""
        w, h = self.width, self.height
        aspect = w / h
        sx = (px + 0.5) / w * 2 - 1
        sy = 1 - (py + 0.5) / h * 2
        d = (self.fwd[None, None, :]
             + self.right[None, None, :] * (sx * self.tan_half * aspect)[..., None]
             + self.upv[None, None, :] * (sy * self.tan_half)[..., None])
        if jitter is not None:
            d = d + jitter
        return d


# ------------------------------------------------------------------ scene


class Sphere:
    __slots__ = ("c", "r", "col", "rough", "opacity", "tint")

    def __init__(self, c, r, col, rough=0.18, opacity=0.72, tint=0.35):
        self.c = np.asarray(c, dtype=np.float64)
        self.r = float(r)
        self.col = np.asarray(col, dtype=np.float64)
        self.rough = rough
        self.opacity = opacity
        self.tint = tint


def _shade_sphere(spheres, s_idx, points, normals, cam, noise, bg_fn):
    """Shading for one sphere's hit pixels: membrane + interior + specular."""
    s = spheres[s_idx]
    n = normals
    view = _norm(cam.eye[None, None, :] - points)

    # membrane: a thin translucent shell, lit softly from the upper left
    ldir = _norm(np.array([-0.45, -0.75, 0.5]))
    ldir2 = _norm(np.array([0.6, 0.25, -0.35]))
    ndl = np.clip(n @ ldir, 0, 1)
    ndl2 = np.clip(n @ ldir2, 0, 1)
    diff = 0.62 * ndl + 0.22 * ndl2

    half = _norm(ldir[None, None, :] + view)
    spec = np.clip((n * half).sum(-1), 0, 1) ** (1.0 / max(s.rough, 1e-3))
    half2 = _norm(ldir2[None, None, :] + view)
    spec2 = np.clip((n * half2).sum(-1), 0, 1) ** (1.0 / max(s.rough, 1e-3))

    # rim / Fresnel: bright at grazing angles, the wet-membrane look
    fres = (1.0 - np.clip((n * view).sum(-1), 0, 1)) ** 2.4

    # interior mottling along the local surface
    m = sample_noise(noise, (points[..., 0] * 1.6 + points[..., 1] * 0.3),
                     (points[..., 2] * 1.6 + points[..., 1] * 0.4))
    interior = s.col[None, None, :] * (0.55 + 0.5 * m[..., None])

    base = 0.30 * s.col[None, None, :] + 0.70 * diff[..., None] * s.col[None, None, :]
    col = 0.62 * base + 0.38 * interior
    col = col + (0.55 * spec + 0.25 * spec2)[..., None] * np.array([0.86, 0.95, 1.0])
    col = col + 0.42 * fres[..., None] * np.array([0.72, 0.78, 0.92])

    alpha = np.clip(s.opacity * (0.55 + 0.75 * fres), 0, 1)
    return col, alpha


def render(spheres, cam, width, height, samples=6, seed=0, slab=None,
           background=(0.86, 0.855, 0.85), bgn=0.0, layers=5, zoom_blur=1.0,
           exposure=1.0):
    """Render the scene.  Returns float RGB in [0,1]."""
    rng = np.random.default_rng(seed)
    noise = _valnoise(seed + 11)
    H, W = height, width

    # pixel grid (shared by all aperture samples)
    ys, xs = np.mgrid[0:H, 0:W]
    px = xs.astype(np.float64)
    py = ys.astype(np.float64)

    acc = np.zeros((H, W, 3), dtype=np.float64)

    # pre-project sphere centres once, for culling
    ctr_rel = np.array([s.c - cam.eye for s in spheres])
    depth0 = ctr_rel @ cam.fwd

    for smp in range(samples):
        # depth of field: jitter the origin inside the aperture disc
        if cam.aperture > 0:
            th = rng.random() * 2 * np.pi
            rr = np.sqrt(rng.random()) * cam.aperture * zoom_blur
            off = cam.right * (rr * np.cos(th)) + cam.upv * (rr * np.sin(th))
        else:
            off = np.zeros(3)
        eye = cam.eye + off

        col_buf = np.empty((H, W, 3), dtype=np.float64)
        # background: pale grey with a soft mottled texture
        col_buf[:] = np.asarray(background, dtype=np.float64)
        if bgn > 0:
            nn = sample_noise(noise, px / 900.0 + smp * 0.13, py / 900.0)
            nn2 = sample_noise(noise, px / 210.0, py / 210.0 + smp * 0.07)
            tex = (nn - 0.5) * 0.055 + (nn2 - 0.5) * 0.028
            col_buf += tex[..., None]
        alpha_buf = np.zeros((H, W), dtype=np.float64)
        depth_buf = np.full((H, W), np.inf, dtype=np.float64)

        order = np.argsort(-depth0)          # back to front
        for s_idx in order:
            s = spheres[s_idx]
            rel = s.c - eye
            depth = float(rel @ cam.fwd)
            if depth <= 1e-6:
                continue
            # project the sphere to a screen-space box, with margin
            sx = float(rel @ cam.right) / depth
            sy = float(rel @ cam.upv) / depth
            aspect = W / H
            cx = (sx / (cam.tan_half * aspect) * 0.5 + 0.5) * W
            cy = (0.5 - sy / cam.tan_half * 0.5) * H
            rpx = (s.r / depth) / (cam.tan_half * aspect) * 0.5 * W
            if rpx < 0.6:
                continue
            m = rpx * 1.5 + 2
            x0 = max(0, int(cx - m)); x1 = min(W, int(cx + m) + 1)
            y0 = max(0, int(cy - m)); y1 = min(H, int(cy + m) + 1)
            if x1 <= x0 or y1 <= y0:
                continue

            sub_x = px[y0:y1, x0:x1]
            sub_y = py[y0:y1, x0:x1]
            d = cam.ray_dirs(sub_x, sub_y)

            oc = (eye - s.c).astype(np.float64)
            b = (d * oc[None, None, :]).sum(-1)
            c = float(oc @ oc) - s.r * s.r
            disc = b * b - c
            hit = disc > 0
            if not hit.any():
                continue
            sq = np.sqrt(np.maximum(disc, 0))
            t0 = -b - sq
            t1 = -b + sq
            # front surface; if the front t is behind the camera, no hit here
            t = np.where(t0 > 1e-6, t0, t1)
            valid = hit & (t > 1e-6)
            if not valid.any():
                continue

            pts = eye[None, None, :] + d * t[..., None]
            nrm = _norm(pts - s.c)
            sh_col, sh_a = _shade_sphere(spheres, s_idx, pts, nrm, cam,
                                         noise, None)

            aa = np.where(valid, sh_a, 0.0)
            region = col_buf[y0:y1, x0:x1]
            areg = alpha_buf[y0:y1, x0:x1]
            # back-to-front: this sphere is behind everything already drawn
            # (painter order), so it composites underneath
            new_a = areg + aa * (1 - areg)
            safe = np.maximum(new_a, 1e-9)
            blended = (region * areg[..., None]
                       + sh_col * aa[..., None] * (1 - areg[..., None]))
            region[valid] = (blended / safe[..., None])[valid]
            areg[valid] = new_a[valid]
            col_buf[y0:y1, x0:x1] = region
            alpha_buf[y0:y1, x0:x1] = areg

            dbuf = depth_buf[y0:y1, x0:x1]
            newd = np.where(valid & (t < dbuf), t, dbuf)
            depth_buf[y0:y1, x0:x1] = newd

        # faint translucent slab: only its edges and a whisper of its faces
        if slab is not None:
            col_buf = _draw_slab(col_buf, slab, cam, eye, background)

        acc += col_buf

    img = acc / max(1, samples)
    img = np.clip(img * exposure, 0, 1)
    return img


def _draw_slab(img, slab, cam, eye, background):
    """Twelve soft edges of the bounding box, projected as lines."""
    (mn, mx) = slab
    mn = np.asarray(mn, dtype=np.float64)
    mx = np.asarray(mx, dtype=np.float64)
    corners = np.array([[x, y, z] for x in (mn[0], mx[0])
                        for y in (mn[1], mx[1]) for z in (mn[2], mx[2])])
    edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
             (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    H, W = img.shape[:2]
    aspect = W / H
    for i, j in edges:
        p0, p1 = corners[i], corners[j]
        n = 160
        ts = np.linspace(0, 1, n)
        pts = p0[None, :] * (1 - ts)[:, None] + p1[None, :] * ts[:, None]
        rel = pts - eye
        dep = rel @ cam.fwd
        ok = dep > 1e-6
        if ok.sum() < 2:
            continue
        sx = (rel @ cam.right)[ok] / dep[ok]
        sy = (rel @ cam.upv)[ok] / dep[ok]
        cx = (sx / (cam.tan_half * aspect) * 0.5 + 0.5) * W
        cy = (0.5 - sy / cam.tan_half * 0.5) * H
        # draw as soft dots
        for xx, yy in zip(cx, cy):
            xi, yi = int(xx), int(yy)
            if 1 <= xi < W - 1 and 1 <= yi < H - 1:
                img[yi - 1:yi + 2, xi - 1:xi + 2] = (
                    img[yi - 1:yi + 2, xi - 1:xi + 2] * 0.55
                    + np.asarray([0.62, 0.60, 0.66]) * 0.45)
    return img


def depth_blur(img, depth, focus, strength=1.6, max_r=9):
    """Cheap post-hoc defocus from a depth buffer.

    Aperture sampling already handles the wide views, but a strong close-up needs
    a large blur radius that would cost too many samples; this adds the remainder
    as a depth-dependent box blur.
    """
    from PIL import Image
    H, W = img.shape[:2]
    out = img.copy()
    steps = 5
    for k in range(1, steps + 1):
        w = k / steps
        r = int(round(max_r * w))
        if r < 1:
            continue
        coc = np.clip(np.abs(1.0 / np.maximum(depth, 1e-6)
                             - 1.0 / max(focus, 1e-6)) * focus * strength, 0, 1)
        m = np.clip((coc - w + 1.0 / steps) * steps, 0, 1)
        if m.max() < 1e-3:
            continue
        pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
        blur = np.asarray(pil.filter(__import__("PIL.ImageFilter",
                                                fromlist=["GaussianBlur"])
                                     .GaussianBlur(r))).astype(np.float64) / 255.0
        out = out * (1 - m[..., None]) + blur * m[..., None]
    return out
