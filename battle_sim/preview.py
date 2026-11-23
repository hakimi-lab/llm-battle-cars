"""
Headless Cross-Platform Per-Pixel Z-Buffer 3D Vehicle & Arena Blueprint Renderer (`arena.py preview` / `preview-arena`, v3.1).

Uses a pure vectorized NumPy 2x-SSAA per-pixel Z-buffer 3D rasterizer over exact MuJoCo `mj_forward` / `mj_step`
3D geom poses — eliminating all Painter's-Algorithm occlusion/sorting glitches from every viewing angle while running
100% headless on Linux, macOS, Windows, and cloud sandboxes without requiring OpenGL / EGL / X11 drivers!
"""
from __future__ import annotations

import math
import warnings as py_warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from .arenas import (
    ARENA_META,
    ICE_INNER_HX,
    ICE_INNER_HY,
    ICE_OUTER_HX,
    ICE_OUTER_HY,
    STAGE_MUD_ZONES,
    STAGE_PORTALS,
    STAGE_SPRINGS,
    build_arena_mjcf,
    resolve_stage_key,
    step_arena_hazards,
)
from .catalog import PHYSICS_DT
from .compiler import _terrain_surface_z_at, compile_single_car_model, full_validate_car


def _configure_cjk_font() -> bool:
    preferred = (
        "Noto Sans CJK SC",
        "Noto Serif CJK SC",
        "WenQuanYi Micro Hei",
        "PingFang SC",
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
    )
    available = {f.name for f in fm.fontManager.ttflist}
    for name in preferred:
        if name in available:
            matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return True
    return False


_HAS_CJK_FONT = _configure_cjk_font()


def _safe_label(txt: str) -> str:
    if _HAS_CJK_FONT:
        return txt
    return txt.encode("ascii", errors="ignore").decode("ascii").strip()


def _box_faces(hx: float, hy: float, hz: float) -> List[np.ndarray]:
    v = np.array([
        [-hx, -hy, -hz], [ hx, -hy, -hz], [ hx,  hy, -hz], [-hx,  hy, -hz],
        [-hx, -hy,  hz], [ hx, -hy,  hz], [ hx,  hy,  hz], [-hx,  hy,  hz],
    ], dtype=np.float64)
    idx = [
        [3, 2, 1, 0],  # -Z bottom (outward normal -Z)
        [4, 5, 6, 7],  # +Z top    (outward normal +Z)
        [0, 1, 5, 4],  # -Y side   (outward normal -Y)
        [2, 3, 7, 6],  # +Y side   (outward normal +Y)
        [1, 2, 6, 5],  # +X front  (outward normal +X)
        [3, 0, 4, 7],  # -X back   (outward normal -X)
    ]
    return [v[f] for f in idx]


def _cylinder_faces(rad: float, hh: float, n_seg: int = 24, cone: bool = False) -> List[np.ndarray]:
    angles = np.linspace(0.0, 2.0 * math.pi, n_seg, endpoint=False)
    top_r = 0.04 * rad if cone else rad
    bot = np.stack([rad * np.cos(angles), rad * np.sin(angles), np.full_like(angles, -hh)], axis=1)
    top = np.stack([top_r * np.cos(angles), top_r * np.sin(angles), np.full_like(angles, hh)], axis=1)
    faces: List[np.ndarray] = [bot[::-1], top]
    for i in range(n_seg):
        j = (i + 1) % n_seg
        faces.append(np.array([bot[i], bot[j], top[j], top[i]], dtype=np.float64))
    return faces


def _elliptical_disc_faces(rx: float, ry: float, hh: float, n_seg: int = 36) -> List[np.ndarray]:
    angles = np.linspace(0.0, 2.0 * math.pi, n_seg, endpoint=False)
    bot = np.stack([rx * np.cos(angles), ry * np.sin(angles), np.full_like(angles, -hh)], axis=1)
    top = np.stack([rx * np.cos(angles), ry * np.sin(angles), np.full_like(angles, hh)], axis=1)
    faces: List[np.ndarray] = [bot[::-1], top]
    for i in range(n_seg):
        j = (i + 1) % n_seg
        faces.append(np.array([bot[i], bot[j], top[j], top[i]], dtype=np.float64))
    return faces


def _sphere_faces(rx: float, ry: float, rz: float, n_lat: int = 8, n_lon: int = 14) -> List[np.ndarray]:
    faces: List[np.ndarray] = []
    lats = np.linspace(-0.5 * math.pi, 0.5 * math.pi, n_lat + 1)
    lons = np.linspace(0.0, 2.0 * math.pi, n_lon, endpoint=False)
    for i in range(n_lat):
        th0, th1 = lats[i], lats[i + 1]
        for j in range(n_lon):
            ph0 = lons[j]
            ph1 = lons[(j + 1) % n_lon]
            p00 = [rx * math.cos(th0) * math.cos(ph0), ry * math.cos(th0) * math.sin(ph0), rz * math.sin(th0)]
            p10 = [rx * math.cos(th0) * math.cos(ph1), ry * math.cos(th0) * math.sin(ph1), rz * math.sin(th0)]
            p11 = [rx * math.cos(th1) * math.cos(ph1), ry * math.cos(th1) * math.sin(ph1), rz * math.sin(th1)]
            p01 = [rx * math.cos(th1) * math.cos(ph0), ry * math.cos(th1) * math.sin(ph0), rz * math.sin(th1)]
            faces.append(np.array([p00, p10, p11, p01], dtype=np.float64))
    return faces


# ============================================================================
# Pure-NumPy 2x-SSAA Per-Pixel Z-Buffer 3D Rasterizer
# ============================================================================

def _camera_basis(elev_deg: float, azim_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    el = math.radians(elev_deg)
    az = math.radians(azim_deg)
    cam_back = np.array([
        math.cos(el) * math.cos(az),
        math.cos(el) * math.sin(az),
        math.sin(el),
    ], dtype=np.float64)
    fwd = -cam_back
    fwd /= max(1e-12, float(np.linalg.norm(fwd)))

    if abs(math.sin(el)) > 0.998:
        # Exact top-down / bottom-up view without gimbal lock:
        # For azim=-90 deg, +X is screen right and +Y is screen up!
        up_ref = np.array([-math.sin(az), -math.cos(az) if el < 0 else -math.sin(az + 0.5 * math.pi), 0.0], dtype=np.float64)
        if abs(azim_deg + 90.0) < 1e-3:
            up_ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        else:
            up_ref = np.array([-math.cos(az), -math.sin(az), 0.0], dtype=np.float64)
    else:
        up_ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    right = np.cross(fwd, up_ref)
    rn = float(np.linalg.norm(right))
    if rn < 1e-9:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        right /= rn
    up = np.cross(right, fwd)
    up /= max(1e-12, float(np.linalg.norm(up)))
    return fwd, right, up


def _project_points(
    pts: np.ndarray,
    cam_pos: np.ndarray,
    fwd: np.ndarray,
    right: np.ndarray,
    up: np.ndarray,
    focal_dist: float,
    half_span_x: float,
    half_span_y: float,
    w_ss: int,
    h_ss: int,
    perspective: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rel = pts - cam_pos
    zc = rel @ fwd
    xc = rel @ right
    yc = rel @ up
    zc_safe = np.maximum(0.08, zc)
    if perspective:
        scale_p = focal_dist / zc_safe
        xn = (xc * scale_p) / half_span_x
        yn = (yc * scale_p) / half_span_y
    else:
        xn = xc / half_span_x
        yn = yc / half_span_y
    px = (xn * 0.5 + 0.5) * (w_ss - 1)
    py = (0.5 - yn * 0.5) * (h_ss - 1)
    return px, py, zc


def _draw_depth_line(
    img: np.ndarray,
    zbuf: np.ndarray,
    x0: float,
    y0: float,
    z0: float,
    x1: float,
    y1: float,
    z1: float,
    rgb: Tuple[float, float, float],
    alpha: float = 1.0,
    thickness_ss: int = 1,
    z_bias: float = 0.004,
) -> None:
    h_ss, w_ss, _ = img.shape
    if z0 <= 0.08 and z1 <= 0.08:
        return
    dx = x1 - x0
    dy = y1 - y0
    steps = int(max(abs(dx), abs(dy))) + 1
    if steps <= 0 or steps > 6000:
        return
    t = np.linspace(0.0, 1.0, steps, dtype=np.float64)
    xs = np.rint(x0 + dx * t).astype(np.int32)
    ys = np.rint(y0 + dy * t).astype(np.int32)
    zs = z0 + (z1 - z0) * t - z_bias

    col_arr = np.array(rgb, dtype=np.float32)
    offsets = [(0, 0)] if thickness_ss <= 1 else [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)]
    for ox, oy in offsets:
        xx = xs + ox
        yy = ys + oy
        valid = (xx >= 0) & (xx < w_ss) & (yy >= 0) & (yy < h_ss) & (zs > 0.08)
        if not np.any(valid):
            continue
        xv = xx[valid]
        yv = yy[valid]
        zv = zs[valid]
        vis = zv <= zbuf[yv, xv]
        if not np.any(vis):
            continue
        xvv = xv[vis]
        yvv = yv[vis]
        if alpha >= 0.99:
            img[yvv, xvv] = col_arr
        else:
            img[yvv, xvv] = img[yvv, xvv] * (1.0 - alpha) + col_arr * alpha


def _render_zbuffer_scene(
    polys: Sequence[np.ndarray],
    face_colors: Sequence[Tuple[float, float, float, float]],
    edge_colors: Sequence[Tuple[float, float, float, float]],
    lookat: np.ndarray,
    half_span: float,
    elev_deg: float,
    azim_deg: float,
    width_px: int = 640,
    height_px: int = 480,
    perspective: bool = True,
    bg_rgb: Tuple[float, float, float] = (0.106, 0.129, 0.180),
    lines_3d: Optional[Sequence[Tuple[np.ndarray, np.ndarray, Tuple[float, float, float], int]]] = None,
    overlay_points: Optional[Sequence[Tuple[np.ndarray, Tuple[float, float, float], int, bool]]] = None,
) -> Tuple[np.ndarray, Any]:
    """
    True per-pixel Z-buffered 2x-SSAA software rasterizer.
    Returns `(rgb_float_image_HxWx3, project_fn_2d)` where `project_fn_2d(xyz)` maps world 3D -> `(px, py)` in final pixels.
    """
    ss = 2
    w_ss = width_px * ss
    h_ss = height_px * ss
    aspect = float(width_px) / max(1.0, float(height_px))
    half_span_x = float(half_span)
    half_span_y = float(half_span) / aspect

    fwd, right, up = _camera_basis(elev_deg, azim_deg)
    focal_dist = 4.35 * max(half_span_x, half_span_y)
    cam_pos = np.asarray(lookat, dtype=np.float64) - focal_dist * fwd

    img = np.empty((h_ss, w_ss, 3), dtype=np.float32)
    img[:, :, 0] = bg_rgb[0]
    img[:, :, 1] = bg_rgb[1]
    img[:, :, 2] = bg_rgb[2]
    zbuf = np.full((h_ss, w_ss), np.inf, dtype=np.float64)

    # Split polygons into opaque & translucent triangles + perimeter edges
    opaque_tris: List[np.ndarray] = []
    opaque_rgbs: List[Tuple[float, float, float]] = []
    trans_tris: List[Tuple[float, np.ndarray, Tuple[float, float, float, float]]] = []
    edge_segs: List[Tuple[np.ndarray, np.ndarray, Tuple[float, float, float, float]]] = []

    for poly, fcol, ecol in zip(polys, face_colors, edge_colors):
        n_v = len(poly)
        if n_v < 3:
            continue
        al = float(fcol[3])
        rgb = (float(fcol[0]), float(fcol[1]), float(fcol[2]))
        if al >= 0.94:
            for k in range(1, n_v - 1):
                opaque_tris.append(np.array([poly[0], poly[k], poly[k + 1]], dtype=np.float64))
                opaque_rgbs.append(rgb)
        else:
            c_mid = poly.mean(axis=0)
            depth_c = float((c_mid - cam_pos) @ fwd)
            for k in range(1, n_v - 1):
                trans_tris.append((depth_c, np.array([poly[0], poly[k], poly[k + 1]], dtype=np.float64), fcol))

        if ecol[3] > 0.02:
            for k in range(n_v):
                edge_segs.append((poly[k], poly[(k + 1) % n_v], ecol))

    # 1. Rasterize all opaque triangles with exact per-pixel Z-buffer (two-sided so zero faces ever vanish!)
    if opaque_tris:
        tri_arr = np.asarray(opaque_tris, dtype=np.float64)  # (T, 3, 3)
        flat_pts = tri_arr.reshape(-1, 3)
        px_all, py_all, zc_all = _project_points(
            flat_pts, cam_pos, fwd, right, up, focal_dist, half_span_x, half_span_y, w_ss, h_ss, perspective
        )
        px_t = px_all.reshape(-1, 3)
        py_t = py_all.reshape(-1, 3)
        zc_t = zc_all.reshape(-1, 3)

        for ti in range(len(opaque_tris)):
            z0, z1, z2 = float(zc_t[ti, 0]), float(zc_t[ti, 1]), float(zc_t[ti, 2])
            if z0 <= 0.10 or z1 <= 0.10 or z2 <= 0.10:
                continue
            x0, x1, x2 = float(px_t[ti, 0]), float(px_t[ti, 1]), float(px_t[ti, 2])
            y0, y1, y2 = float(py_t[ti, 0]), float(py_t[ti, 1]), float(py_t[ti, 2])

            xmin = max(0, int(math.floor(min(x0, x1, x2))))
            xmax = min(w_ss - 1, int(math.ceil(max(x0, x1, x2))))
            ymin = max(0, int(math.floor(min(y0, y1, y2))))
            ymax = min(h_ss - 1, int(math.ceil(max(y0, y1, y2))))
            if xmin > xmax or ymin > ymax:
                continue

            denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if abs(denom) < 1e-6:
                continue
            inv_d = 1.0 / denom

            yy, xx = np.mgrid[ymin:ymax + 1, xmin:xmax + 1]
            dx2 = xx - x2
            dy2 = yy - y2
            w0 = ((y1 - y2) * dx2 + (x2 - x1) * dy2) * inv_d
            w1 = ((y2 - y0) * dx2 + (x0 - x2) * dy2) * inv_d
            w2 = 1.0 - w0 - w1

            inside = (w0 >= -1e-5) & (w1 >= -1e-5) & (w2 >= -1e-5)
            if not np.any(inside):
                continue

            if perspective:
                # Perspective-correct per-pixel depth interpolation: 1/z is linear in screen space!
                inv_z = w0 * (1.0 / z0) + w1 * (1.0 / z1) + w2 * (1.0 / z2)
                z_pix = 1.0 / np.maximum(1e-8, inv_z)
            else:
                z_pix = w0 * z0 + w1 * z1 + w2 * z2

            z_sub = zbuf[ymin:ymax + 1, xmin:xmax + 1]
            win = inside & (z_pix < z_sub)
            if np.any(win):
                z_sub[win] = z_pix[win]
                img_sub = img[ymin:ymax + 1, xmin:xmax + 1]
                img_sub[win] = opaque_rgbs[ti]

    # 2. Draw depth-tested 3D reference lines (ground grid, +X nose arrow, launch vectors)
    z_line_bias = 0.0055 * max(0.5, half_span)
    if lines_3d:
        l_pts = np.asarray([p for seg in lines_3d for p in (seg[0], seg[1])], dtype=np.float64)
        lpx, lpy, lzc = _project_points(
            l_pts, cam_pos, fwd, right, up, focal_dist, half_span_x, half_span_y, w_ss, h_ss, perspective
        )
        for idx_l, (_p0, _p1, lrgb, lw_ss) in enumerate(lines_3d):
            i0, i1 = 2 * idx_l, 2 * idx_l + 1
            _draw_depth_line(
                img, zbuf,
                float(lpx[i0]), float(lpy[i0]), float(lzc[i0]),
                float(lpx[i1]), float(lpy[i1]), float(lzc[i1]),
                lrgb, alpha=0.92, thickness_ss=lw_ss, z_bias=z_line_bias,
            )

    # 3. Draw depth-tested polygon boundary edges (only visible front edges pass zbuf check!)
    if edge_segs:
        e_pts = np.asarray([p for seg in edge_segs for p in (seg[0], seg[1])], dtype=np.float64)
        epx, epy, ezc = _project_points(
            e_pts, cam_pos, fwd, right, up, focal_dist, half_span_x, half_span_y, w_ss, h_ss, perspective
        )
        for idx_e, (_p0, _p1, ecol) in enumerate(edge_segs):
            i0, i1 = 2 * idx_e, 2 * idx_e + 1
            _draw_depth_line(
                img, zbuf,
                float(epx[i0]), float(epy[i0]), float(ezc[i0]),
                float(epx[i1]), float(epy[i1]), float(ezc[i1]),
                (ecol[0], ecol[1], ecol[2]), alpha=float(ecol[3]), thickness_ss=1, z_bias=z_line_bias,
            )

    # 4. Composite translucent glass triangles (depth-tested against opaque zbuf)
    if trans_tris:
        trans_tris.sort(key=lambda item: item[0], reverse=True)
        for _d, tri, fcol in trans_tris:
            px_v, py_v, zc_v = _project_points(
                tri, cam_pos, fwd, right, up, focal_dist, half_span_x, half_span_y, w_ss, h_ss, perspective
            )
            z0, z1, z2 = float(zc_v[0]), float(zc_v[1]), float(zc_v[2])
            if z0 <= 0.10 or z1 <= 0.10 or z2 <= 0.10:
                continue
            x0, x1, x2 = float(px_v[0]), float(px_v[1]), float(px_v[2])
            y0, y1, y2 = float(py_v[0]), float(py_v[1]), float(py_v[2])
            xmin = max(0, int(math.floor(min(x0, x1, x2))))
            xmax = min(w_ss - 1, int(math.ceil(max(x0, x1, x2))))
            ymin = max(0, int(math.floor(min(y0, y1, y2))))
            ymax = min(h_ss - 1, int(math.ceil(max(y0, y1, y2))))
            denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if xmin > xmax or ymin > ymax or abs(denom) < 1e-6:
                continue
            inv_d = 1.0 / denom
            yy, xx = np.mgrid[ymin:ymax + 1, xmin:xmax + 1]
            w0 = ((y1 - y2) * (xx - x2) + (x2 - x1) * (yy - y2)) * inv_d
            w1 = ((y2 - y0) * (xx - x2) + (x0 - x2) * (yy - y2)) * inv_d
            w2 = 1.0 - w0 - w1
            inside = (w0 >= 0.0) & (w1 >= 0.0) & (w2 >= 0.0)
            z_pix = w0 * z0 + w1 * z1 + w2 * z2
            vis = inside & (z_pix <= zbuf[ymin:ymax + 1, xmin:xmax + 1])
            if np.any(vis):
                al = float(fcol[3])
                c_rgb = np.array(fcol[:3], dtype=np.float32)
                img_sub = img[ymin:ymax + 1, xmin:xmax + 1]
                img_sub[vis] = img_sub[vis] * (1.0 - al) + c_rgb * al

    # 5. Draw optional markers (depth-tested or X-ray HUD overlay such as CoM)
    if overlay_points:
        for pt3, prgb, rad_ss, depth_test in overlay_points:
            mpx, mpy, mzc = _project_points(
                np.asarray([pt3], dtype=np.float64),
                cam_pos, fwd, right, up, focal_dist, half_span_x, half_span_y, w_ss, h_ss, perspective,
            )
            cx_i, cy_i, cz_f = int(round(float(mpx[0]))), int(round(float(mpy[0]))), float(mzc[0])
            xmin = max(0, cx_i - rad_ss - 2)
            xmax = min(w_ss - 1, cx_i + rad_ss + 2)
            ymin = max(0, cy_i - rad_ss - 2)
            ymax = min(h_ss - 1, cy_i + rad_ss + 2)
            if xmin > xmax or ymin > ymax or cz_f <= 0.1:
                continue
            yy, xx = np.mgrid[ymin:ymax + 1, xmin:xmax + 1]
            d2 = (xx - cx_i) ** 2 + (yy - cy_i) ** 2
            ring = (d2 <= (rad_ss + 2) ** 2) & (d2 > rad_ss ** 2)
            core = d2 <= rad_ss ** 2
            if depth_test:
                z_sub = zbuf[ymin:ymax + 1, xmin:xmax + 1]
                ring = ring & (cz_f - z_line_bias <= z_sub)
                core = core & (cz_f - z_line_bias <= z_sub)
            img_sub = img[ymin:ymax + 1, xmin:xmax + 1]
            img_sub[ring] = (0.06, 0.08, 0.12)
            img_sub[core] = prgb

    # 6. Downsample 2x2 SSAA -> crisp anti-aliased (H, W, 3) image
    aa_img = img.reshape(height_px, ss, width_px, ss, 3).mean(axis=(1, 3))
    aa_img = np.clip(aa_img, 0.0, 1.0)

    def project_2d(pt_xyz: Sequence[float]) -> Tuple[float, float]:
        p_arr = np.asarray([pt_xyz], dtype=np.float64)
        px1, py1, _ = _project_points(
            p_arr, cam_pos, fwd, right, up, focal_dist, half_span_x, half_span_y, width_px, height_px, perspective
        )
        return float(px1[0]), float(py1[0])

    return aa_img, project_2d


# ============================================================================
# Public Preview Entrypoints (`arena.py preview`, `preview-arena`, `test`)
# ============================================================================

def render_car_preview(spec: Dict[str, Any], out_png: Path) -> Dict[str, Any]:
    """Render 4-panel Z-buffered 3D studio blueprint PNG (Isometric, Side, Front, Top) and return diagnostics."""
    ok, errors, _warnings, summary = full_validate_car(spec)
    if not ok:
        raise ValueError("Cannot preview invalid car:\n  - " + "\n  - ".join(errors))

    model, data, diag = compile_single_car_model(spec)
    styles = diag["geom_styles"]

    light_dir = np.array([0.48, -0.55, 0.68], dtype=np.float64)
    light_dir /= np.linalg.norm(light_dir)

    world_polys: List[np.ndarray] = []
    face_colors: List[Tuple[float, float, float, float]] = []
    edge_colors: List[Tuple[float, float, float, float]] = []

    for gi in range(model.ngeom):
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gi) or ""
        if not gname.startswith("A_") or gname.endswith("_flame") or "_bolt_" in gname:
            continue
        gtype = int(model.geom_type[gi])
        sz = model.geom_size[gi]
        pos = data.geom_xpos[gi]
        R = data.geom_xmat[gi].reshape(3, 3)
        st = styles.get(gname, {})
        base_rgb = np.array(st.get("rgb", model.geom_rgba[gi][:3]), dtype=np.float64)
        is_cone = st.get("deco_shape") == "cone"

        if gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
            local_faces = _box_faces(float(sz[0]), float(sz[1]), float(sz[2]))
        elif gtype == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            local_faces = _cylinder_faces(float(sz[0]), float(sz[1]), n_seg=20, cone=is_cone)
        elif gtype == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
            local_faces = _sphere_faces(float(sz[0]), float(sz[1]), float(sz[2]))
        else:
            local_faces = _sphere_faces(float(sz[0]), float(sz[0]), float(sz[0]))

        for lf in local_faces:
            wf = (lf @ R.T) + pos
            nrm = np.cross(wf[1] - wf[0], wf[2] - wf[0])
            nn = float(np.linalg.norm(nrm))
            if nn > 1e-9:
                nrm /= nn
            hemi = 0.50 + 0.50 * float(nrm[2])
            key_diff = max(0.0, float(np.dot(nrm, light_dir)))
            shade = 0.44 + 0.16 * hemi + 0.44 * key_diff
            if st.get("material") == "neon":
                shaded_rgb = np.clip(base_rgb * 1.18 + 0.12, 0.0, 1.0)
            else:
                shaded_rgb = np.clip(base_rgb * shade, 0.0, 1.0)
            alpha = 0.62 if st.get("material") == "glass" else 0.99
            world_polys.append(wf)
            face_colors.append((float(shaded_rgb[0]), float(shaded_rgb[1]), float(shaded_rgb[2]), alpha))
            edge_colors.append((float(shaded_rgb[0] * 0.35), float(shaded_rgb[1] * 0.35), float(shaded_rgb[2] * 0.35), 0.42))

    all_pts = np.concatenate(world_polys, axis=0) if world_polys else np.zeros((2, 3))
    center = 0.5 * (all_pts.min(axis=0) + all_pts.max(axis=0))
    span_xyz = all_pts.max(axis=0) - all_pts.min(axis=0)
    half_span = max(0.42, 0.64 * float(np.max(span_xyz)))
    grid_ext = max(0.80, round(half_span * 1.10, 2))

    grid_lines: List[Tuple[np.ndarray, np.ndarray, Tuple[float, float, float], int]] = []
    for g in np.linspace(-grid_ext, grid_ext, 9):
        col_g = (0.22, 0.27, 0.36) if abs(g) < 1e-6 else (0.16, 0.20, 0.28)
        grid_lines.append((np.array([g, -grid_ext, 0.0]), np.array([g, grid_ext, 0.0]), col_g, 1))
        grid_lines.append((np.array([-grid_ext, g, 0.0]), np.array([grid_ext, g, 0.0]), col_g, 1))
    # +X forward nose indicator arrow on floor
    ax_len = grid_ext * 0.84
    grid_lines.append((np.array([0.0, 0.0, 0.003]), np.array([ax_len, 0.0, 0.003]), (0.17, 0.90, 0.61), 2))
    grid_lines.append((np.array([ax_len, 0.0, 0.003]), np.array([ax_len - 0.08 * grid_ext, 0.05 * grid_ext, 0.003]), (0.17, 0.90, 0.61), 2))
    grid_lines.append((np.array([ax_len, 0.0, 0.003]), np.array([ax_len - 0.08 * grid_ext, -0.05 * grid_ext, 0.003]), (0.17, 0.90, 0.61), 2))

    com = summary["center_of_mass_m"]
    com_overlay = [(np.array(com, dtype=np.float64), (1.0, 0.23, 0.36), 8, False)]

    views = [
        (1, "3D Isometric View (+X = Front, Z-Buffered)", 25.0, -38.0, True),
        (2, "Side Orthographic (Left +Y -> +X Front)", 0.0, -90.0, False),
        (3, "Front Orthographic (Looking at +X Nose)", 0.0, 0.0, False),
        (4, "Top Orthographic (+X Right, +Y Up)", 90.0, -90.0, False),
    ]

    fig = plt.figure(figsize=(13.6, 9.6), dpi=130, facecolor="#141821")
    meta = spec.get("meta", {})
    title_txt = _safe_label(
        f"{meta.get('car_name', 'Unnamed')}  |  Model: {meta.get('model_name', 'Unknown')}  |  "
        f"Points: {summary['build_points']}/{summary['max_build_points']}  |  "
        f"Mass: {summary['total_mass_kg']} kg ({summary['weight_class']})  |  "
        f"Est. Speed: {summary['est_top_speed_mps']} m/s"
    )
    fig.suptitle(title_txt, color="#F3F6FA", fontsize=12.0, fontweight="bold", y=0.965)

    for idx, vtitle, elev, azim, is_persp in views:
        ax = fig.add_subplot(2, 2, idx, facecolor="#1B212E")
        rgb_img, proj2d = _render_zbuffer_scene(
            polys=world_polys,
            face_colors=face_colors,
            edge_colors=edge_colors,
            lookat=center,
            half_span=half_span,
            elev_deg=elev,
            azim_deg=azim,
            width_px=620,
            height_px=420,
            perspective=is_persp,
            bg_rgb=(0.106, 0.129, 0.180),
            lines_3d=grid_lines,
            overlay_points=com_overlay,
        )
        ax.imshow(rgb_img, interpolation="bilinear")
        cx_px, cy_px = proj2d(com)
        ax.text(cx_px + 8, cy_px - 6, "CoM", color="#FF6B81", fontsize=8.0, fontweight="bold")
        nx_px, ny_px = proj2d([ax_len, 0.0, 0.003])
        ax.text(nx_px + 5, ny_px, "+X", color="#2CE69B", fontsize=8.5, fontweight="bold")
        ax.set_title(vtitle, color="#D5DFEE", fontsize=10.5, pad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#2D3748")

    slogan = str(meta.get("slogan", "")).strip()
    footer = f"BBox: {summary['bbox_m'][0]:.2f}m (L) x {summary['bbox_m'][1]:.2f}m (W) x {summary['bbox_m'][2]:.2f}m (H) [Unrestricted]   |   CoM: {com} m"
    if slogan:
        footer += f"   |   Slogan: \"{_safe_label(slogan[:64])}\""
    fig.text(0.5, 0.018, footer, ha="center", color="#9FB0C9", fontsize=9.5)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    with py_warnings.catch_warnings():
        py_warnings.simplefilter("ignore")
        plt.tight_layout(rect=[0.01, 0.04, 0.99, 0.94])
        fig.savefig(out_png, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    return summary


def render_arena_preview(stage: str, out_png: Path, t_sec: float = 0.0) -> Path:
    """
    Render a 2-panel per-pixel Z-buffered diorama blueprint of an arena:
      - Left: 3D Perspective View (showing diorama architecture, slopes, catapults/mud/ice & spawn beacons)
      - Right: Exact 90° Orthographic Top-Down Tactical Map (+X East, +Y North) with coordinate grid
    """
    sk = resolve_stage_key(stage)
    ameta = ARENA_META[sk]
    w_xml, a_xml, trap_names = build_arena_mjcf(sk)
    xml = f"""<mujoco model="arena_preview">
  <compiler angle="degree" autolimits="true"/>
  <option timestep="{PHYSICS_DT:.8f}" gravity="0 0 -9.81" integrator="implicitfast"/>
  <worldbody>{w_xml}</worldbody>
  <actuator>{a_xml}</actuator>
</mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    trap_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, nm) for nm in trap_names]

    if t_sec > 0.0:
        n_steps = max(1, int(round(t_sec / PHYSICS_DT)))
        for s in range(n_steps):
            step_arena_hazards(sk, model, data, s * PHYSICS_DT, trap_ids)
            mujoco.mj_step(model, data)
    else:
        for tid in trap_ids:
            if tid >= 0:
                data.ctrl[tid] = math.radians(28.0)
        for _ in range(60):
            mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

    light_dir = np.array([0.46, -0.56, 0.69], dtype=np.float64)
    light_dir /= np.linalg.norm(light_dir)
    rim_dir = np.array([-0.55, 0.52, 0.65], dtype=np.float64)
    rim_dir /= np.linalg.norm(rim_dir)

    polys: List[np.ndarray] = []
    fcols: List[Tuple[float, float, float, float]] = []
    ecols: List[Tuple[float, float, float, float]] = []

    for gi in range(model.ngeom):
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gi) or ""
        gtype = int(model.geom_type[gi])
        if gtype == int(mujoco.mjtGeom.mjGEOM_PLANE):
            continue
        sz = model.geom_size[gi]
        pos = data.geom_xpos[gi]
        R = data.geom_xmat[gi].reshape(3, 3)
        base_rgb = np.array(model.geom_rgba[gi][:3], dtype=np.float64)

        if gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
            lfaces = _box_faces(float(sz[0]), float(sz[1]), float(sz[2]))
        elif gtype == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            n_c = 48 if ("vis_skin" in gname or "inner_plate" in gname or "plaza" in gname) else 28
            lfaces = _cylinder_faces(float(sz[0]), float(sz[1]), n_seg=n_c)
        elif gtype == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
            if float(sz[2]) <= 0.025:
                lfaces = _elliptical_disc_faces(float(sz[0]), float(sz[1]), float(sz[2]), n_seg=40)
            else:
                lfaces = _sphere_faces(float(sz[0]), float(sz[1]), float(sz[2]), n_lat=8, n_lon=18)
        else:
            lfaces = _sphere_faces(float(sz[0]), float(sz[0]), float(sz[0]), n_lat=8, n_lon=14)

        is_emissive = any(tok in gname for tok in (
            "reactor_glow", "core_ring", "core_bullseye", "spawn_", "hazard_bevel",
            "cliff_rim", "crest_line", "ice_border", "frost_crack", "chev",
        ))
        is_spring = "spring" in gname
        is_ice = ("_ice_" in gname) or ("crystal_" in gname)
        is_mud_wet = ("mud_wet" in gname) or ("mud_bub" in gname)
        suppress_wire = any(tok in gname for tok in (
            "center_deck_", "vis_skin", "inner_plate", "mud_moss", "mud_peat", "mud_wet", "moss_patch",
        ))

        for lf in lfaces:
            wf = (lf @ R.T) + pos
            nrm = np.cross(wf[1] - wf[0], wf[2] - wf[0])
            nn = float(np.linalg.norm(nrm))
            if nn > 1e-9:
                nrm /= nn
            hemi = 0.50 + 0.50 * float(nrm[2])
            key_d = max(0.0, float(np.dot(nrm, light_dir)))
            rim_d = max(0.0, float(np.dot(nrm, rim_dir)))
            shade = 0.44 + 0.16 * hemi + 0.40 * key_d + 0.09 * rim_d
            spec_boost = 0.0
            if is_emissive:
                shade = 1.08
                spec_boost = 0.06
            elif is_spring:
                shade = max(shade, 0.80)
            elif is_ice:
                shade = 0.54 + 0.20 * hemi + 0.34 * key_d
                spec_boost = 0.10 * (key_d ** 3)
            elif is_mud_wet:
                spec_boost = 0.07 * (key_d ** 4)

            srgb = np.clip(base_rgb * shade + spec_boost, 0.0, 1.0)
            polys.append(wf)
            fcols.append((float(srgb[0]), float(srgb[1]), float(srgb[2]), 0.99))
            if suppress_wire:
                ecols.append((0.0, 0.0, 0.0, 0.0))
            else:
                ecols.append((float(srgb[0] * 0.40), float(srgb[1] * 0.40), float(srgb[2] * 0.40), 0.30))

    lines_3d: List[Tuple[np.ndarray, np.ndarray, Tuple[float, float, float], int]] = []
    for g in range(-4, 5):
        lines_3d.append((np.array([float(g), -4.0, -0.52]), np.array([float(g), 4.0, -0.52]), (0.14, 0.18, 0.26), 1))
        lines_3d.append((np.array([-4.0, float(g), -0.52]), np.array([4.0, float(g), -0.52]), (0.14, 0.18, 0.26), 1))

    for sp in STAGE_SPRINGS[sk]:
        sx, sy, sz = float(sp["x"]), float(sp["y"]), float(sp["z"])
        fyaw = math.radians(float(sp["fling_yaw"]))
        dx_u, dy_u = math.cos(fyaw), math.sin(fyaw)
        acol = (1.0, 0.86, 0.24)
        arc_pts = []
        for k_arc in range(7):
            u = k_arc / 6.0
            dist_h = 1.25 * u
            z_arc = sz + 0.14 + 0.56 * math.sin(math.pi * 0.78 * u)
            arc_pts.append(np.array([sx + dist_h * dx_u, sy + dist_h * dy_u, z_arc], dtype=np.float64))
        for k_arc in range(6):
            lines_3d.append((arc_pts[k_arc], arc_pts[k_arc + 1], acol, 2))

    sa = ameta["spawn_a"]
    sb = ameta["spawn_b"]
    za = _terrain_surface_z_at(sk, float(sa[0]), float(sa[1]))
    zb = _terrain_surface_z_at(sk, float(sb[0]), float(sb[1]))
    pa3 = np.array([float(sa[0]), float(sa[1]), za + 0.18], dtype=np.float64)
    pb3 = np.array([float(sb[0]), float(sb[1]), zb + 0.18], dtype=np.float64)
    yawa = math.radians(float(ameta["yaw_a"]))
    yawb = math.radians(float(ameta["yaw_b"]))
    lines_3d.append((np.array([sa[0], sa[1], za + 0.02]), pa3, (0.22, 0.80, 0.99), 2))
    lines_3d.append((pa3, pa3 + np.array([0.45 * math.cos(yawa), 0.45 * math.sin(yawa), 0.0]), (0.22, 0.80, 0.99), 2))
    lines_3d.append((np.array([sb[0], sb[1], zb + 0.02]), pb3, (0.98, 0.44, 0.52), 2))
    lines_3d.append((pb3, pb3 + np.array([0.45 * math.cos(yawb), 0.45 * math.sin(yawb), 0.0]), (0.98, 0.44, 0.52), 2))

    spawn_pts = [
        (pa3, (0.22, 0.80, 0.99), 9, True),
        (pb3, (0.98, 0.44, 0.52), 9, True),
    ]

    if sk == "spring_arena":
        sub_hdr = "Cyber-Neon Colosseum & 4 Edge-Sensitive Catapults"
        p3d_title = "3D Perspective View (Propped 28° Hinged Catapults & Under-Stage Reactor)"
        bg_3d = (0.068, 0.090, 0.138)
    elif sk == "mud_arena":
        sub_hdr = "Emerald Bayou, Stone Causeways & 50%-Slow Mud Bogs"
        p3d_title = "3D Perspective View (Central & Corner Peat Mud Bogs + Dry Flagstone Causeways)"
        bg_3d = (0.072, 0.110, 0.098)
    else:
        sub_hdr = "Glacier Ski-Jump Overpass & Low-Friction Mirror-Ice Rink"
        p3d_title = "3D Perspective View (+0.26m Glacier Ramp + Surrounding mu=0.10 Mirror Ice)"
        bg_3d = (0.075, 0.115, 0.175)

    fig = plt.figure(figsize=(15.4, 7.4), dpi=135, facecolor="#0E131D")
    fig.suptitle(
        _safe_label(f"Stage {ameta['stage_id']}: {ameta['name_zh']} ({sk}) — {sub_hdr}"),
        color="#F8FAFC", fontsize=13.2, fontweight="bold", y=0.96,
    )

    panel_specs = [
        (p3d_title, 30.0, -52.0, True, 4.20),
        ("Orthographic Top-Down Tactical Map (+X East, +Y North)", 90.0, -90.0, False, 3.98),
    ]
    for p_idx, (vtitle, elev, azim, is_persp, hspan) in enumerate(panel_specs, start=1):
        ax = fig.add_subplot(1, 2, p_idx, facecolor="#141B28")
        rgb_img, proj2d = _render_zbuffer_scene(
            polys=polys,
            face_colors=fcols,
            edge_colors=ecols,
            lookat=np.array([0.0, 0.0, -0.04 if is_persp else 0.04], dtype=np.float64),
            half_span=hspan,
            elev_deg=elev,
            azim_deg=azim,
            width_px=700,
            height_px=550,
            perspective=is_persp,
            bg_rgb=bg_3d,
            lines_3d=lines_3d,
            overlay_points=spawn_pts,
        )
        ax.imshow(rgb_img, interpolation="bilinear")
        ax_px, ay_px = proj2d(pa3)
        bx_px, by_px = proj2d(pb3)
        ax.text(ax_px + 8, ay_px - 8, "Spawn A", color="#7DD3FC", fontsize=8.5, fontweight="bold")
        ax.text(bx_px + 8, by_px - 8, "Spawn B", color="#FDA4AF", fontsize=8.5, fontweight="bold")
        for sp in STAGE_SPRINGS[sk]:
            spx, spy = proj2d([sp["x"], sp["y"], sp["z"] + 0.08])
            ax.text(spx, spy + 14, sp["tag"], color="#FDE68A", fontsize=7.4, ha="center", fontweight="bold")
        for mz in STAGE_MUD_ZONES.get(sk, []):
            mpx, mpy = proj2d([mz["x"], mz["y"], 0.04])
            ax.text(mpx, mpy, f"{mz['tag']} (-50% spd)", color="#FDE68A", fontsize=7.5, ha="center", va="center", fontweight="bold")
        if sk == "ice_ridge" and not is_persp:
            cpx, cpy = proj2d([0.0, 0.0, 0.28])
            ax.text(cpx, cpy, "Glacier Ramp (+0.26m)", color="#0F172A", fontsize=7.6, ha="center", va="center", fontweight="bold")
            ipx_w, ipy_w = proj2d([-3.10, 1.80, 0.02])
            ax.text(ipx_w, ipy_w, "Mirror Ice (mu=0.10)", color="#E0F2FE", fontsize=7.3, ha="center", fontweight="bold")
            ipx_e, ipy_e = proj2d([3.10, -1.80, 0.02])
            ax.text(ipx_e, ipy_e, "-8° Ice Chute", color="#E0F2FE", fontsize=7.3, ha="center", fontweight="bold")
        ax.set_title(vtitle, color="#CBD5E1", fontsize=10.6, pad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#2D3748")

    fig.text(0.5, 0.02, _safe_label(ameta["description"]), ha="center", color="#94A3B8", fontsize=7.6)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    with py_warnings.catch_warnings():
        py_warnings.simplefilter("ignore")
        plt.tight_layout(rect=[0.01, 0.05, 0.99, 0.93])
        fig.savefig(out_png, dpi=135, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_png


def _draw_2d_arena_backdrop(ax: Any, stage_key: str) -> None:
    import matplotlib.patches as mpatches

    ax.set_facecolor("#0F141E")
    for g in np.linspace(-3.5, 3.5, 15):
        ax.axhline(g, color="#1C2434", lw=0.5, zorder=0)
        ax.axvline(g, color="#1C2434", lw=0.5, zorder=0)

    if stage_key == "spring_arena":
        ax.add_patch(mpatches.Circle((0, 0), 3.45, facecolor="#2B161E", edgecolor="#F43F5E", lw=1.4, ls="--", zorder=1))
        ax.add_patch(mpatches.Circle((0, 0), 2.45, facecolor="#1A2232", edgecolor="#F59E0B", lw=1.2, zorder=2))
        ax.add_patch(mpatches.Circle((0, 0), 0.62, facecolor="none", edgecolor="#22D3EE", lw=0.9, alpha=0.65, zorder=2))
        ax.text(0, -3.65, "Cyber Colosseum (r=2.45m) | -7.5° Danger Slope (r=3.45m) | 4 Edge-Sensitive Hinged Catapults", color="#94A3B8", fontsize=7.2, ha="center")

    elif stage_key == "mud_arena":
        ax.add_patch(mpatches.Rectangle((-3.60, -2.90), 7.20, 5.80, facecolor="#1E2F23", edgecolor="#EF4444", lw=1.3, ls="--", zorder=1))
        ax.add_patch(mpatches.Rectangle((-3.60, -2.20), 7.20, 4.40, facecolor="#24382B", edgecolor="#4ADE80", lw=0.9, zorder=2))
        for mz in STAGE_MUD_ZONES["mud_arena"]:
            ax.add_patch(mpatches.Ellipse(
                (float(mz["x"]), float(mz["y"])), 2.0 * float(mz["rx"]), 2.0 * float(mz["ry"]),
                facecolor="#3E2414", edgecolor="#A16207", lw=1.4, alpha=0.92, zorder=3,
            ))
            ax.text(float(mz["x"]), float(mz["y"]), "MUD -50%", color="#FDE68A", fontsize=6.6, ha="center", va="center", zorder=4)
        ax.text(0, -3.65, "7.2x5.8m Emerald Bayou | Brown Ellipses = -50% Speed Mud Bogs | -7° N/S Wet Cliff Banks", color="#94A3B8", fontsize=7.1, ha="center")

    else:  # ice_ridge
        ax.add_patch(mpatches.Rectangle((-3.70, -2.90), 7.40, 5.80, facecolor="#25638A", edgecolor="#EF4444", lw=1.3, ls="--", zorder=1))
        ax.add_patch(mpatches.Rectangle((-2.75, -2.90), 5.50, 5.80, facecolor="#3892B8", edgecolor="#7DD3FC", lw=0.9, zorder=1))
        ax.add_patch(mpatches.Rectangle((-ICE_INNER_HX, -ICE_INNER_HY), 2 * ICE_INNER_HX, 2 * ICE_INNER_HY, facecolor="#2B374B", edgecolor="#E0F2FE", lw=1.4, zorder=2))
        ax.add_patch(mpatches.Rectangle((-1.45, -ICE_INNER_HY), 2.90, 2 * ICE_INNER_HY, facecolor="#3B4D68", edgecolor="#93C5FD", lw=0.9, zorder=2))
        ax.add_patch(mpatches.Rectangle((-0.42, -ICE_INNER_HY), 0.84, 2 * ICE_INNER_HY, facecolor="#93B4D4", edgecolor="#38BDF8", lw=1.1, zorder=2))
        ax.text(0, -3.65, "7.4x5.8m Glacier Arena | +0.26m Ski-Jump Crest | Cyan = Mirror Ice (mu=0.10) & -8° Ice Chutes", color="#94A3B8", fontsize=7.1, ha="center")

    for sp in STAGE_SPRINGS.get(stage_key, []):
        sx, sy = float(sp["x"]), float(sp["y"])
        phi = math.radians(float(sp["fling_yaw"]))
        cp, s_p = math.cos(phi), math.sin(phi)
        hx, hy = 0.48, 0.44
        pts = [
            (sx + lx * cp - ly * s_p, sy + lx * s_p + ly * cp)
            for lx, ly in [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
        ]
        ax.add_patch(mpatches.Polygon(pts, closed=True, facecolor="#78350F", edgecolor="#FBBF24", lw=1.2, alpha=0.88, zorder=3))
        h0 = (sx + hx * cp - hy * s_p, sy + hx * s_p + hy * cp)
        h1 = (sx + hx * cp + hy * s_p, sy + hx * s_p - hy * cp)
        ax.plot([h0[0], h1[0]], [h0[1], h1[1]], color="#E2E8F0", lw=2.4, zorder=4)
        ax.arrow(sx - 0.15 * cp, sy - 0.15 * s_p, 0.55 * cp, 0.55 * s_p, head_width=0.14, head_length=0.14, fc="#FBBF24", ec="#0F141E", lw=0.4, zorder=4, length_includes_head=True)


def render_tactical_trace_png(trace_data: Dict[str, Any], out_png: Path) -> Path:
    """
    Render a 2-panel White-Box Tactical Replay PNG (0 extra credits during `arena.py test`):
      - Left: 2D Top-Down Arena Trajectory color-coded by active strategy rule + Energy Cannon shot rays
      - Right: Per-Rule Coverage % Bar Chart (highlighting 0% NEVER_TRIGGERED rules!)
    """
    frames = trace_data.get("frames", [])
    shots = trace_data.get("shots", [])
    rules_cov = trace_data.get("rule_coverage", [])
    stage_key = trace_data.get("stage_key", "spring_arena")

    palette = [
        "#38BDF8", "#FACC15", "#4ADE80", "#C084FC",
        "#FB923C", "#2DD4BF", "#F472B6", "#A3E635", "#94A3B8",
    ]
    rule_color_map: Dict[str, str] = {}
    for idx, rc in enumerate(rules_cov):
        rname = rc["rule"]
        if rname == "default_action":
            rule_color_map[rname] = "#94A3B8"
        else:
            rule_color_map[rname] = palette[idx % (len(palette) - 1)]

    fig = plt.figure(figsize=(14.4, 6.8), dpi=130, facecolor="#0B0F17")
    res_lbl = "WIN" if trace_data.get("winner") == "A" else "LOSS"
    fig.suptitle(
        _safe_label(
            f"White-Box Tactical Trace — {trace_data.get('car_a_name')} vs {trace_data.get('car_b_name')}  |  "
            f"{trace_data.get('stage_name_zh')} ({stage_key})  |  {res_lbl} ({trace_data.get('win_method')}, {trace_data.get('duration_sec')}s)"
        ),
        color="#F8FAFC", fontsize=11.5, fontweight="bold", y=0.965,
    )

    gs = fig.add_gridspec(1, 2, width_ratios=[1.22, 1.0], wspace=0.40)
    ax_map = fig.add_subplot(gs[0, 0])
    _draw_2d_arena_backdrop(ax_map, stage_key)

    if len(frames) >= 2:
        for i in range(len(frames) - 1):
            fb0, fb1 = frames[i], frames[i + 1]
            if math.hypot(fb1["xb"] - fb0["xb"], fb1["yb"] - fb0["yb"]) > 1.25:
                ax_map.plot([fb0["xb"], fb1["xb"]], [fb0["yb"], fb1["yb"]], color="#F472B6", lw=1.1, ls=":", alpha=0.65, zorder=4)
            else:
                ax_map.plot([fb0["xb"], fb1["xb"]], [fb0["yb"], fb1["yb"]], color="#FB7185", lw=1.4, ls="--", alpha=0.55, zorder=4)
        ax_map.scatter([frames[0]["xb"]], [frames[0]["yb"]], color="#FB7185", s=45, marker="s", zorder=5)
        ax_map.scatter([frames[-1]["xb"]], [frames[-1]["yb"]], color="#FB7185", s=65, marker="X", zorder=5)

        for i in range(len(frames) - 1):
            f0, f1 = frames[i], frames[i + 1]
            rcol = rule_color_map.get(f0["rule_a"], "#38BDF8")
            if math.hypot(f1["xa"] - f0["xa"], f1["ya"] - f0["ya"]) > 1.25:
                ax_map.plot([f0["xa"], f1["xa"]], [f0["ya"], f1["ya"]], color="#22D3EE", lw=1.6, ls=":", alpha=0.90, zorder=6)
            else:
                ax_map.plot([f0["xa"], f1["xa"]], [f0["ya"], f1["ya"]], color=rcol, lw=2.6, alpha=0.95, zorder=6)

        for idx_f, f in enumerate(frames):
            if idx_f == 0 or (idx_f + 1) % 60 == 0 or idx_f == len(frames) - 1:
                rcol = rule_color_map.get(f["rule_a"], "#38BDF8")
                dx = 0.24 * math.cos(f["yawa"])
                dy = 0.24 * math.sin(f["yawa"])
                ax_map.arrow(
                    f["xa"], f["ya"], dx, dy,
                    head_width=0.11, head_length=0.11, fc=rcol, ec="#0B0F17", lw=0.5, zorder=8, length_includes_head=True,
                )
                lbl_t = "0s" if idx_f == 0 else (f"{f['t']:.1f}s" if idx_f == len(frames) - 1 else f"{int(round(f['t']))}s")
                ax_map.text(f["xa"] + 0.08, f["ya"] + 0.08, lbl_t, color="#F8FAFC", fontsize=7.0, fontweight="bold", zorder=9)

    for s in shots:
        if s.get("owner") != "A":
            continue
        x0, y0 = s["p0"]
        x1, y1 = s["p1"]
        outc = s["outcome"]
        if outc == "hit_hull":
            ax_map.plot([x0, x1], [y0, y1], color="#22C55E", lw=1.5, alpha=0.9, zorder=7)
            ax_map.scatter([x1], [y1], color="#22C55E", s=55, marker="*", zorder=9)
        elif outc == "blocked_by_terrain":
            ax_map.plot([x0, x1], [y0, y1], color="#F59E0B", lw=1.1, ls=":", alpha=0.75, zorder=7)
            ax_map.scatter([x1], [y1], color="#F59E0B", s=36, marker="x", zorder=9)
        elif outc == "warped":
            ax_map.plot([x0, x1], [y0, y1], color="#E879F9", lw=1.3, ls="-.", alpha=0.85, zorder=7)
        else:
            ax_map.plot([x0, x1], [y0, y1], color="#38BDF8", lw=0.8, ls="--", alpha=0.35, zorder=6)

    ax_map.set_xlim(-3.85, 3.85)
    ax_map.set_ylim(-3.85, 3.85)
    ax_map.set_aspect("equal")
    ax_map.set_title("Top-Down Path by Active Rule  (Green*=BoltHit, Dotted=PortalWarp, Boxes/Rings=Hazards)", color="#CBD5E1", fontsize=8.8)
    ax_map.tick_params(colors="#64748B", labelsize=7.5)
    for sp in ax_map.spines.values():
        sp.set_color("#334155")

    ax_bar = fig.add_subplot(gs[0, 1], facecolor="#111827")
    labels: List[str] = []
    pcts: List[float] = []
    bar_cols: List[str] = []
    annots: List[str] = []

    for rc in reversed(rules_cov):
        rname = rc["rule"]
        prio = f"#{rc['priority']} " if rc["priority"] != "fallback" else "[Fallback] "
        labels.append(_safe_label(f"{prio}{rname[:27]}"))
        pcts.append(float(rc["active_pct"]))
        if rc["status"] == "NEVER_TRIGGERED":
            bar_cols.append("#EF4444")
            annots.append("0.0s (NEVER TRIGGERED!)")
        elif str(rc["status"]).startswith("SKIPPED_"):
            bar_cols.append("#475569")
            annots.append(f"0.0s ({rc['status']})")
        else:
            bar_cols.append(rule_color_map.get(rname, "#38BDF8"))
            shot_str = ""
            if "cannon_shots" in rc and rc["cannon_shots"]["fired"] > 0:
                cs = rc["cannon_shots"]
                shot_str = f" | shots:{cs['hit_hull']}/{cs['fired']}hit"
            annots.append(f"{rc['active_sec']}s ({rc['active_pct']}%, {rc['distance_driven_m']}m){shot_str}")

    y_pos = np.arange(len(labels))
    ax_bar.barh(y_pos, np.maximum(pcts, 1.2), color=bar_cols, alpha=0.88, height=0.58)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(labels, color="#E2E8F0", fontsize=8.5)
    ax_bar.set_xlim(0, 118)
    ax_bar.set_xlabel("Active Match Time (%)", color="#94A3B8", fontsize=8.5)
    ax_bar.set_title("Strategy Rule Coverage & Attribution (Top-to-Bottom Priority)", color="#CBD5E1", fontsize=9.5)
    ax_bar.tick_params(colors="#64748B", labelsize=7.5)
    for sp in ax_bar.spines.values():
        sp.set_color("#334155")

    for yi, (pv, note) in enumerate(zip(pcts, annots)):
        txt_col = "#FCA5A5" if "NEVER TRIGGERED" in note else "#F1F5F9"
        ax_bar.text(min(72.0, pv + 2.2), yi, note, va="center", color=txt_col, fontsize=7.6, fontweight="bold" if "NEVER" in note else "normal")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    with py_warnings.catch_warnings():
        py_warnings.simplefilter("ignore")
        plt.tight_layout(rect=[0.01, 0.02, 0.99, 0.93])
        fig.savefig(out_png, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_png
