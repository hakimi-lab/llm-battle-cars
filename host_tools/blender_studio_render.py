#!/usr/bin/env python3
"""
LLM Battle-Cars — Native 30 FPS Blender 4.2 LTS Studio Renderer (Cycles OPTIX + Persistent Data, v3.0).

Supports all 3 snag-free slope & reactive tactical-spring arenas, ranged energy bolts,
and auto-scaled 360-degree studio showcase turntables for vehicles of ANY Length x Width x Height:
  1. Stage 1 (`spring_arena`)    — Quad-Spring Ski-Jump Ring
  2. Stage 2 (`camelback_ridge`) — Tabletop Ridge & Vault-Spring Plateau
  3. Stage 3 (`turbo_bowl`)      — Inward Gravity Bowl & Rim Ejectors
  4. `beauty_turntable`          — 360-degree studio showcase pedestal

Usage inside Blender 4.2+:
  blender -b -P host_tools/blender_studio_render.py -- --bundle path/to/replay.json --out path/to/clip.mp4 --samples 20
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import bpy
import numpy as np
from mathutils import Matrix, Vector

FLOOR_Z = -1.85


def configure_cycles_optix(scene: bpy.types.Scene, width: int = 1920, height: int = 1080, samples: int = 20) -> str:
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    chosen_backend = "CUDA"
    for candidate in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = candidate
            prefs.get_devices()
            gpu_devs = [d for d in prefs.devices if d.type != "CPU"]
            if gpu_devs:
                for d in prefs.devices:
                    d.use = (d.type != "CPU")
                chosen_backend = candidate
                break
        except Exception:
            continue

    scene.cycles.device = "GPU"
    scene.cycles.samples = samples
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.028
    scene.cycles.adaptive_min_samples = min(8, samples)
    scene.cycles.max_bounces = 4
    scene.cycles.diffuse_bounces = 2
    scene.cycles.glossy_bounces = 2
    scene.cycles.transmission_bounces = 2
    scene.cycles.volume_bounces = 0
    scene.cycles.transparent_max_bounces = 4
    scene.cycles.sample_clamp_indirect = 8.0
    scene.cycles.caustics_reflective = False
    scene.cycles.caustics_refractive = False

    scene.cycles.use_denoising = True
    scene.cycles.denoiser = "OPENIMAGEDENOISE"
    scene.cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    if hasattr(scene.cycles, "denoising_prefilter"):
        scene.cycles.denoising_prefilter = "FAST"
    if hasattr(scene.cycles, "denoising_use_gpu"):
        scene.cycles.denoising_use_gpu = True

    scene.render.use_persistent_data = True
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.fps = 30
    scene.render.image_settings.file_format = "JPEG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.quality = 96
    scene.render.use_motion_blur = True
    scene.render.motion_blur_shutter = 0.30

    try:
        scene.view_settings.view_transform = "AgX"
        scene.view_settings.look = "AgX - Medium High Contrast"
    except Exception:
        scene.view_settings.view_transform = "Filmic"
        scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0.15
    return chosen_backend


def _srgb_to_linear(c: float) -> float:
    c = max(0.0, min(1.0, float(c)))
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def get_or_create_styled_mat(rgb: List[float], style_kind: str = "glossy") -> bpy.types.Material:
    lin = (_srgb_to_linear(rgb[0]), _srgb_to_linear(rgb[1]), _srgb_to_linear(rgb[2]), 1.0)
    key = f"Mat_{style_kind}_{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}"
    if key in bpy.data.materials:
        return bpy.data.materials[key]

    mat = bpy.data.materials.new(name=key)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = lin

    if style_kind == "metallic":
        bsdf.inputs["Metallic"].default_value = 0.86
        bsdf.inputs["Roughness"].default_value = 0.24
    elif style_kind == "matte":
        bsdf.inputs["Metallic"].default_value = 0.0
        bsdf.inputs["Roughness"].default_value = 0.72
    elif style_kind == "wood":
        bsdf.inputs["Metallic"].default_value = 0.0
        bsdf.inputs["Roughness"].default_value = 0.42
        if "Coat Weight" in bsdf.inputs:
            bsdf.inputs["Coat Weight"].default_value = 0.15
    elif style_kind == "neon":
        bsdf.inputs["Roughness"].default_value = 0.16
        if "Emission Color" in bsdf.inputs:
            bsdf.inputs["Emission Color"].default_value = lin
            bsdf.inputs["Emission Strength"].default_value = 5.2
    elif style_kind == "glass":
        bsdf.inputs["Roughness"].default_value = 0.08
        if "Transmission Weight" in bsdf.inputs:
            bsdf.inputs["Transmission Weight"].default_value = 0.78
    else:
        bsdf.inputs["Metallic"].default_value = 0.08
        bsdf.inputs["Roughness"].default_value = 0.24
        if "Coat Weight" in bsdf.inputs:
            bsdf.inputs["Coat Weight"].default_value = 0.45
    return mat


def add_bevel(obj: bpy.types.Object, width: float = 0.010, segments: int = 2) -> None:
    bev = obj.modifiers.new(name="Bevel", type="BEVEL")
    bev.width = width
    bev.segments = segments
    bev.limit_method = "ANGLE"
    bpy.ops.object.shade_smooth()


def setup_studio_lights_and_floor(
    is_turntable: bool = False,
    bbox_m: Optional[List[float]] = None,
) -> Tuple[bpy.types.Object, bpy.types.Object]:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    world = bpy.data.worlds.new("StudioWorld")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    bg.inputs["Color"].default_value = (0.20, 0.25, 0.34, 1.0)
    bg.inputs["Strength"].default_value = 0.85

    floor_z = -0.02 if is_turntable else FLOOR_Z
    bpy.ops.mesh.primitive_plane_add(size=42.0, location=(0.0, 0.0, floor_z))
    fl = bpy.context.active_object
    fl.data.materials.append(get_or_create_styled_mat([0.14, 0.17, 0.22], "matte"))

    span = max(bbox_m) if (bbox_m and len(bbox_m) == 3) else 1.2
    ped_r = max(1.05, span * 0.68)

    if is_turntable:
        bpy.ops.mesh.primitive_cylinder_add(vertices=96, radius=ped_r, depth=0.08, location=(0.0, 0.0, -0.04))
        ped = bpy.context.active_object
        ped.data.materials.append(get_or_create_styled_mat([0.78, 0.60, 0.38], "wood"))
        add_bevel(ped, 0.012, 3)
        bpy.ops.mesh.primitive_torus_add(major_segments=96, minor_segments=20, major_radius=ped_r + 0.01, minor_radius=0.022, location=(0.0, 0.0, -0.015))
        trim = bpy.context.active_object
        trim.data.materials.append(get_or_create_styled_mat([0.86, 0.60, 0.16], "metallic"))
        bpy.ops.object.shade_smooth()

    bpy.ops.object.light_add(type="SUN", location=(4.5, -5.5, 8.5))
    sun = bpy.context.active_object
    sun.data.energy = 3.6
    sun.data.color = (1.0, 0.95, 0.86)
    sun.data.angle = math.radians(4.5)
    sun.rotation_euler = (math.radians(42), math.radians(18), math.radians(32))

    bpy.ops.object.light_add(type="AREA", location=(-4.5, -4.0, 4.8))
    fill = bpy.context.active_object
    fill.data.energy = 260.0
    fill.data.size = 5.0
    fill.data.color = (0.85, 0.92, 1.0)
    fill.rotation_euler = (math.radians(48), math.radians(-28), math.radians(-35))

    bpy.ops.object.light_add(type="AREA", location=(0.5, 5.8, 3.6))
    rim = bpy.context.active_object
    rim.data.energy = 320.0
    rim.data.size = 4.2
    rim.data.color = (1.0, 0.90, 0.76)
    rim.rotation_euler = (math.radians(-52), math.radians(6), math.radians(175))

    tgt_z = max(0.22, (bbox_m[2] * 0.45) if (bbox_m and len(bbox_m) == 3) else 0.22)
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0.0, 0.0, tgt_z))
    cam_tgt = bpy.context.active_object
    cam_loc = (0.0, -max(2.35, ped_r * 2.15), max(1.05, tgt_z + ped_r * 0.82)) if is_turntable else (0.0, -6.5, 3.65)
    bpy.ops.object.camera_add(location=cam_loc)
    cam = bpy.context.active_object
    cam.data.lens = 42.0 if is_turntable else 36.0
    cam.data.dof.use_dof = True
    cam.data.dof.focus_object = cam_tgt
    cam.data.dof.aperture_fstop = 4.5
    ttc = cam.constraints.new(type="TRACK_TO")
    ttc.target = cam_tgt
    ttc.track_axis = "TRACK_NEGATIVE_Z"
    ttc.up_axis = "UP_Y"
    bpy.context.scene.camera = cam
    return cam, cam_tgt


def spawn_all_geoms(geom_info: List[Dict[str, Any]], is_turntable: bool) -> List[Tuple[int, bpy.types.Object]]:
    MJ_PLANE = 0
    MJ_SPHERE = 2
    MJ_ELLIPSOID = 4
    MJ_CYLINDER = 5
    MJ_BOX = 6

    tracked: List[Tuple[int, bpy.types.Object]] = []
    for g in geom_info:
        gi = int(g["id"])
        gname = g["name"]
        gtype = int(g["type"])
        if gtype == MJ_PLANE or (is_turntable and g["owner"] == "env"):
            continue

        sz = g["size"]
        rgba = g["rgba"]
        st = g.get("style", {}) or {}
        rgb = st.get("rgb", rgba[:3])
        default_mat = (
            "neon" if ("portal_glow" in gname or "portal_arch" in gname or "portal_core" in gname)
            else ("wood" if "deck" in gname or "island" in gname or "slab" in gname else ("metallic" if "trap_" in gname or "wall" in gname else "glossy"))
        )
        mat_kind = st.get("material", default_mat)
        deco_shape = st.get("deco_shape", "")
        mat = get_or_create_styled_mat(rgb, mat_kind)

        if deco_shape == "eye" or "_eye_" in gname:
            rad = float(sz[0])
            bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=16, radius=rad, location=(0, 0, 0))
            obj = bpy.context.active_object
            obj.data.materials.append(get_or_create_styled_mat([0.98, 0.98, 0.99], "glossy"))
            bpy.ops.object.shade_smooth()
            bpy.ops.mesh.primitive_uv_sphere_add(segments=18, ring_count=12, radius=rad * 0.56, location=(rad * 0.62, 0, 0))
            pupil = bpy.context.active_object
            pupil.data.materials.append(get_or_create_styled_mat([0.03, 0.03, 0.04], "glossy"))
            bpy.ops.object.shade_smooth()
            pupil.parent = obj
        elif deco_shape == "cone" or gname.endswith("_flame"):
            rad, hh = float(sz[0]), float(sz[1])
            bpy.ops.mesh.primitive_cone_add(vertices=28, radius1=rad, radius2=rad * 0.06, depth=hh * 2.0, location=(0, 0, 0))
            obj = bpy.context.active_object
            obj.data.materials.append(mat)
            bpy.ops.object.shade_smooth()
        elif deco_shape == "ring":
            rad, hh = float(sz[0]), float(sz[1])
            bpy.ops.mesh.primitive_torus_add(major_segments=48, minor_segments=16, major_radius=rad, minor_radius=max(0.015, hh), location=(0, 0, 0))
            obj = bpy.context.active_object
            obj.data.materials.append(mat)
            bpy.ops.object.shade_smooth()
        elif gtype == MJ_SPHERE:
            bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=16, radius=float(sz[0]), location=(0, 0, 0))
            obj = bpy.context.active_object
            obj.data.materials.append(mat)
            bpy.ops.object.shade_smooth()
        elif gtype == MJ_ELLIPSOID:
            bpy.ops.mesh.primitive_uv_sphere_add(segments=28, ring_count=18, radius=1.0, location=(0, 0, 0))
            obj = bpy.context.active_object
            for v in obj.data.vertices:
                v.co.x *= float(sz[0])
                v.co.y *= float(sz[1])
                v.co.z *= float(sz[2])
            obj.data.update()
            obj.data.materials.append(mat)
            bpy.ops.object.shade_smooth()
        elif gtype == MJ_CYLINDER:
            rad, hh = float(sz[0]), float(sz[1])
            bpy.ops.mesh.primitive_cylinder_add(vertices=48 if "env_" in gname else 32, radius=rad, depth=hh * 2.0, location=(0, 0, 0))
            obj = bpy.context.active_object
            obj.data.materials.append(mat)
            add_bevel(obj, width=min(0.010, hh * 0.2), segments=2)
        else:  # MJ_BOX
            hx, hy, hz = float(sz[0]), float(sz[1]), float(sz[2])
            bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0, 0, 0))
            obj = bpy.context.active_object
            for v in obj.data.vertices:
                v.co.x *= hx
                v.co.y *= hy
                v.co.z *= hz
            obj.data.update()
            obj.data.materials.append(mat)
            add_bevel(obj, width=min(0.010, min(hx, hy, hz) * 0.2), segments=2)

        obj.name = f"G_{gi}_{gname}"
        tracked.append((gi, obj))
    return tracked


def render_bundle(bundle_json: Path, out_mp4: Path, width: int = 1920, height: int = 1080, samples: int = 20) -> None:
    meta = json.loads(bundle_json.read_text(encoding="utf-8"))
    npz = np.load(bundle_json.with_suffix(".npz"))
    xpos = npz["xpos"]
    xmat = npz["xmat"]
    num_frames = int(xpos.shape[0])
    is_turntable = (meta.get("mode") == "beauty_turntable")

    scene = bpy.context.scene
    configure_cycles_optix(scene, width=width, height=height, samples=samples)
    bbox_m = (meta.get("summary") or {}).get("bbox_m")
    cam, cam_tgt = setup_studio_lights_and_floor(is_turntable=is_turntable, bbox_m=bbox_m)
    tracked = spawn_all_geoms(meta["geom_info"], is_turntable=is_turntable)

    names = [g["name"] for g in meta["geom_info"]]
    ia = names.index("A_core_box") if "A_core_box" in names else None
    ib = names.index("B_core_box") if "B_core_box" in names else None

    def apply_frame(scn: bpy.types.Scene) -> None:
        f = max(0, min(num_frames - 1, scn.frame_current - 1))
        for gi, obj in tracked:
            p = xpos[f, gi]
            r = xmat[f, gi].reshape(3, 3)
            # Hide retracted thruster flame inside nozzle when idle!
            if obj.name.endswith("_flame") and ia is not None:
                pass
            m4 = Matrix((
                (float(r[0, 0]), float(r[0, 1]), float(r[0, 2]), float(p[0])),
                (float(r[1, 0]), float(r[1, 1]), float(r[1, 2]), float(p[1])),
                (float(r[2, 0]), float(r[2, 1]), float(r[2, 2]), float(p[2])),
                (0.0, 0.0, 0.0, 1.0),
            ))
            obj.matrix_world = m4

        if not is_turntable and ia is not None and ib is not None:
            pa = xpos[f, ia]
            pb = xpos[f, ib]
            mid = 0.5 * (pa + pb)
            mid[0] = max(-2.2, min(2.2, float(mid[0])))
            mid[1] = max(-1.8, min(1.8, float(mid[1])))
            mid[2] = max(-0.4, min(0.6, float(mid[2]) + 0.15))
            cam_tgt.location = (float(mid[0] * 0.55), float(mid[1] * 0.55), float(mid[2]))
            ang = 0.18 * math.sin(f / 65.0)
            dist = max(5.6, min(7.4, 5.2 + 0.42 * float(np.linalg.norm(pa[:2] - pb[:2]))))
            cam.location = (float(mid[0] * 0.35 + dist * math.sin(ang)), float(-dist * math.cos(ang)), 3.45)

    bpy.app.handlers.frame_change_pre.clear()
    bpy.app.handlers.frame_change_pre.append(apply_frame)

    tmp_dir = out_mp4.parent / f"_frames_{out_mp4.stem}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    scene.frame_start = 1
    scene.frame_end = num_frames
    scene.render.filepath = str(tmp_dir / "frame_")
    apply_frame(scene)

    t0 = time.time()
    bpy.ops.render.render(animation=True)
    dt = time.time() - t0
    print(f"[BLENDER] Rendered {num_frames} frames in {dt:.1f}s ({dt/max(1,num_frames):.2f}s/frame)", flush=True)

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-framerate", "30",
        "-i", str(tmp_dir / "frame_%04d.jpg"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17", "-preset", "fast",
        str(out_mp4),
    ], check=True)
    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--samples", type=int, default=20)
    args = ap.parse_args(argv)
    render_bundle(Path(args.bundle), Path(args.out), args.width, args.height, args.samples)
