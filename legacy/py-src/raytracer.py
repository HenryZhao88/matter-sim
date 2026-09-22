"""Simple path tracer for Matter Sim — renders Mode A scenes to high-quality images.

Supports:
- Triangle meshes (from mesh.py)
- Sphere primitives (particles)
- Box primitives (scene objects)
- PBR materials (Lambertian diffuse + Cook-Torrance specular + metallic blend)
- Directional + point lights
- Shadow rays
- Up to N bounces of indirect illumination
- Gamma-corrected sRGB output
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from mesh import Mesh, MeshMaterial, Vertex


# ---------------------------------------------------------------------------
# Vector math (pure-python, no numpy dependency for individual rays)
# ---------------------------------------------------------------------------

def _v3(x: float, y: float, z: float) -> tuple[float, float, float]:
    return (x, y, z)

def _add(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

def _sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

def _mul(a: tuple[float, float, float], s: float) -> tuple[float, float, float]:
    return (a[0] * s, a[1] * s, a[2] * s)

def _vmul(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] * b[0], a[1] * b[1], a[2] * b[2])

def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

def _cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])

def _length(a: tuple[float, float, float]) -> float:
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])

def _normalize(a: tuple[float, float, float]) -> tuple[float, float, float]:
    ln = _length(a)
    if ln < 1e-12:
        return (0.0, 0.0, 1.0)
    inv = 1.0 / ln
    return (a[0] * inv, a[1] * inv, a[2] * inv)

def _reflect(d: tuple[float, float, float], n: tuple[float, float, float]) -> tuple[float, float, float]:
    dn = _dot(d, n) * 2.0
    return _sub(d, _mul(n, dn))

def _random_hemisphere(normal: tuple[float, float, float]) -> tuple[float, float, float]:
    """Random direction in hemisphere around normal (cosine-weighted bias)."""
    u1 = random.random()
    u2 = random.random()
    r = math.sqrt(u1)
    theta = 2.0 * math.pi * u2
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    z = math.sqrt(max(0.0, 1.0 - u1))
    # Build tangent frame
    if abs(normal[0]) < 0.9:
        tangent = _normalize(_cross((1, 0, 0), normal))
    else:
        tangent = _normalize(_cross((0, 1, 0), normal))
    bitangent = _cross(normal, tangent)
    return _normalize(_add(_add(_mul(tangent, x), _mul(bitangent, y)), _mul(normal, z)))


# ---------------------------------------------------------------------------
# Scene primitives for the raytracer
# ---------------------------------------------------------------------------

@dataclass
class RTSphere:
    center: tuple[float, float, float]
    radius: float
    material: MeshMaterial


@dataclass
class RTTriangle:
    v0: tuple[float, float, float]
    v1: tuple[float, float, float]
    v2: tuple[float, float, float]
    normal: tuple[float, float, float]
    material: MeshMaterial


@dataclass
class RTBox:
    center: tuple[float, float, float]
    half_extents: tuple[float, float, float]
    rotation_deg: float
    material: MeshMaterial


@dataclass
class RTLight:
    direction: tuple[float, float, float]  # normalised, toward the light
    color: tuple[float, float, float]       # intensity * rgb
    is_point: bool = False
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class RTScene:
    spheres: list[RTSphere] = field(default_factory=list)
    triangles: list[RTTriangle] = field(default_factory=list)
    boxes: list[RTBox] = field(default_factory=list)
    lights: list[RTLight] = field(default_factory=list)
    ambient: tuple[float, float, float] = (0.05, 0.06, 0.08)
    bg_color: tuple[float, float, float] = (0.02, 0.03, 0.05)


# ---------------------------------------------------------------------------
# Intersection tests
# ---------------------------------------------------------------------------

_HIT_NONE = (1e30, (0.0, 0.0, 1.0), None)  # (t, normal, material)


def _intersect_sphere(
    origin: tuple[float, float, float],
    direction: tuple[float, float, float],
    sphere: RTSphere,
) -> tuple[float, tuple[float, float, float], MeshMaterial | None]:
    oc = _sub(origin, sphere.center)
    a = _dot(direction, direction)
    b = 2.0 * _dot(oc, direction)
    c = _dot(oc, oc) - sphere.radius * sphere.radius
    disc = b * b - 4.0 * a * c
    if disc < 0:
        return _HIT_NONE
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2.0 * a)
    t2 = (-b + sq) / (2.0 * a)
    t = t1 if t1 > 1e-4 else t2
    if t < 1e-4:
        return _HIT_NONE
    hit = _add(origin, _mul(direction, t))
    normal = _normalize(_sub(hit, sphere.center))
    return (t, normal, sphere.material)


def _intersect_triangle(
    origin: tuple[float, float, float],
    direction: tuple[float, float, float],
    tri: RTTriangle,
) -> tuple[float, tuple[float, float, float], MeshMaterial | None]:
    """Möller–Trumbore ray-triangle intersection."""
    e1 = _sub(tri.v1, tri.v0)
    e2 = _sub(tri.v2, tri.v0)
    h = _cross(direction, e2)
    det = _dot(e1, h)
    if abs(det) < 1e-10:
        return _HIT_NONE
    inv_det = 1.0 / det
    s = _sub(origin, tri.v0)
    u = _dot(s, h) * inv_det
    if u < 0.0 or u > 1.0:
        return _HIT_NONE
    q = _cross(s, e1)
    v = _dot(direction, q) * inv_det
    if v < 0.0 or u + v > 1.0:
        return _HIT_NONE
    t = _dot(e2, q) * inv_det
    if t < 1e-4:
        return _HIT_NONE
    n = tri.normal
    if _dot(n, direction) > 0:
        n = _mul(n, -1.0)
    return (t, n, tri.material)


def _intersect_box(
    origin: tuple[float, float, float],
    direction: tuple[float, float, float],
    box: RTBox,
) -> tuple[float, tuple[float, float, float], MeshMaterial | None]:
    """AABB intersection after rotating ray into box-local space."""
    rr = math.radians(-box.rotation_deg)
    cr, sr = math.cos(rr), math.sin(rr)
    # Transform ray to box-local (rotate around Z, translate)
    ox, oy, oz = origin[0] - box.center[0], origin[1] - box.center[1], origin[2] - box.center[2]
    lox = ox * cr - oy * sr
    loy = ox * sr + oy * cr
    loz = oz
    ldx = direction[0] * cr - direction[1] * sr
    ldy = direction[0] * sr + direction[1] * cr
    ldz = direction[2]

    hx, hy, hz = box.half_extents
    inv = lambda d: 1.0 / d if abs(d) > 1e-12 else 1e12 * (1.0 if d >= 0 else -1.0)
    idx, idy, idz = inv(ldx), inv(ldy), inv(ldz)

    t1x = (-hx - lox) * idx
    t2x = (hx - lox) * idx
    t1y = (-hy - loy) * idy
    t2y = (hy - loy) * idy
    t1z = (-hz - loz) * idz
    t2z = (hz - loz) * idz

    tmin_x, tmax_x = min(t1x, t2x), max(t1x, t2x)
    tmin_y, tmax_y = min(t1y, t2y), max(t1y, t2y)
    tmin_z, tmax_z = min(t1z, t2z), max(t1z, t2z)

    tmin = max(tmin_x, tmin_y, tmin_z)
    tmax = min(tmax_x, tmax_y, tmax_z)

    if tmin > tmax or tmax < 1e-4:
        return _HIT_NONE
    t = tmin if tmin > 1e-4 else tmax
    if t < 1e-4:
        return _HIT_NONE

    # Compute normal in local space then rotate back
    hit_local = (lox + ldx * t, loy + ldy * t, loz + ldz * t)
    eps = 1e-3
    if abs(hit_local[0] - hx) < eps * hx + eps:
        ln = (1, 0, 0)
    elif abs(hit_local[0] + hx) < eps * hx + eps:
        ln = (-1, 0, 0)
    elif abs(hit_local[1] - hy) < eps * hy + eps:
        ln = (0, 1, 0)
    elif abs(hit_local[1] + hy) < eps * hy + eps:
        ln = (0, -1, 0)
    elif abs(hit_local[2] - hz) < eps * hz + eps:
        ln = (0, 0, 1)
    else:
        ln = (0, 0, -1)

    # Rotate normal back to world
    rr2 = math.radians(box.rotation_deg)
    cr2, sr2 = math.cos(rr2), math.sin(rr2)
    wn = (ln[0] * cr2 - ln[1] * sr2, ln[0] * sr2 + ln[1] * cr2, ln[2])
    if _dot(wn, direction) > 0:
        wn = _mul(wn, -1.0)
    return (t, _normalize(wn), box.material)


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------

def _trace_nearest(
    scene: RTScene, origin: tuple[float, float, float], direction: tuple[float, float, float],
) -> tuple[float, tuple[float, float, float], MeshMaterial | None]:
    best_t, best_n, best_m = _HIT_NONE
    for s in scene.spheres:
        t, n, m = _intersect_sphere(origin, direction, s)
        if t < best_t:
            best_t, best_n, best_m = t, n, m
    for tri in scene.triangles:
        t, n, m = _intersect_triangle(origin, direction, tri)
        if t < best_t:
            best_t, best_n, best_m = t, n, m
    for b in scene.boxes:
        t, n, m = _intersect_box(origin, direction, b)
        if t < best_t:
            best_t, best_n, best_m = t, n, m
    return best_t, best_n, best_m


def _shade(
    scene: RTScene,
    hit_point: tuple[float, float, float],
    normal: tuple[float, float, float],
    view_dir: tuple[float, float, float],
    mat: MeshMaterial,
    bounces_left: int,
) -> tuple[float, float, float]:
    """PBR-ish shade: Lambertian + Blinn-Phong specular + shadow rays + indirect."""
    albedo = mat.albedo
    result = _vmul(albedo, scene.ambient)

    # Emission
    if mat.emission_strength > 0:
        result = _add(result, _mul(mat.emission, mat.emission_strength))

    for light in scene.lights:
        if light.is_point:
            to_light = _sub(light.position, hit_point)
            light_dist = _length(to_light)
            L = _normalize(to_light)
            # Inverse-square attenuation
            atten = 1.0 / max(0.01, light_dist * light_dist)
        else:
            L = light.direction
            atten = 1.0
            light_dist = 1e30

        # Shadow ray
        shadow_origin = _add(hit_point, _mul(normal, 0.01))
        st, _sn, _sm = _trace_nearest(scene, shadow_origin, L)
        if st < light_dist:
            continue  # in shadow

        NdotL = max(0.0, _dot(normal, L))
        if NdotL < 1e-6:
            continue

        # Diffuse
        diff = _mul(_vmul(albedo, light.color), NdotL * (1.0 - mat.metallic) * atten)

        # Specular (Blinn-Phong approximation of Cook-Torrance)
        H = _normalize(_add(L, _mul(view_dir, -1.0)))  # half vector
        NdotH = max(0.0, _dot(normal, H))
        roughness = max(0.04, mat.roughness)
        spec_power = 2.0 / (roughness * roughness) - 2.0
        spec_intensity = math.pow(NdotH, spec_power) * (spec_power + 2.0) / (8.0 * math.pi)
        # Fresnel-Schlick (F0 for dielectrics ≈ 0.04, for metals = albedo)
        f0 = (0.04, 0.04, 0.04) if mat.metallic < 0.5 else albedo
        fresnel = _add(f0, _mul(_sub((1.0, 1.0, 1.0), f0), (1.0 - max(0.0, _dot(H, _mul(view_dir, -1.0)))) ** 5))
        spec = _mul(_vmul(fresnel, light.color), spec_intensity * NdotL * atten)

        result = _add(result, _add(diff, spec))

    # Indirect illumination (one cosine-weighted bounce)
    if bounces_left > 0:
        bounce_dir = _random_hemisphere(normal)
        bounce_origin = _add(hit_point, _mul(normal, 0.01))
        bt, bn, bm = _trace_nearest(scene, bounce_origin, bounce_dir)
        if bm is not None:
            bounce_hit = _add(bounce_origin, _mul(bounce_dir, bt))
            indirect = _shade(scene, bounce_hit, bn, _mul(bounce_dir, -1.0), bm, bounces_left - 1)
            NdotBounce = max(0.0, _dot(normal, bounce_dir))
            result = _add(result, _mul(_vmul(albedo, indirect), NdotBounce * 0.5))

    return result


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

def _linear_to_srgb(c: float) -> int:
    """Linear float → 0-255 sRGB with gamma."""
    c = max(0.0, min(1.0, c))
    if c <= 0.0031308:
        s = c * 12.92
    else:
        s = 1.055 * (c ** (1.0 / 2.4)) - 0.055
    return max(0, min(255, int(s * 255 + 0.5)))


@dataclass
class RenderSettings:
    width: int = 1920
    height: int = 1080
    samples_per_pixel: int = 16
    max_bounces: int = 2
    fov_deg: float = 60.0
    camera_pos: tuple[float, float, float] = (0.0, -500.0, 200.0)
    camera_target: tuple[float, float, float] = (0.0, 0.0, 0.0)
    camera_up: tuple[float, float, float] = (0.0, 0.0, 1.0)


def render(scene: RTScene, settings: RenderSettings, progress_callback=None) -> Image.Image:
    """Path-trace the scene and return a PIL Image.

    Args:
        scene: The scene to render.
        settings: Render settings (resolution, samples, bounces, camera).
        progress_callback: Optional callable(row, total_rows) for progress updates.

    Returns:
        PIL Image in sRGB.
    """
    W, H = settings.width, settings.height
    spp = max(1, settings.samples_per_pixel)
    bounces = settings.max_bounces
    fov_rad = math.radians(settings.fov_deg)
    aspect = W / H

    # Camera basis
    cam_fwd = _normalize(_sub(settings.camera_target, settings.camera_pos))
    cam_right = _normalize(_cross(cam_fwd, settings.camera_up))
    cam_up = _cross(cam_right, cam_fwd)

    half_h = math.tan(fov_rad * 0.5)
    half_w = half_h * aspect

    pixels = np.zeros((H, W, 3), dtype=np.float64)

    for y in range(H):
        for x in range(W):
            color = (0.0, 0.0, 0.0)
            for _ in range(spp):
                # Jittered sub-pixel
                u = (2.0 * (x + random.random()) / W - 1.0) * half_w
                v = (1.0 - 2.0 * (y + random.random()) / H) * half_h

                direction = _normalize(_add(_add(cam_fwd, _mul(cam_right, u)), _mul(cam_up, v)))

                t, normal, mat = _trace_nearest(scene, settings.camera_pos, direction)
                if mat is not None:
                    hit = _add(settings.camera_pos, _mul(direction, t))
                    sample_color = _shade(scene, hit, normal, _mul(direction, -1.0), mat, bounces)
                else:
                    sample_color = scene.bg_color

                color = _add(color, sample_color)

            inv_spp = 1.0 / spp
            pixels[y, x, 0] = color[0] * inv_spp
            pixels[y, x, 1] = color[1] * inv_spp
            pixels[y, x, 2] = color[2] * inv_spp

        if progress_callback and y % 10 == 0:
            progress_callback(y, H)

    # Tonemap (Reinhard) + gamma
    out = np.zeros((H, W, 3), dtype=np.uint8)
    for y in range(H):
        for x in range(W):
            for c in range(3):
                linear = pixels[y, x, c]
                # Reinhard tonemap
                mapped = linear / (1.0 + linear)
                out[y, x, c] = _linear_to_srgb(mapped)

    return Image.fromarray(out, "RGB")


def build_scene_from_app(app) -> tuple[RTScene, RenderSettings]:
    """Extract an RTScene + RenderSettings from an AtomSimulatorApp instance.

    This converts the app's particles, scene objects, lights, and camera
    into raytracer primitives.
    """
    scene = RTScene()
    frame = app.timeline_frame

    # Default directional light
    scene.lights.append(RTLight(
        direction=_normalize((0.4, -0.6, 0.7)),
        color=(1.2, 1.15, 1.05),
    ))

    # Scene objects → boxes or triangles (if mesh)
    for name, obj in app.scene_objects.items():
        if not obj.visible:
            continue
        ox, oy, rot = app._eval_object_transform(obj, frame)
        kind = obj.kind.lower()

        mat = MeshMaterial(
            name=name,
            albedo=_hex_to_float_rgb(obj.color),
            roughness=obj.roughness,
            metallic=obj.metallic,
            emission_strength=obj.emission,
        )
        if obj.emission > 0:
            mat.emission = _hex_to_float_rgb(obj.color)

        if kind == "light":
            # Add as point light
            scene.lights.append(RTLight(
                direction=(0, 0, 0),
                color=_hex_to_float_rgb(obj.color),
                is_point=True,
                position=(ox, oy, obj.z),
            ))
            continue

        if hasattr(obj, "mesh_data") and obj.mesh_data is not None:
            # Render mesh triangles
            mesh = obj.mesh_data
            rr = math.radians(rot)
            cr, sr = math.cos(rr), math.sin(rr)
            scale = obj.size / max(1.0, mesh.extent() * 2.0)
            for face in mesh.faces:
                if len(face.vertex_indices) < 3:
                    continue
                face_mat = mesh.materials.get(face.material_name, mat)
                verts = []
                for vi in face.vertex_indices[:3]:
                    v = mesh.vertices[vi]
                    # Scale + rotate + translate
                    lx, ly, lz = v.x * scale, v.y * scale, v.z * scale
                    wx = ox + lx * cr - ly * sr
                    wy = oy + lx * sr + ly * cr
                    wz = obj.z + lz
                    verts.append((wx, wy, wz))
                e1 = _sub(verts[1], verts[0])
                e2 = _sub(verts[2], verts[0])
                n = _normalize(_cross(e1, e2))
                scene.triangles.append(RTTriangle(verts[0], verts[1], verts[2], n, face_mat))
        else:
            # Render as box
            hx, hy, hz = app._box_dims_for_object(obj)
            scene.boxes.append(RTBox(
                center=(ox, oy, obj.z),
                half_extents=(hx, hy, hz),
                rotation_deg=rot,
                material=mat,
            ))

    # Particles → spheres
    for p in app.world.particles:
        scene.spheres.append(RTSphere(
            center=(p.x, p.y, p.z),
            radius=max(1.0, p.radius * app.render_radius_scale * 0.85),
            material=MeshMaterial(
                name=p.material.name,
                albedo=_hex_to_float_rgb(p.material.color),
                roughness=0.6,
                metallic=0.0,
            ),
        ))

    # Camera
    settings = RenderSettings(
        width=app.world_w,
        height=app.world_h,
        camera_pos=(app.camera_x, app.camera_y, app.camera_z),
    )
    # Compute camera target from orbit pivot
    settings.camera_target = tuple(app.orbit_pivot)
    settings.fov_deg = app.camera_fov_deg

    return scene, settings


def _hex_to_float_rgb(hex_color: str) -> tuple[float, float, float]:
    """'#rrggbb' → linear float (0-1) tuple."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return (0.8, 0.8, 0.8)
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    # sRGB → linear (approximate)
    return (r ** 2.2, g ** 2.2, b ** 2.2)
