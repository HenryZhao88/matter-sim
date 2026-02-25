from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from dataclasses import dataclass

import cupy as cp  # type: ignore
import imageio
import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw

try:
    import tkinter as tk
    from tkinter import ttk
    from tkinter import messagebox
    TK_AVAILABLE = True
except Exception:
    tk = None  # type: ignore[assignment]
    ttk = None  # type: ignore[assignment]
    messagebox = None  # type: ignore[assignment]
    TK_AVAILABLE = False

try:
    from PIL import ImageTk
    IMAGETK_AVAILABLE = True
except Exception:
    ImageTk = None  # type: ignore[assignment]
    IMAGETK_AVAILABLE = False

from quantum_engine import QuantumWorld

# -----------------------------
# Materials / Presets
# -----------------------------


@dataclass(frozen=True)
class Material:
    name: str
    mass: float
    charge: float
    radius: float
    color: str


MATERIALS: dict[str, Material] = {
    "electron": Material("electron", mass=1.0, charge=-1.0, radius=3.0, color="#00e5ff"),
    "proton": Material("proton", mass=1836.0, charge=+1.0, radius=7.0, color="#ff4f4f"),
    "neutron": Material("neutron", mass=1839.0, charge=0.0, radius=7.0, color="#b5b5b5"),
    "ion+": Material("ion+", mass=4000.0, charge=+2.0, radius=9.0, color="#ffb347"),
    "ion-": Material("ion-", mass=4000.0, charge=-2.0, radius=9.0, color="#8f7dff"),
    "aluminum": Material("aluminum", mass=26.98, charge=0.0, radius=2.2, color="#d8dce4"),
}


SCALE_PROFILES: dict[str, dict[str, float]] = {
    "macro": {"render_radius": 1.0, "distance": 1.0, "velocity": 1.0},
    "micro": {"render_radius": 0.55, "distance": 0.7, "velocity": 0.8},
    "nano": {"render_radius": 0.35, "distance": 0.5, "velocity": 0.65},
    "atomic": {"render_radius": 0.2, "distance": 0.35, "velocity": 0.45},
}


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    material: Material
    z: float = 0.0
    vz: float = 0.0

    @property
    def mass(self) -> float:
        return self.material.mass

    @property
    def charge(self) -> float:
        return self.material.charge

    @property
    def radius(self) -> float:
        return self.material.radius


@dataclass
class Bond:
    i: int
    j: int
    rest_length: float
    k: float


@dataclass
class ObjectKeyframe:
    frame: int
    x: float
    y: float
    rot_deg: float


@dataclass
class SceneCube:
    name: str
    x: float
    y: float
    z: float
    size: float
    kind: str = "cube"
    rot_deg: float = 0.0
    color: str = "#9aa4c9"
    roughness: float = 0.55
    metallic: float = 0.0
    emission: float = 0.0
    visible: bool = True
    collection: str = "Scene"
    texture_path: str = ""
    texture_image: object | None = None
    texture_pil: object | None = None
    texture_tk_id: int | None = None
    keyframes: list[ObjectKeyframe] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.keyframes is None:
            self.keyframes = []


class PhysicsWorld:
    def __init__(self, world_w: int, world_h: int) -> None:
        self.world_w = world_w
        self.world_h = world_h
        self.world_d = 2400.0
        self.particles: list[Particle] = []

        self.dt = 0.008
        self.k_coulomb = 1800.0
        self.softening = 10.0
        self.repulsion_k = 16000.0
        self.drag = 0.9995
        self.max_speed = 450.0
        self.bonds: list[Bond] = []
        self.last_note: str = ""
        self.compute_backend = "gpu"

        # Exact-mode transport controls (no hard limits; user can scale arbitrarily).
        self.mode_b_transport_exact = True
        self.photons_per_particle_per_step = 128
        self.photon_max_bounces = 2
        self.light_speed = 299792458.0
        self.photon_packet_energy = 1e-21
        self.radiation_coupling = 1.0

        # Mode-A approximation controls
        self.mode_a_pair_samples = 48
        self.mode_a_bond_factor = 0.35

    def clear_all(self) -> None:
        self.particles.clear()
        self.bonds.clear()

    def _seed_particle_depth(self, span: float = 480.0, vz_span: float = 40.0) -> None:
        half = max(1.0, span * 0.5)
        hv = max(0.0, vz_span * 0.5)
        for p in self.particles:
            p.z = random.uniform(-half, half)
            p.vz = random.uniform(-hv, hv)

    def load_preset(self, name: str, distance_scale: float = 1.0, velocity_scale: float = 1.0) -> None:
        self.clear_all()
        self.last_note = ""

        cx = self.world_w * 0.5
        cy = self.world_h * 0.5

        if name == "Hydrogen":
            self.particles.append(Particle(cx, cy, 0.0, 0.0, MATERIALS["proton"]))
            self.particles.append(Particle(cx + 90.0 * distance_scale, cy, 0.0, -145.0 * velocity_scale, MATERIALS["electron"]))

        elif name == "Helium":
            self.particles.append(Particle(cx - 4.0 * distance_scale, cy, 0.0, 0.0, MATERIALS["proton"]))
            self.particles.append(Particle(cx + 4.0 * distance_scale, cy, 0.0, 0.0, MATERIALS["proton"]))
            self.particles.append(Particle(cx + 70.0 * distance_scale, cy, 0.0, -150.0 * velocity_scale, MATERIALS["electron"]))
            self.particles.append(Particle(cx - 85.0 * distance_scale, cy, 0.0, 130.0 * velocity_scale, MATERIALS["electron"]))

        elif name == "Carbon":
            for i in range(6):
                a = 2 * math.pi * i / 6
                self.particles.append(Particle(cx + 8 * distance_scale * math.cos(a), cy + 8 * distance_scale * math.sin(a), 0.0, 0.0, MATERIALS["proton"]))
            for i in range(6):
                a = 2 * math.pi * i / 6
                self.particles.append(Particle(cx + 15 * distance_scale * math.cos(a), cy + 15 * distance_scale * math.sin(a), 0.0, 0.0, MATERIALS["neutron"]))
            for i in range(2):
                a = 2 * math.pi * i / 2
                self.particles.append(
                    Particle(
                        cx + 80 * distance_scale * math.cos(a),
                        cy + 80 * distance_scale * math.sin(a),
                        -140 * velocity_scale * math.sin(a),
                        140 * velocity_scale * math.cos(a),
                        MATERIALS["electron"],
                    )
                )
            for i in range(4):
                a = 2 * math.pi * i / 4 + 0.3
                self.particles.append(
                    Particle(
                        cx + 150 * distance_scale * math.cos(a),
                        cy + 150 * distance_scale * math.sin(a),
                        -120 * velocity_scale * math.sin(a),
                        120 * velocity_scale * math.cos(a),
                        MATERIALS["electron"],
                    )
                )

        elif name == "Plasma Box":
            for _ in range(80):
                self.particles.append(
                    Particle(
                        random.uniform(150, self.world_w - 150),
                        random.uniform(120, self.world_h - 120),
                        random.uniform(-60, 60) * velocity_scale,
                        random.uniform(-60, 60) * velocity_scale,
                        MATERIALS["ion+"] if random.random() < 0.5 else MATERIALS["ion-"],
                    )
                )
        elif name == "Aluminum Cube (50nm, scaled)":
            self._load_aluminum_cube_50nm_scaled()

        if self.particles:
            self._seed_particle_depth()

    def _load_aluminum_cube_50nm_scaled(self) -> None:
        self.clear_all()

        # Real 50nm^3 Al statistics (metadata + projected 2D crystalline render)
        density_kg_m3 = 2700.0
        molar_mass_kg = 0.0269815
        avogadro = 6.02214076e23
        volume_m3 = (50e-9) ** 3
        moles = density_kg_m3 * volume_m3 / molar_mass_kg
        real_atom_count = int(moles * avogadro)

        # Projected crystalline lattice chunk for rendering in 2D.
        # Large enough to stress-test interactions while still drawable on canvas.
        nx, ny = 96, 96
        spacing = 8.4
        x0 = (self.world_w - (nx - 1) * spacing) * 0.5
        y0 = (self.world_h - (ny - 1) * spacing) * 0.5

        for j in range(ny):
            for i in range(nx):
                offset = (spacing * 0.5) if (j % 2 == 1) else 0.0
                x = x0 + i * spacing + offset
                y = y0 + j * spacing * 0.8660254  # ~sqrt(3)/2 for close-packed projection
                x += random.uniform(-0.25, 0.25)
                y += random.uniform(-0.25, 0.25)
                self.particles.append(Particle(x, y, 0.0, 0.0, MATERIALS["aluminum"]))

        # Bonds: nearest-neighbor links for crystal-like structure.
        for j in range(ny):
            for i in range(nx):
                idx = j * nx + i
                if i + 1 < nx:
                    self.bonds.append(Bond(idx, j * nx + (i + 1), spacing, 35.0))
                if j + 1 < ny:
                    self.bonds.append(Bond(idx, (j + 1) * nx + i, spacing * 0.8660254, 35.0))
                if j + 1 < ny and i + (1 if j % 2 == 0 else -1) >= 0 and i + (1 if j % 2 == 0 else -1) < nx:
                    ni = i + (1 if j % 2 == 0 else -1)
                    self.bonds.append(Bond(idx, (j + 1) * nx + ni, spacing, 35.0))

        downsample_factor = real_atom_count / max(1, len(self.particles))
        self.last_note = (
            f"50nm^3 Al contains ~{real_atom_count:.3e} atoms. "
            f"Showing {len(self.particles)} atoms (downsample ~{downsample_factor:.3e}x) with lattice bonds."
        )

    def spawn_random(self, material_name: str, count: int) -> None:
        mat = MATERIALS[material_name]
        for _ in range(max(1, count)):
            self.particles.append(
                Particle(
                    random.uniform(50, self.world_w - 50),
                    random.uniform(50, self.world_h - 50),
                    random.uniform(-100, 100),
                    random.uniform(-100, 100),
                    mat,
                    random.uniform(-280.0, 280.0),
                    random.uniform(-40.0, 40.0),
                )
            )

    def step(self, mode: str = "B") -> None:
        if mode.upper() == "A":
            self._step_approximate()
        else:
            if self.mode_b_transport_exact:
                self._step_exact_transport()
            else:
                self._step_exact()

    def _ray_circle_intersection(
        self,
        ox: float,
        oy: float,
        dx: float,
        dy: float,
        cx: float,
        cy: float,
        radius: float,
    ) -> float | None:
        lx = cx - ox
        ly = cy - oy
        tca = lx * dx + ly * dy
        if tca <= 0:
            return None
        d2 = lx * lx + ly * ly - tca * tca
        r2 = radius * radius
        if d2 > r2:
            return None
        thc = math.sqrt(max(0.0, r2 - d2))
        t0 = tca - thc
        t1 = tca + thc
        if t0 > 1e-9:
            return t0
        if t1 > 1e-9:
            return t1
        return None

    def _step_exact_transport(self) -> None:
        # Exact classical pairwise + bond forces first.
        self._step_exact()

        n = len(self.particles)
        if n <= 1:
            return

        impulses_x = [0.0] * n
        impulses_y = [0.0] * n

        photons_pp = max(1, int(self.photons_per_particle_per_step))
        max_bounces = max(0, int(self.photon_max_bounces))

        for i, src in enumerate(self.particles):
            for k in range(photons_pp):
                frac = ((k + 0.5) / photons_pp + (i * 0.6180339887498949)) % 1.0
                angle = 2.0 * math.pi * frac
                dx = math.cos(angle)
                dy = math.sin(angle)

                ox, oy = src.x, src.y
                bounce = 0
                alive = True

                while alive:
                    hit_j = -1
                    hit_t = float("inf")

                    for j, tgt in enumerate(self.particles):
                        if j == i:
                            continue
                        t = self._ray_circle_intersection(ox, oy, dx, dy, tgt.x, tgt.y, tgt.radius)
                        if t is not None and t < hit_t:
                            hit_t = t
                            hit_j = j

                    if hit_j < 0:
                        break

                    hit = self.particles[hit_j]
                    hx = ox + dx * hit_t
                    hy = oy + dy * hit_t

                    px = self.photon_packet_energy / self.light_speed * dx
                    py = self.photon_packet_energy / self.light_speed * dy

                    impulses_x[i] -= px
                    impulses_y[i] -= py
                    impulses_x[hit_j] += px
                    impulses_y[hit_j] += py

                    if bounce >= max_bounces:
                        break

                    nx = hx - hit.x
                    ny = hy - hit.y
                    nlen = math.hypot(nx, ny)
                    if nlen < 1e-12:
                        break
                    nx /= nlen
                    ny /= nlen

                    dot = dx * nx + dy * ny
                    dx = dx - 2.0 * dot * nx
                    dy = dy - 2.0 * dot * ny

                    ox = hx + dx * 1e-6
                    oy = hy + dy * 1e-6
                    bounce += 1
                    alive = True

        for i, p in enumerate(self.particles):
            mass = max(1e-12, p.mass)
            p.vx += self.radiation_coupling * impulses_x[i] / mass
            p.vy += self.radiation_coupling * impulses_y[i] / mass

    # ------------------------------------------------------------------
    # Spatial hash for O(n) short-range force lookups  (#7)
    # ------------------------------------------------------------------
    def _build_spatial_hash(self, cell_size: float) -> dict[tuple[int, int, int], list[int]]:
        grid: dict[tuple[int, int, int], list[int]] = {}
        inv = 1.0 / max(1e-6, cell_size)
        for i, p in enumerate(self.particles):
            key = (int(math.floor(p.x * inv)), int(math.floor(p.y * inv)), int(math.floor(p.z * inv)))
            if key in grid:
                grid[key].append(i)
            else:
                grid[key] = [i]
        return grid

    @staticmethod
    def _neighbor_keys(cx: int, cy: int, cz: int) -> list[tuple[int, int, int]]:
        keys: list[tuple[int, int, int]] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    keys.append((cx + dx, cy + dy, cz + dz))
        return keys

    def _step_exact(self) -> None:
        if len(self.particles) <= 1:
            return

        n = len(self.particles)
        fx = [0.0] * n
        fy = [0.0] * n
        fz = [0.0] * n

        # Use spatial hash for short-range repulsion, full pairwise for Coulomb (#7)
        repulsion_range = max(p.radius for p in self.particles) * 2.0 + 4.0
        cell_size = max(repulsion_range, 20.0)
        grid = self._build_spatial_hash(cell_size)
        inv = 1.0 / max(1e-6, cell_size)

        # Full pairwise Coulomb + spatial-hash repulsion
        for i in range(n):
            pi = self.particles[i]
            for j in range(i + 1, n):
                pj = self.particles[j]

                dx = pj.x - pi.x
                dy = pj.y - pi.y
                dz = pj.z - pi.z
                r2 = dx * dx + dy * dy + dz * dz + self.softening
                r = math.sqrt(r2)
                nx = dx / r
                ny = dy / r
                nz = dz / r

                f_c = self.k_coulomb * pi.charge * pj.charge / r2
                f_total = f_c

                fx_i = f_total * nx
                fy_i = f_total * ny
                fz_i = f_total * nz

                fx[i] += fx_i
                fy[i] += fy_i
                fz[i] += fz_i
                fx[j] -= fx_i
                fy[j] -= fy_i
                fz[j] -= fz_i

        # Short-range repulsion via spatial hash
        visited: set[tuple[int, int]] = set()
        for key, indices in grid.items():
            cx, cy_cell, cz_cell = key
            for nk in self._neighbor_keys(cx, cy_cell, cz_cell):
                if nk not in grid:
                    continue
                for i in indices:
                    pi = self.particles[i]
                    for j in grid[nk]:
                        if j <= i:
                            continue
                        pair = (i, j)
                        if pair in visited:
                            continue
                        visited.add(pair)
                        pj = self.particles[j]
                        dx = pj.x - pi.x
                        dy = pj.y - pi.y
                        dz = pj.z - pi.z
                        r = math.sqrt(dx * dx + dy * dy + dz * dz + 1e-12)
                        overlap = (pi.radius + pj.radius) - r
                        if overlap <= 0:
                            continue
                        nx = dx / r
                        ny = dy / r
                        nz = dz / r
                        f_rep = self.repulsion_k * overlap
                        fx[i] -= f_rep * nx
                        fy[i] -= f_rep * ny
                        fz[i] -= f_rep * nz
                        fx[j] += f_rep * nx
                        fy[j] += f_rep * ny
                        fz[j] += f_rep * nz

        # Bond spring forces for crystalline structures (3D).
        for b in self.bonds:
            if b.i < 0 or b.j < 0 or b.i >= n or b.j >= n:
                continue
            pi = self.particles[b.i]
            pj = self.particles[b.j]
            dx = pj.x - pi.x
            dy = pj.y - pi.y
            dz = pj.z - pi.z
            r = math.sqrt(dx * dx + dy * dy + dz * dz)
            if r < 1e-9:
                continue
            nx = dx / r
            ny = dy / r
            nz = dz / r
            extension = r - b.rest_length
            f = b.k * extension
            fx[b.i] += f * nx
            fy[b.i] += f * ny
            fz[b.i] += f * nz
            fx[b.j] -= f * nx
            fy[b.j] -= f * ny
            fz[b.j] -= f * nz

        self._integrate_forces(fx, fy, fz)

    def _step_approximate(self) -> None:
        if len(self.particles) <= 1:
            return

        n = len(self.particles)
        fx = [0.0] * n
        fy = [0.0] * n
        fz = [0.0] * n

        samples = max(4, min(self.mode_a_pair_samples, n - 1))
        scale_back = n / float(samples)

        # (#5) Random partner sampling — unbiased, no stride artifacts.
        for i in range(n):
            pi = self.particles[i]
            # Build sample list excluding self, random.sample is O(samples).
            pool_size = n - 1
            if pool_size <= samples:
                partner_indices = [j for j in range(n) if j != i]
            else:
                partner_indices = []
                attempts = 0
                seen: set[int] = set()
                while len(partner_indices) < samples and attempts < samples * 3:
                    j = random.randint(0, n - 1)
                    attempts += 1
                    if j != i and j not in seen:
                        seen.add(j)
                        partner_indices.append(j)
            for j in partner_indices:
                pj = self.particles[j]

                dx = pj.x - pi.x
                dy = pj.y - pi.y
                dz = pj.z - pi.z
                r2 = dx * dx + dy * dy + dz * dz + self.softening
                r = math.sqrt(r2)
                nx = dx / r
                ny = dy / r
                nz = dz / r

                f_c = self.k_coulomb * pi.charge * pj.charge / r2
                overlap = (pi.radius + pj.radius) - r
                f_rep = self.repulsion_k * overlap if overlap > 0 else 0.0
                f_total = (f_c - f_rep) * scale_back

                fx[i] += f_total * nx
                fy[i] += f_total * ny
                fz[i] += f_total * nz

        # Keep bonded structure but soften in mode A (3D).
        for b in self.bonds:
            if b.i < 0 or b.j < 0 or b.i >= n or b.j >= n:
                continue
            pi = self.particles[b.i]
            pj = self.particles[b.j]
            dx = pj.x - pi.x
            dy = pj.y - pi.y
            dz = pj.z - pi.z
            r = math.sqrt(dx * dx + dy * dy + dz * dz)
            if r < 1e-9:
                continue
            nx = dx / r
            ny = dy / r
            nz = dz / r
            extension = r - b.rest_length
            f = b.k * self.mode_a_bond_factor * extension
            fx[b.i] += f * nx
            fy[b.i] += f * ny
            fz[b.i] += f * nz
            fx[b.j] -= f * nx
            fy[b.j] -= f * ny
            fz[b.j] -= f * nz

        self._integrate_forces(fx, fy, fz)

    def _integrate_forces(self, fx: list[float], fy: list[float], fz: list[float] | None = None) -> None:
        # (#6) Framerate-independent drag: drag_factor = drag^dt instead of bare drag.
        drag_factor = self.drag ** self.dt if self.drag < 1.0 else self.drag
        for i, p in enumerate(self.particles):
            ax = fx[i] / max(1e-9, p.mass)
            ay = fy[i] / max(1e-9, p.mass)
            az = (fz[i] / max(1e-9, p.mass)) if fz is not None else 0.0

            p.vx = (p.vx + ax * self.dt) * drag_factor
            p.vy = (p.vy + ay * self.dt) * drag_factor
            p.vz = (p.vz + az * self.dt) * drag_factor

            speed = math.sqrt(p.vx * p.vx + p.vy * p.vy + p.vz * p.vz)
            if speed > self.max_speed:
                s = self.max_speed / speed
                p.vx *= s
                p.vy *= s
                p.vz *= s

            p.x += p.vx * self.dt
            p.y += p.vy * self.dt
            p.z += p.vz * self.dt

            if p.x < p.radius:
                p.x = p.radius
                p.vx *= -0.9
            elif p.x > self.world_w - p.radius:
                p.x = self.world_w - p.radius
                p.vx *= -0.9

            if p.y < p.radius:
                p.y = p.radius
                p.vy *= -0.9
            elif p.y > self.world_h - p.radius:
                p.y = self.world_h - p.radius
                p.vy *= -0.9

            z_lim = self.world_d * 0.5
            if p.z < -z_lim:
                p.z = -z_lim
                p.vz *= -0.9
            elif p.z > z_lim:
                p.z = z_lim
                p.vz *= -0.9


class AtomSimulatorApp:
    def __init__(self) -> None:
        if not TK_AVAILABLE:
            raise RuntimeError("Tk is unavailable")

        self.root = tk.Tk()
        self.root.title("Matter Sim - Physics World")
        self.root.geometry("3900x2260")
        self.root.resizable(False, False)

        # 4K UHD viewport (16:9)
        self.world_w = 3840
        self.world_h = 2160
        self.world = PhysicsWorld(self.world_w, self.world_h)
        self.running = True

        self.selected_index: int | None = None
        self.drag_start: tuple[float, float] | None = None
        self.drag_current: tuple[float, float] | None = None
        self.viewport_drag_last: tuple[float, float] | None = None
        self.view_zoom = 1.0
        self.view_pan_x = 0.0
        self.view_pan_y = 0.0

        self.command_specs: dict[str, dict[str, str]] = {
            "help": {
                "usage": "help [query]",
                "desc": "Show all commands or help for matching commands.",
            },
            "spawn": {
                "usage": "spawn <material> <count>",
                "desc": "Spawn random particles using a material. Materials: electron, proton, neutron, ion+, ion-, aluminum",
            },
            "preset": {
                "usage": "preset <Hydrogen|Helium|Carbon|Plasma Box|Aluminum Cube (50nm, scaled)>",
                "desc": "Load a full scene preset.",
            },
            "scale": {
                "usage": "scale <macro|micro|nano|atomic>",
                "desc": "Set rendering/spacing scale profile.",
            },
            "mode": {
                "usage": "mode <A|B|status>",
                "desc": "Switch simulation mode. A=Blender-like fast viewport, B=full pairwise equations.",
            },
            "exact": {
                "usage": "exact <status|transport <on|off>|photons <n>|bounces <n>|energy <v>|coupling <v>>",
                "desc": "Configure exact Mode-B photon transport and atom-photon momentum coupling.",
            },
            "view": {
                "usage": "view <home|zoom <factor>|pan <dx> <dy>|depth <value>|status>",
                "desc": "Viewport controls similar to DCC tools (frame/home, zoom, pan, 3D depth).",
            },
            "obj": {
                "usage": "obj <addcube|addplane|addsphere|addlight|addcamera> <name> <x> <y> [z] <size> | obj z <name> <z> | obj del <name> | obj list",
                "desc": "Manage Mode-A scene objects (primitives, light, camera).",
            },
            "key": {
                "usage": "key set <name> <frame> <x> <y> <rot_deg>",
                "desc": "Set keyframe on an object for timeline animation.",
            },
            "timeline": {
                "usage": "timeline <play|pause|frame <n>|fps <n>|len <n>|status>",
                "desc": "Control Mode-A timeline playback.",
            },
            "tex": {
                "usage": "tex load <name> <image_path>",
                "desc": "Load and assign texture image to an object.",
            },
            "undo": {
                "usage": "undo",
                "desc": "Undo latest scene-object edit in Mode A.",
            },
            "redo": {
                "usage": "redo",
                "desc": "Redo latest undone scene-object edit in Mode A.",
            },
            "snap": {
                "usage": "snap <on|off|grid <size>|status>",
                "desc": "Toggle Blender-like transform snapping and set grid size.",
            },
            "rotvel": {
                "usage": "rotvel <deg_per_sec|status>",
                "desc": "Set or query arrow-key angular velocity used while in rotate mode.",
            },
            "camera": {
                "usage": "camera <status|pos <x> <y> <z>|rot <yaw> <pitch> <roll>|fov <deg>|speed <move|turn> <v>|reset>",
                "desc": "Configure full 3D camera transform, optics, and speed.",
            },
            "bg": {
                "usage": "bg <none|clear_sky_earth|deep_space|status>",
                "desc": "Set Mode-A sky object. Sun is a separate environment object.",
            },
            "export": {
                "usage": "export mp4 <path> [start end fps] | export test4k <png_path>",
                "desc": "Export Mode-A animation to MP4 or single-frame 4K test image.",
            },
            "step": {
                "usage": "step [n]",
                "desc": "Advance simulation by n steps (default 1).",
            },
            "pause": {
                "usage": "pause",
                "desc": "Pause continuous simulation.",
            },
            "run": {
                "usage": "run",
                "desc": "Resume continuous simulation.",
            },
            "clear": {
                "usage": "clear",
                "desc": "Delete all particles.",
            },
            "list": {
                "usage": "list [count]",
                "desc": "Log first N particles (default 12).",
            },
            "setv": {
                "usage": "setv <idx> <vx> <vy>",
                "desc": "Set velocity for one particle index.",
            },
            "physics": {
                "usage": "physics <k|rep|drag|dt> <value>",
                "desc": "Set physics parameter directly.",
            },
            "emergency": {
                "usage": "emergency <on|off|status>",
                "desc": "Configure emergency pause warning when interaction count is near crash territory.",
            },
            "quantum": {
                "usage": "quantum <set <n> <l> <m>|grid <n>|field <ex> <ey> <ez>|dt <v>|steps <n>|superpose <n> <l> <m> <w>|measure|view <slice|project> [axis]|Z <n>|info|reset>",
                "desc": "Mode C quantum TDSE controls. Set quantum numbers, grid resolution, electric field (Stark effect), create superpositions, measure position (Born rule collapse), or change visualization.",
            },
        }

        self.sim_mode_var = tk.StringVar(value="A")
        self.scale_profile_var = tk.StringVar(value="micro")
        self.render_radius_scale = SCALE_PROFILES["micro"]["render_radius"]
        self.scene_objects: dict[str, SceneCube] = {}
        self.selected_object_name: str | None = None
        self.timeline_frame = 0
        self.timeline_length = 240
        self.timeline_fps = 60.0
        self.timeline_playing = False
        self.timeline_accum = 0.0
        self.background_preset_var = tk.StringVar(value="clear_sky_earth")
        self.sun_object_name = "sun_earth_view"
        self.sun_object_enabled = True
        self.object_assets_dir = os.path.join(os.path.dirname(__file__), "objects")
        self.object_defs: dict[str, dict] = {}
        self.snap_enabled = True
        self.snap_grid = 12.0
        self.transform_mode: str | None = None
        self.transform_axis: str | None = None
        self.transform_anchor_world: tuple[float, float] | None = None
        self.transform_initial: tuple[float, float, float, float] | None = None
        self.rotate_key_left = False
        self.rotate_key_right = False
        self.rotate_key_up = False
        self.rotate_key_down = False
        self.rotate_key_undo_armed = False
        self.rotate_key_angular_velocity_dps = 90.0
        self.view_depth = 1200.0
        self.camera_x = 0.0
        self.camera_y = 0.0
        self.camera_z = -1200.0
        self.camera_yaw_deg = 0.0
        self.camera_pitch_deg = 0.0
        self.camera_roll_deg = 0.0
        self.camera_fov_deg = 60.0
        self.camera_near = 1.0
        self.camera_move_speed = 500.0
        self.camera_turn_speed = 75.0
        self.camera_look_sensitivity = 0.22
        self.camera_look_drag_last: tuple[float, float] | None = None
        # (#10) Orbit camera state
        self.orbit_pivot = [500.0, 430.0, 0.0]  # world xyz the camera orbits around
        self.orbit_distance = 1400.0
        self.orbit_yaw = 0.0  # degrees
        self.orbit_pitch = 15.0  # degrees
        self.orbit_drag_last: tuple[float, float] | None = None  # right-click drag
        self.pan_drag_last: tuple[float, float] | None = None  # middle-click drag
        # (#11) Cached background image to avoid re-rendering every frame
        self._cached_bg_pil: Image.Image | None = None
        self._cached_bg_preset: str = ""
        self._cached_bg_sun_enabled: bool = False
        self._mode_a_photo = None  # prevent GC of live-render PhotoImage
        self.cam_key_forward = False
        self.cam_key_back = False
        self.cam_key_left = False
        self.cam_key_right = False
        self.cam_key_up = False
        self.cam_key_down = False
        self.cam_key_yaw_left = False
        self.cam_key_yaw_right = False
        self.cam_key_pitch_up = False
        self.cam_key_pitch_down = False
        self.cam_key_roll_left = False
        self.cam_key_roll_right = False
        self.reset_camera()
        # (#10) Initial orbit camera sync
        self._sync_camera_from_orbit()
        self.undo_stack: list[dict[str, dict]] = []
        self.redo_stack: list[dict[str, dict]] = []
        self.emergency_pause_enabled = True
        self.emergency_suppress = False
        self.emergency_grace_ticks = 0
        self.emergency_pair_threshold = 30_000_000

        # Mode C: quantum engine (lazy-initialized on first use)
        self.quantum_world: QuantumWorld | None = None
        self._quantum_photo = None  # prevent GC of Tk PhotoImage

        self._build_ui()
        self._build_command_center()
        self._bind_events()
        self.world.load_preset("Hydrogen")
        self._load_default_scene_objects()
        self.selected_index = 0 if self.world.particles else None

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.command_window.protocol("WM_DELETE_WINDOW", self._on_close)
        self._tick()

    # -----------------------------
    # UI
    # -----------------------------
    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=8)
        main.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main, text="Physics World", font=("TkDefaultFont", 12, "bold")).pack(anchor="w", pady=(0, 4))
        self.canvas = tk.Canvas(main, width=self.world_w, height=self.world_h, bg="#0b0f14", highlightthickness=0)
        self.canvas.pack(anchor="nw")

    def _build_command_center(self) -> None:
        self.command_window = tk.Toplevel(self.root)
        self.command_window.title("Matter Sim - Command Center")
        self.command_window.geometry("540x920")

        panel = ttk.Frame(self.command_window, padding=10)
        panel.pack(fill=tk.BOTH, expand=True)

        ttk.Label(panel, text="Simulation", font=("TkDefaultFont", 13, "bold")).pack(anchor="w", pady=(0, 6))

        mode_row = ttk.Frame(panel)
        mode_row.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(mode_row, text="Mode").pack(side=tk.LEFT)
        self.mode_combo = ttk.Combobox(mode_row, textvariable=self.sim_mode_var, values=["A", "B", "C"], state="readonly", width=8)
        self.mode_combo.pack(side=tk.LEFT, padx=6)
        self.mode_combo.bind("<<ComboboxSelected>>", lambda _e: self.apply_sim_mode(self.sim_mode_var.get()))

        self.run_btn = ttk.Button(panel, text="Pause", command=self.toggle_running)
        self.run_btn.pack(fill=tk.X, pady=2)

        ttk.Button(panel, text="Step Once", command=self.step_once).pack(fill=tk.X, pady=2)
        ttk.Button(panel, text="Reset (Current Preset)", command=self.reload_preset).pack(fill=tk.X, pady=2)

        ttk.Separator(panel).pack(fill=tk.X, pady=8)

        ttk.Label(panel, text="Presets").pack(anchor="w")
        self.preset_var = tk.StringVar(value="Hydrogen")
        preset_values = ["Hydrogen", "Helium", "Carbon", "Plasma Box", "Aluminum Cube (50nm, scaled)"]
        self.preset_combo = ttk.Combobox(panel, textvariable=self.preset_var, values=preset_values, state="readonly")
        self.preset_combo.pack(fill=tk.X, pady=2)
        ttk.Button(panel, text="Load Preset", command=lambda: self.load_preset(self.preset_var.get())).pack(fill=tk.X, pady=2)

        ttk.Label(panel, text="Scale Profile").pack(anchor="w", pady=(6, 0))
        self.scale_combo = ttk.Combobox(panel, textvariable=self.scale_profile_var, values=list(SCALE_PROFILES.keys()), state="readonly")
        self.scale_combo.pack(fill=tk.X, pady=2)
        self.scale_combo.bind("<<ComboboxSelected>>", lambda _e: self.apply_scale_profile(self.scale_profile_var.get()))

        ttk.Label(panel, text="Background Preset").pack(anchor="w", pady=(6, 0))
        self.bg_combo = ttk.Combobox(
            panel,
            textvariable=self.background_preset_var,
            values=["none", "clear_sky_earth", "deep_space"],
            state="readonly",
        )
        self.bg_combo.pack(fill=tk.X, pady=2)
        self.bg_combo.bind("<<ComboboxSelected>>", lambda _e: self._log(f"background={self.background_preset_var.get()}"))
        ttk.Button(
            panel,
            text="Export MP4 (timeline)",
            command=lambda: self.execute_external_command(f"export mp4 output_4k.mp4 0 {self.timeline_length} {int(self.timeline_fps)}"),
        ).pack(fill=tk.X, pady=2)

        ttk.Separator(panel).pack(fill=tk.X, pady=8)

        ttk.Label(panel, text="Spawn Material").pack(anchor="w")
        self.material_var = tk.StringVar(value="electron")
        self.material_combo = ttk.Combobox(panel, textvariable=self.material_var, values=list(MATERIALS.keys()), state="readonly")
        self.material_combo.pack(fill=tk.X, pady=2)

        ttk.Label(panel, text="Spawn Count").pack(anchor="w")
        self.spawn_count_var = tk.IntVar(value=10)
        ttk.Spinbox(panel, from_=1, to=500, textvariable=self.spawn_count_var, width=8).pack(anchor="w", pady=2)

        ttk.Button(panel, text="Spawn Random", command=self.spawn_random).pack(fill=tk.X, pady=2)
        ttk.Button(panel, text="Clear All", command=self.clear_all).pack(fill=tk.X, pady=2)

        ttk.Separator(panel).pack(fill=tk.X, pady=8)

        ttk.Label(panel, text="Assign Velocity (selected)").pack(anchor="w")
        vel_row = ttk.Frame(panel)
        vel_row.pack(fill=tk.X, pady=2)
        ttk.Label(vel_row, text="vx").pack(side=tk.LEFT)
        self.vx_var = tk.DoubleVar(value=0.0)
        ttk.Entry(vel_row, textvariable=self.vx_var, width=8).pack(side=tk.LEFT, padx=4)
        ttk.Label(vel_row, text="vy").pack(side=tk.LEFT)
        self.vy_var = tk.DoubleVar(value=0.0)
        ttk.Entry(vel_row, textvariable=self.vy_var, width=8).pack(side=tk.LEFT, padx=4)
        ttk.Button(panel, text="Apply Velocity", command=self.apply_velocity_to_selected).pack(fill=tk.X, pady=2)

        ttk.Separator(panel).pack(fill=tk.X, pady=8)

        ttk.Label(panel, text="Physics").pack(anchor="w")
        self.k_var = tk.DoubleVar(value=self.world.k_coulomb)
        self.rep_var = tk.DoubleVar(value=self.world.repulsion_k)
        self.drag_var = tk.DoubleVar(value=self.world.drag)
        self.dt_var = tk.DoubleVar(value=self.world.dt)

        self._add_slider(panel, "Coulomb K", self.k_var, 100.0, 10000.0)
        self._add_slider(panel, "Repulsion K", self.rep_var, 1000.0, 100000.0)
        self._add_slider(panel, "Drag", self.drag_var, 0.95, 1.0)
        self._add_slider(panel, "Time Step", self.dt_var, 0.001, 0.03)

        self.info_var = tk.StringVar(value="Ready")
        ttk.Label(panel, textvariable=self.info_var, wraplength=330, foreground="#225").pack(fill=tk.X, pady=(8, 0))

        ttk.Separator(panel).pack(fill=tk.X, pady=8)
        ttk.Label(panel, text="Command Console", font=("TkDefaultFont", 12, "bold")).pack(anchor="w")

        cmd_row = ttk.Frame(panel)
        cmd_row.pack(fill=tk.X, pady=(4, 2))
        self.command_var = tk.StringVar(value="spawn electron 10")
        self.command_entry = ttk.Entry(cmd_row, textvariable=self.command_var)
        self.command_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.command_entry.bind("<Return>", lambda _e: self.execute_command())
        ttk.Button(cmd_row, text="Run", command=self.execute_command).pack(side=tk.LEFT, padx=4)

        help_row = ttk.Frame(panel)
        help_row.pack(fill=tk.X, pady=(2, 2))
        ttk.Label(help_row, text="Help Search:").pack(side=tk.LEFT)
        self.help_query_var = tk.StringVar(value="")
        help_entry = ttk.Entry(help_row, textvariable=self.help_query_var, width=18)
        help_entry.pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)
        help_entry.bind("<KeyRelease>", lambda _e: self.refresh_help_list())

        list_and_desc = ttk.Frame(panel)
        list_and_desc.pack(fill=tk.BOTH, expand=True, pady=(2, 2))

        self.help_list = tk.Listbox(list_and_desc, height=8)
        self.help_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.help_list.bind("<<ListboxSelect>>", lambda _e: self.show_selected_help())

        scroll_help = ttk.Scrollbar(list_and_desc, orient=tk.VERTICAL, command=self.help_list.yview)
        scroll_help.pack(side=tk.LEFT, fill=tk.Y)
        self.help_list.configure(yscrollcommand=scroll_help.set)

        self.help_text = tk.Text(panel, height=6, wrap=tk.WORD)
        self.help_text.pack(fill=tk.X)
        self.help_text.configure(state=tk.DISABLED)

        ttk.Label(panel, text="Output Log:").pack(anchor="w", pady=(6, 0))
        self.output_text = tk.Text(panel, height=10, wrap=tk.WORD)
        self.output_text.pack(fill=tk.BOTH, expand=True)
        self.output_text.configure(state=tk.DISABLED)

        ttk.Label(panel, text="Outliner (first 300):").pack(anchor="w", pady=(6, 0))
        self.outliner_list = tk.Listbox(panel, height=8)
        self.outliner_list.pack(fill=tk.BOTH, expand=False)
        self.outliner_list.bind("<<ListboxSelect>>", lambda _e: self._on_outliner_select())

        self.refresh_help_list()
        self._log("Command center ready. Type a command and press Enter.")

    def _add_slider(self, panel: ttk.Frame, label: str, var: tk.DoubleVar, mn: float, mx: float) -> None:
        ttk.Label(panel, text=label).pack(anchor="w")
        scale = ttk.Scale(panel, from_=mn, to=mx, variable=var)
        scale.pack(fill=tk.X, pady=(0, 4))

    def _bind_events(self) -> None:
        self.canvas.bind("<Button-1>", self.on_left_down)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_up)
        self.canvas.bind("<Button-3>", self.on_right_down_camera_look)
        self.canvas.bind("<B3-Motion>", self.on_right_drag_camera_look)
        self.canvas.bind("<ButtonRelease-3>", self.on_right_up_camera_look)
        self.canvas.bind("<Button-2>", self.on_middle_down)
        self.canvas.bind("<B2-Motion>", self.on_middle_drag)
        self.canvas.bind("<ButtonRelease-2>", self.on_middle_up)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Button-4>", lambda e: self._zoom_at(e.x, e.y, 1.1))
        self.canvas.bind("<Button-5>", lambda e: self._zoom_at(e.x, e.y, 1.0 / 1.1))
        self.root.bind("<space>", lambda _e: self.toggle_running())
        self.root.bind("<Delete>", lambda _e: self.delete_selected())
        self.root.bind("<g>", lambda _e: self.start_transform_mode("move"))
        self.root.bind("<r>", lambda _e: self.start_transform_mode("rotate"))
        self.root.bind("<s>", lambda _e: self.start_transform_mode("scale"))
        self.root.bind("<x>", lambda _e: self.set_transform_axis("x"))
        self.root.bind("<y>", lambda _e: self.set_transform_axis("y"))
        self.root.bind("<Escape>", lambda _e: self.cancel_transform())
        self.root.bind("<Return>", lambda _e: self.confirm_transform())
        self.root.bind("<Shift-D>", lambda _e: self.duplicate_selected_object())
        self.root.bind("<Control-z>", lambda _e: self.undo_scene_edit())
        self.root.bind("<Control-y>", lambda _e: self.redo_scene_edit())
        self.root.bind("<Shift-Tab>", lambda _e: self.toggle_snap())
        self.root.bind("<KeyPress-Left>", lambda _e: self._on_rotate_arrow_press("left"))
        self.root.bind("<KeyPress-Right>", lambda _e: self._on_rotate_arrow_press("right"))
        self.root.bind("<KeyPress-Up>", lambda _e: self._on_rotate_arrow_press("up"))
        self.root.bind("<KeyPress-Down>", lambda _e: self._on_rotate_arrow_press("down"))
        self.root.bind("<KeyRelease-Left>", lambda _e: self._on_rotate_arrow_release("left"))
        self.root.bind("<KeyRelease-Right>", lambda _e: self._on_rotate_arrow_release("right"))
        self.root.bind("<KeyRelease-Up>", lambda _e: self._on_rotate_arrow_release("up"))
        self.root.bind("<KeyRelease-Down>", lambda _e: self._on_rotate_arrow_release("down"))
        self.root.bind("<FocusOut>", lambda _e: (self._clear_rotate_key_state(), self._clear_camera_key_state(), self._clear_camera_look_state()))
        self.root.bind("<KeyPress-w>", lambda _e: self._on_camera_key_press("forward"))
        self.root.bind("<KeyPress-s>", lambda _e: self._on_camera_key_press("back"))
        self.root.bind("<KeyPress-a>", lambda _e: self._on_camera_key_press("left"))
        self.root.bind("<KeyPress-d>", lambda _e: self._on_camera_key_press("right"))
        self.root.bind("<KeyPress-q>", lambda _e: self._on_camera_key_press("up"))
        self.root.bind("<KeyPress-e>", lambda _e: self._on_camera_key_press("down"))
        self.root.bind("<KeyPress-j>", lambda _e: self._on_camera_key_press("yaw_left"))
        self.root.bind("<KeyPress-l>", lambda _e: self._on_camera_key_press("yaw_right"))
        self.root.bind("<KeyPress-i>", lambda _e: self._on_camera_key_press("pitch_up"))
        self.root.bind("<KeyPress-k>", lambda _e: self._on_camera_key_press("pitch_down"))
        self.root.bind("<KeyPress-u>", lambda _e: self._on_camera_key_press("roll_left"))
        self.root.bind("<KeyPress-o>", lambda _e: self._on_camera_key_press("roll_right"))
        self.root.bind("<KeyRelease-w>", lambda _e: self._on_camera_key_release("forward"))
        self.root.bind("<KeyRelease-s>", lambda _e: self._on_camera_key_release("back"))
        self.root.bind("<KeyRelease-a>", lambda _e: self._on_camera_key_release("left"))
        self.root.bind("<KeyRelease-d>", lambda _e: self._on_camera_key_release("right"))
        self.root.bind("<KeyRelease-q>", lambda _e: self._on_camera_key_release("up"))
        self.root.bind("<KeyRelease-e>", lambda _e: self._on_camera_key_release("down"))
        self.root.bind("<KeyRelease-j>", lambda _e: self._on_camera_key_release("yaw_left"))
        self.root.bind("<KeyRelease-l>", lambda _e: self._on_camera_key_release("yaw_right"))
        self.root.bind("<KeyRelease-i>", lambda _e: self._on_camera_key_release("pitch_up"))
        self.root.bind("<KeyRelease-k>", lambda _e: self._on_camera_key_release("pitch_down"))
        self.root.bind("<KeyRelease-u>", lambda _e: self._on_camera_key_release("roll_left"))
        self.root.bind("<KeyRelease-o>", lambda _e: self._on_camera_key_release("roll_right"))

    # -----------------------------
    # Presets
    # -----------------------------
    def clear_all(self) -> None:
        self.world.particles.clear()
        self.world.bonds.clear()
        self.selected_index = None
        self.selected_object_name = None

    def apply_scale_profile(self, profile_name: str) -> None:
        if profile_name not in SCALE_PROFILES:
            self._log(f"error: unknown scale profile {profile_name}")
            return
        cfg = SCALE_PROFILES[profile_name]
        self.render_radius_scale = cfg["render_radius"]
        self._log(
            f"scale={profile_name} render_radius={cfg['render_radius']} distance={cfg['distance']} velocity={cfg['velocity']}"
        )

    def world_to_screen(self, x: float, y: float) -> tuple[float, float]:
        sx = (x + self.view_pan_x) * self.view_zoom
        sy = (y + self.view_pan_y) * self.view_zoom
        return sx, sy

    @staticmethod
    def _rot_x(x: float, y: float, z: float, deg: float) -> tuple[float, float, float]:
        a = math.radians(deg)
        c = math.cos(a)
        s = math.sin(a)
        return x, (y * c - z * s), (y * s + z * c)

    @staticmethod
    def _rot_y(x: float, y: float, z: float, deg: float) -> tuple[float, float, float]:
        a = math.radians(deg)
        c = math.cos(a)
        s = math.sin(a)
        return (x * c + z * s), y, (-x * s + z * c)

    @staticmethod
    def _rot_z(x: float, y: float, z: float, deg: float) -> tuple[float, float, float]:
        a = math.radians(deg)
        c = math.cos(a)
        s = math.sin(a)
        return (x * c - y * s), (x * s + y * c), z

    def world3_to_screen(self, x: float, y: float, z: float) -> tuple[float, float, float, float] | None:
        # World uses y-down. Convert to camera math y-up.
        wx = x - self.camera_x
        wy = -(y - self.camera_y)
        wz = z - self.camera_z

        # Inverse camera orientation: world -> camera space.
        cx, cy, cz = self._rot_z(wx, wy, wz, -self.camera_roll_deg)
        cx, cy, cz = self._rot_x(cx, cy, cz, -self.camera_pitch_deg)
        cx, cy, cz = self._rot_y(cx, cy, cz, -self.camera_yaw_deg)

        if cz <= max(0.1, self.camera_near):
            return None

        fov = max(10.0, min(160.0, self.camera_fov_deg))
        focal = (self.world_h * 0.5) / math.tan(math.radians(fov * 0.5)) * self.view_zoom
        sx = (self.world_w * 0.5) + (cx / cz) * focal + self.view_pan_x
        sy = (self.world_h * 0.5) - (cy / cz) * focal + self.view_pan_y
        scale = focal / cz
        return sx, sy, scale, cz

    def screen_to_world(self, sx: float, sy: float) -> tuple[float, float]:
        x = sx / self.view_zoom - self.view_pan_x
        y = sy / self.view_zoom - self.view_pan_y
        return x, y

    def _zoom_at(self, sx: float, sy: float, scale: float) -> None:
        wx, wy = self.screen_to_world(float(sx), float(sy))
        self.view_zoom = max(0.1, min(8.0, self.view_zoom * scale))
        nsx, nsy = self.world_to_screen(wx, wy)
        self.view_pan_x += (float(sx) - nsx) / self.view_zoom
        self.view_pan_y += (float(sy) - nsy) / self.view_zoom

    def reset_view(self) -> None:
        self.view_zoom = 1.0
        self.view_pan_x = 0.0
        self.view_pan_y = 0.0

    def _camera_look_at(self, tx: float, ty: float, tz: float) -> None:
        dx = tx - self.camera_x
        dy_up = -(ty - self.camera_y)
        dz = tz - self.camera_z
        dist = math.sqrt(dx * dx + dy_up * dy_up + dz * dz)
        if dist < 1e-9:
            return
        self.camera_yaw_deg = math.degrees(math.atan2(dx, dz))
        self.camera_pitch_deg = -math.degrees(math.asin(max(-1.0, min(1.0, dy_up / dist))))
        self.camera_pitch_deg = max(-89.0, min(89.0, self.camera_pitch_deg))

        # Refine yaw/pitch numerically so the target lands at screen center.
        cx_t = self.world_w * 0.5
        cy_t = self.world_h * 0.5
        eps = 0.05
        for _ in range(24):
            proj = self.world3_to_screen(tx, ty, tz)
            if proj is None:
                break
            sx, sy, _sc, _d = proj
            ex = sx - cx_t
            ey = sy - cy_t
            if abs(ex) + abs(ey) < 0.5:
                break

            y0 = self.camera_yaw_deg
            p0 = self.camera_pitch_deg

            self.camera_yaw_deg = y0 + eps
            p_yaw = self.world3_to_screen(tx, ty, tz)
            self.camera_yaw_deg = y0
            self.camera_pitch_deg = p0 + eps
            p_pitch = self.world3_to_screen(tx, ty, tz)
            self.camera_pitch_deg = p0
            if p_yaw is None or p_pitch is None:
                break

            sx_y, sy_y, _a, _b = p_yaw
            sx_p, sy_p, _c, _d2 = p_pitch
            j11 = (sx_y - sx) / eps
            j21 = (sy_y - sy) / eps
            j12 = (sx_p - sx) / eps
            j22 = (sy_p - sy) / eps
            det = j11 * j22 - j12 * j21
            if abs(det) < 1e-9:
                break

            # Solve J * delta = -error
            dyaw = (-ex * j22 + j12 * ey) / det
            dpitch = (-j11 * ey + ex * j21) / det
            dyaw = max(-8.0, min(8.0, dyaw))
            dpitch = max(-8.0, min(8.0, dpitch))

            self.camera_yaw_deg += dyaw
            self.camera_pitch_deg += dpitch
            self.camera_pitch_deg = max(-89.0, min(89.0, self.camera_pitch_deg))

    def reset_camera(self) -> None:
        # (#10) Orbit camera: reset orbit params and derive camera transform.
        self.orbit_pivot = [500.0, 430.0, 0.0]
        self.orbit_distance = 1400.0
        self.orbit_yaw = 0.0
        self.orbit_pitch = 15.0
        self.camera_roll_deg = 0.0
        self.camera_fov_deg = 60.0
        self._sync_camera_from_orbit()

    def _sync_camera_from_orbit(self) -> None:
        """Derive camera_x/y/z and yaw/pitch from orbit pivot + distance + angles."""
        yaw_r = math.radians(self.orbit_yaw)
        pitch_r = math.radians(self.orbit_pitch)
        # Camera position = pivot + offset in spherical coords (y-down world).
        dx = self.orbit_distance * math.sin(yaw_r) * math.cos(pitch_r)
        dy_up = self.orbit_distance * math.sin(pitch_r)
        dz = -self.orbit_distance * math.cos(yaw_r) * math.cos(pitch_r)
        self.camera_x = self.orbit_pivot[0] + dx
        self.camera_y = self.orbit_pivot[1] - dy_up  # y-down
        self.camera_z = self.orbit_pivot[2] + dz
        # Look at pivot
        self._camera_look_at(self.orbit_pivot[0], self.orbit_pivot[1], self.orbit_pivot[2])

    def camera_status_text(self) -> str:
        return (
            f"camera pos=({self.camera_x:.2f}, {self.camera_y:.2f}, {self.camera_z:.2f}) "
            f"rot(yaw,pitch,roll)=({self.camera_yaw_deg:.2f}, {self.camera_pitch_deg:.2f}, {self.camera_roll_deg:.2f}) "
            f"fov={self.camera_fov_deg:.2f} move_speed={self.camera_move_speed:.2f} turn_speed={self.camera_turn_speed:.2f}"
        )

    def delete_selected(self) -> None:
        if self.selected_object_name is not None and self.selected_object_name in self.scene_objects:
            self.delete_object(self.selected_object_name)
            self.selected_object_name = None
            return
        if self.selected_index is None:
            return
        if 0 <= self.selected_index < len(self.world.particles):
            del self.world.particles[self.selected_index]
        self.selected_index = None
        self._log("deleted selected object")

    def apply_sim_mode(self, mode: str) -> None:
        m = mode.upper()
        if m not in {"A", "B", "C"}:
            self._log("error: mode must be A, B, or C")
            return
        self.sim_mode_var.set(m)
        if m == "A":
            self._log("mode=A (Blender-like viewport, approximated simulation + proxy ray transport)")
        elif m == "B":
            self._log(f"mode=B (full equations + exact transport, backend={self.world.compute_backend})")
        else:
            if self.quantum_world is None:
                self.quantum_world = QuantumWorld()
                self._log("Quantum engine initialized (64\u00b3 grid, hydrogen Z=1)")
            self._log("mode=C (quantum TDSE split-operator, exact Schr\u00f6dinger evolution)")

    def load_preset(self, name: str) -> None:
        cfg = SCALE_PROFILES.get(self.scale_profile_var.get(), SCALE_PROFILES["micro"])
        self.world.load_preset(name, distance_scale=cfg["distance"], velocity_scale=cfg["velocity"])
        self.selected_index = 0 if self.world.particles else None
        if self.world.last_note:
            self._log(self.world.last_note)

    def reload_preset(self) -> None:
        self.load_preset(self.preset_var.get())

    def spawn_random(self) -> None:
        mat = MATERIALS[self.material_var.get()]
        count = max(1, int(self.spawn_count_var.get()))
        for _ in range(count):
            self.world.particles.append(
                Particle(
                    random.uniform(50, self.world_w - 50),
                    random.uniform(50, self.world_h - 50),
                    random.uniform(-100, 100) * SCALE_PROFILES[self.scale_profile_var.get()]["velocity"],
                    random.uniform(-100, 100) * SCALE_PROFILES[self.scale_profile_var.get()]["velocity"],
                    mat,
                    random.uniform(-280.0, 280.0),
                    random.uniform(-40.0, 40.0),
                )
            )
        self._log(f"spawned {count} {mat.name}")

    # -----------------------------
    # Interaction
    # -----------------------------
    def find_particle(self, x: float, y: float) -> int | None:
        best_i = None
        best_d = 1e9
        for i, p in enumerate(self.world.particles):
            sx, sy = self.world_to_screen(p.x, p.y)
            d = math.hypot(sx - x, sy - y)
            hit_r = max(4.0, p.radius * self.render_radius_scale * self.view_zoom + 6)
            if d <= hit_r and d < best_d:
                best_d = d
                best_i = i
        return best_i

    def on_left_down(self, event) -> None:
        sx, sy = float(event.x), float(event.y)
        if self.sim_mode_var.get().upper() == "A":
            obj_name = self.find_scene_object(sx, sy)
            if obj_name is not None:
                self.selected_object_name = obj_name
                self.selected_index = None
                self.transform_anchor_world = self.screen_to_world(sx, sy)
                return
            self.selected_object_name = None
        idx = self.find_particle(sx, sy)
        self.selected_index = idx
        if idx is not None:
            wx, wy = self.screen_to_world(sx, sy)
            self.drag_start = (wx, wy)
            self.drag_current = (wx, wy)
        else:
            self.drag_start = None
            self.drag_current = None

    def on_drag(self, event) -> None:
        if self.sim_mode_var.get().upper() == "A" and self.transform_mode and self.selected_object_name:
            obj = self.scene_objects.get(self.selected_object_name)
            if obj is None:
                return
            wx, wy = self.screen_to_world(float(event.x), float(event.y))
            if self.transform_anchor_world is None:
                self.transform_anchor_world = (wx, wy)
            if self.transform_initial is None:
                self.transform_initial = (obj.x, obj.y, obj.rot_deg, obj.size)
                self._push_undo()
            x0, y0, r0, s0 = self.transform_initial
            ax, ay = self.transform_anchor_world
            dx, dy = wx - ax, wy - ay
            if self.transform_axis == "x":
                dy = 0.0
            elif self.transform_axis == "y":
                dx = 0.0

            if self.transform_mode == "move":
                nx = x0 + dx
                ny = y0 + dy
                if self.snap_enabled:
                    g = max(0.01, self.snap_grid)
                    nx = round(nx / g) * g
                    ny = round(ny / g) * g
                obj.x = nx
                obj.y = ny
            elif self.transform_mode == "rotate":
                a0 = math.degrees(math.atan2(ay - y0, ax - x0))
                a1 = math.degrees(math.atan2(wy - y0, wx - x0))
                nr = r0 + (a1 - a0)
                if self.snap_enabled:
                    nr = round(nr / 5.0) * 5.0
                obj.rot_deg = nr
            elif self.transform_mode == "scale":
                dist = math.hypot(dx, dy)
                ns = max(4.0, s0 + dist)
                if self.snap_enabled:
                    g = max(1.0, self.snap_grid)
                    ns = round(ns / g) * g
                obj.size = ns
            return
        if self.drag_start is not None:
            self.drag_current = self.screen_to_world(float(event.x), float(event.y))

    def on_left_up(self, event) -> None:
        if self.transform_mode is not None:
            return
        if self.drag_start is None or self.selected_index is None:
            return
        sx, sy = self.drag_start
        ex, ey = self.screen_to_world(float(event.x), float(event.y))
        scale = 3.0
        vx = (sx - ex) * scale
        vy = (sy - ey) * scale
        p = self.world.particles[self.selected_index]
        p.vx = vx
        p.vy = vy
        self.vx_var.set(vx)
        self.vy_var.set(vy)
        self.drag_start = None
        self.drag_current = None

    # (#10) Right-click = orbit/rotate around pivot
    def on_right_down_camera_look(self, event) -> None:
        if self.sim_mode_var.get().upper() != "A":
            return
        self.orbit_drag_last = (float(event.x), float(event.y))

    def on_right_drag_camera_look(self, event) -> None:
        if self.sim_mode_var.get().upper() != "A":
            return
        if self.orbit_drag_last is None:
            self.orbit_drag_last = (float(event.x), float(event.y))
            return
        lx, ly = self.orbit_drag_last
        nx, ny = float(event.x), float(event.y)
        dx = nx - lx
        dy = ny - ly
        self.orbit_yaw += dx * self.camera_look_sensitivity
        self.orbit_pitch = max(-89.0, min(89.0, self.orbit_pitch + dy * self.camera_look_sensitivity))
        self._sync_camera_from_orbit()
        self.orbit_drag_last = (nx, ny)

    def on_right_up_camera_look(self, event) -> None:
        _ = event
        self.orbit_drag_last = None

    # (#10) Middle-click = pan (translate pivot)
    def on_middle_down(self, event) -> None:
        self.pan_drag_last = (float(event.x), float(event.y))

    def on_middle_drag(self, event) -> None:
        if self.pan_drag_last is None:
            return
        lx, ly = self.pan_drag_last
        nx, ny = float(event.x), float(event.y)
        dx, dy = nx - lx, ny - ly
        # Pan in camera-local right/up plane
        yaw_r = math.radians(self.orbit_yaw)
        rx = math.cos(yaw_r)
        rz = math.sin(yaw_r)
        pan_speed = self.orbit_distance * 0.001
        self.orbit_pivot[0] -= dx * rx * pan_speed
        self.orbit_pivot[2] -= dx * rz * pan_speed
        self.orbit_pivot[1] += dy * pan_speed
        self._sync_camera_from_orbit()
        self.pan_drag_last = (nx, ny)

    def on_middle_up(self, event) -> None:
        self.pan_drag_last = None

    def on_mouse_wheel(self, event) -> None:
        # (#10) Scroll = zoom (change orbit distance)
        delta = event.delta if hasattr(event, "delta") else 0
        if delta > 0:
            self.orbit_distance = max(50.0, self.orbit_distance * 0.9)
        elif delta < 0:
            self.orbit_distance = min(50000.0, self.orbit_distance * 1.1)
        self._sync_camera_from_orbit()

    def apply_velocity_to_selected(self) -> None:
        if self.selected_index is None:
            return
        p = self.world.particles[self.selected_index]
        p.vx = float(self.vx_var.get())
        p.vy = float(self.vy_var.get())
        self._log(f"setv idx={self.selected_index} vx={p.vx:.3f} vy={p.vy:.3f}")

    def _scene_snapshot(self) -> dict[str, dict]:
        snap: dict[str, dict] = {}
        for n, o in self.scene_objects.items():
            snap[n] = {
                "name": o.name,
                "x": o.x,
                "y": o.y,
                "z": o.z,
                "size": o.size,
                "kind": o.kind,
                "rot_deg": o.rot_deg,
                "color": o.color,
                "roughness": o.roughness,
                "metallic": o.metallic,
                "emission": o.emission,
                "visible": o.visible,
                "collection": o.collection,
                "texture_path": o.texture_path,
                "keyframes": [{"frame": k.frame, "x": k.x, "y": k.y, "rot_deg": k.rot_deg} for k in o.keyframes],
            }
        return snap

    def _restore_scene_snapshot(self, snap: dict[str, dict]) -> None:
        self.scene_objects.clear()
        for n, d in snap.items():
            obj = SceneCube(
                name=str(d.get("name", n)),
                x=float(d.get("x", 0.0)),
                y=float(d.get("y", 0.0)),
                z=float(d.get("z", 0.0)),
                size=float(d.get("size", 80.0)),
                kind=str(d.get("kind", "cube")),
                rot_deg=float(d.get("rot_deg", 0.0)),
                color=str(d.get("color", "#9aa4c9")),
                roughness=float(d.get("roughness", 0.55)),
                metallic=float(d.get("metallic", 0.0)),
                emission=float(d.get("emission", 0.0)),
                visible=bool(d.get("visible", True)),
                collection=str(d.get("collection", "Scene")),
            )
            obj.keyframes = [
                ObjectKeyframe(frame=int(k["frame"]), x=float(k["x"]), y=float(k["y"]), rot_deg=float(k["rot_deg"]))
                for k in d.get("keyframes", [])
            ]
            self.scene_objects[n] = obj

    def _push_undo(self) -> None:
        self.undo_stack.append(self._scene_snapshot())
        if len(self.undo_stack) > 200:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def undo_scene_edit(self) -> None:
        if not self.undo_stack:
            self._log("undo: nothing to undo")
            return
        self.redo_stack.append(self._scene_snapshot())
        snap = self.undo_stack.pop()
        self._restore_scene_snapshot(snap)
        self._log("undo: scene restored")

    def redo_scene_edit(self) -> None:
        if not self.redo_stack:
            self._log("redo: nothing to redo")
            return
        self.undo_stack.append(self._scene_snapshot())
        snap = self.redo_stack.pop()
        self._restore_scene_snapshot(snap)
        self._log("redo: scene restored")

    def toggle_snap(self) -> None:
        self.snap_enabled = not self.snap_enabled
        self._log(f"snap enabled={self.snap_enabled} grid={self.snap_grid:.2f}")

    def _clear_rotate_key_state(self) -> None:
        self.rotate_key_left = False
        self.rotate_key_right = False
        self.rotate_key_up = False
        self.rotate_key_down = False

    def _on_rotate_arrow_press(self, direction: str) -> None:
        if direction == "left":
            self.rotate_key_left = True
        elif direction == "right":
            self.rotate_key_right = True
        elif direction == "up":
            self.rotate_key_up = True
        elif direction == "down":
            self.rotate_key_down = True

        if self.sim_mode_var.get().upper() != "A":
            return
        if self.transform_mode != "rotate":
            return
        if self.selected_object_name is None or self.selected_object_name not in self.scene_objects:
            return
        if not self.rotate_key_undo_armed:
            self._push_undo()
            self.rotate_key_undo_armed = True

    def _on_rotate_arrow_release(self, direction: str) -> None:
        if direction == "left":
            self.rotate_key_left = False
        elif direction == "right":
            self.rotate_key_right = False
        elif direction == "up":
            self.rotate_key_up = False
        elif direction == "down":
            self.rotate_key_down = False

    def _apply_rotate_key_hold(self, dt_seconds: float) -> None:
        if self.sim_mode_var.get().upper() != "A":
            return
        if self.transform_mode != "rotate":
            return
        if self.selected_object_name is None:
            return
        obj = self.scene_objects.get(self.selected_object_name)
        if obj is None:
            return

        direction = 0
        if self.rotate_key_left or self.rotate_key_down:
            direction -= 1
        if self.rotate_key_right or self.rotate_key_up:
            direction += 1
        if direction == 0:
            return

        obj.rot_deg += direction * self.rotate_key_angular_velocity_dps * max(0.0, dt_seconds)

    def _on_camera_key_press(self, key: str) -> None:
        if key == "forward":
            self.cam_key_forward = True
        elif key == "back":
            self.cam_key_back = True
        elif key == "left":
            self.cam_key_left = True
        elif key == "right":
            self.cam_key_right = True
        elif key == "up":
            self.cam_key_up = True
        elif key == "down":
            self.cam_key_down = True
        elif key == "yaw_left":
            self.cam_key_yaw_left = True
        elif key == "yaw_right":
            self.cam_key_yaw_right = True
        elif key == "pitch_up":
            self.cam_key_pitch_up = True
        elif key == "pitch_down":
            self.cam_key_pitch_down = True
        elif key == "roll_left":
            self.cam_key_roll_left = True
        elif key == "roll_right":
            self.cam_key_roll_right = True

    def _on_camera_key_release(self, key: str) -> None:
        if key == "forward":
            self.cam_key_forward = False
        elif key == "back":
            self.cam_key_back = False
        elif key == "left":
            self.cam_key_left = False
        elif key == "right":
            self.cam_key_right = False
        elif key == "up":
            self.cam_key_up = False
        elif key == "down":
            self.cam_key_down = False
        elif key == "yaw_left":
            self.cam_key_yaw_left = False
        elif key == "yaw_right":
            self.cam_key_yaw_right = False
        elif key == "pitch_up":
            self.cam_key_pitch_up = False
        elif key == "pitch_down":
            self.cam_key_pitch_down = False
        elif key == "roll_left":
            self.cam_key_roll_left = False
        elif key == "roll_right":
            self.cam_key_roll_right = False

    def _clear_camera_key_state(self) -> None:
        self.cam_key_forward = False
        self.cam_key_back = False
        self.cam_key_left = False
        self.cam_key_right = False
        self.cam_key_up = False
        self.cam_key_down = False
        self.cam_key_yaw_left = False
        self.cam_key_yaw_right = False
        self.cam_key_pitch_up = False
        self.cam_key_pitch_down = False
        self.cam_key_roll_left = False
        self.cam_key_roll_right = False

    def _clear_camera_look_state(self) -> None:
        self.camera_look_drag_last = None

    def _apply_camera_key_hold(self, dt_seconds: float) -> None:
        if self.sim_mode_var.get().upper() != "A":
            return
        if self.transform_mode is not None:
            return

        s = max(0.0, dt_seconds)
        move = self.camera_move_speed * s
        turn = self.camera_turn_speed * s

        # Orientation controls
        if self.cam_key_yaw_left:
            self.camera_yaw_deg -= turn
        if self.cam_key_yaw_right:
            self.camera_yaw_deg += turn
        if self.cam_key_pitch_up:
            self.camera_pitch_deg += turn
        if self.cam_key_pitch_down:
            self.camera_pitch_deg -= turn
        if self.cam_key_roll_left:
            self.camera_roll_deg -= turn
        if self.cam_key_roll_right:
            self.camera_roll_deg += turn

        self.camera_pitch_deg = max(-89.0, min(89.0, self.camera_pitch_deg))

        # Movement in camera-relative basis (y-up math space), then convert to world y-down.
        yaw = math.radians(self.camera_yaw_deg)
        pitch = math.radians(self.camera_pitch_deg)
        fx = math.sin(yaw) * math.cos(pitch)
        fy = math.sin(pitch)
        fz = math.cos(yaw) * math.cos(pitch)
        rx, ry, rz = fz, 0.0, -fx
        rlen = max(1e-6, math.sqrt(rx * rx + ry * ry + rz * rz))
        rx, ry, rz = rx / rlen, ry / rlen, rz / rlen
        ux = (ry * fz - rz * fy)
        uy = (rz * fx - rx * fz)
        uz = (rx * fy - ry * fx)

        fw = (1.0 if self.cam_key_forward else 0.0) - (1.0 if self.cam_key_back else 0.0)
        st = (1.0 if self.cam_key_right else 0.0) - (1.0 if self.cam_key_left else 0.0)
        vv = (1.0 if self.cam_key_up else 0.0) - (1.0 if self.cam_key_down else 0.0)
        if fw == 0.0 and st == 0.0 and vv == 0.0:
            return

        dx = (fx * fw + rx * st + ux * vv) * move
        dy_up = (fy * fw + ry * st + uy * vv) * move
        dz = (fz * fw + rz * st + uz * vv) * move
        self.camera_x += dx
        self.camera_y -= dy_up
        self.camera_z += dz

    def start_transform_mode(self, mode: str) -> None:
        if self.selected_object_name is None or self.selected_object_name not in self.scene_objects:
            return
        obj = self.scene_objects[self.selected_object_name]
        self.transform_mode = mode
        self.transform_axis = None
        self.transform_initial = (obj.x, obj.y, obj.rot_deg, obj.size)
        self.rotate_key_undo_armed = False
        self._clear_rotate_key_state()
        self._log(f"transform {mode}: {obj.name} (axis: free)")

    def set_transform_axis(self, axis: str) -> None:
        if self.transform_mode is None:
            return
        self.transform_axis = axis.lower()
        self._log(f"transform axis={self.transform_axis}")

    def cancel_transform(self) -> None:
        if self.transform_mode is None or self.selected_object_name is None or self.transform_initial is None:
            return
        obj = self.scene_objects.get(self.selected_object_name)
        if obj is None:
            return
        x, y, r, s = self.transform_initial
        obj.x, obj.y, obj.rot_deg, obj.size = x, y, r, s
        self.transform_mode = None
        self.transform_axis = None
        self.transform_anchor_world = None
        self.transform_initial = None
        self.rotate_key_undo_armed = False
        self._clear_rotate_key_state()
        self._log("transform canceled")

    def confirm_transform(self) -> None:
        if self.transform_mode is None:
            return
        self.transform_mode = None
        self.transform_axis = None
        self.transform_anchor_world = None
        self.transform_initial = None
        self.rotate_key_undo_armed = False
        self._clear_rotate_key_state()
        self._log("transform confirmed")

    def duplicate_selected_object(self) -> None:
        if self.selected_object_name is None or self.selected_object_name not in self.scene_objects:
            return
        self._push_undo()
        src = self.scene_objects[self.selected_object_name]
        base = f"{src.name}_copy"
        candidate = base
        k = 1
        while candidate in self.scene_objects:
            k += 1
            candidate = f"{base}{k}"
        dup = copy.deepcopy(src)
        dup.name = candidate
        dup.x += 28.0
        dup.y += 18.0
        self.scene_objects[candidate] = dup
        self.selected_object_name = candidate
        self._log(f"duplicated object: {src.name} -> {candidate}")

    def find_scene_object(self, sx: float, sy: float) -> str | None:
        names = list(self.scene_objects.keys())
        for name in reversed(names):
            obj = self.scene_objects[name]
            if not obj.visible:
                continue
            ox, oy, _rot = self._eval_object_transform(obj, self.timeline_frame)
            p3 = self.world3_to_screen(ox, oy, obj.z)
            if p3 is None:
                continue
            sox, soy, perspective, _cam_depth = p3
            half = max(4.0, obj.size * 0.5 * perspective)
            if math.hypot(sx - sox, sy - soy) <= half * 1.2:
                return name
        return None

    # -----------------------------
    # Mode-A scene objects + timeline
    # -----------------------------
    def _load_default_scene_objects(self) -> None:
        path = os.path.join(self.object_assets_dir, "scene_defaults.json")
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            self._log(f"warning: failed to load scene defaults: {exc}")
            return

        if not isinstance(data, dict):
            return
        names = data.get("objects", [])
        if not isinstance(names, list):
            return
        for item in names:
            if not isinstance(item, str):
                continue
            obj = self._load_object_def(item)
            if obj is None:
                self._log(f"warning: scene object not found: {item}")
                continue
            self._spawn_scene_object_from_def(obj)

    def _spawn_scene_object_from_def(self, data: dict) -> None:
        if str(data.get("type", "")).lower() != "scene_cube":
            return
        name = str(data.get("name", "")).strip()
        if not name:
            return

        x = self.world_w * float(data.get("x_ratio", 0.5))
        y = self.world_h * float(data.get("y_ratio", 0.5))
        z = float(data.get("z", 0.0))
        size = min(self.world_w, self.world_h) * float(data.get("size_ratio", 0.1))
        rot = float(data.get("rot_deg", 0.0))
        color = str(data.get("color", "#9aa4c9"))
        kind = str(data.get("kind", "cube")).lower()
        roughness = float(data.get("roughness", 0.55))
        metallic = float(data.get("metallic", 0.0))
        emission = float(data.get("emission", 0.0))
        collection = str(data.get("collection", "Scene"))

        if name in self.scene_objects:
            obj = self.scene_objects[name]
            obj.x = x
            obj.y = y
            obj.z = z
            obj.size = size
            obj.kind = kind
            obj.rot_deg = rot
            obj.color = color
            obj.roughness = roughness
            obj.metallic = metallic
            obj.emission = emission
            obj.collection = collection
            obj.texture_path = ""
            obj.texture_image = None
            obj.texture_pil = None
            obj.keyframes = []
            return

        obj = SceneCube(name=name, x=x, y=y, z=z, size=size, kind=kind)
        obj.rot_deg = rot
        obj.color = color
        obj.roughness = roughness
        obj.metallic = metallic
        obj.emission = emission
        obj.collection = collection
        obj.texture_path = ""
        obj.texture_image = None
        obj.texture_pil = None
        obj.keyframes = []
        self.scene_objects[name] = obj

    def _add_scene_object(self, kind: str, name: str, x: float, y: float, z: float, size: float) -> None:
        if name in self.scene_objects:
            raise ValueError(f"object already exists: {name}")
        self._push_undo()
        self.scene_objects[name] = SceneCube(name=name, x=x, y=y, z=z, size=size, kind=kind)
        self.selected_object_name = name
        self.selected_index = None
        self._log(f"object added: {name} kind={kind} at ({x:.1f}, {y:.1f}, {z:.1f}) size={size:.1f}")

    def add_cube_object(self, name: str, x: float, y: float, z: float, size: float) -> None:
        self._add_scene_object("cube", name, x, y, z, size)

    def add_plane_object(self, name: str, x: float, y: float, z: float, size: float) -> None:
        self._add_scene_object("plane", name, x, y, z, size)

    def add_sphere_object(self, name: str, x: float, y: float, z: float, size: float) -> None:
        self._add_scene_object("sphere", name, x, y, z, size)

    def add_light_object(self, name: str, x: float, y: float, z: float, size: float) -> None:
        self._add_scene_object("light", name, x, y, z, size)
        self.scene_objects[name].emission = 1.0
        self.scene_objects[name].color = "#ffe1a3"

    def add_camera_object(self, name: str, x: float, y: float, z: float, size: float) -> None:
        self._add_scene_object("camera", name, x, y, z, size)
        self.scene_objects[name].color = "#9ad0ff"

    def delete_object(self, name: str) -> None:
        if name not in self.scene_objects:
            raise ValueError(f"object not found: {name}")
        self._push_undo()
        obj = self.scene_objects.pop(name)
        obj.texture_image = None
        if self.selected_object_name == name:
            self.selected_object_name = None
        self._log(f"object deleted: {name}")

    def set_object_keyframe(self, name: str, frame: int, x: float, y: float, rot_deg: float) -> None:
        if name not in self.scene_objects:
            raise ValueError(f"object not found: {name}")
        self._push_undo()
        obj = self.scene_objects[name]
        obj.keyframes = [k for k in obj.keyframes if k.frame != frame]
        obj.keyframes.append(ObjectKeyframe(frame=frame, x=x, y=y, rot_deg=rot_deg))
        obj.keyframes.sort(key=lambda k: k.frame)
        self._log(f"keyframe set: {name} frame={frame} pos=({x:.1f},{y:.1f}) rot={rot_deg:.1f}")

    def assign_texture_to_object(self, name: str, path: str) -> None:
        if name not in self.scene_objects:
            raise ValueError(f"object not found: {name}")
        if not os.path.exists(path):
            raise ValueError(f"texture file not found: {path}")
        self._push_undo()
        img = tk.PhotoImage(file=path)
        pil_img = Image.open(path).convert("RGBA")
        obj = self.scene_objects[name]
        obj.texture_path = path
        obj.texture_image = img
        obj.texture_pil = pil_img
        self._log(f"texture assigned: {name} <- {path}")

    def _eval_object_transform(self, obj: SceneCube, frame: int) -> tuple[float, float, float]:
        if not obj.keyframes:
            return obj.x, obj.y, obj.rot_deg

        keys = obj.keyframes
        if frame <= keys[0].frame:
            return keys[0].x, keys[0].y, keys[0].rot_deg
        if frame >= keys[-1].frame:
            return keys[-1].x, keys[-1].y, keys[-1].rot_deg

        left = keys[0]
        right = keys[-1]
        seg_idx = 0
        for i in range(len(keys) - 1):
            if keys[i].frame <= frame <= keys[i + 1].frame:
                left = keys[i]
                right = keys[i + 1]
                seg_idx = i
                break

        span = max(1, right.frame - left.frame)
        t = (frame - left.frame) / span

        # (#13) Cubic Hermite (Catmull-Rom) interpolation for smooth easing.
        # Use neighboring keyframes to compute tangent vectors.
        n_keys = len(keys)
        if n_keys >= 3:
            # Tangent at left
            if seg_idx > 0:
                prev = keys[seg_idx - 1]
                m0x = 0.5 * (right.x - prev.x)
                m0y = 0.5 * (right.y - prev.y)
                m0r = 0.5 * (right.rot_deg - prev.rot_deg)
            else:
                m0x = right.x - left.x
                m0y = right.y - left.y
                m0r = right.rot_deg - left.rot_deg
            # Tangent at right
            if seg_idx + 2 < n_keys:
                nxt = keys[seg_idx + 2]
                m1x = 0.5 * (nxt.x - left.x)
                m1y = 0.5 * (nxt.y - left.y)
                m1r = 0.5 * (nxt.rot_deg - left.rot_deg)
            else:
                m1x = right.x - left.x
                m1y = right.y - left.y
                m1r = right.rot_deg - left.rot_deg

            t2 = t * t
            t3 = t2 * t
            h00 = 2.0 * t3 - 3.0 * t2 + 1.0
            h10 = t3 - 2.0 * t2 + t
            h01 = -2.0 * t3 + 3.0 * t2
            h11 = t3 - t2

            x = h00 * left.x + h10 * m0x + h01 * right.x + h11 * m1x
            y = h00 * left.y + h10 * m0y + h01 * right.y + h11 * m1y
            r = h00 * left.rot_deg + h10 * m0r + h01 * right.rot_deg + h11 * m1r
        else:
            # Fallback for 2 keyframes: use smoothstep
            t = t * t * (3.0 - 2.0 * t)
            x = left.x + (right.x - left.x) * t
            y = left.y + (right.y - left.y) * t
            r = left.rot_deg + (right.rot_deg - left.rot_deg) * t

        return x, y, r

    def _advance_timeline(self) -> None:
        if not self.timeline_playing:
            return
        self.timeline_accum += 1.0 / 60.0
        step = 1.0 / max(1.0, self.timeline_fps)
        while self.timeline_accum >= step:
            self.timeline_accum -= step
            self.timeline_frame += 1
            if self.timeline_frame > self.timeline_length:
                self.timeline_frame = 0

    # -----------------------------
    # Physics
    # -----------------------------
    def step_once(self) -> None:
        self._update_physics()
        self._draw()

    def toggle_running(self) -> None:
        self.running = not self.running
        self.run_btn.configure(text="Pause" if self.running else "Run")
        self._log("running" if self.running else "paused")

    def _update_physics(self) -> None:
        if self.sim_mode_var.get().upper() == "C":
            if self.quantum_world is None:
                self.quantum_world = QuantumWorld()
            self.quantum_world.step_multi()
            return
        self.world.k_coulomb = float(self.k_var.get())
        self.world.repulsion_k = float(self.rep_var.get())
        self.world.drag = float(self.drag_var.get())
        self.world.dt = float(self.dt_var.get())
        self.world.step(self.sim_mode_var.get())

    # -----------------------------
    # Rendering
    # -----------------------------
    def _draw(self) -> None:
        mode = self.sim_mode_var.get().upper()
        if mode == "A":
            self._draw_mode_a()
        elif mode == "C":
            self._draw_mode_c()
        else:
            self._draw_mode_b()

    def _draw_mode_b(self) -> None:
        self.canvas.delete("all")

        # background grid
        grid_step = 40
        for x in range(0, self.world_w, grid_step):
            sx0, sy0 = self.world_to_screen(x, 0)
            sx1, sy1 = self.world_to_screen(x, self.world_h)
            self.canvas.create_line(sx0, sy0, sx1, sy1, fill="#101827")
        for y in range(0, self.world_h, grid_step):
            sx0, sy0 = self.world_to_screen(0, y)
            sx1, sy1 = self.world_to_screen(self.world_w, y)
            self.canvas.create_line(sx0, sy0, sx1, sy1, fill="#101827")

        # bonds
        for b in self.world.bonds:
            if b.i < 0 or b.j < 0 or b.i >= len(self.world.particles) or b.j >= len(self.world.particles):
                continue
            p1 = self.world.particles[b.i]
            p2 = self.world.particles[b.j]
            x1, y1 = self.world_to_screen(p1.x, p1.y)
            x2, y2 = self.world_to_screen(p2.x, p2.y)
            self.canvas.create_line(x1, y1, x2, y2, fill="#4f5f7a")

        for i, p in enumerate(self.world.particles):
            sx, sy = self.world_to_screen(p.x, p.y)
            r = max(1.0, p.radius * self.render_radius_scale * self.view_zoom)
            self.canvas.create_oval(sx - r, sy - r, sx + r, sy + r, fill=p.material.color, outline="")

            if i == self.selected_index:
                rr = r + 4
                self.canvas.create_oval(sx - rr, sy - rr, sx + rr, sy + rr, outline="#ffffff")

        # velocity drag arrow
        if self.drag_start and self.drag_current:
            sx, sy = self.drag_start
            cx, cy = self.drag_current
            dsx, dsy = self.world_to_screen(sx, sy)
            dcx, dcy = self.world_to_screen(cx, cy)
            self.canvas.create_line(dsx, dsy, dcx, dcy, fill="#ffffff", width=2, arrow=tk.LAST)

        self.info_var.set(
            f"Mode: B | Particles: {len(self.world.particles)} Bonds: {len(self.world.bonds)} | Running: {self.running} | "
            f"Timeline: {self.timeline_frame}/{self.timeline_length} @ {self.timeline_fps:.1f}fps play={self.timeline_playing} | "
            f"Selected: {self.selected_index if self.selected_index is not None else 'None'}"
        )
        self._draw_axis_overlay()
        self._refresh_outliner()

    def _draw_mode_c(self) -> None:
        """Render Mode C: quantum wavefunction probability density."""
        self.canvas.delete("all")

        if self.quantum_world is None:
            self.quantum_world = QuantumWorld()

        qw = self.quantum_world

        # Black background
        self.canvas.create_rectangle(0, 0, self.world_w, self.world_h, fill="#020408", outline="")

        # Get 2D density (slice or projection)
        density = qw.get_density_2d()

        # Normalize for colormap
        max_val = float(density.max())
        if max_val > 1e-30:
            normalized = density / max_val
        else:
            normalized = density

        # Apply fire colormap -> RGB uint8 array
        rgb = self._quantum_fire_colormap(normalized)

        # Create PIL image and scale to fit canvas
        img = Image.fromarray(rgb, mode="RGB")
        display_size = min(self.world_w, self.world_h) - 80
        img = img.resize((display_size, display_size), Image.Resampling.BICUBIC)

        # Convert to Tk PhotoImage and display centered
        if IMAGETK_AVAILABLE:
            self._quantum_photo = ImageTk.PhotoImage(img)
        else:
            # Fallback: save to temp PPM for Tk
            import io
            buf = io.BytesIO()
            img.save(buf, format="PPM")
            buf.seek(0)
            self._quantum_photo = tk.PhotoImage(data=buf.read())

        cx = self.world_w // 2
        cy = self.world_h // 2
        self.canvas.create_image(cx, cy, image=self._quantum_photo)

        # Determine axis labels based on slice/projection axis
        half = display_size // 2
        ax = qw.slice_axis
        if ax == "x":
            xlabel, ylabel = "y", "z"
        elif ax == "y":
            xlabel, ylabel = "x", "z"
        else:
            xlabel, ylabel = "x", "y"

        ext = qw.extent
        fontsize_label = max(9, min(14, self.world_h // 160))
        fontsize_info = max(9, min(13, self.world_h // 170))

        # Axis labels
        self.canvas.create_text(
            cx + half + 30, cy,
            text=f"{xlabel} \u2192\n{ext:.0f} a\u2080",
            fill="#ffffff", font=("TkDefaultFont", fontsize_label),
        )
        self.canvas.create_text(
            cx, cy - half - 20,
            text=f"\u2191 {ylabel} ({ext:.0f} a\u2080)",
            fill="#ffffff", font=("TkDefaultFont", fontsize_label),
        )

        # Scale bar
        self.canvas.create_line(
            cx - half, cy + half + 15, cx + half, cy + half + 15,
            fill="#ffffff", width=2,
        )
        self.canvas.create_text(
            cx, cy + half + 32,
            text=f"{2 * ext:.0f} a\u2080  ({2 * ext * 0.529:.1f} \u00c5)",
            fill="#aabbcc", font=("TkDefaultFont", fontsize_label - 1),
        )

        # Quantum info overlay
        info = qw.get_info_text()
        self.canvas.create_text(
            20, 20, text=info, fill="#00e5ff",
            font=("TkDefaultFont", fontsize_info), anchor="nw",
        )

        view_info = f"View: {qw.view_mode} along {qw.slice_axis}-axis | E_field=({qw.electric_field[0]:.3f}, {qw.electric_field[1]:.3f}, {qw.electric_field[2]:.3f})"
        self.canvas.create_text(
            20, 20 + fontsize_info + 12, text=view_info, fill="#8fa8ff",
            font=("TkDefaultFont", fontsize_info - 1), anchor="nw",
        )

        # Exact vs computed energy comparison
        exact_E = -float(qw.nuclear_Z) ** 2 / (2.0 * qw.current_n ** 2)
        energy_err = abs(qw.last_energy - exact_E)
        energy_text = (
            f"Energy check: computed={qw.last_energy:.6f} Ha | exact={exact_E:.6f} Ha | "
            f"\u0394E={energy_err:.6f} Ha ({energy_err * 27.211:.4f} eV)"
        )
        self.canvas.create_text(
            20, 20 + 2 * (fontsize_info + 8), text=energy_text, fill="#63d26e",
            font=("TkDefaultFont", fontsize_info - 1), anchor="nw",
        )

        self.info_var.set(info)
        self._refresh_outliner()

    @staticmethod
    def _quantum_fire_colormap(values: np.ndarray) -> np.ndarray:
        """Map values in [0,1] to RGB uint8 using fire/inferno colormap."""
        colors = np.array(
            [
                [0, 0, 4],
                [40, 0, 100],
                [160, 0, 0],
                [255, 100, 0],
                [255, 220, 0],
                [255, 255, 255],
            ],
            dtype=np.float64,
        )
        v = np.clip(values, 0.0, 1.0)
        n_colors = len(colors)
        scaled = v * (n_colors - 1)
        i = np.floor(scaled).astype(int)
        i = np.clip(i, 0, n_colors - 2)
        t = (scaled - i)[..., np.newaxis]
        c0 = colors[i]
        c1 = colors[np.minimum(i + 1, n_colors - 1)]
        rgb = np.clip(c0 + t * (c1 - c0), 0, 255).astype(np.uint8)
        return rgb

    def _draw_mode_a(self) -> None:
        # (#11) Unified PIL renderer — renders to Image then displays as single PhotoImage.
        # (#3) Frustum culling with 40° cushion on all sides.
        # (#8) Unified depth sort — particles and objects interleaved by camera depth.
        frame = self._render_mode_a_frame_pil(self.timeline_frame)
        if IMAGETK_AVAILABLE:
            self._mode_a_photo = ImageTk.PhotoImage(frame)
        else:
            import io
            buf = io.BytesIO()
            frame.save(buf, format="PPM")
            buf.seek(0)
            self._mode_a_photo = tk.PhotoImage(data=buf.read())

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self._mode_a_photo, anchor="nw")

        # Lightweight canvas overlays (axis gizmo, selection rings, drag arrow) stay on canvas
        # because they need sharp vector lines and per-frame interactivity.
        self._draw_axis_overlay()

        # Selection outlines — objects
        if self.selected_object_name and self.selected_object_name in self.scene_objects:
            obj = self.scene_objects[self.selected_object_name]
            ox, oy, _rot = self._eval_object_transform(obj, self.timeline_frame)
            p3 = self.world3_to_screen(ox, oy, obj.z)
            if p3 is not None:
                sx, sy, perspective, _cd = p3
                half = max(2.0, obj.size * 0.5 * perspective)
                rr = half + 6
                self.canvas.create_oval(sx - rr, sy - rr, sx + rr, sy + rr, outline="#ffffff")

        # Selection outlines — particles
        if self.selected_index is not None and 0 <= self.selected_index < len(self.world.particles):
            p = self.world.particles[self.selected_index]
            proj = self.world3_to_screen(p.x, p.y, p.z)
            if proj is not None:
                sx, sy, perspective, _d = proj
                r = max(1.0, p.radius * self.render_radius_scale * 0.85 * perspective)
                rr = r + 5
                self.canvas.create_oval(sx - rr, sy - rr, sx + rr, sy + rr, outline="#ffffff")

        # Drag arrow
        if self.drag_start and self.drag_current:
            sx, sy = self.drag_start
            cx, cy = self.drag_current
            dsx, dsy = self.world_to_screen(sx, sy)
            dcx, dcy = self.world_to_screen(cx, cy)
            self.canvas.create_line(dsx, dsy, dcx, dcy, fill="#ffffff", width=2, arrow=tk.LAST)

        edit_state = f"T:{self.transform_mode or '-'} A:{self.transform_axis or '-'} snap={self.snap_enabled} g={self.snap_grid:.1f}"
        cam_state = (
            f"orbit d={self.orbit_distance:.0f} yaw={self.orbit_yaw:.1f} pitch={self.orbit_pitch:.1f} "
            f"pivot=({self.orbit_pivot[0]:.0f},{self.orbit_pivot[1]:.0f},{self.orbit_pivot[2]:.0f}) "
            f"fov={self.camera_fov_deg:.1f}"
        )
        self.info_var.set(
            f"Mode: A | Particles: {len(self.world.particles)} Bonds: {len(self.world.bonds)} | Running: {self.running} | "
            f"Timeline: {self.timeline_frame}/{self.timeline_length} @ {self.timeline_fps:.1f}fps play={self.timeline_playing} | "
            f"Selected Particle: {self.selected_index if self.selected_index is not None else 'None'} | "
            f"Selected Object: {self.selected_object_name or 'None'} | {edit_state} | {cam_state}"
        )
        self._refresh_outliner()

    def _is_in_frustum(self, sx: float, sy: float, cushion: float = 0.0) -> bool:
        """(#3) Return True if screen-space point is within viewport + cushion pixels."""
        return -cushion <= sx <= self.world_w + cushion and -cushion <= sy <= self.world_h + cushion

    def _get_frustum_cushion_px(self) -> float:
        """(#3) Convert 40° angular cushion to approximate pixel margin at screen center."""
        fov_rad = math.radians(max(10.0, min(160.0, self.camera_fov_deg)))
        cushion_rad = math.radians(40.0)
        focal = (self.world_h * 0.5) / math.tan(fov_rad * 0.5)
        return focal * math.tan(cushion_rad)

    def _render_mode_a_frame_pil(self, frame: int) -> Image.Image:
        """Render a full Mode A frame to a PIL Image (used for both live display and export)."""
        # Background (cached)
        bg_key = self.background_preset_var.get().lower()
        sun_key = self.sun_object_enabled
        if self._cached_bg_pil is None or self._cached_bg_preset != bg_key or self._cached_bg_sun_enabled != sun_key:
            self._cached_bg_pil = self._render_background_pil()
            self._cached_bg_preset = bg_key
            self._cached_bg_sun_enabled = sun_key

        img = self._cached_bg_pil.copy()
        draw = ImageDraw.Draw(img, "RGBA")
        cushion = self._get_frustum_cushion_px()

        # --- Collect all drawable items into a single depth-sorted list (#8) ---
        # Items are (cam_depth, kind_tag, payload)
        # kind_tag: "particle", "bond", "object"
        draw_list: list[tuple[float, str, object]] = []

        n = len(self.world.particles)
        stride_p = max(1, n // 3500)
        for i in range(0, n, stride_p):
            p = self.world.particles[i]
            proj = self.world3_to_screen(p.x, p.y, p.z)
            if proj is None:
                continue
            sx, sy, perspective, depth = proj
            # (#3) Frustum culling with 40° cushion
            if not self._is_in_frustum(sx, sy, cushion):
                continue
            r = max(1, int(round(p.radius * self.render_radius_scale * 0.85 * perspective)))
            draw_list.append((depth, "particle", (i, p, sx, sy, r, perspective)))

        # Bonds
        bond_stride = max(1, len(self.world.bonds) // 3000)
        for bi in range(0, len(self.world.bonds), bond_stride):
            b = self.world.bonds[bi]
            if b.i < 0 or b.j < 0 or b.i >= n or b.j >= n:
                continue
            p1 = self.world.particles[b.i]
            p2 = self.world.particles[b.j]
            pp1 = self.world3_to_screen(p1.x, p1.y, p1.z)
            pp2 = self.world3_to_screen(p2.x, p2.y, p2.z)
            if pp1 is None or pp2 is None:
                continue
            x1, y1, _s1, d1 = pp1
            x2, y2, _s2, d2 = pp2
            avg_depth = (d1 + d2) * 0.5
            # Frustum cull: skip if BOTH endpoints are off-screen beyond cushion
            if not self._is_in_frustum(x1, y1, cushion) and not self._is_in_frustum(x2, y2, cushion):
                continue
            draw_list.append((avg_depth, "bond", (x1, y1, x2, y2)))

        # Scene objects
        for name, obj in self.scene_objects.items():
            if not obj.visible:
                continue
            ox, oy, rot = self._eval_object_transform(obj, frame)
            p3 = self.world3_to_screen(ox, oy, obj.z)
            if p3 is None:
                continue
            sx, sy, perspective, cam_depth = p3
            if not self._is_in_frustum(sx, sy, cushion):
                continue
            draw_list.append((cam_depth, "object", (name, obj, ox, oy, rot, sx, sy, perspective)))

        # Sort far to near (painter's algorithm)
        draw_list.sort(key=lambda it: it[0], reverse=True)

        # --- Draw all items in depth order ---
        for _depth, kind, payload in draw_list:
            if kind == "bond":
                x1, y1, x2, y2 = payload
                draw.line((x1, y1, x2, y2), fill=(95, 122, 160, 255), width=1)

            elif kind == "particle":
                _i, p, sx, sy, r, _persp = payload
                col = self._hex_to_rgb(p.material.color)
                # Glow ring
                draw.ellipse((sx - r - 2, sy - r - 2, sx + r + 2, sy + r + 2), outline=(122, 166, 255, 120))
                # Main body
                draw.ellipse((sx - r, sy - r, sx + r, sy + r), fill=(col[0], col[1], col[2], 255))
                # Velocity streak
                speed = math.sqrt(p.vx * p.vx + p.vy * p.vy)
                trail = min(16.0, speed * 0.06)
                if speed > 1e-4 and trail > 0.5:
                    tx = p.x - (p.vx / speed) * trail
                    ty = p.y - (p.vy / speed) * trail
                    tproj = self.world3_to_screen(tx, ty, p.z)
                    if tproj is not None:
                        tsx, tsy, _tp, _td = tproj
                        draw.line((sx, sy, int(tsx), int(tsy)), fill=(223, 232, 255, 200), width=1)

            elif kind == "object":
                name, obj, ox, oy, rot, sx, sy, perspective = payload
                self._draw_export_object(draw, obj, ox, oy, rot, img)
                half = max(2.0, obj.size * 0.5 * perspective)
                draw.text(
                    (int(sx - half), int(sy - half - 12)),
                    f"{name} ({obj.kind}) z={obj.z:.1f}",
                    fill=(210, 220, 230, 255),
                )

        return img

    def _render_background_pil(self) -> Image.Image:
        """Render the background (sky + optional sun) to a PIL Image. Cached."""
        img = Image.new("RGB", (self.world_w, self.world_h), color=(11, 15, 20))
        draw = ImageDraw.Draw(img, "RGBA")
        bg = self.background_preset_var.get().lower()
        if bg != "none":
            sky = self._load_object_def(bg)
            if sky is not None:
                self._draw_export_sky_object(draw, sky)
        if self.sun_object_enabled:
            sun = self._load_object_def(self.sun_object_name)
            if sun is not None:
                self._draw_export_sun_object(draw, sun)
        return img

    @staticmethod
    def _shade_rgb(col: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
        return (
            max(0, min(255, int(col[0] * factor))),
            max(0, min(255, int(col[1] * factor))),
            max(0, min(255, int(col[2] * factor))),
        )

    def _box_dims_for_object(self, obj: SceneCube) -> tuple[float, float, float]:
        kind = obj.kind.lower()
        hx = max(4.0, obj.size * 0.5)
        if kind == "plane":
            return hx, max(2.0, hx * 0.55), max(1.0, hx * 0.08)
        if kind == "light":
            return max(3.0, hx * 0.35), max(3.0, hx * 0.35), max(3.0, hx * 0.35)
        if kind == "camera":
            return max(3.0, hx * 0.8), max(3.0, hx * 0.5), max(3.0, hx * 0.4)
        if kind == "sphere":
            return hx, hx, hx
        return hx, hx, max(2.0, hx * 0.75)

    def _project_object_box(
        self,
        obj: SceneCube,
        ox: float,
        oy: float,
        rot_deg: float,
    ) -> tuple[list[tuple[float, float, float]], list[tuple[tuple[int, ...], float]]] | None:
        hx, hy, hz = self._box_dims_for_object(obj)
        corners_local = [
            (-hx, -hy, -hz),
            (hx, -hy, -hz),
            (hx, hy, -hz),
            (-hx, hy, -hz),
            (-hx, -hy, hz),
            (hx, -hy, hz),
            (hx, hy, hz),
            (-hx, hy, hz),
        ]

        rr = math.radians(rot_deg)
        c = math.cos(rr)
        s = math.sin(rr)
        projected: list[tuple[float, float, float]] = []
        for lx, ly, lz in corners_local:
            wx = ox + (lx * c - ly * s)
            wy = oy + (lx * s + ly * c)
            wz = obj.z + lz
            p = self.world3_to_screen(wx, wy, wz)
            if p is None:
                return None
            sx, sy, _scale, cam_depth = p
            projected.append((sx, sy, cam_depth))

        # Faces — face normals computed per-frame for dot-product shading (#9).
        face_defs: list[tuple[tuple[int, ...], tuple[float, float, float]]] = [
            ((4, 5, 6, 7), (0.0, 0.0, 1.0)),   # +z top
            ((0, 1, 2, 3), (0.0, 0.0, -1.0)),  # -z bottom
            ((0, 3, 7, 4), (-1.0, 0.0, 0.0)),  # -x left
            ((1, 2, 6, 5), (1.0, 0.0, 0.0)),   # +x right
            ((3, 2, 6, 7), (0.0, 1.0, 0.0)),   # +y front
            ((0, 1, 5, 4), (0.0, -1.0, 0.0)),  # -y back
        ]

        # Rotate face normals by object rotation and compute shade via dot product with light direction.
        rr = math.radians(rot_deg)
        cr = math.cos(rr)
        sr = math.sin(rr)
        # Default light direction (towards upper-right-front, normalized).
        lx, ly, lz = 0.4, -0.6, 0.7
        ln = math.sqrt(lx * lx + ly * ly + lz * lz)
        lx, ly, lz = lx / ln, ly / ln, lz / ln

        faces: list[tuple[tuple[int, ...], float]] = []
        for idxs, (fnx, fny, fnz) in face_defs:
            # Rotate normal by object yaw (around z).
            rnx = fnx * cr - fny * sr
            rny = fnx * sr + fny * cr
            rnz = fnz
            dot = max(0.0, rnx * lx + rny * ly + rnz * lz)
            shade = 0.25 + 0.75 * dot  # ambient 0.25, diffuse 0.75
            faces.append((idxs, shade))

        faces.sort(key=lambda it: sum(projected[i][2] for i in it[0]) / len(it[0]), reverse=True)
        return projected, faces

    def _draw_mode_a_object_canvas(self, obj: SceneCube, ox: float, oy: float, rot_deg: float, sx: float, sy: float, half: float) -> None:
        kind = obj.kind.lower()
        col = self._hex_to_rgb(obj.color)

        mesh = self._project_object_box(obj, ox, oy, rot_deg)
        if mesh is None:
            return
        projected, faces = mesh
        for idxs, shade in faces:
            pts: list[float] = []
            for i in idxs:
                pts.extend([projected[i][0], projected[i][1]])
            shaded = self._shade_rgb(col, shade)
            self.canvas.create_polygon(*pts, fill=self._rgb_to_hex(shaded), outline="#d7e2ff")

        # Additional type cues.
        if kind == "light":
            r2 = half * 1.2
            self.canvas.create_line(sx - r2, sy, sx + r2, sy, fill="#ffe7ad")
            self.canvas.create_line(sx, sy - r2, sx, sy + r2, fill="#ffe7ad")
        elif kind == "camera":
            self.canvas.create_line(sx, sy, sx + half * 1.4, sy, fill="#d8ebff")

    def _draw_export_object(
        self,
        draw: ImageDraw.ImageDraw,
        obj: SceneCube,
        ox: float,
        oy: float,
        rot: float,
        img_pil: Image.Image,
    ) -> None:
        col = self._hex_to_rgb(obj.color)
        mesh = self._project_object_box(obj, ox, oy, rot)
        if mesh is None:
            return
        projected, faces = mesh
        for idxs, shade in faces:
            pts = [(int(round(projected[i][0])), int(round(projected[i][1]))) for i in idxs]
            shaded = self._shade_rgb(col, shade)
            draw.polygon(pts, fill=(shaded[0], shaded[1], shaded[2], 255), outline=(215, 226, 255, 255))

        if obj.texture_pil is not None and obj.kind.lower() == "cube":
            center = self.world3_to_screen(ox, oy, obj.z)
            if center is not None:
                csx, csy, scale, _depth = center
                tex_size = max(8, int(round(obj.size * max(0.05, scale))))
                tex = obj.texture_pil.resize((tex_size, tex_size), Image.Resampling.BICUBIC)
                tex = tex.rotate(-rot, expand=True, resample=Image.Resampling.BICUBIC)
                px = int(round(csx - tex.width * 0.5))
                py = int(round(csy - tex.height * 0.5))
                img_pil.paste(tex, (px, py), tex)

    def _camera_space_dir(self, wx: float, wy_up: float, wz: float) -> tuple[float, float, float]:
        cx, cy, cz = self._rot_z(wx, wy_up, wz, -self.camera_roll_deg)
        cx, cy, cz = self._rot_x(cx, cy, cz, -self.camera_pitch_deg)
        cx, cy, cz = self._rot_y(cx, cy, cz, -self.camera_yaw_deg)
        return cx, cy, cz

    def _draw_axis_overlay(self) -> None:
        # World-origin axes in scene space.
        axis_len_world = 280.0
        origin = self.world3_to_screen(0.0, 0.0, 0.0)
        if origin is not None:
            ox, oy, _s, _d = origin
            px = self.world3_to_screen(axis_len_world, 0.0, 0.0)
            py = self.world3_to_screen(0.0, axis_len_world, 0.0)
            pz = self.world3_to_screen(0.0, 0.0, axis_len_world)
            if px is not None:
                self.canvas.create_line(ox, oy, px[0], px[1], fill="#ff5c5c", width=3)
                self.canvas.create_text(px[0] + 10, px[1], text="X", fill="#ff8080", font=("TkDefaultFont", 10, "bold"))
            if py is not None:
                self.canvas.create_line(ox, oy, py[0], py[1], fill="#63d26e", width=3)
                self.canvas.create_text(py[0], py[1] - 10, text="Y", fill="#7fe48a", font=("TkDefaultFont", 10, "bold"))
            if pz is not None:
                self.canvas.create_line(ox, oy, pz[0], pz[1], fill="#5e87ff", width=3)
                self.canvas.create_text(pz[0] + 10, pz[1] - 10, text="Z", fill="#86a4ff", font=("TkDefaultFont", 10, "bold"))
            self.canvas.create_oval(ox - 4, oy - 4, ox + 4, oy + 4, fill="#ffffff", outline="")
            self.canvas.create_text(ox + 32, oy + 12, text="Origin (0,0,0)", fill="#e9eefc", anchor="w")

        # Corner orientation gizmo (always visible).
        gx = 90.0
        gy = self.world_h - 90.0
        gl = 48.0
        self.canvas.create_oval(gx - 3, gy - 3, gx + 3, gy + 3, fill="#ffffff", outline="")

        dirs = [
            ("X", (1.0, 0.0, 0.0), "#ff5c5c"),
            ("Y", (0.0, -1.0, 0.0), "#63d26e"),  # -Y because world Y is down
            ("Z", (0.0, 0.0, 1.0), "#5e87ff"),
        ]
        for label, vec, color in dirs:
            cx, cy, _cz = self._camera_space_dir(vec[0], vec[1], vec[2])
            norm = max(1e-6, math.hypot(cx, cy))
            ex = gx + (cx / norm) * gl
            ey = gy - (cy / norm) * gl
            self.canvas.create_line(gx, gy, ex, ey, fill=color, width=3)
            self.canvas.create_text(ex + 10, ey, text=label, fill=color, font=("TkDefaultFont", 10, "bold"))

    def _refresh_outliner(self) -> None:
        if not hasattr(self, "outliner_list"):
            return
        prev_sel = self.selected_index
        self.outliner_list.delete(0, tk.END)
        sorted_names = sorted(self.scene_objects.keys())[:150]
        for name in sorted_names:
            obj = self.scene_objects[name]
            ox, oy, rot = self._eval_object_transform(obj, self.timeline_frame)
            self.outliner_list.insert(tk.END, f"OBJ  | {name} [{obj.kind}] [{obj.collection}] | ({ox:.1f}, {oy:.1f}, {obj.z:.1f}) r={rot:.1f}")

        limit = min(150, len(self.world.particles))
        for i in range(limit):
            p = self.world.particles[i]
            self.outliner_list.insert(tk.END, f"PART | {i:04d} | {p.material.name} | ({p.x:.1f}, {p.y:.1f})")
        if self.selected_object_name in sorted_names:
            self.outliner_list.selection_set(sorted_names.index(self.selected_object_name))
            return
        if prev_sel is not None and 0 <= prev_sel < limit:
            self.outliner_list.selection_set(len(self.scene_objects) + prev_sel)

    def _on_outliner_select(self) -> None:
        if not hasattr(self, "outliner_list"):
            return
        sel = self.outliner_list.curselection()
        if not sel:
            return
        idx = int(sel[0])
        obj_count = min(150, len(self.scene_objects))
        if idx < obj_count:
            names = sorted(self.scene_objects.keys())[:150]
            if idx < len(names):
                self.selected_object_name = names[idx]
                self.selected_index = None
            return
        pidx = idx - obj_count
        if 0 <= pidx < len(self.world.particles):
            self.selected_object_name = None
            self.selected_index = pidx

    def _tick(self) -> None:
        self._advance_timeline()
        if self.running and self._check_emergency_state():
            self._update_physics()
        elif self.emergency_grace_ticks > 0:
            self.emergency_grace_ticks -= 1
        self._apply_rotate_key_hold(0.016)
        self._apply_camera_key_hold(0.016)
        self._draw()
        self.root.after(16, self._tick)

    def _check_emergency_state(self) -> bool:
        if not self.emergency_pause_enabled or self.emergency_suppress:
            return True
        if self.emergency_grace_ticks > 0:
            self.emergency_grace_ticks -= 1
            return True

        n = len(self.world.particles)
        est_pairs = n * (n - 1) // 2
        est_ops = est_pairs
        if self.sim_mode_var.get().upper() == "B" and self.world.mode_b_transport_exact:
            est_ops += n * max(1, self.world.photons_per_particle_per_step) * max(1, self.world.photon_max_bounces + 1) * max(1, n - 1)

        if est_ops < self.emergency_pair_threshold:
            return True

        self.running = False
        self.run_btn.configure(text="Run")
        self._log(
            f"EMERGENCY: estimated operations={est_ops:,} (threshold={self.emergency_pair_threshold:,}). Paused."
        )

        if messagebox is None:
            return False

        choice = messagebox.askyesnocancel(
            "Emergency performance warning",
            "Simulation load is extremely high and a crash/freeze is likely.\n\n"
            "Yes = Continue now (temporary)\n"
            "No = Continue and suppress future warnings\n"
            "Cancel = Stay paused",
            icon="warning",
        )
        if choice is True:
            self.running = True
            self.run_btn.configure(text="Pause")
            self.emergency_grace_ticks = 180
            self._log("Emergency override: continue temporarily.")
            return True
        if choice is False:
            self.running = True
            self.run_btn.configure(text="Pause")
            self.emergency_suppress = True
            self._log("Emergency warnings suppressed. Continuing at your risk.")
            return True

        self._log("Emergency pause maintained.")
        return False

    def _on_close(self) -> None:
        try:
            self.command_window.destroy()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass

    def _log(self, msg: str) -> None:
        if not hasattr(self, "output_text"):
            return
        self.output_text.configure(state=tk.NORMAL)
        self.output_text.insert(tk.END, msg + "\n")
        self.output_text.see(tk.END)
        self.output_text.configure(state=tk.DISABLED)

    def refresh_help_list(self) -> None:
        q = self.help_query_var.get().strip().lower()
        names = sorted(self.command_specs.keys())
        if q:
            names = [n for n in names if q in n.lower() or q in self.command_specs[n]["usage"].lower() or q in self.command_specs[n]["desc"].lower()]
        self.help_list.delete(0, tk.END)
        for n in names:
            self.help_list.insert(tk.END, n)
        if names:
            self.help_list.selection_clear(0, tk.END)
            self.help_list.selection_set(0)
            self.show_selected_help()
        else:
            self._set_help_text("No commands match your query.")

    def _set_help_text(self, text: str) -> None:
        self.help_text.configure(state=tk.NORMAL)
        self.help_text.delete("1.0", tk.END)
        self.help_text.insert("1.0", text)
        self.help_text.configure(state=tk.DISABLED)

    def execute_external_command(self, command: str) -> None:
        self.command_var.set(command)
        self.execute_command()

    def _normalize_object_def_3d(self, data: dict) -> dict:
        norm = dict(data)
        typ = str(norm.get("type", "")).lower()
        norm["dimensionality"] = "3d"

        if typ == "scene_cube":
            norm["z"] = float(norm.get("z", 0.0))
            norm["x_ratio"] = float(norm.get("x_ratio", 0.5))
            norm["y_ratio"] = float(norm.get("y_ratio", 0.5))
            norm["size_ratio"] = float(norm.get("size_ratio", 0.1))
            norm["kind"] = str(norm.get("kind", "cube")).lower()
        elif typ == "sun_disk":
            norm["z"] = float(norm.get("z", 4500.0))
            norm["x_ratio"] = float(norm.get("x_ratio", 0.82))
            norm["y_ratio"] = float(norm.get("y_ratio", 0.18))
        elif typ in {"sky_gradient", "deep_space"}:
            norm["z"] = float(norm.get("z", 12000.0))

        return norm

    def _load_object_def(self, name: str) -> dict | None:
        if not name:
            return None
        key = name.strip().lower()
        if key in self.object_defs:
            return self.object_defs[key]
        path = os.path.join(self.object_assets_dir, f"{key}.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            norm = self._normalize_object_def_3d(data)
            self.object_defs[key] = norm
            return norm
        return None

    def _draw_canvas_sky_object(self, sky: dict) -> None:
        kind = str(sky.get("type", "")).lower()
        if kind == "sky_gradient":
            top = tuple(sky.get("top_color", [109, 174, 255]))
            bot = tuple(sky.get("bottom_color", [173, 212, 255]))
            for y in range(self.world_h):
                t = y / max(1, self.world_h - 1)
                c = self._lerp_rgb(top, bot, t)
                self.canvas.create_line(0, y, self.world_w, y, fill=self._rgb_to_hex(c))
            return

        if kind == "deep_space":
            base = tuple(sky.get("base_color", [4, 6, 12]))
            self.canvas.create_rectangle(0, 0, self.world_w, self.world_h, fill=self._rgb_to_hex(base), outline="")
            stars = int(sky.get("stars", 900))
            star_color = self._rgb_to_hex(tuple(sky.get("star_color", [219, 231, 255])))
            for i in range(stars):
                x = (i * 73) % self.world_w
                y = (i * 131) % self.world_h
                self.canvas.create_rectangle(x, y, x + 1, y + 1, fill=star_color, outline="")
            return

        self.canvas.create_rectangle(0, 0, self.world_w, self.world_h, fill="#0b0f14", outline="")

    def _draw_canvas_sun_object(self, sun: dict) -> None:
        if str(sun.get("type", "")).lower() != "sun_disk":
            return
        sx = int(self.world_w * float(sun.get("x_ratio", 0.82)))
        sy = int(self.world_h * float(sun.get("y_ratio", 0.18)))
        sun_r = max(8, int(min(self.world_w, self.world_h) * float(sun.get("radius_ratio", 0.03))))

        glow_steps = max(1, int(sun.get("glow_steps_canvas", 14)))
        glow_extent = float(sun.get("glow_extent", 7.0))
        glow_inner = tuple(sun.get("glow_inner", [255, 255, 255]))
        glow_outer = tuple(sun.get("glow_outer", [255, 247, 210]))
        for i in range(glow_steps, 0, -1):
            t = i / glow_steps
            rr = int(sun_r + (sun_r * glow_extent * t))
            c = self._lerp_rgb(glow_outer, glow_inner, 1.0 - t)
            self.canvas.create_oval(sx - rr, sy - rr, sx + rr, sy + rr, fill=self._rgb_to_hex(c), outline="")

        disk_steps = max(1, int(sun.get("disk_steps_canvas", 8)))
        core_inner = tuple(sun.get("core_inner", [255, 251, 236]))
        core_outer = tuple(sun.get("core_outer", [255, 232, 166]))
        for i in range(disk_steps, 0, -1):
            t = i / disk_steps
            rr = int(max(1, sun_r * t))
            c = self._lerp_rgb(core_inner, core_outer, 1.0 - t)
            self.canvas.create_oval(sx - rr, sy - rr, sx + rr, sy + rr, fill=self._rgb_to_hex(c), outline="")

        halo_r = int(sun_r * float(sun.get("halo_ratio", 1.25)))
        halo_color = self._rgb_to_hex(tuple(sun.get("halo_color", [255, 243, 180])))
        self.canvas.create_oval(sx - halo_r, sy - halo_r, sx + halo_r, sy + halo_r, outline=halo_color, width=1)
        streak_color = self._rgb_to_hex(tuple(sun.get("streak_color", [255, 249, 208])))
        streak_scale = float(sun.get("streak_scale", 2.4))
        self.canvas.create_line(sx - int(sun_r * streak_scale), sy, sx + int(sun_r * streak_scale), sy, fill=streak_color)

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
        c = hex_color.lstrip("#")
        if len(c) != 6:
            return (255, 255, 255)
        r = int(c[0:2], 16)
        g = int(c[2:4], 16)
        b = int(c[4:6], 16)
        return (r, g, b)

    @staticmethod
    def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
        r = max(0, min(255, int(rgb[0])))
        g = max(0, min(255, int(rgb[1])))
        b = max(0, min(255, int(rgb[2])))
        return f"#{r:02x}{g:02x}{b:02x}"

    @staticmethod
    def _lerp_rgb(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
        tt = max(0.0, min(1.0, t))
        return (
            int(a[0] + (b[0] - a[0]) * tt),
            int(a[1] + (b[1] - a[1]) * tt),
            int(a[2] + (b[2] - a[2]) * tt),
        )

    def _render_mode_a_frame_np(self, frame: int) -> np.ndarray:
        # (#11) Unified: reuse the same PIL renderer used for live display.
        return np.asarray(self._render_mode_a_frame_pil(frame), dtype=np.uint8)

    def export_mp4_mode_a(self, out_path: str, start_frame: int, end_frame: int, fps: int) -> None:
        if end_frame < start_frame:
            raise ValueError("end_frame must be >= start_frame")
        fps = max(1, int(fps))

        saved_frame = self.timeline_frame
        total = end_frame - start_frame + 1
        with imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8) as writer:
            for idx, fr in enumerate(range(start_frame, end_frame + 1), start=1):
                self.timeline_frame = fr
                frame = self._render_mode_a_frame_np(fr)
                writer.append_data(frame)
                if idx % 30 == 0 or idx == total:
                    self._log(f"export progress: {idx}/{total}")
        self.timeline_frame = saved_frame

    def _draw_export_sky_object(self, draw: ImageDraw.ImageDraw, sky: dict) -> None:
        kind = str(sky.get("type", "")).lower()
        if kind == "sky_gradient":
            top = tuple(sky.get("top_color", [109, 174, 255]))
            bot = tuple(sky.get("bottom_color", [173, 212, 255]))
            for y in range(self.world_h):
                t = y / max(1, self.world_h - 1)
                c = self._lerp_rgb(top, bot, t)
                draw.line((0, y, self.world_w, y), fill=(c[0], c[1], c[2], 255), width=1)
            return

        if kind == "deep_space":
            base = tuple(sky.get("base_color", [4, 6, 12]))
            draw.rectangle((0, 0, self.world_w, self.world_h), fill=(base[0], base[1], base[2], 255))
            stars = int(sky.get("stars", 12000))
            sc = tuple(sky.get("star_color", [220, 240, 255]))
            for i in range(stars):
                x = (i * 73) % self.world_w
                y = (i * 131) % self.world_h
                draw.point((x, y), fill=(sc[0], sc[1], sc[2], 255))
            return

        draw.rectangle((0, 0, self.world_w, self.world_h), fill=(11, 15, 20, 255))

    def _draw_export_sun_object(self, draw: ImageDraw.ImageDraw, sun: dict) -> None:
        if str(sun.get("type", "")).lower() != "sun_disk":
            return
        sx = int(self.world_w * float(sun.get("x_ratio", 0.82)))
        sy = int(self.world_h * float(sun.get("y_ratio", 0.18)))
        sun_r = max(8, int(min(self.world_w, self.world_h) * float(sun.get("radius_ratio", 0.03))))

        glow_steps = max(1, int(sun.get("glow_steps_export", 22)))
        glow_extent = float(sun.get("glow_extent", 7.0))
        glow_inner = tuple(sun.get("glow_inner", [255, 255, 255]))
        glow_outer = tuple(sun.get("glow_outer", [255, 247, 210]))
        for i in range(glow_steps, 0, -1):
            t = i / glow_steps
            rr = int(sun_r + (sun_r * glow_extent * t))
            alpha = int(8 + (1.0 - t) * 72)
            c = self._lerp_rgb(glow_outer, glow_inner, 1.0 - t)
            draw.ellipse((sx - rr, sy - rr, sx + rr, sy + rr), fill=(c[0], c[1], c[2], alpha))

        disk_steps = max(1, int(sun.get("disk_steps_export", 16)))
        core_inner = tuple(sun.get("core_inner", [255, 251, 236]))
        core_outer = tuple(sun.get("core_outer", [255, 232, 166]))
        for i in range(disk_steps, 0, -1):
            t = i / disk_steps
            rr = int(max(1, sun_r * t))
            c = self._lerp_rgb(core_inner, core_outer, 1.0 - t)
            draw.ellipse((sx - rr, sy - rr, sx + rr, sy + rr), fill=(c[0], c[1], c[2], 255))

        halo_r = int(sun_r * float(sun.get("halo_ratio", 1.25)))
        hc = tuple(sun.get("halo_color", [255, 243, 180]))
        draw.ellipse((sx - halo_r, sy - halo_r, sx + halo_r, sy + halo_r), outline=(hc[0], hc[1], hc[2], 210), width=1)
        streak_scale = float(sun.get("streak_scale", 2.4))
        sc = tuple(sun.get("streak_color", [255, 249, 208]))
        draw.line((sx - int(sun_r * streak_scale), sy, sx + int(sun_r * streak_scale), sy), fill=(sc[0], sc[1], sc[2], 170), width=1)

    def export_test4k_png(self, out_path: str) -> None:
        frame = self._render_mode_a_frame_np(self.timeline_frame)
        iio.imwrite(out_path, frame)

    def show_selected_help(self) -> None:
        sel = self.help_list.curselection()
        if not sel:
            return
        cmd = self.help_list.get(sel[0])
        spec = self.command_specs[cmd]
        self._set_help_text(f"Command: {cmd}\nUsage: {spec['usage']}\n\n{spec['desc']}")

    def execute_command(self) -> None:
        raw = self.command_var.get().strip()
        if not raw:
            return

        self._log(f"> {raw}")
        parts = raw.split()
        op = parts[0].lower()

        try:
            if op == "help":
                q = " ".join(parts[1:]).strip().lower() if len(parts) > 1 else ""
                if not q:
                    cmds = ", ".join(sorted(self.command_specs.keys()))
                    self._log(f"commands: {cmds}")
                else:
                    matches = [c for c, s in self.command_specs.items() if q in c.lower() or q in s['usage'].lower() or q in s['desc'].lower()]
                    if matches:
                        for c in sorted(matches):
                            s = self.command_specs[c]
                            self._log(f"{c}: {s['usage']}")
                    else:
                        self._log("no matching command")

            elif op == "spawn":
                if len(parts) != 3:
                    raise ValueError("usage: spawn <material> <count>")
                material = parts[1]
                count = int(parts[2])
                if material not in MATERIALS:
                    raise ValueError(f"unknown material: {material}")
                self.material_var.set(material)
                self.spawn_count_var.set(count)
                self.spawn_random()

            elif op == "preset":
                if len(parts) < 2:
                    raise ValueError("usage: preset <Hydrogen|Helium|Carbon|Plasma Box|Aluminum Cube (50nm, scaled)>")
                name = " ".join(parts[1:])
                self.load_preset(name)
                self.preset_var.set(name)
                self._log(f"loaded preset: {name}")

            elif op == "scale":
                if len(parts) != 2:
                    raise ValueError("usage: scale <macro|micro|nano|atomic>")
                profile = parts[1].lower()
                if profile not in SCALE_PROFILES:
                    raise ValueError("scale profile must be one of: macro, micro, nano, atomic")
                self.scale_profile_var.set(profile)
                self.apply_scale_profile(profile)

            elif op == "mode":
                if len(parts) != 2:
                    raise ValueError("usage: mode <A|B|status>")
                m = parts[1].upper()
                if m == "STATUS":
                    self._log(f"mode={self.sim_mode_var.get().upper()}")
                elif m in {"A", "B"}:
                    self.apply_sim_mode(m)
                else:
                    raise ValueError("usage: mode <A|B|status>")

            elif op == "exact":
                if len(parts) < 2:
                    raise ValueError("usage: exact <status|transport <on|off>|photons <n>|bounces <n>|energy <v>|coupling <v>>")
                sub = parts[1].lower()
                if sub == "status":
                    self._log(
                        f"exact transport={self.world.mode_b_transport_exact} photons={self.world.photons_per_particle_per_step} "
                        f"bounces={self.world.photon_max_bounces} energy={self.world.photon_packet_energy:.3e} "
                        f"coupling={self.world.radiation_coupling:.3e} backend={self.world.compute_backend}"
                    )
                elif sub == "transport" and len(parts) == 3:
                    v = parts[2].lower()
                    if v not in {"on", "off"}:
                        raise ValueError("usage: exact transport <on|off>")
                    self.world.mode_b_transport_exact = (v == "on")
                    self._log(f"exact transport={self.world.mode_b_transport_exact}")
                elif sub == "photons" and len(parts) == 3:
                    self.world.photons_per_particle_per_step = max(1, int(parts[2]))
                    self._log(f"exact photons_per_particle_per_step={self.world.photons_per_particle_per_step}")
                elif sub == "bounces" and len(parts) == 3:
                    self.world.photon_max_bounces = max(0, int(parts[2]))
                    self._log(f"exact photon_max_bounces={self.world.photon_max_bounces}")
                elif sub == "energy" and len(parts) == 3:
                    self.world.photon_packet_energy = float(parts[2])
                    self._log(f"exact photon_packet_energy={self.world.photon_packet_energy:.3e}")
                elif sub == "coupling" and len(parts) == 3:
                    self.world.radiation_coupling = float(parts[2])
                    self._log(f"exact radiation_coupling={self.world.radiation_coupling:.3e}")
                else:
                    raise ValueError("usage: exact <status|transport <on|off>|photons <n>|bounces <n>|energy <v>|coupling <v>>")

            elif op == "view":
                if len(parts) < 2:
                    raise ValueError("usage: view <home|zoom <factor>|pan <dx> <dy>|depth <value>|status>")
                sub = parts[1].lower()
                if sub == "home":
                    self.reset_view()
                    self._log("view reset")
                elif sub == "zoom" and len(parts) == 3:
                    factor = float(parts[2])
                    self.view_zoom = max(0.1, min(8.0, self.view_zoom * factor))
                    self._log(f"view zoom={self.view_zoom:.3f}")
                elif sub == "pan" and len(parts) == 4:
                    dx = float(parts[2])
                    dy = float(parts[3])
                    self.view_pan_x += dx
                    self.view_pan_y += dy
                    self._log(f"view pan=({self.view_pan_x:.2f}, {self.view_pan_y:.2f})")
                elif sub == "depth" and len(parts) == 3:
                    self.view_depth = max(64.0, float(parts[2]))
                    self._log(f"view depth={self.view_depth:.2f}")
                elif sub == "status":
                    self._log(
                        f"view zoom={self.view_zoom:.3f} pan=({self.view_pan_x:.2f}, {self.view_pan_y:.2f}) depth={self.view_depth:.2f}"
                    )
                else:
                    raise ValueError("usage: view <home|zoom <factor>|pan <dx> <dy>|depth <value>|status>")

            elif op == "camera":
                if len(parts) < 2:
                    raise ValueError("usage: camera <status|pos <x> <y> <z>|rot <yaw> <pitch> <roll>|fov <deg>|speed <move|turn> <v>|reset>")
                sub = parts[1].lower()
                if sub == "status":
                    self._log(self.camera_status_text())
                elif sub == "reset":
                    self.reset_camera()
                    self._log("camera reset")
                elif sub == "pos" and len(parts) == 5:
                    self.camera_x = float(parts[2])
                    self.camera_y = float(parts[3])
                    self.camera_z = float(parts[4])
                    self._log(self.camera_status_text())
                elif sub == "rot" and len(parts) == 5:
                    self.camera_yaw_deg = float(parts[2])
                    self.camera_pitch_deg = max(-89.0, min(89.0, float(parts[3])))
                    self.camera_roll_deg = float(parts[4])
                    self._log(self.camera_status_text())
                elif sub == "fov" and len(parts) == 3:
                    self.camera_fov_deg = max(10.0, min(160.0, float(parts[2])))
                    self._log(self.camera_status_text())
                elif sub == "speed" and len(parts) == 4:
                    kind = parts[2].lower()
                    val = max(1.0, float(parts[3]))
                    if kind == "move":
                        self.camera_move_speed = val
                    elif kind == "turn":
                        self.camera_turn_speed = val
                    else:
                        raise ValueError("usage: camera speed <move|turn> <v>")
                    self._log(self.camera_status_text())
                else:
                    raise ValueError("usage: camera <status|pos <x> <y> <z>|rot <yaw> <pitch> <roll>|fov <deg>|speed <move|turn> <v>|reset>")

            elif op == "bg":
                if len(parts) != 2:
                    raise ValueError("usage: bg <none|clear_sky_earth|deep_space|status>")
                sub = parts[1].lower()
                if sub == "status":
                    self._log(f"background={self.background_preset_var.get()}")
                elif sub in {"none", "clear_sky_earth", "deep_space"}:
                    self.background_preset_var.set(sub)
                    self._log(f"background={sub}")
                else:
                    raise ValueError("usage: bg <none|clear_sky_earth|deep_space|status>")

            elif op == "obj":
                if len(parts) < 2:
                    raise ValueError("usage: obj <addcube|addplane|addsphere|addlight|addcamera> <name> <x> <y> [z] <size> | obj z <name> <z> | obj del <name> | obj list")
                sub = parts[1].lower()
                if sub in {"addcube", "addplane", "addsphere", "addlight", "addcamera"} and len(parts) in {6, 7}:
                    name = parts[2]
                    x = float(parts[3])
                    y = float(parts[4])
                    if len(parts) == 6:
                        z = 0.0
                        size = float(parts[5])
                    else:
                        z = float(parts[5])
                        size = float(parts[6])
                    if sub == "addcube":
                        self.add_cube_object(name, x, y, z, size)
                    elif sub == "addplane":
                        self.add_plane_object(name, x, y, z, size)
                    elif sub == "addsphere":
                        self.add_sphere_object(name, x, y, z, size)
                    elif sub == "addlight":
                        self.add_light_object(name, x, y, z, size)
                    elif sub == "addcamera":
                        self.add_camera_object(name, x, y, z, size)
                elif sub == "z" and len(parts) == 4:
                    name = parts[2]
                    z = float(parts[3])
                    if name not in self.scene_objects:
                        raise ValueError(f"object not found: {name}")
                    self._push_undo()
                    self.scene_objects[name].z = z
                    self._log(f"obj {name} z={z:.2f}")
                elif sub == "del" and len(parts) == 3:
                    self.delete_object(parts[2])
                elif sub == "list":
                    self._log(f"objects total={len(self.scene_objects)}")
                    for name in sorted(self.scene_objects.keys())[:100]:
                        obj = self.scene_objects[name]
                        x, y, r = self._eval_object_transform(obj, self.timeline_frame)
                        self._log(
                            f"obj {name} kind={obj.kind} pos=({x:.1f},{y:.1f},{obj.z:.1f}) size={obj.size:.1f} rot={r:.1f} "
                            f"mat(r={obj.roughness:.2f} m={obj.metallic:.2f} e={obj.emission:.2f})"
                        )
                else:
                    raise ValueError("usage: obj <addcube|addplane|addsphere|addlight|addcamera> <name> <x> <y> [z] <size> | obj z <name> <z> | obj del <name> | obj list")

            elif op == "undo":
                self.undo_scene_edit()

            elif op == "redo":
                self.redo_scene_edit()

            elif op == "snap":
                if len(parts) < 2:
                    raise ValueError("usage: snap <on|off|grid <size>|status>")
                sub = parts[1].lower()
                if sub == "on":
                    self.snap_enabled = True
                    self._log(f"snap enabled={self.snap_enabled} grid={self.snap_grid:.2f}")
                elif sub == "off":
                    self.snap_enabled = False
                    self._log(f"snap enabled={self.snap_enabled} grid={self.snap_grid:.2f}")
                elif sub == "grid" and len(parts) == 3:
                    self.snap_grid = max(0.1, float(parts[2]))
                    self._log(f"snap grid={self.snap_grid:.2f}")
                elif sub == "status":
                    self._log(f"snap enabled={self.snap_enabled} grid={self.snap_grid:.2f}")
                else:
                    raise ValueError("usage: snap <on|off|grid <size>|status>")

            elif op == "rotvel":
                if len(parts) != 2:
                    raise ValueError("usage: rotvel <deg_per_sec|status>")
                if parts[1].lower() == "status":
                    self._log(f"rotvel={self.rotate_key_angular_velocity_dps:.2f} deg/s")
                else:
                    self.rotate_key_angular_velocity_dps = max(1.0, float(parts[1]))
                    self._log(f"rotvel={self.rotate_key_angular_velocity_dps:.2f} deg/s")

            elif op == "key":
                if len(parts) != 7 or parts[1].lower() != "set":
                    raise ValueError("usage: key set <name> <frame> <x> <y> <rot_deg>")
                name = parts[2]
                frame = int(parts[3])
                x = float(parts[4])
                y = float(parts[5])
                rot = float(parts[6])
                self.set_object_keyframe(name, frame, x, y, rot)

            elif op == "timeline":
                if len(parts) < 2:
                    raise ValueError("usage: timeline <play|pause|frame <n>|fps <n>|len <n>|status>")
                sub = parts[1].lower()
                if sub == "play":
                    self.timeline_playing = True
                    self._log("timeline playing")
                elif sub == "pause":
                    self.timeline_playing = False
                    self._log("timeline paused")
                elif sub == "frame" and len(parts) == 3:
                    self.timeline_frame = max(0, int(parts[2]))
                    self._log(f"timeline frame={self.timeline_frame}")
                elif sub == "fps" and len(parts) == 3:
                    self.timeline_fps = max(1.0, float(parts[2]))
                    self._log(f"timeline fps={self.timeline_fps:.2f}")
                elif sub == "len" and len(parts) == 3:
                    self.timeline_length = max(1, int(parts[2]))
                    self._log(f"timeline length={self.timeline_length}")
                elif sub == "status":
                    self._log(
                        f"timeline frame={self.timeline_frame}/{self.timeline_length} fps={self.timeline_fps:.2f} playing={self.timeline_playing}"
                    )
                else:
                    raise ValueError("usage: timeline <play|pause|frame <n>|fps <n>|len <n>|status>")

            elif op == "tex":
                if len(parts) < 2:
                    raise ValueError("usage: tex load <name> <image_path>")
                if parts[1].lower() == "load" and len(parts) >= 4:
                    name = parts[2]
                    path = " ".join(parts[3:])
                    self.assign_texture_to_object(name, path)
                else:
                    raise ValueError("usage: tex load <name> <image_path>")

            elif op == "export":
                if len(parts) < 2:
                    raise ValueError("usage: export mp4 <path> [start end fps] | export test4k <png_path>")
                sub = parts[1].lower()
                if sub == "mp4" and len(parts) >= 3:
                    path = parts[2]
                    if len(parts) >= 6:
                        start_f = int(parts[3])
                        end_f = int(parts[4])
                        fps = int(parts[5])
                    else:
                        start_f = 0
                        end_f = self.timeline_length
                        fps = int(self.timeline_fps)
                    self._log(f"exporting mp4: {path} frames={start_f}-{end_f} fps={fps} res={self.world_w}x{self.world_h}")
                    self.export_mp4_mode_a(path, start_f, end_f, fps)
                    self._log(f"export done: {path}")
                elif sub == "test4k" and len(parts) >= 3:
                    path = parts[2]
                    self.export_test4k_png(path)
                    self._log(f"4k test frame written: {path}")
                else:
                    raise ValueError("usage: export mp4 <path> [start end fps] | export test4k <png_path>")

            elif op == "step":
                n = int(parts[1]) if len(parts) > 1 else 1
                for _ in range(max(1, n)):
                    self._update_physics()
                self._draw()
                self._log(f"stepped {max(1, n)}")

            elif op == "pause":
                self.running = False
                self.run_btn.configure(text="Run")
                self._log("paused")

            elif op == "run":
                self.running = True
                self.run_btn.configure(text="Pause")
                self._log("running")

            elif op == "clear":
                self.clear_all()
                self._log("cleared all particles")

            elif op == "list":
                count = int(parts[1]) if len(parts) > 1 else 12
                take = self.world.particles[:max(1, count)]
                self._log(f"objects total={len(self.scene_objects)}")
                self._log(f"particles total={len(self.world.particles)}")
                for i, p in enumerate(take):
                    self._log(f"[{i}] {p.material.name} pos=({p.x:.2f},{p.y:.2f},{p.z:.2f}) vel=({p.vx:.2f},{p.vy:.2f},{p.vz:.2f})")

            elif op == "setv":
                if len(parts) != 4:
                    raise ValueError("usage: setv <idx> <vx> <vy>")
                idx = int(parts[1])
                vx = float(parts[2])
                vy = float(parts[3])
                if idx < 0 or idx >= len(self.world.particles):
                    raise ValueError("index out of range")
                self.world.particles[idx].vx = vx
                self.world.particles[idx].vy = vy
                self.selected_index = idx
                self.vx_var.set(vx)
                self.vy_var.set(vy)
                self._log(f"setv idx={idx} vx={vx:.3f} vy={vy:.3f}")

            elif op == "physics":
                if len(parts) != 3:
                    raise ValueError("usage: physics <k|rep|drag|dt> <value>")
                key = parts[1].lower()
                val = float(parts[2])
                if key == "k":
                    self.k_var.set(val)
                elif key == "rep":
                    self.rep_var.set(val)
                elif key == "drag":
                    self.drag_var.set(val)
                elif key == "dt":
                    self.dt_var.set(val)
                else:
                    raise ValueError("physics key must be one of: k, rep, drag, dt")
                self._log(f"physics {key}={val}")

            elif op == "emergency":
                if len(parts) != 2:
                    raise ValueError("usage: emergency <on|off|status>")
                mode = parts[1].lower()
                if mode == "on":
                    self.emergency_pause_enabled = True
                    self.emergency_suppress = False
                    self._log("emergency warning: enabled")
                elif mode == "off":
                    self.emergency_pause_enabled = False
                    self._log("emergency warning: disabled (unsafe)")
                elif mode == "status":
                    self._log(
                        f"emergency enabled={self.emergency_pause_enabled} suppressed={self.emergency_suppress} "
                        f"threshold={self.emergency_pair_threshold:,}"
                    )
                else:
                    raise ValueError("usage: emergency <on|off|status>")

            elif op == "quantum":
                if len(parts) < 2:
                    raise ValueError("usage: quantum <set|grid|field|dt|steps|superpose|measure|view|Z|info|reset>")
                if self.quantum_world is None:
                    self.quantum_world = QuantumWorld()
                    self._log("Quantum engine initialized (64\u00b3 grid, hydrogen Z=1)")
                qw = self.quantum_world
                sub = parts[1].lower()

                if sub == "set" and len(parts) == 5:
                    n_q = int(parts[2])
                    l_q = int(parts[3])
                    m_q = int(parts[4])
                    qw.init_eigenstate(n_q, l_q, m_q)
                    self._log(f"quantum: initialized \u03c8_({qw.current_n},{qw.current_l},{qw.current_m})")
                elif sub == "grid" and len(parts) == 3:
                    gn = int(parts[2])
                    if gn not in {32, 48, 64, 96, 128, 192, 256}:
                        raise ValueError("grid size should be 32, 48, 64, 96, 128, 192, or 256")
                    self._log(f"quantum: resizing grid to {gn}\u00b3 (this may take a moment)...")
                    qw.resize_grid(gn)
                    self._log(f"quantum: grid resized to {gn}\u00b3, eigenstate reinitialized")
                elif sub == "field" and len(parts) == 5:
                    ex = float(parts[2])
                    ey = float(parts[3])
                    ez = float(parts[4])
                    qw.set_electric_field(ex, ey, ez)
                    self._log(f"quantum: electric field set to ({ex}, {ey}, {ez}) atomic units")
                elif sub == "dt" and len(parts) == 3:
                    dt_val = float(parts[2])
                    qw.set_dt(dt_val)
                    self._log(f"quantum: dt={dt_val} atomic units ({dt_val * 24.188:.2f} attoseconds)")
                elif sub == "steps" and len(parts) == 3:
                    s = max(1, int(parts[2]))
                    qw.steps_per_tick = s
                    self._log(f"quantum: steps_per_tick={s}")
                elif sub == "superpose" and len(parts) == 6:
                    n2 = int(parts[2])
                    l2 = int(parts[3])
                    m2 = int(parts[4])
                    w = float(parts[5])
                    qw.superpose_state(n2, l2, m2, w)
                    self._log(f"quantum: superposed with \u03c8_({n2},{l2},{m2}) weight={w:.3f}")
                elif sub == "measure":
                    x0, y0, z0 = qw.measure_position()
                    self._log(
                        f"quantum: MEASUREMENT collapsed \u03c8 at "
                        f"({x0:.3f}, {y0:.3f}, {z0:.3f}) a\u2080 "
                        f"= ({x0 * 0.529:.3f}, {y0 * 0.529:.3f}, {z0 * 0.529:.3f}) \u00c5"
                    )
                elif sub == "view" and len(parts) >= 3:
                    vm = parts[2].lower()
                    if vm in {"slice", "project"}:
                        qw.view_mode = vm
                        if len(parts) >= 4 and parts[3].lower() in {"x", "y", "z"}:
                            qw.slice_axis = parts[3].lower()
                        self._log(f"quantum: view={qw.view_mode} axis={qw.slice_axis}")
                    else:
                        raise ValueError("usage: quantum view <slice|project> [x|y|z]")
                elif sub == "z" and len(parts) == 3:
                    Z_val = int(parts[2])
                    qw.set_nuclear_Z(Z_val)
                    self._log(f"quantum: nuclear charge Z={Z_val}")
                elif sub == "info":
                    info = qw.get_info_dict()
                    for k, v in info.items():
                        self._log(f"  {k}: {v}")
                elif sub == "reset":
                    qw.init_eigenstate(qw.current_n, qw.current_l, qw.current_m)
                    self._log(f"quantum: reset to \u03c8_({qw.current_n},{qw.current_l},{qw.current_m})")
                else:
                    raise ValueError("usage: quantum <set|grid|field|dt|steps|superpose|measure|view|Z|info|reset>")

            else:
                self._log("unknown command. use: help")

        except Exception as exc:
            self._log(f"error: {exc}")

    def run(self) -> None:
        self.root.mainloop()


class ConsoleSimulatorApp:
    def __init__(self) -> None:
        self.world = PhysicsWorld(1000, 860)
        self.world.load_preset("Hydrogen")
        self.mode = "A"
        self.quantum_world: QuantumWorld | None = None

    def print_state(self) -> None:
        print(f"Particles: {len(self.world.particles)}")
        preview = self.world.particles[:8]
        for i, p in enumerate(preview):
            print(f"[{i}] {p.material.name:8s} pos=({p.x:7.2f},{p.y:7.2f},{p.z:7.2f}) vel=({p.vx:7.2f},{p.vy:7.2f},{p.vz:7.2f})")
        if len(self.world.particles) > len(preview):
            print(f"... ({len(self.world.particles) - len(preview)} more)")

    def run(self) -> None:
        print("Tk is unavailable. Running console mode.")
        print("Commands: help, preset <name>, spawn <material> <count>, step <n>, setv <idx> <vx> <vy>, mode <A|B|C|status>, exact <...>, quantum <...>, clear, list, quit")
        while True:
            try:
                cmd = input("sim> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not cmd:
                continue
            parts = cmd.split()
            op = parts[0].lower()
            try:
                if op == "help":
                    print("preset names: Hydrogen, Helium, Carbon, Plasma Box, Aluminum Cube (50nm, scaled)")
                    print("materials: " + ", ".join(MATERIALS.keys()))
                    print("scale profiles: " + ", ".join(SCALE_PROFILES.keys()))
                    print("modes: A (approximate), B (full equations), C (quantum TDSE)")
                    print("exact: status | transport <on|off> | photons <n> | bounces <n> | energy <v> | coupling <v>")
                    print("quantum: set <n> <l> <m> | grid <n> | field <ex> <ey> <ez> | dt <v> | steps <n> | info | reset")
                elif op == "preset" and len(parts) >= 2:
                    name = " ".join(parts[1:])
                    self.world.load_preset(name)
                    self.print_state()
                elif op == "spawn" and len(parts) == 3:
                    self.world.spawn_random(parts[1], int(parts[2]))
                    self.print_state()
                elif op == "step":
                    steps = int(parts[1]) if len(parts) > 1 else 1
                    for _ in range(max(1, steps)):
                        self.world.step(self.mode)
                    self.print_state()
                elif op == "mode" and len(parts) == 2:
                    m = parts[1].upper()
                    if m == "STATUS":
                        print(f"mode={self.mode}")
                    elif m in {"A", "B", "C"}:
                        self.mode = m
                        if m == "C" and self.quantum_world is None:
                            self.quantum_world = QuantumWorld()
                            print("Quantum engine initialized (64\u00b3 grid, hydrogen Z=1)")
                        print(f"mode={self.mode}")
                    else:
                        print("usage: mode <A|B|C|status>")
                elif op == "exact" and len(parts) >= 2:
                    sub = parts[1].lower()
                    if sub == "status":
                        print(
                            f"exact transport={self.world.mode_b_transport_exact} photons={self.world.photons_per_particle_per_step} "
                            f"bounces={self.world.photon_max_bounces} energy={self.world.photon_packet_energy:.3e} "
                            f"coupling={self.world.radiation_coupling:.3e} backend={self.world.compute_backend}"
                        )
                    elif sub == "transport" and len(parts) == 3:
                        self.world.mode_b_transport_exact = parts[2].lower() == "on"
                        print(f"exact transport={self.world.mode_b_transport_exact}")
                    elif sub == "photons" and len(parts) == 3:
                        self.world.photons_per_particle_per_step = max(1, int(parts[2]))
                        print(f"exact photons={self.world.photons_per_particle_per_step}")
                    elif sub == "bounces" and len(parts) == 3:
                        self.world.photon_max_bounces = max(0, int(parts[2]))
                        print(f"exact bounces={self.world.photon_max_bounces}")
                    elif sub == "energy" and len(parts) == 3:
                        self.world.photon_packet_energy = float(parts[2])
                        print(f"exact energy={self.world.photon_packet_energy:.3e}")
                    elif sub == "coupling" and len(parts) == 3:
                        self.world.radiation_coupling = float(parts[2])
                        print(f"exact coupling={self.world.radiation_coupling:.3e}")
                    else:
                        print("usage: exact <status|transport <on|off>|photons <n>|bounces <n>|energy <v>|coupling <v>>")
                elif op == "setv" and len(parts) == 4:
                    i = int(parts[1])
                    self.world.particles[i].vx = float(parts[2])
                    self.world.particles[i].vy = float(parts[3])
                    self.print_state()
                elif op == "clear":
                    self.world.clear_all()
                    self.print_state()
                elif op == "list":
                    self.print_state()
                elif op in {"quit", "exit"}:
                    break
                else:
                    print("Unknown command.")
            except Exception as exc:
                print(f"error: {exc}")


def run_self_test() -> None:
    world = PhysicsWorld(1000, 860)
    world.load_preset("Hydrogen")
    world.spawn_random("electron", 5)
    for _ in range(25):
        world.step()
    if not world.particles:
        raise RuntimeError("self-test failed: no particles")
    print(f"SELFTEST_OK particles={len(world.particles)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    if TK_AVAILABLE:
        AtomSimulatorApp().run()
    else:
        ConsoleSimulatorApp().run()


if __name__ == "__main__":
    main()
