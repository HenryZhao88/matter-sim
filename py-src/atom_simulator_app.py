from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass

try:
    import cupy as cp  # type: ignore
    CUPY_AVAILABLE = True
except Exception:
    cp = None  # type: ignore[assignment]
    CUPY_AVAILABLE = False

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


class PhysicsWorld:
    def __init__(self, world_w: int, world_h: int) -> None:
        self.world_w = world_w
        self.world_h = world_h
        self.particles: list[Particle] = []

        self.dt = 0.008
        self.k_coulomb = 1800.0
        self.softening = 10.0
        self.repulsion_k = 16000.0
        self.drag = 0.9995
        self.max_speed = 450.0
        self.bonds: list[Bond] = []
        self.last_note: str = ""
        self.compute_backend = "gpu" if CUPY_AVAILABLE else "cpu"

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

    def _step_exact(self) -> None:
        if len(self.particles) <= 1:
            return

        n = len(self.particles)
        fx = [0.0] * n
        fy = [0.0] * n

        for i in range(n):
            pi = self.particles[i]
            for j in range(i + 1, n):
                pj = self.particles[j]

                dx = pj.x - pi.x
                dy = pj.y - pi.y
                r2 = dx * dx + dy * dy + self.softening
                r = math.sqrt(r2)
                nx = dx / r
                ny = dy / r

                f_c = self.k_coulomb * pi.charge * pj.charge / r2
                overlap = (pi.radius + pj.radius) - r
                f_rep = self.repulsion_k * overlap if overlap > 0 else 0.0
                f_total = f_c - f_rep

                fx_i = f_total * nx
                fy_i = f_total * ny

                fx[i] += fx_i
                fy[i] += fy_i
                fx[j] -= fx_i
                fy[j] -= fy_i

        # Bond spring forces for crystalline structures.
        for b in self.bonds:
            if b.i < 0 or b.j < 0 or b.i >= len(self.particles) or b.j >= len(self.particles):
                continue
            pi = self.particles[b.i]
            pj = self.particles[b.j]
            dx = pj.x - pi.x
            dy = pj.y - pi.y
            r = math.hypot(dx, dy)
            if r < 1e-9:
                continue
            nx = dx / r
            ny = dy / r
            extension = r - b.rest_length
            f = b.k * extension
            fx[b.i] += f * nx
            fy[b.i] += f * ny
            fx[b.j] -= f * nx
            fy[b.j] -= f * ny

        self._integrate_forces(fx, fy)

    def _step_approximate(self) -> None:
        if len(self.particles) <= 1:
            return

        n = len(self.particles)
        fx = [0.0] * n
        fy = [0.0] * n

        samples = max(4, min(self.mode_a_pair_samples, n - 1))
        stride = max(1, (n // samples) + 1)
        scale_back = n / float(samples)

        # Approximate interactions: each particle samples only a subset of partners.
        for i in range(n):
            pi = self.particles[i]
            for s in range(1, samples + 1):
                j = (i + s * stride) % n
                if j == i:
                    continue
                pj = self.particles[j]

                dx = pj.x - pi.x
                dy = pj.y - pi.y
                r2 = dx * dx + dy * dy + self.softening
                r = math.sqrt(r2)
                nx = dx / r
                ny = dy / r

                f_c = self.k_coulomb * pi.charge * pj.charge / r2
                overlap = (pi.radius + pj.radius) - r
                f_rep = self.repulsion_k * overlap if overlap > 0 else 0.0
                f_total = (f_c - f_rep) * scale_back

                fx[i] += f_total * nx
                fy[i] += f_total * ny

        # Keep bonded structure but soften in mode A.
        for b in self.bonds:
            if b.i < 0 or b.j < 0 or b.i >= len(self.particles) or b.j >= len(self.particles):
                continue
            pi = self.particles[b.i]
            pj = self.particles[b.j]
            dx = pj.x - pi.x
            dy = pj.y - pi.y
            r = math.hypot(dx, dy)
            if r < 1e-9:
                continue
            nx = dx / r
            ny = dy / r
            extension = r - b.rest_length
            f = b.k * self.mode_a_bond_factor * extension
            fx[b.i] += f * nx
            fy[b.i] += f * ny
            fx[b.j] -= f * nx
            fy[b.j] -= f * ny

        self._integrate_forces(fx, fy)

    def _integrate_forces(self, fx: list[float], fy: list[float]) -> None:
        for i, p in enumerate(self.particles):
            ax = fx[i] / max(1e-9, p.mass)
            ay = fy[i] / max(1e-9, p.mass)

            p.vx = (p.vx + ax * self.dt) * self.drag
            p.vy = (p.vy + ay * self.dt) * self.drag

            speed = math.hypot(p.vx, p.vy)
            if speed > self.max_speed:
                s = self.max_speed / speed
                p.vx *= s
                p.vy *= s

            p.x += p.vx * self.dt
            p.y += p.vy * self.dt

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


class AtomSimulatorApp:
    def __init__(self) -> None:
        if not TK_AVAILABLE:
            raise RuntimeError("Tk is unavailable")

        self.root = tk.Tk()
        self.root.title("Matter Sim - Physics World")
        self.root.geometry("1020x920")

        self.world_w = 1000
        self.world_h = 860
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
                "usage": "view <home|zoom <factor>|pan <dx> <dy>>",
                "desc": "Viewport controls similar to DCC tools (frame/home, zoom, pan).",
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
        }

        self.sim_mode_var = tk.StringVar(value="A")
        self.scale_profile_var = tk.StringVar(value="micro")
        self.render_radius_scale = SCALE_PROFILES["micro"]["render_radius"]
        self.emergency_pause_enabled = True
        self.emergency_suppress = False
        self.emergency_grace_ticks = 0
        self.emergency_pair_threshold = 30_000_000

        self._build_ui()
        self._build_command_center()
        self._bind_events()
        self.world.load_preset("Hydrogen")
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
        self.canvas.pack(fill=tk.BOTH, expand=True)

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
        self.mode_combo = ttk.Combobox(mode_row, textvariable=self.sim_mode_var, values=["A", "B"], state="readonly", width=8)
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
        self.canvas.bind("<Button-3>", self.on_right_click_spawn)
        self.canvas.bind("<Button-2>", self.on_middle_down)
        self.canvas.bind("<B2-Motion>", self.on_middle_drag)
        self.canvas.bind("<ButtonRelease-2>", self.on_middle_up)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Button-4>", lambda e: self._zoom_at(e.x, e.y, 1.1))
        self.canvas.bind("<Button-5>", lambda e: self._zoom_at(e.x, e.y, 1.0 / 1.1))
        self.root.bind("<space>", lambda _e: self.toggle_running())
        self.root.bind("<Delete>", lambda _e: self.delete_selected())

    # -----------------------------
    # Presets
    # -----------------------------
    def clear_all(self) -> None:
        self.world.particles.clear()
        self.world.bonds.clear()
        self.selected_index = None

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

    def delete_selected(self) -> None:
        if self.selected_index is None:
            return
        if 0 <= self.selected_index < len(self.world.particles):
            del self.world.particles[self.selected_index]
        self.selected_index = None
        self._log("deleted selected object")

    def apply_sim_mode(self, mode: str) -> None:
        m = mode.upper()
        if m not in {"A", "B"}:
            self._log("error: mode must be A or B")
            return
        self.sim_mode_var.set(m)
        if m == "A":
            self._log("mode=A (Blender-like viewport, approximated simulation + proxy ray transport)")
        else:
            self._log(f"mode=B (full equations + exact transport, backend={self.world.compute_backend})")

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
        if self.drag_start is not None:
            self.drag_current = self.screen_to_world(float(event.x), float(event.y))

    def on_left_up(self, event) -> None:
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

    def on_right_click_spawn(self, event) -> None:
        mat = MATERIALS[self.material_var.get()]
        wx, wy = self.screen_to_world(float(event.x), float(event.y))
        self.world.particles.append(Particle(wx, wy, 0.0, 0.0, mat))

    def on_middle_down(self, event) -> None:
        self.viewport_drag_last = (float(event.x), float(event.y))

    def on_middle_drag(self, event) -> None:
        if self.viewport_drag_last is None:
            return
        lx, ly = self.viewport_drag_last
        nx, ny = float(event.x), float(event.y)
        dx, dy = nx - lx, ny - ly
        self.view_pan_x += dx / self.view_zoom
        self.view_pan_y += dy / self.view_zoom
        self.viewport_drag_last = (nx, ny)

    def on_middle_up(self, event) -> None:
        self.viewport_drag_last = None

    def on_mouse_wheel(self, event) -> None:
        delta = event.delta if hasattr(event, "delta") else 0
        if delta > 0:
            self._zoom_at(event.x, event.y, 1.1)
        elif delta < 0:
            self._zoom_at(event.x, event.y, 1.0 / 1.1)

    def apply_velocity_to_selected(self) -> None:
        if self.selected_index is None:
            return
        p = self.world.particles[self.selected_index]
        p.vx = float(self.vx_var.get())
        p.vy = float(self.vy_var.get())
        self._log(f"setv idx={self.selected_index} vx={p.vx:.3f} vy={p.vy:.3f}")

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
        self.world.k_coulomb = float(self.k_var.get())
        self.world.repulsion_k = float(self.rep_var.get())
        self.world.drag = float(self.drag_var.get())
        self.world.dt = float(self.dt_var.get())
        self.world.step(self.sim_mode_var.get())

    # -----------------------------
    # Rendering
    # -----------------------------
    def _draw(self) -> None:
        if self.sim_mode_var.get().upper() == "A":
            self._draw_mode_a()
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
            f"Selected: {self.selected_index if self.selected_index is not None else 'None'}"
        )
        self._refresh_outliner()

    def _draw_mode_a(self) -> None:
        self.canvas.delete("all")

        # Background + coarse ray-marched density proxy.
        self.canvas.create_rectangle(0, 0, self.world_w, self.world_h, fill="#090c12", outline="")

        cell = 14
        gw = (self.world_w // cell) + 1
        gh = (self.world_h // cell) + 1
        bins = [0.0] * (gw * gh)

        for p in self.world.particles:
            sx, sy = self.world_to_screen(p.x, p.y)
            ix = int(sx // cell)
            iy = int(sy // cell)
            if 0 <= ix < gw and 0 <= iy < gh:
                idx = iy * gw + ix
                speed = math.hypot(p.vx, p.vy)
                bins[idx] += 1.0 + min(3.0, speed / 130.0)

        max_bin = max(bins) if bins else 0.0
        if max_bin > 0.0:
            for iy in range(gh):
                y0 = iy * cell
                y1 = y0 + cell
                for ix in range(gw):
                    v = bins[iy * gw + ix]
                    if v <= 0.0:
                        continue
                    t = min(1.0, v / max_bin)
                    r = int(20 + 220 * t)
                    g = int(40 + 150 * (t ** 0.6))
                    b = int(90 + 120 * (1.0 - t * 0.5))
                    color = f"#{r:02x}{g:02x}{b:02x}"
                    x0 = ix * cell
                    x1 = x0 + cell
                    self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="")

        # Bonds sampled for speed.
        bond_stride = max(1, len(self.world.bonds) // 3000)
        for bi in range(0, len(self.world.bonds), bond_stride):
            b = self.world.bonds[bi]
            if b.i < 0 or b.j < 0 or b.i >= len(self.world.particles) or b.j >= len(self.world.particles):
                continue
            p1 = self.world.particles[b.i]
            p2 = self.world.particles[b.j]
            x1, y1 = self.world_to_screen(p1.x, p1.y)
            x2, y2 = self.world_to_screen(p2.x, p2.y)
            self.canvas.create_line(x1, y1, x2, y2, fill="#5f7aa0")

        # Particle sample with glow and ray-like velocity streak.
        n = len(self.world.particles)
        stride = max(1, n // 3500)
        for i in range(0, n, stride):
            p = self.world.particles[i]
            sx, sy = self.world_to_screen(p.x, p.y)
            r = max(1.0, p.radius * self.render_radius_scale * 0.85 * self.view_zoom)
            speed = math.hypot(p.vx, p.vy)
            trail = min(16.0, speed * 0.06)
            if speed > 1e-4:
                tx = p.x - (p.vx / speed) * trail
                ty = p.y - (p.vy / speed) * trail
                tsx, tsy = self.world_to_screen(tx, ty)
                self.canvas.create_line(sx, sy, tsx, tsy, fill="#dfe8ff")

            self.canvas.create_oval(sx - (r + 2), sy - (r + 2), sx + (r + 2), sy + (r + 2), outline="#7aa6ff")
            self.canvas.create_oval(sx - r, sy - r, sx + r, sy + r, fill=p.material.color, outline="")

            if i == self.selected_index:
                rr = r + 5
                self.canvas.create_oval(sx - rr, sy - rr, sx + rr, sy + rr, outline="#ffffff")

        if self.drag_start and self.drag_current:
            sx, sy = self.drag_start
            cx, cy = self.drag_current
            dsx, dsy = self.world_to_screen(sx, sy)
            dcx, dcy = self.world_to_screen(cx, cy)
            self.canvas.create_line(dsx, dsy, dcx, dcy, fill="#ffffff", width=2, arrow=tk.LAST)

        self.info_var.set(
            f"Mode: A | Particles: {len(self.world.particles)} Bonds: {len(self.world.bonds)} | Running: {self.running} | "
            f"Selected: {self.selected_index if self.selected_index is not None else 'None'}"
        )
        self._refresh_outliner()

    def _refresh_outliner(self) -> None:
        if not hasattr(self, "outliner_list"):
            return
        prev_sel = self.selected_index
        self.outliner_list.delete(0, tk.END)
        limit = min(300, len(self.world.particles))
        for i in range(limit):
            p = self.world.particles[i]
            self.outliner_list.insert(tk.END, f"{i:04d} | {p.material.name} | ({p.x:.1f}, {p.y:.1f})")
        if prev_sel is not None and 0 <= prev_sel < limit:
            self.outliner_list.selection_set(prev_sel)

    def _on_outliner_select(self) -> None:
        if not hasattr(self, "outliner_list"):
            return
        sel = self.outliner_list.curselection()
        if not sel:
            return
        idx = int(sel[0])
        if 0 <= idx < len(self.world.particles):
            self.selected_index = idx

    def _tick(self) -> None:
        if self.running and self._check_emergency_state():
            self._update_physics()
        elif self.emergency_grace_ticks > 0:
            self.emergency_grace_ticks -= 1
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
                    raise ValueError("usage: view <home|zoom <factor>|pan <dx> <dy>>")
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
                else:
                    raise ValueError("usage: view <home|zoom <factor>|pan <dx> <dy>>")

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
                self._log(f"particles total={len(self.world.particles)}")
                for i, p in enumerate(take):
                    self._log(f"[{i}] {p.material.name} pos=({p.x:.2f},{p.y:.2f}) vel=({p.vx:.2f},{p.vy:.2f})")

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

    def print_state(self) -> None:
        print(f"Particles: {len(self.world.particles)}")
        preview = self.world.particles[:8]
        for i, p in enumerate(preview):
            print(f"[{i}] {p.material.name:8s} pos=({p.x:7.2f},{p.y:7.2f}) vel=({p.vx:7.2f},{p.vy:7.2f})")
        if len(self.world.particles) > len(preview):
            print(f"... ({len(self.world.particles) - len(preview)} more)")

    def run(self) -> None:
        print("Tk is unavailable. Running console mode.")
        print("Commands: help, preset <name>, spawn <material> <count>, step <n>, setv <idx> <vx> <vy>, mode <A|B|status>, exact <...>, clear, list, quit")
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
                    print("modes: A (approximate), B (full equations)")
                    print("exact: status | transport <on|off> | photons <n> | bounces <n> | energy <v> | coupling <v>")
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
                    elif m in {"A", "B"}:
                        self.mode = m
                        print(f"mode={self.mode}")
                    else:
                        print("usage: mode <A|B|status>")
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
