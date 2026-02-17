from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass

try:
    import tkinter as tk
    from tkinter import ttk
    TK_AVAILABLE = True
except Exception:
    tk = None  # type: ignore[assignment]
    ttk = None  # type: ignore[assignment]
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

    def clear_all(self) -> None:
        self.particles.clear()

    def load_preset(self, name: str) -> None:
        self.clear_all()

        cx = self.world_w * 0.5
        cy = self.world_h * 0.5

        if name == "Hydrogen":
            self.particles.append(Particle(cx, cy, 0.0, 0.0, MATERIALS["proton"]))
            self.particles.append(Particle(cx + 90.0, cy, 0.0, -145.0, MATERIALS["electron"]))

        elif name == "Helium":
            self.particles.append(Particle(cx - 4.0, cy, 0.0, 0.0, MATERIALS["proton"]))
            self.particles.append(Particle(cx + 4.0, cy, 0.0, 0.0, MATERIALS["proton"]))
            self.particles.append(Particle(cx + 70.0, cy, 0.0, -150.0, MATERIALS["electron"]))
            self.particles.append(Particle(cx - 85.0, cy, 0.0, 130.0, MATERIALS["electron"]))

        elif name == "Carbon":
            for i in range(6):
                a = 2 * math.pi * i / 6
                self.particles.append(Particle(cx + 8 * math.cos(a), cy + 8 * math.sin(a), 0.0, 0.0, MATERIALS["proton"]))
            for i in range(6):
                a = 2 * math.pi * i / 6
                self.particles.append(Particle(cx + 15 * math.cos(a), cy + 15 * math.sin(a), 0.0, 0.0, MATERIALS["neutron"]))
            for i in range(2):
                a = 2 * math.pi * i / 2
                self.particles.append(Particle(cx + 80 * math.cos(a), cy + 80 * math.sin(a), -140 * math.sin(a), 140 * math.cos(a), MATERIALS["electron"]))
            for i in range(4):
                a = 2 * math.pi * i / 4 + 0.3
                self.particles.append(Particle(cx + 150 * math.cos(a), cy + 150 * math.sin(a), -120 * math.sin(a), 120 * math.cos(a), MATERIALS["electron"]))

        elif name == "Plasma Box":
            for _ in range(80):
                self.particles.append(
                    Particle(
                        random.uniform(150, self.world_w - 150),
                        random.uniform(120, self.world_h - 120),
                        random.uniform(-60, 60),
                        random.uniform(-60, 60),
                        MATERIALS["ion+"] if random.random() < 0.5 else MATERIALS["ion-"],
                    )
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

    def step(self) -> None:
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
        self.root.title("Matter Sim - Atom Simulator (starter)")
        self.root.geometry("1400x900")

        self.world_w = 1000
        self.world_h = 860
        self.world = PhysicsWorld(self.world_w, self.world_h)
        self.running = True

        self.selected_index: int | None = None
        self.drag_start: tuple[float, float] | None = None
        self.drag_current: tuple[float, float] | None = None

        self._build_ui()
        self._bind_events()
        self.world.load_preset("Hydrogen")
        self.selected_index = 0 if self.world.particles else None
        self._tick()

    # -----------------------------
    # UI
    # -----------------------------
    def _build_ui(self) -> None:
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(main, width=self.world_w, height=self.world_h, bg="#0b0f14", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        panel = ttk.Frame(main, padding=10)
        panel.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Label(panel, text="Simulation", font=("TkDefaultFont", 13, "bold")).pack(anchor="w", pady=(0, 6))

        self.run_btn = ttk.Button(panel, text="Pause", command=self.toggle_running)
        self.run_btn.pack(fill=tk.X, pady=2)

        ttk.Button(panel, text="Step Once", command=self.step_once).pack(fill=tk.X, pady=2)
        ttk.Button(panel, text="Reset (Current Preset)", command=self.reload_preset).pack(fill=tk.X, pady=2)

        ttk.Separator(panel).pack(fill=tk.X, pady=8)

        ttk.Label(panel, text="Presets").pack(anchor="w")
        self.preset_var = tk.StringVar(value="Hydrogen")
        preset_values = ["Hydrogen", "Helium", "Carbon", "Plasma Box"]
        self.preset_combo = ttk.Combobox(panel, textvariable=self.preset_var, values=preset_values, state="readonly")
        self.preset_combo.pack(fill=tk.X, pady=2)
        ttk.Button(panel, text="Load Preset", command=lambda: self.load_preset(self.preset_var.get())).pack(fill=tk.X, pady=2)

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

    def _add_slider(self, panel: ttk.Frame, label: str, var: tk.DoubleVar, mn: float, mx: float) -> None:
        ttk.Label(panel, text=label).pack(anchor="w")
        scale = ttk.Scale(panel, from_=mn, to=mx, variable=var)
        scale.pack(fill=tk.X, pady=(0, 4))

    def _bind_events(self) -> None:
        self.canvas.bind("<Button-1>", self.on_left_down)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_up)
        self.canvas.bind("<Button-3>", self.on_right_click_spawn)

    # -----------------------------
    # Presets
    # -----------------------------
    def clear_all(self) -> None:
        self.world.particles.clear()
        self.selected_index = None

    def load_preset(self, name: str) -> None:
        self.world.load_preset(name)
        self.selected_index = 0 if self.world.particles else None

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
                    random.uniform(-100, 100),
                    random.uniform(-100, 100),
                    mat,
                )
            )

    # -----------------------------
    # Interaction
    # -----------------------------
    def find_particle(self, x: float, y: float) -> int | None:
        best_i = None
        best_d = 1e9
        for i, p in enumerate(self.world.particles):
            d = math.hypot(p.x - x, p.y - y)
            if d <= p.radius + 6 and d < best_d:
                best_d = d
                best_i = i
        return best_i

    def on_left_down(self, event) -> None:
        x, y = float(event.x), float(event.y)
        idx = self.find_particle(x, y)
        self.selected_index = idx
        if idx is not None:
            self.drag_start = (x, y)
            self.drag_current = (x, y)
        else:
            self.drag_start = None
            self.drag_current = None

    def on_drag(self, event) -> None:
        if self.drag_start is not None:
            self.drag_current = (float(event.x), float(event.y))

    def on_left_up(self, event) -> None:
        if self.drag_start is None or self.selected_index is None:
            return
        sx, sy = self.drag_start
        ex, ey = float(event.x), float(event.y)
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
        self.world.particles.append(Particle(float(event.x), float(event.y), 0.0, 0.0, mat))

    def apply_velocity_to_selected(self) -> None:
        if self.selected_index is None:
            return
        p = self.world.particles[self.selected_index]
        p.vx = float(self.vx_var.get())
        p.vy = float(self.vy_var.get())

    # -----------------------------
    # Physics
    # -----------------------------
    def step_once(self) -> None:
        self._update_physics()
        self._draw()

    def toggle_running(self) -> None:
        self.running = not self.running
        self.run_btn.configure(text="Pause" if self.running else "Run")

    def _update_physics(self) -> None:
        self.world.k_coulomb = float(self.k_var.get())
        self.world.repulsion_k = float(self.rep_var.get())
        self.world.drag = float(self.drag_var.get())
        self.world.dt = float(self.dt_var.get())
        self.world.step()

    # -----------------------------
    # Rendering
    # -----------------------------
    def _draw(self) -> None:
        self.canvas.delete("all")

        # background grid
        grid_step = 40
        for x in range(0, self.world_w, grid_step):
            self.canvas.create_line(x, 0, x, self.world_h, fill="#101827")
        for y in range(0, self.world_h, grid_step):
            self.canvas.create_line(0, y, self.world_w, y, fill="#101827")

        for i, p in enumerate(self.world.particles):
            r = p.radius
            self.canvas.create_oval(p.x - r, p.y - r, p.x + r, p.y + r, fill=p.material.color, outline="")

            if i == self.selected_index:
                rr = r + 4
                self.canvas.create_oval(p.x - rr, p.y - rr, p.x + rr, p.y + rr, outline="#ffffff")

        # velocity drag arrow
        if self.drag_start and self.drag_current:
            sx, sy = self.drag_start
            cx, cy = self.drag_current
            self.canvas.create_line(sx, sy, cx, cy, fill="#ffffff", width=2, arrow=tk.LAST)

        self.info_var.set(
            f"Particles: {len(self.world.particles)} | Running: {self.running} | "
            f"Selected: {self.selected_index if self.selected_index is not None else 'None'}"
        )

    def _tick(self) -> None:
        if self.running:
            self._update_physics()
        self._draw()
        self.root.after(16, self._tick)

    def run(self) -> None:
        self.root.mainloop()


class ConsoleSimulatorApp:
    def __init__(self) -> None:
        self.world = PhysicsWorld(1000, 860)
        self.world.load_preset("Hydrogen")

    def print_state(self) -> None:
        print(f"Particles: {len(self.world.particles)}")
        preview = self.world.particles[:8]
        for i, p in enumerate(preview):
            print(f"[{i}] {p.material.name:8s} pos=({p.x:7.2f},{p.y:7.2f}) vel=({p.vx:7.2f},{p.vy:7.2f})")
        if len(self.world.particles) > len(preview):
            print(f"... ({len(self.world.particles) - len(preview)} more)")

    def run(self) -> None:
        print("Tk is unavailable. Running console mode.")
        print("Commands: help, preset <name>, spawn <material> <count>, step <n>, setv <idx> <vx> <vy>, clear, list, quit")
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
                    print("preset names: Hydrogen, Helium, Carbon, Plasma Box")
                    print("materials: " + ", ".join(MATERIALS.keys()))
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
                        self.world.step()
                    self.print_state()
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
