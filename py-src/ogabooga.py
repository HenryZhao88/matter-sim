from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import glfw
from OpenGL.GL import *

C = 299792458.0 / 100000000.0
WIDTH, HEIGHT = 800, 600


def draw_filled_circle(x: float, y: float, r: float, segments: int = 50) -> None:
    glBegin(GL_TRIANGLE_FAN)
    glVertex2f(x, y)
    for i in range(segments + 1):
        a = 2.0 * math.pi * i / segments
        glVertex2f(x + r * math.cos(a), y + r * math.sin(a))
    glEnd()


@dataclass
class Proton:
    x: float
    y: float


@dataclass
class Neutron:
    x: float
    y: float


@dataclass
class Electron:
    orbit_radius: float
    angle: float
    speed: float


@dataclass
class Atom:
    protons: int
    electrons: int
    neutrons: int
    x: float
    y: float
    proton_list: list[Proton] = field(default_factory=list)
    neutron_list: list[Neutron] = field(default_factory=list)
    electron_list: list[Electron] = field(default_factory=list)

    def __post_init__(self) -> None:
        for _ in range(self.protons):
            self.proton_list.append(Proton(self.x + random.randint(-5, 5), self.y + random.randint(-5, 5)))
        for _ in range(self.neutrons):
            self.neutron_list.append(Neutron(self.x + random.randint(-5, 5), self.y + random.randint(-5, 5)))

        shell_cap = [2, 8, 18, 32]
        base_radius = 45.0
        remaining = self.electrons
        for s in range(4):
            if remaining <= 0:
                break
            count = min(remaining, shell_cap[s])
            for i in range(count):
                angle = i * (2.0 * math.pi / max(1, count))
                self.electron_list.append(Electron(base_radius * (s + 1), angle, C))
            remaining -= count

    def update(self) -> None:
        dt = 1.0 / 60.0
        for e in self.electron_list:
            e.angle += e.speed * dt
            if e.angle > 2.0 * math.pi:
                e.angle -= 2.0 * math.pi

    def draw(self) -> None:
        for p in self.proton_list:
            glColor3f(1.0, 0.0, 0.0)
            draw_filled_circle(p.x, p.y, 12)
        for n in self.neutron_list:
            glColor3f(0.5, 0.5, 0.5)
            draw_filled_circle(n.x, n.y, 12)

        glColor3f(0.0, 0.6, 1.0)
        for e in self.electron_list:
            ex = math.cos(e.angle) * e.orbit_radius + self.x
            ey = math.sin(e.angle) * e.orbit_radius + self.y
            draw_filled_circle(ex, ey, 6)


atoms = [
    Atom(1, 1, 2, -200, 0),
    Atom(1, 1, 2, 200, 0),
    Atom(1, 1, 2, 0, 200),
    Atom(1, 1, 2, 0, -200),
    Atom(6, 6, 12, 0, 0),
]


def main() -> None:
    if not glfw.init():
        raise RuntimeError("failed to init glfw")
    window = glfw.create_window(WIDTH, HEIGHT, "Atom Sim (python)", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("failed to create window")

    glfw.make_context_current(window)

    while not glfw.window_should_close(window):
        glClear(GL_COLOR_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-WIDTH, WIDTH, -HEIGHT, HEIGHT, -1.0, 1.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        for atom in atoms:
            atom.draw()
            atom.update()

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.destroy_window(window)
    glfw.terminate()


if __name__ == "__main__":
    main()
