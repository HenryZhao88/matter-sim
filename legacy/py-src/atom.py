from __future__ import annotations

import math
from dataclasses import dataclass, field

import glfw
from OpenGL.GL import *

ORBIT_DISTANCE = 200.0
WIDTH, HEIGHT = 800, 600


@dataclass
class Particle:
    x: float
    y: float
    charge: int
    angle: float = math.pi
    energy: float = -13.6
    n: int = 1

    def draw(self, cx: float, cy: float, segments: int = 50) -> None:
        if self.charge == -1:
            glLineWidth(0.4)
            glBegin(GL_LINE_LOOP)
            glColor3f(0.4, 0.4, 0.4)
            for i in range(5000 + 1):
                r = ORBIT_DISTANCE * self.n
                a = 2.0 * math.pi * i / 5000
                glVertex2f(math.cos(a) * r + cx, math.sin(a) * r + cy)
            glEnd()

        if self.charge == -1:
            r = 10
            glColor3f(0.0, 1.0, 1.0)
        elif self.charge == 1:
            r = 50
            glColor3f(1.0, 0.0, 0.0)
        else:
            r = 10
            glColor3f(0.5, 0.5, 0.5)

        glBegin(GL_TRIANGLE_FAN)
        glVertex2f(self.x, self.y)
        for i in range(segments + 1):
            a = 2.0 * math.pi * i / segments
            glVertex2f(self.x + math.cos(a) * r, self.y + math.sin(a) * r)
        glEnd()

    def update(self, cx: float, cy: float) -> None:
        r = ORBIT_DISTANCE * self.n
        self.angle += 0.05
        self.x = math.cos(self.angle) * r + cx
        self.y = math.sin(self.angle) * r + cy


@dataclass
class Atom:
    x: float
    y: float
    particles: list[Particle] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.particles:
            self.particles = [
                Particle(self.x, self.y, 1),
                Particle(self.x - ORBIT_DISTANCE, self.y, -1),
            ]


atoms = [Atom(0.0, 0.0)]


def key_callback(window, key, scancode, action, mods):
    if action not in (glfw.PRESS, glfw.REPEAT):
        return
    delta = 0.0
    if key == glfw.KEY_W:
        delta = 0.01
    elif key == glfw.KEY_S:
        delta = -0.01
    elif key == glfw.KEY_E:
        delta = 0.1
    elif key == glfw.KEY_D:
        delta = -0.1
    elif key == glfw.KEY_R:
        delta = 1.0
    elif key == glfw.KEY_F:
        delta = -1.0

    if delta != 0.0:
        for atom in atoms:
            for p in atom.particles:
                p.energy += delta
                if key == glfw.KEY_W:
                    p.angle = 0.0
                print(f"Particle energy: {p.energy}")


def main() -> None:
    if not glfw.init():
        raise RuntimeError("failed to init glfw")

    window = glfw.create_window(WIDTH, HEIGHT, "2D atom sim by kavan (python)", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("failed to create window")

    glfw.make_context_current(window)
    glfw.set_key_callback(window, key_callback)

    while not glfw.window_should_close(window):
        glClear(GL_COLOR_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-WIDTH / 2.0, WIDTH / 2.0, -HEIGHT / 2.0, HEIGHT / 2.0, -1.0, 1.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        for atom in atoms:
            for p in atom.particles:
                p.draw(atom.x, atom.y)
                if p.charge == -1:
                    p.update(atom.x, atom.y)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.destroy_window(window)
    glfw.terminate()


if __name__ == "__main__":
    main()
