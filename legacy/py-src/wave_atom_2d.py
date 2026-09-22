from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

import glfw
from OpenGL.GL import *

ORBITAL_RAD = 25.0
WIDTH, HEIGHT = 800, 600


def draw_circle(x: float, y: float, r: float, segments: int = 100) -> None:
    glLineWidth(0.4)
    glColor3f(0.4, 0.4, 0.4)
    glBegin(GL_LINE_LOOP)
    for i in range(segments + 1):
        a = 2.0 * math.pi * i / segments
        glVertex2f(x + r * math.cos(a), y + r * math.sin(a))
    glEnd()


def draw_filled_circle(x: float, y: float, r: float, segments: int = 50) -> None:
    glBegin(GL_TRIANGLE_FAN)
    glVertex2f(x, y)
    for i in range(segments + 1):
        a = 2.0 * math.pi * i / segments
        glVertex2f(x + r * math.cos(a), y + r * math.sin(a))
    glEnd()


@dataclass
class WavePoint:
    x: float
    y: float
    dx: float
    dy: float


@dataclass
class Wave:
    energy: float
    x: float
    y: float
    dir_x: float
    dir_y: float
    sigma: float = 40.0
    k: float = 0.4
    phase: float = 0.0
    a: float = 10.0
    points: list[WavePoint] = field(default_factory=list)

    def __post_init__(self) -> None:
        mag = math.hypot(self.dir_x, self.dir_y) or 1.0
        self.dir_x /= mag
        self.dir_y /= mag
        x = -self.sigma
        while x <= self.sigma:
            self.points.append(WavePoint(self.x + x * self.dir_x, self.y + x * self.dir_y, self.dir_x * 200.0, self.dir_y * 200.0))
            x += 0.1

    def draw(self) -> None:
        if self.energy == 4.2:
            glColor3f(1.0, 1.0, 0.0)
        else:
            glColor3f(0.0, 1.0, 1.0)
        glBegin(GL_LINE_STRIP)
        for p in self.points:
            px, py = -p.dy, p.dx
            mag = math.hypot(px, py) or 1.0
            px /= mag
            py /= mag
            y_disp = self.a * math.sin(self.k * math.hypot(p.x, p.y) - self.phase)
            glVertex2f(p.x + px * y_disp, p.y + py * y_disp)
        glEnd()

    def update(self, dt: float) -> None:
        self.phase += 30.0 * dt
        for p in self.points:
            p.x += p.dx * dt
            p.y += p.dy * dt

            if p.x < -400:
                p.x = -400
                p.dx *= -1
            if p.x > 400:
                p.x = 400
                p.dx *= -1
            if p.y < -300:
                p.y = -300
                p.dy *= -1
            if p.y > 300:
                p.y = 300
                p.dy *= -1


@dataclass
class Particle:
    x: float
    y: float
    charge: int
    theta: float = 0.0
    n: int = 1
    excite_timer: float = 0.0

    def __post_init__(self) -> None:
        self.theta = math.atan2(self.y, self.x)

    def draw(self) -> None:
        draw_circle(0.0, 0.0, math.hypot(self.x, self.y), 100)
        if self.charge == 1:
            glColor3f(1.0, 0.0, 0.0)
            draw_filled_circle(self.x, self.y, 10)
        elif self.charge == -1:
            glColor3f(0.0, 1.0, 1.0)
            draw_filled_circle(self.x, self.y, 2)
        else:
            glColor3f(0.5, 0.5, 0.5)
            draw_filled_circle(self.x, self.y, 10)

    def electron_update(self, speed: float, waves: list[Wave]) -> None:
        self.theta += speed
        self.x = (ORBITAL_RAD * (self.n * self.n)) * math.cos(self.theta)
        self.y = (ORBITAL_RAD * (self.n * self.n)) * math.sin(self.theta)
        if self.n != 1 and self.excite_timer <= 0.0:
            self.n -= 1
            self.excite_timer = 0.003
            dx = -1.0 if random.randint(0, 1) == 0 else 1.0
            dy = -1.0 if random.randint(0, 1) == 0 else 1.0
            waves.append(Wave(4.2, self.x, self.y, dx, dy))

    def check_absorption(self, wave: Wave) -> None:
        if wave.energy != 3.2:
            return
        for wp in wave.points:
            if math.hypot(self.x - wp.x, self.y - wp.y) < 20.0:
                wave.energy = 0.0
                self.n += 1
                self.excite_timer += 0.003
                break


def main() -> None:
    if not glfw.init():
        raise RuntimeError("failed to init glfw")
    window = glfw.create_window(WIDTH, HEIGHT, "Wave Atom 2D (python)", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("failed to create window")

    glfw.make_context_current(window)
    particles = [Particle(0, 0, 1), Particle(50, 50, -1)]
    waves: list[Wave] = []

    for _ in range(25):
        x = random.random() * 800.0 - 400.0
        y = random.random() * 600.0 - 300.0
        waves.append(Wave(3.2, x, y, y, x))

    last_t = time.perf_counter()
    while not glfw.window_should_close(window):
        now = time.perf_counter()
        dt = now - last_t
        last_t = now

        glClear(GL_COLOR_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-WIDTH / 2.0, WIDTH / 2.0, -HEIGHT / 2.0, HEIGHT / 2.0, -1.0, 1.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        for p in particles:
            if p.excite_timer > 0:
                p.excite_timer -= dt / 100.0
            p.draw()
            if p.charge == -1:
                p.electron_update(5.0 * dt, waves)
            for w in waves:
                p.check_absorption(w)

        for w in waves:
            if w.energy <= 0.0:
                continue
            w.draw()
            w.update(dt)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.destroy_window(window)
    glfw.terminate()


if __name__ == "__main__":
    main()
