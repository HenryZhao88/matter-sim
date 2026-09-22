from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import glfw
from OpenGL.GL import *
from OpenGL.GLU import gluLookAt, gluPerspective

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common_quantum import Camera

WIDTH, HEIGHT = 800, 600
BOHR_TO_PM = 52.9
camera = Camera(radius=500.0, zoom_speed=125.0)


class Particle:
    def __init__(self, x: float, y: float, z: float, size: float = 2.0):
        self.x = x
        self.y = y
        self.z = z
        self.size = size


def load_wavefunction(filename: str) -> list[Particle]:
    candidates = [
        Path(__file__).resolve().parents[2] / "src" / "orbitals" / filename,
        Path(__file__).resolve().parents[2] / "orbitals" / filename,
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(filename)

    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    pts: list[Particle] = []
    for p in obj.get("points", []):
        x = float(p[0]) * BOHR_TO_PM
        y = float(p[1]) * BOHR_TO_PM
        z = float(p[2]) * BOHR_TO_PM
        pts.append(Particle(x, y, z, 2.0))
    return pts


def draw_grid(size: float = 500.0, divisions: int = 20) -> None:
    half = size / 2.0
    step = size / divisions
    glColor4f(1.0, 1.0, 1.0, 0.05)
    glBegin(GL_LINES)
    for i in range(divisions + 1):
        v = -half + i * step
        glVertex3f(-half, 0.0, v)
        glVertex3f(half, 0.0, v)
        glVertex3f(v, 0.0, -half)
        glVertex3f(v, 0.0, half)
    glEnd()


def mouse_button_callback(window, button, action, mods):
    if button in (glfw.MOUSE_BUTTON_LEFT, glfw.MOUSE_BUTTON_MIDDLE):
        if action == glfw.PRESS:
            camera.dragging = True
            camera.last_x, camera.last_y = glfw.get_cursor_pos(window)
        elif action == glfw.RELEASE:
            camera.dragging = False


def cursor_callback(window, x, y):
    dx = x - camera.last_x
    dy = y - camera.last_y
    if camera.dragging:
        camera.azimuth += dx * camera.orbit_speed
        camera.elevation -= dy * camera.orbit_speed
    camera.last_x, camera.last_y = x, y


def scroll_callback(window, xoff, yoff):
    camera.radius = max(10.0, camera.radius - yoff * camera.zoom_speed)


def main() -> None:
    if not glfw.init():
        raise RuntimeError("failed to init glfw")
    win = glfw.create_window(WIDTH, HEIGHT, "Orbital Visualizer Raw (python)", None, None)
    if not win:
        glfw.terminate()
        raise RuntimeError("failed to create window")

    glfw.make_context_current(win)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    glfw.set_mouse_button_callback(win, mouse_button_callback)
    glfw.set_cursor_pos_callback(win, cursor_callback)
    glfw.set_scroll_callback(win, scroll_callback)

    particles = load_wavefunction("orbital_n3_l2_m2.json")

    while not glfw.window_should_close(win):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, WIDTH / HEIGHT, 0.1, 10000.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        cx, cy, cz = camera.position()
        gluLookAt(cx, cy, cz, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)

        draw_grid()

        glPointSize(2.0)
        glColor4f(1.0, 1.0, 0.0, 1.0)
        glBegin(GL_POINTS)
        for p in particles:
            glVertex3f(p.x, p.y, p.z)
        glEnd()

        glfw.swap_buffers(win)
        glfw.poll_events()

    glfw.destroy_window(win)
    glfw.terminate()


if __name__ == "__main__":
    main()
