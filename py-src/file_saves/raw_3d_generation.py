from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
import sys

import glfw
from OpenGL.GL import *
from OpenGL.GLU import gluLookAt, gluPerspective

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common_quantum import Camera, spherical_to_cartesian

A0 = 52.9
ELECTRON_R = 3.0
WIDTH, HEIGHT = 800, 600
camera = Camera(radius=500.0, zoom_speed=125.0)


@dataclass
class Particle:
    x: float
    y: float
    z: float
    r: float
    g: float
    b: float
    a: float


def radial_probability_1s(r: float) -> float:
    rb = r / A0
    return 4.0 * rb * rb * math.exp(-2.0 * rb)


def radial_probability_2s(r: float) -> float:
    rb = r / A0
    return 0.5 * (rb**2) * ((1.0 - rb / 2.0) ** 2) * math.exp(-rb)


def radial_probability_3s(r: float) -> float:
    x = r / A0
    rv = (1.0 - (2.0 / 3.0) * x + (2.0 / 27.0) * x * x) * math.exp(-x / 3.0)
    return rv * rv * r * r


def radial_probability_2p(r: float) -> float:
    rb = r / A0
    return (rb**4 / 24.0) * math.exp(-rb)


def radial_probability_3p(r: float) -> float:
    x = r / A0
    rv = x * (1.0 - x / 6.0) * math.exp(-x / 3.0)
    return rv * rv * r * r


def radial_probability_3d(r: float) -> float:
    x = r / A0
    c = 2.0 * math.sqrt(5.0) / 405.0
    rv = c * (x * x) / (A0**3) * math.exp(-x / 3.0)
    return (r * r) * (rv * rv)


def radial_probability_4f(r: float) -> float:
    x = r / A0
    c = 1.0 / (96.0 * math.sqrt(70.0) * (A0**3))
    rv = c * (x**3) * math.exp(-x / 4.0)
    return r * r * rv * rv


def sample_from_pdf(pdf, r_max: float, p_max: float) -> float:
    while True:
        r = random.random() * r_max
        y = random.random() * p_max
        if y <= pdf(r):
            return r


def sample_r1s() -> float:
    return sample_from_pdf(radial_probability_1s, 5.0 * A0, radial_probability_1s(A0))


def sample_r2s() -> float:
    return sample_from_pdf(radial_probability_2s, 10.0 * A0, radial_probability_2s(A0))


def sample_r3s() -> float:
    return sample_from_pdf(radial_probability_3s, 20.0 * A0, radial_probability_3s(3.0 * A0))


def sample_r2p() -> float:
    return sample_from_pdf(radial_probability_2p, 15.0 * A0, radial_probability_2p(4.0 * A0))


def sample_r3p() -> float:
    return sample_from_pdf(radial_probability_3p, 25.0 * A0, radial_probability_3p(8.0 * A0))


def sample_r3d() -> float:
    return sample_from_pdf(radial_probability_3d, 30.0 * A0, radial_probability_3d(9.0 * A0))


def sample_r4f() -> float:
    return sample_from_pdf(radial_probability_4f, 30.0 * A0, 0.00015)


def random_theta_phi() -> tuple[float, float]:
    theta = math.acos(1.0 - 2.0 * random.random())
    phi = 2.0 * math.pi * random.random()
    return theta, phi


def sample_orbital(name: str) -> Particle:
    if name == "1s":
        r = sample_r1s()
        theta, phi = random_theta_phi()
        col = (1.0, 0.0, 1.0, 1.0)
    elif name == "2s":
        r = sample_r2s()
        theta, phi = random_theta_phi()
        col = (0.0, 1.0, 1.0, 1.0)
    elif name == "3s":
        r = sample_r3s()
        theta, phi = random_theta_phi()
        col = (0.0, 1.0, 1.0, 1.0)
    elif name in {"2p_x", "2p_y", "2p_z"}:
        r = sample_r2p()
        while True:
            theta, phi = random_theta_phi()
            if name == "2p_x":
                prob = (math.sin(theta) ** 2) * (math.cos(phi) ** 2)
            elif name == "2p_y":
                prob = (math.sin(theta) ** 2) * (math.sin(phi) ** 2)
            else:
                prob = math.cos(theta) ** 2
            if random.random() <= prob:
                break
        col = (0.0, 1.0, 1.0, 1.0)
    elif name in {"3dxy", "3dxz", "3dyz", "3dx2y2", "3dz2"}:
        r = sample_r3d()
        while True:
            theta, phi = random_theta_phi()
            s = math.sin(theta)
            c = math.cos(theta)
            if name == "3dxy":
                prob = (s**4) * (math.sin(2.0 * phi) ** 2)
                col = (1.0, 0.15, 0.0, 1.0)
            elif name == "3dxz":
                prob = (s * s) * (c * c) * (math.cos(phi) ** 2)
                col = (0.0, 1.0, 0.3, 1.0)
            elif name == "3dyz":
                prob = (s * s) * (c * c) * (math.sin(phi) ** 2)
                col = (1.0, 0.0, 0.5, 1.0)
            elif name == "3dx2y2":
                prob = (s**4) * (math.cos(2.0 * phi) ** 2)
                col = (0.2, 0.7, 1.0, 1.0)
            else:
                t = 3.0 * c * c - 1.0
                prob = t * t
                col = (1.0, 0.3, 0.1, 1.0)
            if random.random() <= prob:
                break
    else:  # 4f_x3y2 default
        r = sample_r4f()
        while True:
            theta, phi = random_theta_phi()
            s = math.sin(theta)
            term = s * s * s * math.cos(phi) * (math.cos(phi) ** 2 - 3.0 * (math.sin(phi) ** 2))
            prob = term * term
            if random.random() <= prob:
                break
        col = (0.2, 1.0, 0.8, 1.0)

    x, y, z = spherical_to_cartesian(r, theta, phi)
    return Particle(x, y, z, *col)


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
    camera.radius = max(20.0, camera.radius - yoff * camera.zoom_speed)


def main() -> None:
    if not glfw.init():
        raise RuntimeError("failed to init glfw")
    win = glfw.create_window(WIDTH, HEIGHT, "Raw 3D Orbital Generation (python)", None, None)
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

    particles: list[Particle] = [sample_orbital("4f_x3y2") for _ in range(10000)]

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

        glPointSize(ELECTRON_R)
        glBegin(GL_POINTS)
        for p in particles:
            glColor4f(p.r, p.g, p.b, p.a)
            glVertex3f(p.x, p.y, p.z)
        glEnd()

        glfw.swap_buffers(win)
        glfw.poll_events()

    glfw.destroy_window(win)
    glfw.terminate()


if __name__ == "__main__":
    main()
