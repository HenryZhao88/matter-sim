from __future__ import annotations

import math
from dataclasses import dataclass

import glfw
from OpenGL.GL import *
from OpenGL.GLU import gluLookAt, gluPerspective

from common_quantum import Camera, inferno_color, probability_flow, sample_phi, sample_r, sample_theta, spherical_to_cartesian

WIDTH, HEIGHT = 800, 600
N = 100000
n, l, m = 2, 1, 0
ELECTRON_R = 1.5
DT = 0.5


@dataclass
class Particle:
    x: float
    y: float
    z: float
    r: float
    g: float
    b: float
    a: float


particles: list[Particle] = []
camera = Camera(radius=50.0, zoom_speed=10.0)


def generate_particles(count: int) -> None:
    particles.clear()
    for _ in range(max(1, count)):
        rr = sample_r(n, l)
        theta = sample_theta(l, m)
        phi = sample_phi()
        x, y, z = spherical_to_cartesian(rr, theta, phi)
        cr, cg, cb, ca = inferno_color(rr, theta, phi, n, l, m)
        particles.append(Particle(x, y, z, cr, cg, cb, ca))


def draw_grid(size: float = 500.0, divisions: int = 20) -> None:
    half = size / 2.0
    step = size / divisions
    glColor4f(1.0, 1.0, 1.0, 0.15)
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
    camera.radius = max(1.0, camera.radius - yoff * camera.zoom_speed)


def key_callback(window, key, scancode, action, mods):
    global n, l, m, N, ELECTRON_R
    if action not in (glfw.PRESS, glfw.REPEAT):
        return

    if key == glfw.KEY_W:
        n += 1
    elif key == glfw.KEY_S:
        n = max(1, n - 1)
    elif key == glfw.KEY_E:
        l += 1
    elif key == glfw.KEY_D:
        l = max(0, l - 1)
    elif key == glfw.KEY_R:
        m += 1
    elif key == glfw.KEY_F:
        m -= 1
    elif key == glfw.KEY_T:
        N += 100000
    elif key == glfw.KEY_G:
        N = max(1000, N - 100000)
    else:
        return

    l = min(max(0, l), n - 1)
    m = min(max(m, -l), l)
    ELECTRON_R = n / 3.0
    print(f"Quantum numbers updated: n={n} l={l} m={m} N={N}")
    generate_particles(N)


def main() -> None:
    if not glfw.init():
        raise RuntimeError("failed to init glfw")
    win = glfw.create_window(WIDTH, HEIGHT, "Atom Prob-Flow (python)", None, None)
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
    glfw.set_key_callback(win, key_callback)

    generate_particles(250000)
    print("Starting simulation...")

    while not glfw.window_should_close(win):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, WIDTH / HEIGHT, 0.1, 2000.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        cx, cy, cz = camera.position()
        gluLookAt(cx, cy, cz, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)

        draw_grid()

        glPointSize(max(1.0, ELECTRON_R))
        glBegin(GL_POINTS)
        for p in particles:
            rr = math.sqrt(p.x * p.x + p.y * p.y + p.z * p.z)
            if rr > 1e-6:
                theta = math.acos(max(-1.0, min(1.0, p.y / rr)))
                vx, vy, vz = probability_flow(p.x, p.y, p.z, m)
                tx, ty, tz = p.x + vx * DT, p.y + vy * DT, p.z + vz * DT
                new_phi = math.atan2(tz, tx)
                p.x, p.y, p.z = spherical_to_cartesian(rr, theta, new_phi)

            if p.x < 0 and p.y > 0:
                continue
            glColor4f(p.r, p.g, p.b, p.a)
            glVertex3f(p.x, p.y, p.z)
        glEnd()

        glfw.swap_buffers(win)
        glfw.poll_events()

    glfw.destroy_window(win)
    glfw.terminate()


if __name__ == "__main__":
    main()
