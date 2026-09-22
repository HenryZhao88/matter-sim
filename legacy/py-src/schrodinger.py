# ------- import libs -------
import numpy as np
import json
from scipy.special import sph_harm, factorial
from math import sqrt
import scipy.special as sp
import os

a0 = 1.0


def R(n, l, r):
    rho = 2.0 * r / (n * a0)
    norm = sqrt((2.0 / (n * a0)) ** 3 * factorial(n - l - 1) / (2 * n * factorial(n + l)))
    lag = sp.assoc_laguerre(rho, n - l - 1, 2 * l + 1)
    return norm * np.exp(-rho / 2.0) * rho ** l * lag


def psi(n, l, m, r, theta, phi):
    y = sph_harm(m, l, phi, theta)
    return R(n, l, r) * y


def probability(n, l, m, r, theta, phi):
    return abs(psi(n, l, m, r, theta, phi)) ** 2


def sample_points(n, l, m, N=50000):
    points = []
    max_prob = None

    for _ in range(N):
        r = np.random.exponential(scale=n ** 2)
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2 * np.pi)

        p = probability(n, l, m, r, theta, phi)
        if max_prob is None or p > max_prob:
            max_prob = p

        if np.random.rand() < p / max_prob:
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)

            psi_val = psi(n, l, m, r, theta, phi)
            points.append({"x": x, "y": y, "z": z, "psi_re": psi_val.real, "psi_im": psi_val.imag})
    return points


def main():
    print("\n=== Hydrogen Orbital Generator (complex) ===\n")
    n = int(input("Enter n (1..7): "))
    l = int(input(f"Enter l (0..{n-1}): "))
    m = int(input(f"Enter m (-{l}..{l}): "))
    N = int(input("How many particle samples? (ex: 50000): "))

    print("\nGenerating samples... please wait...")
    pts = sample_points(n, l, m, N)

    out_dir = os.path.join(os.path.dirname(__file__), "orbitals")
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, f"orbital_n{n}_l{l}_m{m}_complex.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump({"n": n, "l": l, "m": m, "points": pts}, f)

    print(f"\nSaved {len(pts)} samples to: {filename}")


if __name__ == "__main__":
    main()
