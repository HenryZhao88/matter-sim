"""Block LOBPCG eigensolver for the lowest eigenstates of a grid Hamiltonian.

Heavy work (H applications, linear combinations, Gram matrices) runs on the
backend (GPU for mlx). The small Rayleigh–Ritz problem runs on the CPU in
float64. The subspace is orthonormalised by eigen-decomposition of its Gram
matrix with numerical-rank truncation, which keeps LOBPCG stable in float32.
"""

from __future__ import annotations

import numpy as np

from ..core.backend import Backend
from ..core.grid import Grid


def _flat(A):
    return A.reshape(A.shape[0], -1)


def _gram(b: Backend, A, B) -> np.ndarray:
    return b.to_numpy(_flat(A) @ _flat(B).T)


def _combine(b: Backend, coefs: np.ndarray, S, shape):
    """Rows = coefs.T @ S (coefs is m x k, S has m rows)."""
    C = b.asarray(coefs.T)
    return (C @ _flat(S)).reshape((coefs.shape[1],) + shape)


def _normalize_rows(b: Backend, A):
    norms = b.xp.sqrt(b.xp.sum(_flat(A) * _flat(A), axis=1))
    norms = b.xp.maximum(norms, 1e-30)
    return A / norms.reshape((-1, 1, 1, 1))


def orthonormalize(b: Backend, X):
    G = _gram(b, X, X)
    s, U = np.linalg.eigh(0.5 * (G + G.T))
    T = U / np.sqrt(np.maximum(s, 1e-30))
    return _combine(b, T, X, X.shape[1:])


def teter_preconditioner(grid: Grid, e_ref: float = 1.0):
    """Kinetic-energy preconditioner (Teter, Payne & Allan 1989)."""
    b = grid.backend
    x = grid.k2_np * 0.5 / e_ref
    num = 27 + 18 * x + 12 * x ** 2 + 8 * x ** 3
    K = b.asarray(num / (num + 16 * x ** 4))

    def apply(R):
        return b.irfftn(K * b.rfftn(R), R.shape[-3:])

    return apply


def lobpcg(backend: Backend, apply_H, X, precond, tol: float = 1e-4, maxiter: int = 40,
           n_check: int | None = None):
    """Return (eigenvalues ndarray[nb], eigenvectors (nb, N, N, N), max residual).

    ``X`` is the starting block (any rank-full set). ``n_check`` bands must
    converge to residual norm < tol (default: all).
    """
    b = backend
    nb = X.shape[0]
    shape = X.shape[1:]
    n_check = nb if n_check is None else n_check
    rank_tol = 1e-5 if b.is_gpu else 1e-12

    X = orthonormalize(b, X)
    HX = apply_H(X)
    b.eval(X, HX)
    A = _gram(b, X, HX)
    lam, C = np.linalg.eigh(0.5 * (A + A.T))
    X = _combine(b, C, X, shape)
    HX = _combine(b, C, HX, shape)
    P = HP = None
    rmax = np.inf

    for _ in range(maxiter):
        R = HX - b.asarray(lam).reshape((-1, 1, 1, 1)) * X
        rn = np.sqrt(np.maximum(np.diag(_gram(b, R, R)), 0.0))
        rmax = float(rn[:n_check].max())
        if rmax < tol:
            break

        W = precond(R)
        W = W - _combine(b, _gram(b, X, W), X, shape)  # orthogonalise against X
        W = _normalize_rows(b, W)
        HW = apply_H(W)
        blocks, hblocks = [X, W], [HX, HW]
        if P is not None:
            blocks.append(P)
            hblocks.append(HP)
        S = b.xp.concatenate(blocks, axis=0)
        HS = b.xp.concatenate(hblocks, axis=0)
        b.eval(S, HS)

        G = _gram(b, S, S)
        G = 0.5 * (G + G.T)
        Hm = _gram(b, S, HS)
        Hm = 0.5 * (Hm + Hm.T)
        s, U = np.linalg.eigh(G)
        keep = s > rank_tol * s.max()
        T = U[:, keep] / np.sqrt(s[keep])
        e, Cr = np.linalg.eigh(T.T @ Hm @ T)
        coefs = T @ Cr[:, :nb]
        lam = e[:nb]

        coefs_p = coefs.copy()
        coefs_p[:nb] = 0.0
        X = _combine(b, coefs, S, shape)
        HX = _combine(b, coefs, HS, shape)
        P = _combine(b, coefs_p, S, shape)
        HP = _combine(b, coefs_p, HS, shape)
        pn = np.sqrt(np.maximum(np.diag(_gram(b, P, P)), 1e-30))
        P = P / b.asarray(pn).reshape((-1, 1, 1, 1))
        HP = HP / b.asarray(pn).reshape((-1, 1, 1, 1))
        b.eval(X, HX, P, HP)

    return np.asarray(lam), X, rmax
