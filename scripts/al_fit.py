"""Fit the learned potential to the DFT labels, check it against DFT, then run the experiments.

    uv run python scripts/al_fit.py [tag] [force_weight]

Stage 1 fits (MLX where available, then float64 least squares everywhere) and compares with
the DFT the potential is supposed to reproduce: lattice constant, bulk modulus, elastic
constants, and how far bcc sits above fcc. If those disagree, stop — the dynamics that follow
would only inherit the error.

Stage 2 measures the metal: thermal expansion, heat capacity, the melting point by solid-liquid
coexistence, and the latent heat. Results land in .cache/materials/al_results.json, which the
Materials workspace and `matter-sim validate --part materials` read.
"""
import glob, json, pickle, sys, time
from pathlib import Path
import numpy as np
from engine.materials.eam import fit, refine, energy_forces
from engine.crystal.periodic import cubic
from engine.materials.experiments import run_all
HA_EV = 27.211386245988; BOHR_A = 0.529177210903; GPA = 29421.02648438959
C = Path(__file__).resolve().parents[1] / ".cache" / "materials"
tag = sys.argv[1] if len(sys.argv) > 1 else "round1"
w_force = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
# only the labels computed with the converged, uniform k-mesh (dataset.K_SPACING = 45)
from engine.materials.dataset import K_SPACING
data = []
for f in sorted(glob.glob(str(C / "dft/*.pkl"))):
    d = pickle.loads(open(f, "rb").read())
    if d["converged"] and d.get("kspacing", K_SPACING) == K_SPACING:
        data.append(d)
print(len(data), "labels", flush=True)
rng = np.random.default_rng(1)
idx = rng.permutation(len(data)); nt = max(4, len(data) // 7)
test = [data[i] for i in idx[:nt]]; train = [data[i] for i in idx[nt:]]
t = time.time()
m = fit(train, iters=12000, e_scale_eV=0.3, log=lambda i, l: print("fit", i, f"{l:.3g}", f"{time.time()-t:.0f}s", flush=True))
m = refine(m, train, w_force=w_force, e_scale_eV=0.3, log=print)
def errs(ds):
    de, df = [], []
    for d in ds:
        E, F = energy_forces(m, d["cell"], d["positions"])
        de.append((E - d["energy"]) / len(d["positions"])); df.append((F - d["forces"]).ravel())
    return float(np.sqrt(np.mean(np.square(de))) * HA_EV * 1000), float(np.sqrt(np.mean(np.concatenate(df) ** 2)) * HA_EV / BOHR_A)
etr, ftr = errs(train); ete, fte = errs(test)
print(f"train E {etr:.1f} meV/atom F {ftr:.3f} eV/A | test E {ete:.1f} F {fte:.3f}", flush=True)
a0 = 3.955 / BOHR_A
def E_cell(a, strain=(0, 0, 0)):
    c = cubic("fcc", a, 13, strain=strain); return energy_forces(m, c.cell, c.positions)[0] / 4
aa = a0 * np.linspace(0.95, 1.05, 11)
c = np.polyfit(aa, [E_cell(a) for a in aa], 4)
r = np.roots(np.polyder(c)); r = r[np.isreal(r)].real; amin = float(r[np.argmin(np.abs(r - a0))])
B = 4 * np.polyval(np.polyder(c, 2), amin) / (9 * amin) * GPA
V = amin ** 3 / 4
eps = np.linspace(-0.02, 0.02, 9)
c11_c12 = np.polyfit(eps, [E_cell(amin, (x, x, -2 * x)) for x in eps], 2)[0] / V / 3 * GPA
C11 = B + 2 * c11_c12 / 3; C12 = B - c11_c12 / 3
bcc = cubic("bcc", (amin ** 3 / 2) ** (1 / 3), 13)
dE_bcc = (energy_forces(m, bcc.cell, bcc.positions)[0] / 2 - E_cell(amin)) * HA_EV * 1000
out = {"n_train": len(train), "n_test": len(test), "E_meV": ete, "F_eVA": fte, "E_meV_train": etr,
       "F_eVA_train": ftr, "a0_A": amin * BOHR_A, "B_GPa": float(B), "C11": float(C11), "C12": float(C12),
       "bcc_minus_fcc_meV": float(dE_bcc)}
print("EAM vs DFT (3.955 A, 84.5 GPa, C11 124, C12 65, bcc-fcc 108 meV):", out, flush=True)
(C / "al_eam_errors.json").write_text(json.dumps(out, indent=1))
m.save(C / f"al_eam_{tag}.npz")
res = run_all(m, amin * BOHR_A, log=lambda k, r: print(k, r, flush=True))
print("RESULT", json.dumps({k: v for k, v in res.items() if k != "thermal"}), flush=True)
