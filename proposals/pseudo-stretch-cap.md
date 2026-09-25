# Proposal: cap the pseudopotential ghost-avoidance stretch at 1.25×

**Status:** applied 2026-09-25 — see `DECISIONS.md` (D1).
**Author:** Claude, Linux cloud container, 2026-09-25. Branch `claude/jolly-carson-f8nwao`.
**Touches:** `engine/atoms/pseudo.py` (`generate`), `engine/atoms/species.py` (`CACHE_VERSION`).

## The problem

Iron's generated pseudopotential passes every free-atom check (ghost-free, 22 meV transferability)
and collapses in the solid. Non-magnetic bcc Fe, h = 0.24 bohr, `xc_grid = 2`, 6³ k, energy per
atom stepping 8 bohr³ at a time from V = 56 to 80 bohr³/atom:

| Fe pseudopotential | E steps (mHa/atom) | Shape |
|---|---|---|
| current: s-local, r_s = 3.45 (1.5× base) | +17.7, +17.2, +17.6 | concave, no minimum |
| s-local, r_s = 2.88 (1.25×) | −15.2, −0.6, +7.0, +10.9 | convex, minimum a ≈ 2.75 Å |
| s-local, r_s = 2.30 (1.0×) | −20.9, −4.1, +4.8 | convex, minimum a ≈ 2.75 Å |

At h = 0.30 the current potential's curve stays concave out to V = 170 bohr³/atom, and at V = 60 it
sits 7.5 eV/atom below the free atom (measured cohesive energy 4.3 eV; LDA typically ~6.3). It
is not the grid (same at h = 0.24 with `xc_grid = 2`), not the partial core (a softer core collapses
identically) and not a ghost band (both potentials' band structures agree to 0.02 Ha).

**Why the generator picks it.** `generate` tries the default local channel, then the others,
stretching the local channel's radius 1.0×, 1.25×, 1.5× until a candidate is ghost-free. It
returns the first ghost-free one under 50 meV, else the best ghost-free one. For iron only the
1.5× s-local candidate gets under 50 meV. The free-atom test rewards a softer potential and never
compresses the atom, so it can't see what happens where neighbouring atoms overlap.

## The change

Drop 1.5 from the stretch loop in `pseudo.generate`:

```python
for scale in (1.0, 1.25):          # was (1.0, 1.25, 1.5)
```

Bump `species.CACHE_VERSION` (4 → 5) so both machines regenerate. Everything else is the
generator's existing rule; no element is chosen by hand.

## What it changes (measured here)

Only the elements whose current pseudopotential used the 1.5× stretch change. Of K–Kr that is
exactly **Ti, V, Cr, Mn, Fe**. Co and Ni are at 1.0× and Cu at 1.25×, so they are unaffected,
and so are copper's grid, lattice constant (3.548 Å) and any copper labels.

| Z | now: stretch, free-atom error | under the cap | limit 150 meV |
|---|---|---|---|
| 22 Ti | 1.5×, 154 meV (already fails) | 1.25× p-local, **280 meV** | fails, worse |
| 23 V | 1.5×, 36 | 1.25× s-local, 77 | passes |
| 24 Cr | 1.5×, 33 | 1.25× s-local, 69.5 | passes |
| 25 Mn | 1.5×, 21 | 1.25× s-local, 58 | passes |
| 26 Fe | 1.5×, 22 | 1.25× s-local (r_s 2.88), 73 | passes |

(`scripts/pp_check.py candidates Z` reproduces each row.)

## Costs and risks

- **Titanium gets worse** on its own check (154 → 280 meV). It fails today too, so it stays greyed
  out either way.
- **V, Cr, Mn become softer-checked but unverified in the solid.** Only Fe (and Cu, unchanged) have
  had their E(V) looked at. The cap is evidence from one failing element, not a proof.
- **The atoms validation changes for Ti–Fe** (ionisation energies and the like): rerun
  `uv run matter-sim validate --part atoms` and report any row that moves.
- **Regeneration** of every pseudopotential on each machine after the version bump (deterministic,
  so unaffected elements come back identical; a few minutes per transition metal).

## Alternatives considered

- **Prefer the hardest ghost-free candidate within the 150 meV limit.** It would also fix iron
  (r_s = 2.30), but it changes copper (1.25× → 1.0×, 53 → 129 meV) under running labels.
- **An override for iron only.** It's the smallest blast radius, but it's a hand choice for one
  element, and V–Mn would keep the construction that failed for iron.
- **A compressed-solid check inside `generate`** (reject a candidate whose bcc/fcc E(V) has no
  minimum). It's the principled fix, but it runs a crystal calculation inside pseudopotential
  generation: minutes per candidate, and a crystal-code dependency in `atoms/`. It's worth doing
  eventually; the cap is the cheap step that unblocks iron now.

## After it lands

Check E(V) for Fe (and ideally V, Cr, Mn) with `scripts/pp_check.py eos`. Converge iron's grid
like copper's (egg-box and strain energies against h). Then run `scripts/fe_magnetism.py`
(spin-polarised, all phases; clear `.cache/fe_magnetism` first).
