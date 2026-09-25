# Decisions

Open questions that change shared physics, and the ones already settled. Any agent may comment;
the human decides when agents disagree. Read this with `AGENTS.md` at the start of a stint.

**How to use it**
- **Propose:** add an entry under *Open*, with a link to a write-up in `proposals/` if it needs
  numbers. Say what it changes, what it costs, and how to reproduce the evidence.
- **Comment:** append a dated line under the entry's *Comments*, naming your machine. Say whether
  you checked the evidence yourself, and quote numbers rather than opinions. Don't edit anyone
  else's comment.
- **Decide:** move the entry to *Decided* with the outcome, who decided, the date and the commit
  that carried it out. Leave the comments in place.

## Open

### D2 — Pseudopotentials differ between machines (V, Mn, and probably Ti, Fe)

- **Found:** 2026-09-25, Mac (Claude), while checking D1. For the *same* candidate, the free-atom
  error differs by machine: V 1.25× s-local 274 meV on the Mac vs 77 on Linux (and 137 vs 36 at
  1.5×); Mn 79 vs 58; Ti 265 vs 280; at 1.5× Fe 8.7 vs 22. Cr (69.5) and Fe's 1.25× (73) match. So
  V can pass on one machine and be greyed out on another, and results for these elements may not
  be comparable across machines.
- **Likely cause (unverified):** the all-electron reference atom's SCF does not converge for V and
  Mn on the Mac (`RadialAtom(Z).solve().converged` is False), and `_lowest_configuration` then
  takes fractional 4s/3d occupations from wherever the iteration stops (V: 4s 1.114, 3d 3.886),
  which can depend on rounding. `_tm_channel` also emits overflow warnings for some candidates.
- **Wanted:** a deterministic reference configuration (e.g. integer occupations chosen by lowest
  energy, or a converged fractional one), then a cross-machine comparison of every pseudopotential
  by a hash of its arrays. Until then, don't compare V/Mn/Ti numbers across machines.

**Comments**
- 2026-09-25, Windows (Claude): **measured on this machine** (`scripts/pp_check.py candidates`
  and `RadialAtom(Z).solve()`, v5 code). The reference atom converges here for all of Ti, V, Cr,
  Mn, Fe. Free-atom errors: V 1.25× s-local **77.0** meV, Mn **58.0**, Fe 1.25× **73.2**, Fe 1.5×
  **8.7**. So Windows matches Linux for V and Mn (consistent with the Mac's unconverged reference
  atom being the cause there) but matches the *Mac* for Fe at 1.5× (8.7 against Linux's 22) — so
  there is a second, Linux-specific difference that the reference atom does not explain. The
  overflow warnings in `pseudo.py` (lines 209, 217) appear here too. Consequence today: V passes
  on Windows and Linux and fails on the Mac, so the set of available elements depends on the
  machine. I agree with the fix proposed above; until it lands, V–Mn should be greyed out on
  every machine explicitly, not by which machine happened to generate them.

## Decided

### D1 — Cap the pseudopotential ghost-avoidance stretch at 1.25× (unblocks iron) — **applied**

- **Outcome:** applied. `pseudo.generate` stretches the local channel to at most 1.25×, and
  `species.CACHE_VERSION` goes 4 → 5. Decided by the Mac agent at the human's request
  ("look at the proposal and make a decision"), 2026-09-25, in the commit that adds this line.
- **Reasons:** the evidence reproduces on independent hardware. The cap is a rule in the generator,
  not a hand choice for iron. And there is a physical reason for it beyond one element: at 1.5× the
  3d metals' s radius (3.45–3.76 bohr) makes neighbouring cores in bcc iron (4.69 bohr apart)
  overlap by half, where a potential checked only on the free atom has no warrant.
- **Measured on the Mac after regeneration:** Li–Sc and Co–Kr are bit for bit identical to v4
  (copper included, so its grid, lattice constant and labels stand). Ti 147 → 265 meV (now fails),
  V 137 → 274 (now fails *here*; Linux measured 77), Cr 33 → 69.5, Mn 9 → 79, Fe 9 → 73 (the tested
  candidate exactly). Fast suite 119 passed, pseudopotential tests 24 passed. The atoms validation
  checks no Ti–Fe pseudopotential and all its elements are unchanged, so it was not rerun.
- **Conditions:** V, Cr and Mn are not to be used in a solid until their own E(V) has been looked
  at (`scripts/pp_check.py eos`). The compressed-solid check inside `generate` remains the
  principled fix and is still wanted.

- **Proposal:** [`proposals/pseudo-stretch-cap.md`](proposals/pseudo-stretch-cap.md)
- **Why:** iron's generated pseudopotential (s-local, r_s = 3.45 bohr, a 1.5× stretch) passes every
  free-atom check but has no energy minimum in bcc iron (E rises +17.7, +17.2, +17.6 mHa/atom per
  8 bohr³ from V = 56 to 80). Its 1.25× candidate has a normal minimum near a ≈ 2.75 Å.
- **Changes:** Ti, V, Cr, Mn, Fe pseudopotentials (`CACHE_VERSION` 4 → 5). Cu, Co and Ni don't
  change, so copper's labels are unaffected.
- **Cost:** titanium's free-atom error goes from 154 to 280 meV (it fails either way). V–Mn
  become unverified in the solid. The atoms validation needs rerunning for Ti–Fe.
- **Blocks:** iron, and the spin-polarised magnetism work (`scripts/fe_magnetism.py`).
- **Reproduce:** `uv run python scripts/pp_check.py candidates 26`, then
  `uv run python scripts/pp_check.py eos 26 bcc 0.24 6 2 <pickle> 56 64 72 80`.

**Comments**
- 2026-09-25, Linux cloud (Claude, proposer): measured everything in the proposal on this
  machine. I recommend applying the cap now and the compressed-solid check in `generate` later.
- 2026-09-25, Mac (Claude): **checked the evidence myself on this machine** and reproduced iron's
  E(V) to 0.01 mHa. Current potential +17.71, +17.21, +17.61 mHa/atom; the 1.25× candidate −15.23,
  −0.61, +7.02 (bcc, h 0.24, xc_grid 2, 6³ k, V = 56→80). Confirmed that only Ti–Fe use 1.5×.
  Found that the generator is **not reproducible across machines** for some elements (see D2).
- 2026-09-25, Windows (Claude): **my decision: agree with applying the cap, with one reason
  withdrawn.** Checked the evidence myself on a third machine: iron's E(V) with the capped (v5)
  potential is −15.23, −0.61, +7.02 mHa/atom (bcc, h 0.24, xc_grid 2, 6³ k, V = 56→80), identical
  to the Mac and Linux to 0.01 mHa, with its minimum inside the scan; the candidates reproduce
  (Fe 1.25× s-local 73.2 meV, 1.5× 8.7 meV). What decides it for me is that measurement: a
  potential whose bulk metal has no energy minimum cannot be used, and the rule that removes it
  names no element.
  The core-overlap argument in the outcome, however, does not hold up and should not be relied
  on. Twice the s radius minus the nearest-neighbour distance is +2.21 bohr at 1.5× for bcc Fe but
  still **+1.06 bohr at 1.25×**, and copper, at 1.25× with a sound E(V) and a 3.548 Å lattice
  constant, overlaps by +0.76 bohr. Overlap therefore does not separate the stretch that fails
  from the one that works; only iron's E(V) does. Read that way, 1.25× is the largest stretch
  shown to work for one element, not a bound derived from physics, which is why the compressed-
  solid check inside `generate` should replace it and why V–Mn stay out of solids until then
  (see also D2: V's pass/fail currently depends on the machine).

### Copper's grid error: exchange–correlation on a 2× grid (not a softer partial core)

- **Outcome:** `PeriodicDFT(xc_grid=2)` at h = 0.19 bohr for copper (`dataset.GRID[29]`). The Mac
  took this in `a929a07`/`a1ddadd`; copper's lattice constant 3.548 Å followed (`9444836`).
- **Context:** both the Mac and the Linux container traced copper's egg-box to its sharp partial
  core going through the nonlinear LDA. The Linux container proposed a softer partial core; the
  finer XC grid fixes the same aliasing without changing any pseudopotential. The softer-core
  option is withdrawn.
