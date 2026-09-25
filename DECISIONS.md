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

### D1 — Cap the pseudopotential ghost-avoidance stretch at 1.25× (unblocks iron)

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

## Decided

### Copper's grid error: exchange–correlation on a 2× grid (not a softer partial core)

- **Outcome:** `PeriodicDFT(xc_grid=2)` at h = 0.19 bohr for copper (`dataset.GRID[29]`). The Mac
  took this in `a929a07`/`a1ddadd`; copper's lattice constant 3.548 Å followed (`9444836`).
- **Context:** both the Mac and the Linux container traced copper's egg-box to its sharp partial
  core going through the nonlinear LDA. The Linux container proposed a softer partial core; the
  finer XC grid fixes the same aliasing without changing any pseudopotential. The softer-core
  option is withdrawn.
