import { BOHR_A, HARTREE_EV, el, num, sci, signed, sub } from "../format";
import type { ElementInfo, PresetInfo, ScfProgress, Snapshot } from "../types";

const SVG = "http://www.w3.org/2000/svg";
function svg<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number> = {}) {
  const e = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  return e;
}

const COMPONENT_NAMES: Record<string, string> = {
  kinetic: "Kinetic",
  electron_nuclear: "Electron–nucleus",
  hartree: "Electron–electron",
  exchange_correlation: "Exchange & correlation",
  nuclear_nuclear: "Nucleus–nucleus",
};

// Right rail: everything here is measured from the simulation's output.
export class MeasureRail {
  private energyValue = el("div", { class: "big-value" });
  private energySub = el("div", { class: "big-sub" });
  private energyRef = el("div", { class: "ref-row" });
  private components = el("dl", { class: "table" });
  private shape = el("dl", { class: "table" });
  private shapeNote = el("p", { class: "hint" });
  private levels = svg("svg", { class: "levels", viewBox: "0 0 260 200", role: "img" });
  private levelsNote = el("p", { class: "hint" });
  private trace = svg("svg", { class: "trace", viewBox: "0 0 260 64", preserveAspectRatio: "none", role: "img" });
  private traceNote = el("p", { class: "hint" });
  private elements = new Map<number, ElementInfo>();

  constructor(root: HTMLElement) {
    root.append(
      el("section", {}, el("h2", {}, "Energy"), this.energyValue, this.energySub, this.energyRef, this.components),
      el("section", {}, el("h2", {}, "Shape"), this.shape, this.shapeNote),
      el("section", {}, el("h2", {}, "Energy levels"), this.levels, this.levelsNote),
      el("section", {}, el("h2", {}, "History"), this.trace, this.traceNote),
    );
  }

  setElements(list: ElementInfo[]): void {
    for (const e of list) this.elements.set(e.Z, e);
  }

  update(snap: Snapshot, preset: PresetInfo | null): void {
    const m = snap.meta;
    const ref = preset?.reference;

    // ---- energy
    const E = m.total_energy ?? m.energy;
    this.energyValue.replaceChildren(num(E, 5), el("span", { class: "unit" }, " Ha"));
    this.energySub.textContent = E !== undefined ? `${num(E * HARTREE_EV, 3)} eV` : "";
    if (ref?.energy_ha !== undefined && E !== undefined) {
      const d = (E - ref.energy_ha) * 1000;
      this.energyRef.replaceChildren(
        el("span", {}, `Reference ${num(ref.energy_ha, 5)} Ha`),
        el("span", { class: "delta" }, `${signed(d, 1)} mHa`));
      this.energyRef.title = ref.note ?? "";
    } else this.energyRef.replaceChildren();
    const rows: (Node | string)[] = [];
    for (const [k, v] of Object.entries(m.components ?? {})) {
      if (v === 0 && k !== "kinetic") continue;
      rows.push(el("dt", {}, COMPONENT_NAMES[k] ?? k), el("dd", {}, num(v, 4)));
    }
    this.components.replaceChildren(...rows);

    // ---- shape (pure measurement from nuclear positions)
    this.renderShape(snap, ref);

    // ---- levels
    this.renderLevels(snap);

    // ---- history
    this.renderTrace(snap);
  }

  private label(i: number, Z: number): string {
    return `${this.elements.get(Z)?.symbol ?? "?"}${sub(i + 1)}`;
  }

  private renderShape(snap: Snapshot, ref: PresetInfo["reference"] | undefined): void {
    const atoms = snap.meta.atoms;
    const P = atoms.map((a) => a.pos.map((x) => x * BOHR_A));
    const d = (i: number, j: number) => Math.hypot(P[i][0] - P[j][0], P[i][1] - P[j][1], P[i][2] - P[j][2]);
    const pairs: { i: number; j: number; r: number }[] = [];
    for (let i = 0; i < P.length; i++) for (let j = i + 1; j < P.length; j++) pairs.push({ i, j, r: d(i, j) });
    pairs.sort((a, b) => a.r - b.r);
    const refD = [...(ref?.distances_A ?? [])].sort((a, b) => a - b);
    const rows: (Node | string)[] = [];
    pairs.slice(0, 6).forEach((p, k) => {
      const dd = el("dd", {}, `${num(p.r, 3)} Å`);
      if (refD[k] !== undefined) dd.append(el("span", { class: "ref" }, ` ref ${num(refD[k], 3)}`));
      rows.push(el("dt", {}, `${this.label(p.i, atoms[p.i].Z)}–${this.label(p.j, atoms[p.j].Z)}`), dd);
    });
    const angleAt = (c: number, a: number, b: number) => {
      const u = P[a].map((x, k) => x - P[c][k]);
      const v = P[b].map((x, k) => x - P[c][k]);
      const cos = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / (Math.hypot(...u) * Math.hypot(...v));
      return (Math.acos(Math.max(-1, Math.min(1, cos))) * 180) / Math.PI;
    };
    const refA = ref?.angles_deg?.[0];
    const refSpan = (dd: HTMLElement) => {
      if (refA !== undefined) dd.append(el("span", { class: "ref" }, ` ref ${num(refA, 1)}`));
      return dd;
    };
    if (P.length >= 3) {
      // Angles around the most central nucleus (smallest summed distance to the others).
      const sumD = (k: number) => P.reduce((acc, _, j) => acc + (j === k ? 0 : d(k, j)), 0);
      const centre = [...Array(P.length).keys()].reduce((best, k) => (sumD(k) < sumD(best) ? k : best), 0);
      const others = [...Array(P.length).keys()].filter((k) => k !== centre);
      const tri: [number, number, number][] = [];
      for (let a = 0; a < others.length; a++)
        for (let b = a + 1; b < others.length; b++) tri.push([others[a], others[b], angleAt(centre, others[a], others[b])]);
      if (tri.length <= 3) {
        for (const [i, j, ang] of tri)
          rows.push(el("dt", {}, `∠ ${this.label(i, atoms[i].Z)}${this.label(centre, atoms[centre].Z)}${this.label(j, atoms[j].Z)}`),
            refSpan(el("dd", {}, `${num(ang, 1)}°`)));
      } else {
        const angs = tri.map((t) => t[2]);
        rows.push(el("dt", {}, `Angles at ${this.label(centre, atoms[centre].Z)}`),
          refSpan(el("dd", {}, `${num(Math.min(...angs), 1)}–${num(Math.max(...angs), 1)}°`)));
      }
    }
    this.shape.replaceChildren(...rows);
    const fmax = Math.max(0, ...atoms.map((a) => Math.hypot(...a.force)));
    const mode = snap.meta.mode;
    this.shapeNote.textContent =
      P.length < 2 ? "One nucleus: nothing to measure yet." :
      mode === "relax" ? (snap.meta.relaxed ? `Settled. Largest force ${sci(fmax)} Ha/bohr.` : `Still settling. Largest force ${sci(fmax)} Ha/bohr.`) :
      mode === "dynamics" ? `t = ${num(snap.meta.time_fs, 2)} fs` : "Nuclei held in place.";
    if (ref?.note && P.length >= 2) this.shapeNote.textContent += ` Reference: ${ref.note}.`;
    if (ref?.method_note && P.length >= 2) this.shapeNote.textContent += ` ${ref.method_note}`;
  }

  private renderLevels(snap: Snapshot): void {
    const o = snap.meta.orbitals;
    this.levels.replaceChildren();
    if (!o) return;
    const all = [...o.up.energy, ...o.down.energy];
    const occE = [...o.up.energy.filter((_, i) => o.up.occ[i] > 0.01), ...o.down.energy.filter((_, i) => o.down.occ[i] > 0.01)];
    if (!all.length) return;
    // Linear in eV, from the deepest level to a little above the highest filled one.
    const f = (e: number) => e * HARTREE_EV;
    const top = occE.length ? Math.max(...occE) : Math.min(...all);
    const empty = all.filter((e) => e > top + 1e-4);
    const lumo = empty.length ? Math.min(...empty) : top;
    const lo = f(Math.min(...all)) - 1.5;
    const hi = f(Math.min(Math.max(lumo, top) + 0.08, top + 0.6)) + 1.5;
    // ticks every 2, 5 or 10 eV
    const stepEv = [2, 5, 10, 20, 50].find((s) => (hi - lo) / s <= 6) ?? 100;
    for (let v = Math.ceil(lo / stepEv) * stepEv; v <= hi; v += stepEv) {
      const yy = 186 - ((v - lo) / (hi - lo)) * 172;
      this.levels.append(svg("line", { x1: 40, x2: 44, y1: yy, y2: yy, class: "tickmark" }));
      const t = svg("text", { x: 36, y: yy + 3, class: "tick", "text-anchor": "end" });
      t.textContent = `${v === 0 ? "0" : num(v, 0)}`;
      this.levels.append(t);
    }
    this.levels.append(svg("line", { x1: 44, x2: 44, y1: 14, y2: 186, class: "axis" }));
    const unit = svg("text", { x: 36, y: 8, class: "tick", "text-anchor": "end" });
    unit.textContent = "eV";
    this.levels.append(unit);
    const y = (e: number) => 186 - ((f(e) - lo) / (hi - lo)) * 172;
    // zero line
    if (lo < 0 && hi > 0) this.levels.append(svg("line", { x1: 44, x2: 256, y1: y(0), y2: y(0), class: "zero" }));
    const column = (levels: { energy: number[]; occ: number[] }, x0: number, cls: string, arrow: string) => {
      // group near-degenerate levels side by side
      const groups: { e: number; occ: number[] }[] = [];
      levels.energy.forEach((e, i) => {
        if (f(e) > hi) return;
        const g = groups.find((gg) => Math.abs(gg.e - e) < 2e-3);
        if (g) g.occ.push(levels.occ[i]);
        else groups.push({ e, occ: [levels.occ[i]] });
      });
      for (const g of groups) {
        const w = 18, gap = 5;
        const total = g.occ.length * w + (g.occ.length - 1) * gap;
        g.occ.forEach((occ, k) => {
          const x = x0 + 44 - total / 2 + k * (w + gap);
          this.levels.append(svg("line", { x1: x, x2: x + w, y1: y(g.e), y2: y(g.e), class: `level ${cls}` }));
          if (occ > 0.01) {
            const t = svg("text", { x: x + w / 2, y: y(g.e) - 3, class: `arrow ${cls}`, "text-anchor": "middle", opacity: Math.max(0.25, occ).toFixed(2) });
            t.textContent = arrow;
            this.levels.append(t);
          }
        });
      }
    };
    column(o.up, 56, "up", "↑");
    column(o.down, 156, "down", "↓");
    for (const [x, s] of [[100, "spin ↑"], [200, "spin ↓"]] as const) {
      const t = svg("text", { x, y: 198, class: "tick", "text-anchor": "middle" });
      t.textContent = s;
      this.levels.append(t);
    }
    const homo = top;
    this.levelsNote.textContent = empty.length
      ? `Highest filled ${num(homo * HARTREE_EV, 2)} eV. Gap to the next empty level ${num((lumo - homo) * HARTREE_EV, 2)} eV.`
      : `Highest filled ${num(homo * HARTREE_EV, 2)} eV.`;
  }

  private renderTrace(snap: Snapshot): void {
    const tr = snap.meta.energy_trace.filter((x): x is number => x !== null);
    this.trace.replaceChildren();
    if (tr.length < 2) {
      this.traceNote.textContent = snap.meta.mode === "frozen" ? "Energy per step appears here when nuclei move." : "Collecting steps…";
      return;
    }
    const last = tr[tr.length - 1];
    const lo = Math.min(...tr), hi = Math.max(...tr);
    const span = Math.max(hi - lo, 1e-6);
    const pts = tr.map((e, i) => `${(i / (tr.length - 1)) * 260},${58 - ((e - lo) / span) * 52}`).join(" ");
    this.trace.append(svg("polyline", { points: pts, class: "trace-line" }));
    this.traceNote.textContent = `${tr.length} steps. Range ${num(span * HARTREE_EV * 1000, 1)} meV, now ${num(last, 5)} Ha.`;
  }

  progress(p: ScfProgress): string {
    return `Solving for the electrons: iteration ${p.iter}, density change ${sci(p.drho)}`;
  }
}
