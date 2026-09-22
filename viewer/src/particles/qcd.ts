import { el, num } from "../format";

type Send = (cmd: Record<string, unknown>) => void;

type QcdEvent =
  | { type: "qcd.progress"; sweep: number; sweeps: number; plaquette: number; thermalised: boolean; measurements: number }
  | { type: "qcd.result"; beta: number; L: number; plaquette: number; R: number[]; V: number[];
      fit: { A: number; alpha: number; sigma: number }; measurements: number; seconds: number }
  | { type: "qcd.scan_start"; betas: number[]; L: number; Nt: number }
  | { type: "qcd.scan_row"; beta: number; polyakov: number; error: number }
  | { type: "qcd.hadrons"; beta: number; L: number; T: number; kappas: number[]; pion: number[]; rho: number[];
      kappa_c: number; rho_chiral: number; meff_pion: number[][]; meff_rho: number[][] }
  | { type: "qcd.hadron_progress"; done: number; total: number; seconds: number }
  | { type: "qcd.done"; cancelled?: boolean };

const PUBLISHED_PLAQUETTE: Record<number, number> = { 5.7: 0.54934, 5.8: 0.56771, 5.9: 0.58184, 6.0: 0.59368 };
const SQRT_SIGMA_MEV = 440;   // measured scale used only to convert lattice units to femtometres
const HBARC = 197.327;

// Gluons on a 4D grid of spacetime, from the QCD Lagrangian alone.
export class LatticeQCD {
  private mode: "confinement" | "deconfinement" | "hadrons" = "confinement";
  private hadrons: Extract<QcdEvent, { type: "qcd.hadrons" }> | null = null;
  private beta = 5.7;
  private L = 8;
  private plaq: number[] = [];
  private result: Extract<QcdEvent, { type: "qcd.result" }> | null = null;
  private scan: { beta: number; p: number; e: number }[] = [];

  private modes = el("div", { class: "beam-list", role: "radiogroup", "aria-label": "Experiment" });
  private blurb = el("p", { class: "blurb" });
  private betaSection = el("section", {});
  private betaChips = el("div", { class: "chips", role: "radiogroup", "aria-label": "Coupling" });
  private sizeChips = el("div", { class: "chips", role: "radiogroup", "aria-label": "Lattice size" });
  private runBtn = el("button", { class: "primary" }, "Run");
  private readout = el("div", { class: "readout" });
  private main = el("div", { class: "q-main" });
  private caption = el("p", { class: "u-axis-note" });
  private trace = el("div", { class: "u-chart" });
  private facts = el("dl", { class: "table" });
  private note = el("p", { class: "hint" });

  constructor(left: HTMLElement, stage: HTMLElement, right: HTMLElement, bar: HTMLElement, private send: Send) {
    this.runBtn.onclick = () => this.run();
    this.betaSection.append(el("h2", {}, "Coupling β"), this.betaChips,
      el("p", { class: "hint" }, "β = 6/g². Larger β means a finer lattice spacing."));
    left.append(
      el("section", {}, el("h2", {}, "Experiment"), this.modes, this.blurb),
      this.betaSection,
      el("section", {}, el("h2", {}, "Lattice"), this.sizeChips,
        el("p", { class: "hint" }, "Points along each side of the 4D spacetime box. Every link carries a 3×3 matrix: the gluon field.")),
      el("section", {}, el("h2", {}, "What goes in"),
        el("p", { class: "hint" },
          "The QCD Lagrangian for gluons, written on a grid (the Wilson action). Configurations are drawn with the probability the path integral gives them. This version leaves out quark loops (the quenched approximation).")),
    );
    stage.append(el("div", { class: "q-wrap" }, this.main, this.caption));
    right.append(
      el("section", {}, el("h2", {}, "Result"), this.facts, this.note),
      el("section", {}, el("h2", {}, "Thermalising"), this.trace,
        el("p", { class: "hint" }, "Average plaquette per sweep. It settles once the lattice has forgotten its starting point.")),
    );
    bar.append(el("div", { class: "run-group" }, this.runBtn), el("div", { class: "run-group end" }, this.readout));
    this.renderControls();
    this.renderMain();
  }

  onEvent(e: QcdEvent): void {
    switch (e.type) {
      case "qcd.progress":
        this.plaq.push(e.plaquette);
        this.readout.textContent = `Sweep ${e.sweep} of ${e.sweeps}. ${e.measurements} measurements.`;
        this.renderTrace();
        break;
      case "qcd.result":
        this.result = e;
        this.readout.textContent = `Done in ${num(e.seconds, 0)} s.`;
        this.runBtn.textContent = "Run";
        this.renderMain();
        this.renderFacts();
        break;
      case "qcd.scan_start":
        this.scan = [];
        this.renderMain();
        break;
      case "qcd.scan_row":
        this.scan.push({ beta: e.beta, p: e.polyakov, e: e.error });
        this.readout.textContent = `β = ${e.beta} done.`;
        this.renderMain();
        this.renderFacts();
        break;
      case "qcd.hadrons":
        this.hadrons = e;
        this.readout.textContent = `Lattice ${e.L}³×${e.T} at β = ${e.beta}.`;
        this.renderMain();
        this.renderFacts();
        break;
      case "qcd.hadron_progress":
        this.readout.textContent = `Gluon configuration ${e.done} of ${e.total}, ${num(e.seconds / 60, 0)} min so far.`;
        break;
      case "qcd.done":
        this.runBtn.textContent = "Run";
        if (this.mode === "deconfinement" && !e.cancelled) this.readout.textContent = "Scan complete.";
        break;
    }
  }

  private renderControls(): void {
    const opts: [typeof this.mode, string, string][] = [
      ["confinement", "Confinement", "Pull a quark and an antiquark apart and measure the energy it costs. If it keeps rising with distance, they can never be separated."],
      ["deconfinement", "Melting protons", "Heat the gluon field (a short time direction is a high temperature) and watch confinement switch off: a quark–gluon plasma."],
      ["hadrons", "Hadron masses", "Send quarks through the gluon field and time how their correlations fade. The rate is a mass: pions and rho mesons, from nothing but quarks and gluons."],
    ];
    this.modes.replaceChildren(...opts.map(([id, name]) => {
      const b = el("button", { class: "preset single", role: "radio", "aria-checked": String(id === this.mode) }, el("span", { class: "preset-name" }, name));
      b.onclick = () => { this.mode = id; this.renderControls(); this.renderMain(); this.renderFacts(); };
      return b;
    }));
    this.blurb.textContent = opts.find((o) => o[0] === this.mode)![2];
    this.betaSection.hidden = this.mode !== "confinement";
    this.sizeChips.parentElement!.hidden = this.mode === "hadrons";
    this.betaChips.replaceChildren(...[5.7, 5.8, 5.9, 6.0].map((b) => {
      const c = el("button", { role: "radio", "aria-checked": String(b === this.beta) }, b.toFixed(1));
      c.onclick = () => { this.beta = b; this.renderControls(); };
      return c;
    }));
    this.sizeChips.replaceChildren(...[6, 8, 10].map((n) => {
      const c = el("button", { role: "radio", "aria-checked": String(n === this.L) }, `${n}⁴`);
      c.onclick = () => { this.L = n; this.renderControls(); };
      return c;
    }));
  }

  private run(): void {
    this.plaq = [];
    this.runBtn.textContent = "Restart";
    this.readout.textContent = "Starting…";
    if (this.mode === "confinement") {
      this.result = null;
      this.send({ type: "qcd.run", mode: "confinement", beta: this.beta, L: this.L, sweeps: 90 });
    } else if (this.mode === "hadrons") {
      this.send({ type: "qcd.run", mode: "hadrons", fresh: this.hadrons !== null });
    } else {
      this.send({ type: "qcd.run", mode: "deconfinement", L: this.L, sweeps: 120 });
    }
    this.renderMain();
  }

  private svg(W: number, H: number) {
    const ns = "http://www.w3.org/2000/svg";
    const s = document.createElementNS(ns, "svg");
    s.setAttribute("viewBox", `0 0 ${W} ${H}`);
    s.setAttribute("class", "q-chart");
    const add = (tag: string, attrs: Record<string, string | number>, text?: string) => {
      const n = document.createElementNS(ns, tag);
      for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, String(v));
      if (text) n.textContent = text;
      s.append(n);
      return n;
    };
    return { s, add };
  }

  private renderMain(): void {
    const W = 560, H = 360, L = 56, B = 44, T = 20, R = 20;
    const { s, add } = this.svg(W, H);
    if (this.mode === "confinement") {
      const r = this.result;
      const Rmax = (this.L / 2) + 0.5;
      const Vs = r ? r.V.filter(Number.isFinite) : [];
      const vmax = Math.max(1.4, ...Vs) * 1.1;
      const x = (R_: number) => L + (R_ / Rmax) * (W - L - R);
      const y = (v: number) => H - B - (v / vmax) * (H - B - T);
      add("line", { x1: L, x2: W - R, y1: H - B, y2: H - B, class: "axis" });
      add("line", { x1: L, x2: L, y1: T, y2: H - B, class: "axis" });
      add("text", { x: (L + W - R) / 2, y: H - 10, class: "q-label", "text-anchor": "middle" }, "Distance between the quarks (lattice spacings)");
      add("text", { x: 14, y: (T + H - B) / 2, class: "q-label", "text-anchor": "middle", transform: `rotate(-90 14 ${(T + H - B) / 2})` }, "Energy of the pair");
      for (let k = 1; k <= this.L / 2; k++) add("text", { x: x(k), y: H - B + 16, class: "tick", "text-anchor": "middle" }, String(k));
      if (r && Number.isFinite(r.fit.sigma)) {
        const pts = [];
        for (let R_ = 0.6; R_ <= Rmax; R_ += 0.05) pts.push(`${x(R_)},${y(r.fit.A - r.fit.alpha / R_ + r.fit.sigma * R_)}`);
        add("polyline", { points: pts.join(" "), class: "q-fit" });
        const lin = [];
        for (let R_ = 0.6; R_ <= Rmax; R_ += 0.05) lin.push(`${x(R_)},${y(r.fit.A + r.fit.sigma * R_ - r.fit.alpha / Rmax)}`);
        add("polyline", { points: lin.join(" "), class: "q-linear" });
      }
      if (r) r.R.forEach((R_, i) => Number.isFinite(r.V[i]) && add("circle", { cx: x(R_), cy: y(r.V[i]), r: 5, class: "q-dot" }));
      this.caption.textContent = r
        ? "Dots: measured energy of a static quark–antiquark pair. Solid: Coulomb pull plus a straight rising line. Dashed: the straight part alone."
        : "Run the experiment to measure the energy between two quarks.";
    } else if (this.mode === "hadrons") {
      const h = this.hadrons;
      // x: bare quark mass a·m_q = 1/(2κ) − 1/(2κ_c); y: masses squared
      const kc = h?.kappa_c ?? 0.1695;
      const mq = (k: number) => 0.5 / k - 0.5 / kc;
      const xmax = 0.3;
      const ymax = 1.0;
      const x = (m: number) => L + (m / xmax) * (W - L - R);
      const y = (v: number) => H - B - (v / ymax) * (H - B - T);
      add("line", { x1: L, x2: W - R, y1: H - B, y2: H - B, class: "axis" });
      add("line", { x1: L, x2: L, y1: T, y2: H - B, class: "axis" });
      add("text", { x: (L + W - R) / 2, y: H - 10, class: "q-label", "text-anchor": "middle" }, "Quark mass (lattice units; zero where the pion becomes massless)");
      add("text", { x: 14, y: (T + H - B) / 2, class: "q-label", "text-anchor": "middle", transform: `rotate(-90 14 ${(T + H - B) / 2})` }, "Mass squared (lattice units)");
      for (const m of [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]) add("text", { x: x(m), y: H - B + 16, class: "tick", "text-anchor": "middle" }, m.toFixed(2));
      if (h) {
        const P = h.kappas.map((k, i) => [mq(k), h.pion[i] ** 2]);
        const Rr = h.kappas.map((k, i) => [mq(k), h.rho[i] ** 2]);
        add("polyline", { points: [[0, 0], ...P].map(([a, b]) => `${x(a)},${y(b)}`).join(" "), class: "q-linear" });
        add("polyline", { points: [[0, h.rho_chiral ** 2], ...Rr].map(([a, b]) => `${x(a)},${y(b)}`).join(" "), class: "q-linear" });
        for (const [a, b] of P) add("circle", { cx: x(a), cy: y(b), r: 5, class: "q-dot" });
        for (const [a, b] of Rr) add("rect", { x: x(a) - 4.5, y: y(b) - 4.5, width: 9, height: 9, class: "q-dot" });
        const last = P[P.length - 1], lastR = Rr[Rr.length - 1];
        add("text", { x: x(last[0]) + 10, y: y(last[1]) + 4, class: "q-label" }, "pion");
        add("text", { x: x(lastR[0]) + 10, y: y(lastR[1]) + 4, class: "q-label" }, "rho");
      }
      this.caption.textContent = h
        ? "Circles: pion. Squares: rho meson. As the quarks get lighter the pion's mass squared falls in a straight line to zero, while the rho stays heavy: the pion is the Goldstone boson of broken chiral symmetry."
        : "Run it to measure hadron masses. A stored result from a longer run loads first; running again computes a fresh, smaller one (about a quarter of an hour).";
    } else {
      const x = (b: number) => L + ((b - 5.4) / (6.25 - 5.4)) * (W - L - R);
      const pmax = Math.max(0.2, ...this.scan.map((p) => p.p + p.e)) * 1.1;
      const y = (p: number) => H - B - (p / pmax) * (H - B - T);
      add("line", { x1: L, x2: W - R, y1: H - B, y2: H - B, class: "axis" });
      add("line", { x1: L, x2: L, y1: T, y2: H - B, class: "axis" });
      add("text", { x: (L + W - R) / 2, y: H - 10, class: "q-label", "text-anchor": "middle" }, "β (higher = hotter at this lattice size)");
      add("text", { x: 14, y: (T + H - B) / 2, class: "q-label", "text-anchor": "middle", transform: `rotate(-90 14 ${(T + H - B) / 2})` }, "Polyakov loop |P|");
      for (const b of [5.5, 5.7, 5.9, 6.1]) add("text", { x: x(b), y: H - B + 16, class: "tick", "text-anchor": "middle" }, b.toFixed(1));
      const pts = [...this.scan].sort((a, b) => a.beta - b.beta);
      if (pts.length > 1) add("polyline", { points: pts.map((p) => `${x(p.beta)},${y(p.p)}`).join(" "), class: "q-fit" });
      for (const p of pts) {
        add("line", { x1: x(p.beta), x2: x(p.beta), y1: y(p.p - p.e), y2: y(p.p + p.e), class: "q-err" });
        add("circle", { cx: x(p.beta), cy: y(p.p), r: 5, class: "q-dot" });
      }
      this.caption.textContent = "|P| near zero: quarks are confined. It lifts off when the gluon field is hot enough to melt hadrons into a quark–gluon plasma.";
    }
    this.main.replaceChildren(s);
  }

  private renderTrace(): void {
    const W = 260, H = 90;
    const { s, add } = this.svg(W, H);
    s.setAttribute("class", "scan");
    if (this.plaq.length > 1) {
      const lo = Math.min(...this.plaq), hi = Math.max(...this.plaq);
      const pts = this.plaq.map((p, i) => `${6 + (i / (this.plaq.length - 1)) * (W - 12)},${H - 10 - ((p - lo) / Math.max(hi - lo, 1e-6)) * (H - 20)}`);
      add("polyline", { points: pts.join(" "), class: "trace-line" });
    }
    this.trace.replaceChildren(s);
  }

  private renderFacts(): void {
    if (this.mode === "confinement") {
      const r = this.result;
      if (!r) {
        this.facts.replaceChildren();
        this.note.textContent = "";
        return;
      }
      const a_fm = (Math.sqrt(Math.max(r.fit.sigma, 1e-9)) * HBARC) / SQRT_SIGMA_MEV;
      const pub = PUBLISHED_PLAQUETTE[r.beta];
      this.facts.replaceChildren(
        el("dt", {}, "String tension σa²"), el("dd", {}, num(r.fit.sigma, 3)),
        el("dt", {}, "Coulomb term α"), el("dd", {}, `${num(r.fit.alpha, 2)}`, el("span", { class: "ref" }, " theory π/12 = 0.26")),
        el("dt", {}, "Plaquette"), el("dd", {}, num(r.plaquette, 4), pub ? el("span", { class: "ref" }, ` published ${num(pub, 4)}`) : ""),
        el("dt", {}, "Lattice spacing"), el("dd", {}, `${num(a_fm, 3)} fm`),
        el("dt", {}, "Measurements"), el("dd", {}, String(r.measurements)),
      );
      this.note.textContent =
        `A string tension above zero means the energy grows without limit as the quarks separate: confinement. The spacing in femtometres uses one measured number, √σ ≈ ${SQRT_SIGMA_MEV} MeV. Small lattices and few measurements make σ read low by about a quarter.`;
    } else if (this.mode === "hadrons") {
      const h = this.hadrons;
      if (!h) { this.facts.replaceChildren(); this.note.textContent = ""; return; }
      this.facts.replaceChildren(
        ...h.kappas.flatMap((k, i) => [el("dt", {}, `κ = ${k}`),
          el("dd", {}, `π ${num(h.pion[i], 3)}  ρ ${num(h.rho[i], 3)}  ratio ${num(h.pion[i] / h.rho[i], 2)}`)]),
        el("dt", {}, "Pion massless at κ"), el("dd", {}, num(h.kappa_c, 4), el("span", { class: "ref" }, " published 0.1694")),
        el("dt", {}, "Rho there, m·a"), el("dd", {}, num(h.rho_chiral, 3), el("span", { class: "ref" }, " published ≈ 0.56")),
      );
      this.note.textContent =
        "Masses in units of the lattice spacing, quenched Wilson quarks. In nature the pion is 0.18 of the rho; reaching that needs lighter quarks, bigger lattices and quark loops, which is a question of computer time.";
    } else {
      const pts = [...this.scan].sort((a, b) => a.beta - b.beta);
      this.facts.replaceChildren(...pts.flatMap((p) => [el("dt", {}, `β = ${p.beta}`), el("dd", {}, `${num(p.p, 3)} ± ${num(p.e, 3)}`)]));
      this.note.textContent = "Published: for a time extent of 4 lattice spacings the transition sits at β ≈ 5.69, about 270 MeV in temperature for pure gluons.";
    }
  }
}
