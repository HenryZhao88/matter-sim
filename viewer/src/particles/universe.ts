import { el, num, sci } from "../format";

type Send = (cmd: Record<string, unknown>) => void;

interface Frame {
  t: number;
  number: number[];
  charge: number[];
  field: number[];
  particles: number;
  energy_drift: number;
}

type LatticeEvent =
  | { type: "lattice.hello"; scenarios: Record<string, { name: string; blurb: string }> }
  | { type: "lattice.start"; scenario: string; N: number; mass: number; dim: number; frames: number; t_max: number; vacuum_particles: number }
  | ({ type: "lattice.frame" } & Frame)
  | { type: "lattice.done"; cancelled?: boolean; seconds?: number };

const STRENGTH: Record<string, { label: string; values: number[]; initial: number; hint: string }> = {
  pair_creation: { label: "Field strength", values: [0.5, 1, 1.5, 2], initial: 1,
    hint: "The electric field switched on at the start, in units of one charge's field." },
  string_breaking: { label: "Charge on each end", values: [0.5, 1, 1.5], initial: 1,
    hint: "The two fixed charges holding the string, in units of the particle's charge." },
  collision: { label: "Collision speed", values: [0.6, 0.8, 1, 1.15], initial: 1,
    hint: "Momentum of each incoming meson, as a fraction of the fastest this lattice allows." },
};

// A universe with one dimension of space, evolved exactly. Time runs up the page.
export class Universe {
  private scenarios: Record<string, { name: string; blurb: string }> = {};
  private scenario = "pair_creation";
  private mass = 0.4;
  private N = 18;
  private strength = 1;
  private frames: Frame[] = [];
  private meta: Extract<LatticeEvent, { type: "lattice.start" }> | null = null;
  private ready = false;

  private list = el("div", { class: "beam-list", role: "radiogroup", "aria-label": "Scenario" });
  private blurb = el("p", { class: "blurb" });
  private massChips = el("div", { class: "chips", role: "radiogroup", "aria-label": "Mass" });
  private strengthTitle = el("h2", {});
  private strengthChips = el("div", { class: "chips", role: "radiogroup" });
  private strengthHint = el("p", { class: "hint" });
  private sizeChips = el("div", { class: "chips", role: "radiogroup", "aria-label": "Universe size" });
  private sizeHint = el("p", { class: "hint" });
  private runBtn = el("button", { class: "primary" }, "Run");
  private readout = el("div", { class: "readout" });
  private matter = document.createElement("canvas");
  private field = document.createElement("canvas");
  private chart = el("div", { class: "u-chart" });
  private facts = el("dl", { class: "table" });

  constructor(left: HTMLElement, stage: HTMLElement, right: HTMLElement, bar: HTMLElement, private send: Send) {
    this.runBtn.onclick = () => this.run();
    left.append(
      el("section", {}, el("h2", {}, "Experiment"), this.list, this.blurb),
      el("section", {}, el("h2", {}, "Mass of the particles"), this.massChips,
        el("p", { class: "hint" }, "In units of the coupling. Light particles are easy to create out of the vacuum; heavy ones are not.")),
      el("section", {}, this.strengthTitle, this.strengthChips, this.strengthHint),
      el("section", {}, el("h2", {}, "Size of the universe"), this.sizeChips, this.sizeHint),
    );
    this.matter.className = "u-canvas";
    this.field.className = "u-canvas";
    stage.append(
      el("div", { class: "u-panels" },
        el("figure", { class: "u-panel" }, this.matter,
          el("figcaption", {}, "Charge added to the vacuum: ", el("span", { class: "u-key plus" }, "positive"), " and ", el("span", { class: "u-key minus" }, "negative"))),
        el("figure", { class: "u-panel" }, this.field,
          el("figcaption", {}, "Electric field along the line"))),
      el("p", { class: "u-axis-note" }, "Space runs across, time runs up."),
    );
    right.append(
      el("section", {}, el("h2", {}, "Particles above the vacuum"), this.chart),
      el("section", {}, el("h2", {}, "The whole state"), this.facts),
      el("section", {}, el("h2", {}, "What goes in"),
        el("p", { class: "hint" },
          "The Hamiltonian of electromagnetism in one dimension of space, on a lattice: electrons and positrons hopping between sites, and the energy stored in the electric field between them. Gauss's law fixes the field from the charges. Nothing says when particles appear.")),
    );
    bar.append(el("div", { class: "run-group" }, this.runBtn), el("div", { class: "run-group end" }, this.readout));
    new ResizeObserver(() => this.draw()).observe(stage);
  }

  activate(): void {
    if (!this.ready) this.send({ type: "lattice.hello" });
  }

  onEvent(e: LatticeEvent): void {
    switch (e.type) {
      case "lattice.hello":
        this.scenarios = e.scenarios;
        this.ready = true;
        this.renderControls();
        break;
      case "lattice.start":
        this.meta = e;
        this.frames = [];
        this.renderFacts();
        break;
      case "lattice.frame":
        this.frames.push(e);
        this.draw();
        this.renderChart();
        this.readout.textContent = `t = ${num(e.t, 1)} of ${num(this.meta?.t_max ?? 0, 0)}`;
        break;
      case "lattice.done":
        this.runBtn.textContent = "Run";
        if (!e.cancelled) this.readout.textContent = `Done in ${num(e.seconds ?? 0, 1)} s.`;
        this.renderFacts();
        break;
    }
  }

  private chips(root: HTMLElement, values: number[], current: number, onPick: (v: number) => void, fmt = (v: number) => String(v)) {
    root.replaceChildren(...values.map((v) => {
      const b = el("button", { role: "radio", "aria-checked": String(v === current) }, fmt(v));
      b.onclick = () => { onPick(v); this.renderControls(); };
      return b;
    }));
  }

  private renderControls(): void {
    this.list.replaceChildren(...Object.entries(this.scenarios).map(([id, s]) => {
      const b = el("button", { class: "preset single", role: "radio", "aria-checked": String(id === this.scenario) },
        el("span", { class: "preset-name" }, s.name));
      b.onclick = () => {
        this.scenario = id;
        this.strength = STRENGTH[id].initial;
        this.renderControls();
      };
      return b;
    }));
    this.blurb.textContent = this.scenarios[this.scenario]?.blurb ?? "";
    this.chips(this.massChips, [0.1, 0.25, 0.4, 0.7, 1], this.mass, (v) => (this.mass = v));
    const st = STRENGTH[this.scenario];
    this.strengthTitle.textContent = st.label;
    this.strengthHint.textContent = st.hint;
    this.chips(this.strengthChips, st.values, this.strength, (v) => (this.strength = v));
    this.chips(this.sizeChips, [12, 14, 16, 18, 20], this.N, (v) => (this.N = v), (v) => `${v / 2}`);
    const dim = binom(this.N, this.N / 2);
    this.sizeHint.textContent = `${this.N / 2} sites, each holding a particle or an antiparticle. Tracked exactly: ${dim.toLocaleString()} quantum amplitudes.`;
  }

  private run(): void {
    this.runBtn.textContent = "Restart";
    this.frames = [];
    this.draw();
    this.readout.textContent = "Starting…";
    this.send({ type: "lattice.run", scenario: this.scenario, N: this.N, mass: this.mass,
      strength: this.strength, t_max: 10, frames: 81 });
  }

  // ---------------------------------------------------------------- drawing
  private draw(): void {
    for (const [canvas, key] of [[this.matter, "charge"], [this.field, "field"]] as const) {
      const w = canvas.clientWidth, h = canvas.clientHeight;
      if (!w || !h) continue;
      const dpr = Math.min(devicePixelRatio, 2);
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      const g = canvas.getContext("2d")!;
      g.scale(dpr, dpr);
      g.fillStyle = "#0a1f30";
      g.fillRect(0, 0, w, h);
      const total = this.meta?.frames ?? 81;
      const rowH = h / total;
      for (const [i, f] of this.frames.entries()) {
        const vals = key === "charge" ? f.charge : f.field;
        const colW = w / vals.length;
        for (const [j, v] of vals.entries()) {
          g.fillStyle = key === "charge" ? chargeColour(v) : fieldColour(v);
          g.fillRect(j * colW, h - (i + 1) * rowH, colW + 0.5, rowH + 0.5);
        }
      }
    }
  }

  private renderChart(): void {
    const W = 260, H = 110, L = 28, B = 18, T = 8, R = 6;
    const ts = this.frames.map((f) => f.t), ps = this.frames.map((f) => f.particles);
    const tmax = this.meta?.t_max ?? 10;
    const pmin = Math.min(0, ...ps), pmax = Math.max(1, ...ps) * 1.1;
    const x = (t: number) => L + (t / tmax) * (W - L - R);
    const y = (p: number) => H - B - ((p - pmin) / (pmax - pmin)) * (H - B - T);
    const svgNS = "http://www.w3.org/2000/svg";
    const s = document.createElementNS(svgNS, "svg");
    s.setAttribute("viewBox", `0 0 ${W} ${H}`);
    s.setAttribute("class", "scan");
    const add = (tag: string, attrs: Record<string, string | number>, text?: string) => {
      const n = document.createElementNS(svgNS, tag);
      for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, String(v));
      if (text) n.textContent = text;
      s.append(n);
    };
    add("line", { x1: L, x2: W - R, y1: y(0), y2: y(0), class: "axis" });
    add("line", { x1: L, x2: L, y1: T, y2: H - B, class: "axis" });
    add("text", { x: W - R, y: H - 4, class: "tick", "text-anchor": "end" }, `t = ${tmax}`);
    add("text", { x: L - 4, y: y(pmax / 1.1) + 3, class: "tick", "text-anchor": "end" }, num(pmax / 1.1, 1));
    add("text", { x: L - 4, y: y(0) + 3, class: "tick", "text-anchor": "end" }, "0");
    if (ts.length > 1) add("polyline", { points: ts.map((t, i) => `${x(t)},${y(ps[i])}`).join(" "), class: "trace-line" });
    this.chart.replaceChildren(s);
  }

  private renderFacts(): void {
    const m = this.meta;
    if (!m) return;
    const last = this.frames[this.frames.length - 1];
    this.facts.replaceChildren(
      el("dt", {}, "Quantum amplitudes"), el("dd", {}, m.dim.toLocaleString()),
      el("dt", {}, "Particles in the vacuum"), el("dd", {}, num(m.vacuum_particles, 2)),
      el("dt", {}, "Energy change"), el("dd", {}, last ? sci(Math.abs(last.energy_drift)) : "—"),
    );
  }
}

function mix(a: number[], b: number[], t: number): string {
  const c = a.map((x, i) => Math.round(x + (b[i] - x) * t));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

const BG = [10, 31, 48];
function chargeColour(v: number): string {
  const t = Math.min(1, Math.abs(v) * 1.6);
  return v >= 0 ? mix(BG, [255, 138, 122], t) : mix(BG, [127, 216, 255], t);
}

function fieldColour(v: number): string {
  const t = Math.min(1, Math.abs(v));
  return v >= 0 ? mix(BG, [240, 180, 85], t) : mix(BG, [120, 150, 255], t);
}

function binom(n: number, k: number): number {
  let r = 1;
  for (let i = 1; i <= k; i++) r = (r * (n - k + i)) / i;
  return Math.round(r);
}
