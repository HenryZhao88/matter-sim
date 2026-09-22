import { el, num, rich, sci } from "../format";
import { EventDisplay, FAMILY_COLOUR } from "./display";
import type { BeamInfo, CEvent, ColliderEvent, OutcomeRow, ParticleMeta } from "./types";

type Send = (cmd: Record<string, unknown>) => void;

const SVG = "http://www.w3.org/2000/svg";
function svg<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number> = {}) {
  const e = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  return e;
}

function pb(x: number): string {
  if (x >= 1e6) return `${num(x / 1e6, 2)} μb`;
  if (x >= 1e3) return `${num(x / 1e3, 2)} nb`;
  if (x >= 1) return `${num(x, 2)} pb`;
  return `${num(x * 1e3, 1)} fb`;
}

function lifetime(s: number): string {
  if (s > 1e-3) return `${num(s, 2)} s`;
  if (s > 1e-9) return `${num(s * 1e6, 2)} μs`;
  return `${sci(s)} s`;
}

export class Collider {
  private display: EventDisplay;
  private meta: Record<string, ParticleMeta> = {};
  private beams: BeamInfo[] = [];
  private energies: number[] = [];
  private beam: BeamInfo | null = null;
  private energy = 200;
  private outcomes: OutcomeRow[] = [];
  private tally = new Map<string, number>();
  private collisions = 0;
  private scan = new Map<number, number>();
  private scanning = false;
  private ready = false;

  private beamList = el("div", { class: "beam-list", role: "radiogroup", "aria-label": "Beams" });
  private beamNote = el("p", { class: "blurb" });
  private energyChips = el("div", { class: "chips", role: "radiogroup", "aria-label": "Collision energy" });
  private collideBtn = el("button", { class: "primary" }, "Collide");
  private many = el("button", { class: "quiet" }, "Collide 50 times");
  private eventTitle = el("div", { class: "event-title" });
  private tree = el("ol", { class: "decay-tree" });
  private outcomeTable = el("div", { class: "outcomes" });
  private outcomeHead = el("p", { class: "hint" });
  private tallyTable = el("div", { class: "outcomes" });
  private scanSvg = svg("svg", { class: "scan", viewBox: "0 0 260 150", role: "img" });
  private scanBtn = el("button", { class: "quiet small" }, "Scan energies");
  private scanNote = el("p", { class: "hint" });
  private barText = el("div", { class: "readout" });
  private particleInfo = el("section", { class: "particle-info" });
  private scanSection = el("section", {});
  private triggerBox = el("input", { type: "checkbox" });
  private triggerField = el("label", { class: "check" }, this.triggerBox,
    el("span", {}, "Trigger: keep only collisions that make leptons, photons, W, Z, top quarks or the Higgs"));
  private triggerNote = el("p", { class: "hint" });
  private showAll = el("button", { class: "quiet small" }, "Show all");
  private allShown = false;
  private totalPb = 0;

  constructor(
    left: HTMLElement,
    right: HTMLElement,
    bar: HTMLElement,
    canvas: HTMLCanvasElement,
    labels: HTMLElement,
    private legend: HTMLElement,
    private progress: HTMLElement,
    private empty: HTMLElement,
    private send: Send,
  ) {
    this.display = new EventDisplay(canvas, labels);
    this.collideBtn.onclick = () => this.collide(1);
    this.many.onclick = () => this.collide(50);
    this.scanBtn.onclick = () => this.startScan();
    this.scanSection.append(el("h2", {}, "Rate against energy"), this.scanSvg,
      el("div", { class: "spin-row" }, this.scanNote, this.scanBtn));
    this.showAll.onclick = () => { this.allShown = !this.allShown; this.renderOutcomes(this.totalPb); };
    this.triggerBox.onchange = () => { this.tally.clear(); this.collisions = 0; this.renderTally(); };
    left.append(
      el("section", {}, el("h2", {}, "Beams"), this.beamList, this.beamNote),
      el("section", {}, el("h2", {}, "Collision energy"), this.energyChips,
        el("p", { class: "hint" }, "Total energy of the two beams in their centre-of-mass frame, in GeV.")),
      el("section", { class: "trigger" }, el("h2", {}, "Trigger"), this.triggerField, this.triggerNote),
      el("section", {}, el("h2", {}, "What goes in"),
        el("p", { class: "hint" },
          "One equation, the Standard Model Lagrangian, plus 18 measured constants such as the electron's mass and the strength of each force. Particles, charges, masses of the W and Z, decays and lifetimes are all computed from it.")),
      el("section", {}, el("h2", {}, "Where it stops"),
        el("p", { class: "hint" },
          "Quarks and gluons are drawn as cones marked confined. In nature each becomes a spray of hadrons. No known method computes that step from first principles in real time, so it is left out rather than faked.")),
    );
    right.append(
      el("section", {}, el("h2", {}, "This collision"), this.eventTitle, this.tree),
      this.particleInfo,
      el("section", {}, el("h2", {}, "What these beams can make"), this.outcomeHead, this.outcomeTable, this.showAll),
      el("section", {}, el("h2", {}, "Counted so far"), this.tallyTable),
      this.scanSection,
    );
    bar.append(el("div", { class: "run-group" }, this.collideBtn, this.many), el("div", { class: "run-group end" }, this.barText));
    this.renderLegend();
    this.setBusy(true);
  }

  activate(): void {
    if (!this.ready) this.send({ type: "collider.hello" });
  }

  onEvent(e: ColliderEvent): void {
    switch (e.type) {
      case "collider.hello":
        this.meta = e.particles;
        this.display.meta = e.particles;
        this.beams = e.beams;
        this.energies = e.energies;
        this.ready = true;
        this.renderBeams();
        this.selectBeam(this.beams.find((b) => b.id === "ee") ?? this.beams[0]);
        break;
      case "collider.outcomes":
        if (!this.isCurrent(e.beams, e.sqrt_s)) return;
        this.outcomes = e.rows;
        this.renderOutcomes(e.total_pb);
        this.triggerNote.textContent = e.rare_pb
          ? `Most collisions only make jets of quarks and gluons. The rest are ${pb(e.rare_pb)} out of ${pb(e.total_pb)}: about one in ${Math.round(e.total_pb / e.rare_pb).toLocaleString()}. Real detectors use triggers for the same reason. With it on, collisions are drawn from exactly that rarer share.`
          : "";
        this.setBusy(false);
        break;
      case "collider.events":
        if (!this.isCurrent(e.beams, e.sqrt_s)) return;
        for (const ev of e.events) this.count(ev);
        this.showEvent(e.events[e.events.length - 1]);
        this.renderTally();
        this.setBusy(false);
        break;
      case "collider.scan_row":
        if (this.beam && e.beams.join() === this.beam.pair.join()) {
          this.scan.set(e.sqrt_s, e.total_pb);
          this.renderScan();
        }
        break;
      case "collider.scan_done":
        this.scanning = false;
        this.scanBtn.removeAttribute("disabled");
        this.scanNote.textContent = "Each point is every final state these beams can make, summed.";
        break;
      case "collider.progress":
        this.progress.hidden = false;
        this.progress.textContent = e.message;
        break;
      case "collider.particle":
        this.renderParticle(e);
        break;
    }
  }

  private isCurrent(beams: [string, string], E: number): boolean {
    return !!this.beam && beams.join() === this.beam.pair.join() && Math.abs(E - this.energy) < 1e-6;
  }

  private sym(name: string): string {
    return this.meta[name]?.symbol ?? name;
  }

  // ----------------------------------------------------------------- controls
  private renderBeams(): void {
    this.beamList.replaceChildren(
      ...this.beams.map((b) => {
        const btn = el("button", { class: "preset", role: "radio", "data-id": b.id },
          el("span", { class: "preset-formula" }, `${this.sym(b.pair[0])} ${this.sym(b.pair[1])}`),
          el("span", { class: "preset-name" }, b.label));
        btn.onclick = () => this.selectBeam(b);
        return btn;
      }),
    );
  }

  private selectBeam(b: BeamInfo): void {
    this.beam = b;
    const Es = b.energies ?? this.energies;
    this.energyChips.replaceChildren(
      ...Es.map((E) => {
        const chip = el("button", { role: "radio", "data-e": String(E) }, E >= 1000 ? `${E / 1000} TeV` : `${E}`);
        chip.onclick = () => this.selectEnergy(E);
        return chip;
      }),
    );
    if (!Es.includes(this.energy)) this.energy = Es.includes(200) ? 200 : Es[0];
    const protons = b.id === "pp";
    this.scanSection.hidden = protons;
    (this.triggerField.closest("section") as HTMLElement | null)?.toggleAttribute("hidden", !protons);
    this.beamNote.textContent = b.note;
    for (const x of this.beamList.children) x.setAttribute("aria-checked", String((x as HTMLElement).dataset.id === b.id));
    this.scan.clear();
    this.renderScan();
    this.scanNote.textContent = "Not scanned yet.";
    this.selectEnergy(this.energy);
  }

  private selectEnergy(E: number): void {
    this.energy = E;
    for (const x of this.energyChips.children) x.setAttribute("aria-checked", String(Number((x as HTMLElement).dataset.e) === E));
    this.tally.clear();
    this.collisions = 0;
    this.renderTally();
    this.outcomeTable.replaceChildren();
    this.outcomeHead.textContent = "";
    this.setBusy(true);
    this.send({ type: "collider.select", beams: this.beam!.pair, sqrt_s: E });
    this.renderScan();
  }

  private collide(n: number): void {
    if (!this.beam) return;
    this.setBusy(true);
    this.send({ type: "collider.collide", beams: this.beam.pair, sqrt_s: this.energy, n,
      trigger: this.beam.id === "pp" && this.triggerBox.checked });
  }

  private startScan(): void {
    if (!this.beam || this.scanning) return;
    this.scanning = true;
    this.scanBtn.setAttribute("disabled", "");
    this.scanNote.textContent = "Scanning. The first scan of a beam pair takes a few minutes.";
    this.send({ type: "collider.scan", beams: this.beam.pair });
  }

  private setBusy(busy: boolean): void {
    for (const b of [this.collideBtn, this.many]) {
      if (busy) b.setAttribute("disabled", "");
      else b.removeAttribute("disabled");
    }
    if (!busy) this.progress.hidden = true;
    const E = this.energy >= 1000 ? `${this.energy / 1000} TeV` : `${this.energy} GeV`;
    this.barText.textContent = this.beam
      ? `${this.sym(this.beam.pair[0])} ${this.sym(this.beam.pair[1])} at ${E}. ${this.collisions} collisions.`
      : "";
  }

  // ----------------------------------------------------------------- panels
  private finalLabel(pair: [string, string]): string {
    return `${this.sym(pair[0])} ${this.sym(pair[1])}`;
  }

  private renderOutcomes(total: number): void {
    const b = this.beam!;
    this.totalPb = total;
    const E = this.energy >= 1000 ? `${this.energy / 1000} TeV` : `${this.energy} GeV`;
    this.outcomeHead.textContent = this.outcomes.length
      ? `${this.outcomes.length} possible outcomes at ${E}. Total ${pb(total)}.`
      : `These beams cannot make anything at ${E}.`;
    const shown = this.allShown ? this.outcomes : this.outcomes.slice(0, 10);
    this.outcomeTable.replaceChildren(
      ...shown.map((r) => this.outcomeRow(`${this.sym(b.pair[0])} ${this.sym(b.pair[1])} → ${this.finalLabel(r.final)}`, r.share, pb(r.pb))),
    );
    this.showAll.hidden = this.outcomes.length <= 10;
    this.showAll.textContent = this.allShown ? "Show fewer" : `Show all ${this.outcomes.length}`;
  }

  private outcomeRow(label: string, share: number, value: string): HTMLElement {
    const barEl = el("span", { class: "share-bar" });
    barEl.style.width = `${Math.max(1, share * 100)}%`;
    return el("div", { class: "outcome" },
      el("span", { class: "outcome-label" }, rich(label)),
      el("span", { class: "outcome-value" }, value),
      el("span", { class: "share-track" }, barEl));
  }

  private count(ev: CEvent): void {
    this.collisions += 1;
    if (!ev.channel) return;
    const key = this.finalLabel(ev.channel);
    this.tally.set(key, (this.tally.get(key) ?? 0) + 1);
  }

  private renderTally(): void {
    if (!this.collisions) {
      this.tallyTable.replaceChildren(el("p", { class: "hint" }, "Nothing collided yet."));
      return;
    }
    const expected = new Map(this.outcomes.map((r) => [this.finalLabel(r.final), r.share]));
    const rows = [...this.tally.entries()].sort((a, b) => b[1] - a[1]).slice(0, 8);
    this.tallyTable.replaceChildren(
      el("p", { class: "hint" }, `${this.collisions} collisions. Bars show observed share; the number in brackets is the predicted share.`),
      ...rows.map(([k, n]) => this.outcomeRow(k, n / this.collisions,
        `${n} (${num(100 * (expected.get(k) ?? 0), 0)}%)`)),
    );
    this.setBusy(false);
  }

  private showEvent(ev: CEvent): void {
    this.empty.hidden = true;
    this.display.show(ev);
    const b = ev.beams;
    this.eventTitle.replaceChildren(rich(ev.channel
      ? `${this.sym(b[0])} ${this.sym(b[1])} → ${this.finalLabel(ev.channel)}`
      : "Nothing was produced."));
    const byParent = new Map<number | null, number[]>();
    ev.particles.forEach((p, i) => {
      const k = p.parent;
      byParent.set(k, [...(byParent.get(k) ?? []), i]);
    });
    const node = (i: number): HTMLElement => {
      const p = ev.particles[i];
      const meta = this.meta[p.name];
      const sw = el("span", { class: "swatch" });
      sw.style.background = FAMILY_COLOUR[meta?.family ?? "other"];
      const status = p.status === "decayed" ? "decayed" : p.status === "confined" ? "confined" :
        p.status === "showered" ? "radiated" : p.status === "invisible" ? "unseen" : "seen";
      const btn = el("button", { class: "tree-node", title: "Show this particle's decays" },
        sw, el("span", { class: "tree-sym" }, rich(this.sym(p.name))),
        el("span", { class: "tree-energy" }, `${num(p.p[0], 1)} GeV`),
        el("span", { class: `tree-status ${status}` }, status));
      btn.onclick = () => this.send({ type: "collider.particle", name: p.name });
      const li = el("li", {}, btn);
      const kids = byParent.get(i);
      if (kids?.length && p.status === "showered") {
        li.append(el("p", { class: "tree-shower hint" }, `radiated into ${kids.length} quarks and gluons (parton shower)`));
      } else if (kids?.length) li.append(el("ol", {}, ...kids.map(node)));
      return li;
    };
    this.tree.replaceChildren(...(byParent.get(null) ?? []).map(node));
    if (ev.partons && ev.x) {
      this.tree.prepend(el("li", { class: "tree-partons hint" },
        rich(`Inside the protons: ${this.sym(ev.partons[0])} (carrying ${num(100 * ev.x[0], 1)}% of proton 1) met ${this.sym(ev.partons[1])} (${num(100 * ev.x[1], 1)}% of proton 2) with ${num(ev.sqrt_shat ?? 0, 0)} GeV.`)));
    }
  }

  private renderParticle(e: Extract<ColliderEvent, { type: "collider.particle" }>): void {
    const rows: (Node | string)[] = [el("h2", {}, rich(this.sym(e.name)))];
    const facts = el("dl", { class: "table" },
      el("dt", {}, "Mass"), el("dd", {}, e.mass > 0 ? `${num(e.mass, e.mass < 1 ? 4 : 2)} GeV` : "0"));
    if (e.lifetime_s !== undefined) facts.append(el("dt", {}, "Lifetime"), el("dd", {}, lifetime(e.lifetime_s)));
    rows.push(facts);
    if (e.confined) rows.push(el("p", { class: "hint" }, "Confined: never seen alone. It would form hadrons."));
    if (e.decays?.length) {
      rows.push(el("p", { class: "hint" }, "Computed decays:"));
      rows.push(el("div", { class: "outcomes" },
        ...e.decays.map((d) => this.outcomeRow(d.products.map((x) => this.sym(x)).join(" "), d.br, `${num(100 * d.br, 1)}%`))));
    }
    this.particleInfo.replaceChildren(...rows);
  }

  private renderScan(): void {
    this.scanSvg.replaceChildren();
    const pts = [...this.scan.entries()].filter(([, s]) => s > 0).sort((a, b) => a[0] - b[0]);
    const W = 260, H = 150, L = 34, B = 20, T = 8, R = 6;
    const xE = [4, 1100], yS = pts.length ? [Math.min(...pts.map((p) => p[1])) / 2, Math.max(...pts.map((p) => p[1])) * 2] : [1, 1e6];
    const x = (E: number) => L + ((Math.log10(E) - Math.log10(xE[0])) / (Math.log10(xE[1]) - Math.log10(xE[0]))) * (W - L - R);
    const y = (s: number) => H - B - ((Math.log10(s) - Math.log10(yS[0])) / (Math.log10(yS[1]) - Math.log10(yS[0]))) * (H - B - T);
    this.scanSvg.append(svg("line", { x1: L, x2: W - R, y1: H - B, y2: H - B, class: "axis" }));
    this.scanSvg.append(svg("line", { x1: L, x2: L, y1: T, y2: H - B, class: "axis" }));
    for (const E of [10, 100, 1000]) {
      const t = svg("text", { x: x(E), y: H - 6, class: "tick", "text-anchor": "middle" });
      t.textContent = `${E} GeV`;
      this.scanSvg.append(t);
    }
    const yt = svg("text", { x: 4, y: T + 8, class: "tick" });
    yt.textContent = "σ";
    this.scanSvg.append(yt);
    if (pts.length > 1) {
      this.scanSvg.append(svg("polyline", { points: pts.map(([E, s]) => `${x(E)},${y(s)}`).join(" "), class: "trace-line" }));
    }
    for (const [E, s] of pts) this.scanSvg.append(svg("circle", { cx: x(E), cy: y(s), r: 1.8, class: "scan-dot" }));
    if (this.energy) this.scanSvg.append(svg("line", { x1: x(this.energy), x2: x(this.energy), y1: T, y2: H - B, class: "now" }));
  }

  private renderLegend(): void {
    const items: [string, string][] = [["lepton", "electrons, muons, taus"], ["photon", "photons"],
      ["quark", "quarks (confined)"], ["gluon", "gluons (confined)"], ["neutrino", "neutrinos (unseen)"]];
    this.legend.replaceChildren(...items.map(([f, t]) => {
      const sw = el("span", { class: "swatch" });
      sw.style.background = FAMILY_COLOUR[f];
      return el("span", { class: "legend-item" }, sw, t);
    }));
  }
}
