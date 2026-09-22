import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { el, num, sci } from "../format";

type Send = (cmd: Record<string, unknown>) => void;

type Chain = {
  al_crystal?: { a0_A: number; B_GPa: number; E_coh_eV: number; C11: number; C12: number; C44: number; bcc_minus_fcc_meV: number };
  al_results?: { thermal: { T: number; a_A: number; H_eV: number }[]; melting: { T_melt: number; bracket: number[] };
    latent: { latent_eV: number; dV_melt_frac: number } };
  al_eam_errors?: { n_train: number; n_test: number; E_meV: number; F_eVA: number };
};
type Summary = Record<string, number>;
type MatterEvent =
  | { type: "matter.hello"; ready: boolean; chain: Chain; block: Summary | null; reference: Record<string, number> }
  | { type: "matter.start"; N: number; T: number; P_GPa: number; start: string }
  | { type: "matter.frame"; step: number; t_ps: number; pos: number[]; box: number[]; T: number; P_GPa: number;
      U_eV: number; E_eV: number; a_A: number; melting_in: number }
  | { type: "matter.done"; cancelled?: boolean; error?: string }
  | { type: "matter.block"; available: boolean; T?: number; side_cm?: number; density?: number; heat_J?: number;
      melted?: boolean };

// Aluminium: the materials rung (many atoms moving on forces learned from DFT) and the
// everyday rung (a centimetre cube whose properties come from those atoms).
export class Matter {
  private ready = false;
  private running = false;
  private T = 300;
  private P = 0;
  private cells = 5;
  private start = "crystal";
  private blockT = 293;

  private chainList = el("ol", { class: "m-chain" });
  private tChips = el("div", { class: "chips", role: "radiogroup", "aria-label": "Temperature" });
  private startChips = el("div", { class: "chips", role: "radiogroup", "aria-label": "Starting state" });
  private sizeChips = el("div", { class: "chips", role: "radiogroup", "aria-label": "Number of atoms" });
  private pChips = el("div", { class: "chips", role: "radiogroup", "aria-label": "Pressure" });
  private runBtn = el("button", { class: "primary" }, "Run dynamics");
  private readout = el("div", { class: "readout" });
  private live = el("dl", { class: "table" });
  private blockTable = el("dl", { class: "table" });
  private blockChips = el("div", { class: "chips", role: "radiogroup", "aria-label": "Block temperature" });
  private blockNow = el("p", { class: "hint" });
  private empty = el("div", { class: "overlay p-empty" },
    el("p", { class: "offline-title" }, "A few hundred aluminium atoms"),
    el("p", {}, "Pick a temperature and run. The atoms move on forces the engine learned from its own quantum-mechanical calculations; whether the metal stays solid or melts is up to them."));
  private legend = el("div", { class: "overlay m-legend" },
    el("span", { class: "m-swatch crystal" }), "ordered like a crystal", el("span", { class: "m-swatch liquid" }), "disordered, liquid-like");

  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(35, 1, 0.1, 1000);
  private controls: OrbitControls;
  private atoms: THREE.InstancedMesh | null = null;
  private boxLines: THREE.LineSegments | null = null;
  private tmp = new THREE.Object3D();
  private cCrystal = new THREE.Color("#7fd8ff");
  private cLiquid = new THREE.Color("#f0b455");

  constructor(left: HTMLElement, private stage: HTMLElement, right: HTMLElement, bar: HTMLElement, private send: Send) {
    const canvas = document.createElement("canvas");
    canvas.className = "m-canvas";
    stage.append(canvas, this.legend, this.empty);
    this.legend.hidden = true;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.setClearColor("#0a1f30");
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.autoRotate = true;
    this.controls.autoRotateSpeed = 0.6;
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    const key = new THREE.DirectionalLight(0xffffff, 1.4);
    key.position.set(1, 1.4, 1.2);
    this.scene.add(key);

    left.append(
      el("section", {}, el("h2", {}, "From atoms to a block"), this.chainList),
      el("section", {}, el("h2", {}, "Temperature"), this.tChips,
        el("p", { class: "hint" }, "Held by a thermostat that exchanges energy with the atoms at random, as a heat bath would.")),
      el("section", {}, el("h2", {}, "Start from"), this.startChips),
      el("section", {}, el("h2", {}, "Pressure"), this.pChips),
      el("section", {}, el("h2", {}, "Atoms"), this.sizeChips),
    );
    right.append(
      el("section", {}, el("h2", {}, "The metal right now"), this.live),
      el("section", {}, el("h2", {}, "One cubic centimetre"), this.blockChips, this.blockNow, this.blockTable,
        el("p", { class: "hint" },
          "About 6 × 10²² atoms is far too many to move one by one, and there is no need: the block is a continuum whose behaviour is set by the per-atom properties measured on the left. Amber numbers are laboratory measurements, shown only for comparison.")),
    );
    this.runBtn.onclick = () => this.toggle();
    bar.append(el("div", { class: "run-group" }, this.runBtn), el("div", { class: "run-group end" }, this.readout));
    new ResizeObserver(() => this.resize()).observe(stage);
    this.renderControls();
    const tick = () => {
      requestAnimationFrame(tick);
      if (this.stage.offsetParent === null) return;
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    };
    tick();
  }

  activate(): void {
    if (!this.ready) this.send({ type: "matter.hello" });
    this.resize();
  }

  private toggle(): void {
    if (this.running) {
      this.send({ type: "matter.stop" });
      return;
    }
    this.send({ type: "matter.md", T: this.T, P_GPa: this.P, cells: this.cells, start: this.start });
  }

  onEvent(e: MatterEvent): void {
    switch (e.type) {
      case "matter.hello":
        this.ready = true;
        this.renderChain(e.chain, e.reference);
        this.renderBlock(e.block, e.reference);
        this.runBtn.toggleAttribute("disabled", !e.ready);
        if (!e.ready) this.readout.textContent = "The learned potential has not been trained yet.";
        this.send({ type: "matter.block", T: this.blockT });
        break;
      case "matter.start":
        this.running = true;
        this.runBtn.textContent = "Stop";
        this.runBtn.classList.add("is-busy");
        this.empty.hidden = true;
        this.legend.hidden = false;
        this.makeAtoms(e.N);
        break;
      case "matter.frame":
        this.drawFrame(e.pos, e.box);
        this.live.replaceChildren(
          ...row("Temperature", `${num(e.T, 0)} K`),
          ...row("Pressure", `${num(e.P_GPa, 2)} GPa`),
          ...row("Lattice spacing", `${num(e.a_A, 3)} Å`),
          ...row("Potential energy", `${num(e.U_eV, 3)} eV/atom`),
          ...row("Time simulated", `${num(e.t_ps, 2)} ps`),
        );
        this.readout.textContent = e.melting_in > 0 ? "Melting it first at 2500 K…" : `${num(e.t_ps, 1)} ps of motion`;
        break;
      case "matter.done":
        this.running = false;
        this.runBtn.textContent = "Run dynamics";
        this.runBtn.classList.remove("is-busy");
        if (e.error) this.readout.textContent = e.error;
        break;
      case "matter.block":
        if (!e.available) { this.blockNow.textContent = ""; break; }
        this.blockNow.textContent = e.melted
          ? `At ${num(e.T!, 0)} K it has melted. Getting there from room temperature took ${num(e.heat_J! / 1000, 2)} kJ.`
          : `At ${num(e.T!, 0)} K each edge is ${num(e.side_cm!, 4)} cm; warming it from room temperature takes ${num(e.heat_J!, 0)} J.`;
        break;
    }
  }

  // ---------------------------------------------------------------- rails
  private chips(root: HTMLElement, values: (number | string)[], current: number | string, pick: (v: never) => void,
                fmt: (v: never) => string = (v) => String(v)) {
    root.replaceChildren(...values.map((v) => {
      const b = el("button", { role: "radio", "aria-checked": String(v === current) }, fmt(v as never));
      b.onclick = () => { pick(v as never); this.renderControls(); };
      return b;
    }));
  }

  private renderControls(): void {
    this.chips(this.tChips, [300, 600, 900, 1200, 1500], this.T, (v: number) => (this.T = v), (v: number) => `${v} K`);
    this.chips(this.startChips, ["crystal", "liquid"], this.start, (v: string) => (this.start = v),
      (v: string) => (v === "crystal" ? "Crystal" : "Liquid"));
    this.chips(this.pChips, [0, 5, 20], this.P, (v: number) => (this.P = v), (v: number) => (v === 0 ? "None" : `${v} GPa`));
    this.chips(this.sizeChips, [4, 5, 6], this.cells, (v: number) => (this.cells = v), (v: number) => String(4 * v ** 3));
    this.chips(this.blockChips, [293, 373, 600, 1000], this.blockT, (v: number) => {
      this.blockT = v;
      this.send({ type: "matter.block", T: v });
    }, (v: number) => `${v} K`);
  }

  private renderChain(c: Chain, ref: Record<string, number>): void {
    const step = (title: string, body: string) => el("li", {}, el("span", { class: "m-step" }, title), el("span", { class: "m-body" }, body));
    const items: HTMLElement[] = [];
    const k = c.al_crystal;
    items.push(step("Electrons in a crystal", k
      ? `Periodic DFT on one unit cell: spacing ${num(k.a0_A, 3)} Å, stiffness ${num(k.B_GPa, 0)} GPa, cohesion ${num(k.E_coh_eV, 2)} eV. Face-centred cubic wins over body-centred by ${num(k.bcc_minus_fcc_meV, 0)} meV.`
      : "Not computed yet."));
    const f = c.al_eam_errors;
    items.push(step("Forces learned from it", f
      ? `A potential fitted to ${f.n_train} DFT calculations reproduces ${f.n_test} it never saw to ${num(f.E_meV, 1)} meV per atom.`
      : "Not trained yet."));
    const r = c.al_results;
    items.push(step("Hundreds of atoms moving", r
      ? `Melts at ${num(r.melting.T_melt, 0)} K (measured ${num(ref.T_melt, 0)} K), absorbing ${num(r.latent.latent_eV * 1000, 0)} meV per atom.`
      : "Molecular dynamics not run yet."));
    items.push(step("A block you could hold", r ? "Built from the numbers above; see the right." : "Waits for the steps above."));
    this.chainList.replaceChildren(...items);
  }

  private renderBlock(b: Summary | null, ref: Record<string, number>): void {
    if (!b) {
      this.blockTable.replaceChildren(...row("Status", "waiting for the molecular dynamics"));
      return;
    }
    const line = (label: string, value: string, refv?: string) => {
      const dd = el("dd", {}, value);
      if (refv) dd.append(el("br"), el("span", { class: "ref" }, refv));
      return [el("dt", {}, label), dd];
    };
    this.blockTable.replaceChildren(
      ...line("Atoms", sci(b.atoms, 2)),
      ...line("Mass", `${num(b.mass_g, 2)} g`),
      ...line("Density", `${num(b.density / 1000, 3)} g/cm³`, `${num(ref.density / 1000, 3)}`),
      ...line("Expansion", `${num(b.alpha_per_K * 1e6, 1)} ×10⁻⁶ /K`, `${num(ref.alpha_per_K * 1e6, 1)}`),
      ...line("Specific heat", `${num(b.c_J_per_gK, 3)} J/g·K`, `${num(ref.c_J_per_gK, 3)}`),
      ...line("Melts at", `${num(b.T_melt, 0)} K`, `${num(ref.T_melt, 0)}`),
      ...line("Heat to melt it", `${num(b.heat_to_melt_J / 1000, 2)} kJ`),
      ...line("Bulk modulus", `${num(b.B_GPa, 0)} GPa`, `${num(ref.B_GPa, 0)}`),
      ...line("Young's modulus ⟨100⟩", `${num(b.young_GPa, 0)} GPa`, `${num(ref.young_GPa, 0)}`),
      ...line("Sound, lengthwise", `${num(b.sound_long, 0)} m/s`, `${num(ref.sound_long, 0)}`),
      ...line("Sound, shear", `${num(b.sound_trans, 0)} m/s`, `${num(ref.sound_trans, 0)}`),
      ...line("Energy to vaporise into atoms", `${num(b.cohesive_J / 1000, 0)} kJ`),
    );
  }

  // ---------------------------------------------------------------- 3D
  private makeAtoms(n: number): void {
    if (this.atoms) { this.scene.remove(this.atoms); this.atoms.geometry.dispose(); }
    const geo = new THREE.SphereGeometry(1, 14, 10);
    const mat = new THREE.MeshStandardMaterial({ roughness: 0.35, metalness: 0.25 });
    this.atoms = new THREE.InstancedMesh(geo, mat, n);
    this.atoms.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.scene.add(this.atoms);
  }

  private drawFrame(pos: number[], box: number[]): void {
    if (!this.atoms) return;
    const n = pos.length / 3;
    const c = [box[0] / 2, box[1] / 2, box[2] / 2];
    // Local order: in a close-packed crystal every atom has 12 neighbours near 2.86 Å.
    const cut2 = 3.35 * 3.35;
    const count = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        let d2 = 0;
        for (let a = 0; a < 3; a++) {
          let d = pos[3 * j + a] - pos[3 * i + a];
          d -= box[a] * Math.round(d / box[a]);
          d2 += d * d;
        }
        if (d2 < cut2) { count[i]++; count[j]++; }
      }
    }
    const col = new THREE.Color();
    for (let i = 0; i < n; i++) {
      this.tmp.position.set(pos[3 * i] - c[0], pos[3 * i + 1] - c[1], pos[3 * i + 2] - c[2]);
      this.tmp.scale.setScalar(0.62);
      this.tmp.updateMatrix();
      this.atoms.setMatrixAt(i, this.tmp.matrix);
      const order = Math.max(0, 1 - Math.abs(count[i] - 12) / 4);
      col.copy(this.cLiquid).lerpHSL(this.cCrystal, order);
      this.atoms.setColorAt(i, col);
    }
    this.atoms.instanceMatrix.needsUpdate = true;
    if (this.atoms.instanceColor) this.atoms.instanceColor.needsUpdate = true;
    if (!this.boxLines || Math.abs((this.boxLines.userData.L ?? 0) - box[0]) > 0.02) {
      if (this.boxLines) this.scene.remove(this.boxLines);
      this.boxLines = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(box[0], box[1], box[2])),
        new THREE.LineBasicMaterial({ color: "#2a5470" }));
      this.boxLines.userData.L = box[0];
      this.scene.add(this.boxLines);
      if (!this.camera.userData.framed) {
        this.camera.position.set(box[0] * 1.5, box[0] * 1.1, box[0] * 1.9);
        this.camera.userData.framed = true;
      }
    }
  }

  private resize(): void {
    const w = this.stage.clientWidth, h = this.stage.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }
}

function row(label: string, value: string): HTMLElement[] {
  return [el("dt", {}, label), el("dd", {}, value)];
}
