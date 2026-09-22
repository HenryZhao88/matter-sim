import { el, num } from "../format";
import type { Hello, Mode, Snapshot, Status } from "../types";

type Send = (cmd: Record<string, unknown>) => void;

const MODE_LABEL: Record<Mode, [string, string]> = {
  frozen: ["Fixed", "Hold the nuclei still and solve only the electrons"],
  relax: ["Relax", "Let the nuclei slide downhill to the nearest stable shape"],
  dynamics: ["Dynamics", "Move the nuclei in real time with their real masses"],
};
const METHOD_LABEL: Record<string, string> = {
  lda: "Density functional (LDA)",
  hf: "Hartree–Fock",
  none: "Single electron (exact)",
};
const QUALITY_LABEL: Record<string, string> = { draft: "Draft", standard: "Standard", fine: "Fine" };

export class RunBar {
  private play = el("button", { class: "primary play" });
  private stepBtn = el("button", { class: "quiet" }, "Step");
  private modes = el("div", { class: "segmented", role: "radiogroup", "aria-label": "What the nuclei do" });
  private quality = el("select", { "aria-label": "Grid resolution" });
  private method = el("select", { "aria-label": "Electron method" });
  private temp = el("input", { type: "number", min: "0", max: "5000", step: "50", value: "0", "aria-label": "Temperature in kelvin" });
  private tempField = el("label", { class: "inline-field" }, "Heat bath ", this.temp, " K");
  private verify = el("button", { class: "quiet" }, "Check precision");
  private readout = el("div", { class: "readout" });
  private status: Status | null = null;

  constructor(root: HTMLElement, private send: Send) {
    this.play.onclick = () => this.send({ type: this.status?.running ? "pause" : "run" });
    this.stepBtn.onclick = () => this.send({ type: "step" });
    this.quality.onchange = () => this.send({ type: "set_params", params: { quality: this.quality.value } });
    this.method.onchange = () => this.send({ type: "set_params", params: { functional: this.method.value } });
    this.temp.onchange = () => this.send({ type: "set_params", params: { temperature_K: Number(this.temp.value) || 0 } });
    this.verify.onclick = () => this.send({ type: "verify" });
    this.verify.title = "Re-solve this state in 64-bit on the CPU and compare with the 32-bit GPU result";
    root.append(
      el("div", { class: "run-group" }, this.play, this.stepBtn),
      this.modes,
      el("div", { class: "run-group" },
        el("label", { class: "inline-field" }, "Resolution ", this.quality),
        el("label", { class: "inline-field" }, "Electrons ", this.method),
        this.tempField),
      el("div", { class: "run-group end" }, this.verify, this.readout),
    );
  }

  setHello(h: Hello): void {
    this.modes.replaceChildren(
      ...h.modes.map((m) => {
        const b = el("button", { role: "radio", "data-mode": m, title: MODE_LABEL[m][1] }, MODE_LABEL[m][0]);
        b.onclick = () => this.send({ type: "set_params", params: { mode: m } });
        return b;
      }),
    );
    this.quality.replaceChildren(
      ...Object.entries(h.qualities).map(([k, hv]) => el("option", { value: k }, `${QUALITY_LABEL[k] ?? k} (${hv} bohr)`)),
    );
    this.method.replaceChildren(...h.functionals.map((f) => el("option", { value: f }, METHOD_LABEL[f] ?? f)));
  }

  update(status: Status | null, snap: Snapshot | null): void {
    this.status = status;
    const running = !!status?.running;
    const idle = !!status?.idle;
    this.play.textContent = running ? (idle ? "Settled" : "Pause") : "Run";
    this.play.setAttribute("aria-pressed", String(running));
    this.play.classList.toggle("is-busy", !!status?.busy);
    const p = status?.params;
    if (p) {
      for (const b of this.modes.querySelectorAll<HTMLButtonElement>("button"))
        b.setAttribute("aria-checked", String(b.dataset.mode === p.mode));
      if (document.activeElement !== this.quality) this.quality.value = p.quality;
      if (document.activeElement !== this.method) this.method.value = p.functional;
      if (document.activeElement !== this.temp) this.temp.value = String(p.temperature_K);
      this.tempField.hidden = p.mode !== "dynamics";
    }
    if (snap) {
      const m = snap.meta;
      const dev = m.backend === "mlx" ? "Metal GPU, 32-bit" : "CPU, 64-bit";
      this.readout.replaceChildren(
        el("span", {}, dev),
        el("span", {}, `grid ${m.grid.N}³`),
        el("span", {}, `${num(m.step_seconds, 2)} s per step`),
      );
    }
  }
}
