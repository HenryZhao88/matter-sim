import { el, sub } from "../format";
import type { ElementInfo, Hello, PresetInfo, Snapshot, Status } from "../types";

type Send = (cmd: Record<string, unknown>) => void;

// Left rail: choose a starting state, add nuclei, set electron count and spin.
export class BuildRail {
  private presetList = el("ul", { class: "presets" });
  private blurb = el("p", { class: "blurb" });
  private table = el("div", { class: "ptable", role: "group", "aria-label": "Elements" });
  private charge = el("output", { class: "stepper-value" });
  private spin = el("div", { class: "segmented small", role: "radiogroup", "aria-label": "Unpaired spins" });
  private selection = el("section", { class: "selection" });
  private findSpin = el("button", { class: "quiet small", title: "Solve every allowed spin here and keep the lowest energy" }, "Find lowest");
  private presets: PresetInfo[] = [];
  private elements: ElementInfo[] = [];
  private snap: Snapshot | null = null;
  selected: number | null = null;

  constructor(root: HTMLElement, private send: Send) {
    const minus = el("button", { class: "stepper-btn", "aria-label": "Remove an electron" }, "−");
    const plus = el("button", { class: "stepper-btn", "aria-label": "Add an electron" }, "+");
    this.findSpin.onclick = () => this.send({ type: "find_spin" });
    minus.onclick = () => this.send({ type: "set_charge", charge: (this.snap?.meta.system.charge ?? 0) + 1 });
    plus.onclick = () => this.send({ type: "set_charge", charge: (this.snap?.meta.system.charge ?? 0) - 1 });

    root.append(
      el("section", {},
        el("h2", {}, "Start from"),
        this.presetList,
        this.blurb),
      el("section", {},
        el("h2", {}, "Add a nucleus"),
        this.table,
        el("p", { class: "hint" },
          "Hydrogen and helium are bare nuclei. Lithium to argon carry pseudopotentials the engine derived from its own all-electron atoms.")),
      el("section", {},
        el("h2", {}, "Electrons"),
        el("div", { class: "field" },
          el("span", { class: "field-label" }, "Net charge"),
          el("div", { class: "stepper" }, minus, this.charge, plus)),
        el("div", { class: "field column" },
          el("span", { class: "field-label" }, "Unpaired spins"),
          el("div", { class: "spin-row" }, this.spin, this.findSpin))),
      this.selection,
    );
  }

  setHello(h: Hello): void {
    this.presets = h.presets;
    this.elements = h.elements;
    const groups = [...new Set(h.presets.map((p) => p.group))];
    this.presetList.replaceChildren(
      ...groups.flatMap((g) => [
        el("li", { class: "preset-group" }, g),
        ...h.presets.filter((p) => p.group === g).map((p) => {
          const b = el("button", { class: "preset", "data-id": p.id },
            el("span", { class: "preset-formula" }, p.formula),
            el("span", { class: "preset-name" }, p.name));
          b.onclick = () => this.send({ type: "load_preset", id: p.id });
          return el("li", {}, b);
        }),
      ]),
    );
    // Periodic-table layout: H and He on the first row's ends, Li–Ne below.
    const cell = (e: ElementInfo, col: number, row: number) => {
      const b = el("button", {
        class: "element", style: `grid-column:${col};grid-row:${row};--el:${e.color}`,
        title: e.available ? `Add ${e.name}` : `${e.name}: needs a pseudopotential (not built yet)`,
        "aria-label": e.available ? `Add ${e.name}` : `${e.name}, unavailable`,
      }, el("span", { class: "el-sym" }, e.symbol));
      if (!e.available) b.setAttribute("disabled", "");
      b.onclick = () => this.send({ type: "add_atom", Z: e.Z });
      return b;
    };
    const byZ = new Map(h.elements.map((e) => [e.Z, e]));
    const cells: HTMLElement[] = [];
    if (byZ.get(1)) cells.push(cell(byZ.get(1)!, 1, 1));
    if (byZ.get(2)) cells.push(cell(byZ.get(2)!, 8, 1));
    for (let z = 3; z <= 10; z++) if (byZ.get(z)) cells.push(cell(byZ.get(z)!, z - 2, 2));
    for (let z = 11; z <= 18; z++) if (byZ.get(z)) cells.push(cell(byZ.get(z)!, z - 10, 3));
    this.table.replaceChildren(...cells);
  }

  update(status: Status | null, snap: Snapshot | null): void {
    this.snap = snap;
    const current = status?.preset ?? null;
    for (const b of this.presetList.querySelectorAll<HTMLButtonElement>(".preset")) {
      b.setAttribute("aria-pressed", String(b.dataset.id === current));
    }
    const p = this.presets.find((x) => x.id === current);
    this.blurb.textContent = p ? p.blurb : "Your own arrangement. Nothing here was set up in advance.";

    if (!snap) return;
    const sys = snap.meta.system;
    this.charge.textContent = sys.charge > 0 ? `+${sys.charge}` : sys.charge < 0 ? `−${-sys.charge}` : "0";
    const ne = sys.n_electrons;
    const options: number[] = [];
    for (let u = ne % 2; u <= Math.min(ne, 4); u += 2) options.push(u);
    this.spin.replaceChildren(
      ...options.map((u) => {
        const b = el("button", { role: "radio", "aria-checked": String(sys.multiplicity - 1 === u) }, String(u));
        b.title = u === 0 ? "All spins paired" : `${u} more spin-up than spin-down`;
        b.onclick = () => this.send({ type: "set_spin", multiplicity: u + 1 });
        return b;
      }),
    );
    this.renderSelection();
  }

  setSelected(i: number | null): void {
    this.selected = i;
    this.renderSelection();
  }

  private renderSelection(): void {
    const snap = this.snap;
    if (!snap) return;
    const i = this.selected;
    if (i === null || i >= snap.meta.atoms.length) {
      this.selection.replaceChildren(el("p", { class: "hint" }, "Drag a nucleus to move it. Click one to select it."));
      return;
    }
    const Z = snap.meta.atoms[i].Z;
    const e = this.elements.find((x) => x.Z === Z);
    const remove = el("button", { class: "quiet" }, "Remove");
    remove.onclick = () => this.send({ type: "remove_atom", index: i });
    if (snap.meta.atoms.length <= 1) remove.setAttribute("disabled", "");
    this.selection.replaceChildren(
      el("div", { class: "field" },
        el("span", { class: "selected-name" }, `${e?.symbol ?? "?"}${sub(i + 1)}`, el("span", { class: "muted" }, ` ${e?.name ?? ""}`)),
        remove),
    );
  }
}
