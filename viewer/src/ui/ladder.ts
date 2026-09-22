import { el } from "../format";

export type Workspace = "particles" | "atoms" | "matter";

// The scales of matter this project climbs. The two lit rungs open their workspace.
const RUNGS: { name: string; size: string; workspace?: Workspace; note: string }[] = [
  { name: "Particles and forces", size: "10⁻¹⁸ m", workspace: "particles",
    note: "The Standard Model: collide particles and see what the Lagrangian makes." },
  { name: "Electrons and nuclei", size: "10⁻¹⁰ m", workspace: "atoms",
    note: "Quantum electrons and moving nuclei: atoms and molecules." },
  { name: "Materials", size: "10⁻⁸ m", workspace: "matter",
    note: "Hundreds of atoms moving on forces learned from the rung below: melting, expansion, heat." },
  { name: "Everyday matter", size: "10⁻² m", workspace: "matter",
    note: "A cubic centimetre of aluminium, from the properties the atoms produced." },
];

export function renderLadder(root: HTMLElement, current: Workspace, onPick: (w: Workspace) => void): void {
  root.replaceChildren(
    ...RUNGS.map((r) => {
      const live = r.workspace !== undefined;
      const inner = [el("span", { class: "rung-name" }, r.name), el("span", { class: "rung-size" }, r.size)];
      const node = live
        ? el("button", { class: "rung-btn", title: r.note, "aria-pressed": String(r.workspace === current) }, ...inner)
        : el("span", { class: "rung-btn", title: r.note }, ...inner);
      if (live) node.addEventListener("click", () => onPick(r.workspace!));
      return el("li", { class: `rung${live ? " live" : ""}${r.workspace === current ? " current" : ""}` }, node);
    }),
  );
}
