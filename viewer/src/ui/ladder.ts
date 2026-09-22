import { el } from "../format";

// The scales of matter this project climbs. Only rung 2 exists today.
const RUNGS = [
  { name: "Quarks and nuclei", size: "10⁻¹⁵ m", live: false, note: "Not simulated. Nuclei enter as measured charge and mass." },
  { name: "Electrons and nuclei", size: "10⁻¹⁰ m", live: true, note: "Running now: quantum electrons, moving nuclei." },
  { name: "Materials", size: "10⁻⁸ m", live: false, note: "Next: forces learned from this rung drive many-atom simulations." },
  { name: "Everyday matter", size: "10⁻² m", live: false, note: "The long goal: a cubic centimetre of aluminium." },
];

export function renderLadder(root: HTMLElement): void {
  root.replaceChildren(
    ...RUNGS.map((r) =>
      el("li", { class: r.live ? "rung live" : "rung", title: r.note, "aria-current": r.live ? "step" : "false" },
        el("span", { class: "rung-name" }, r.name),
        el("span", { class: "rung-size" }, r.size)),
    ),
  );
}
