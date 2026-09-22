import "@fontsource/stix-two-text/400.css";
import "@fontsource/stix-two-text/500.css";
import "@fontsource/stix-two-text/400-italic.css";
import "@fontsource/ibm-plex-sans/400.css";
import "@fontsource/ibm-plex-sans/500.css";
import "@fontsource/ibm-plex-sans/600.css";
import "./style.css";

import { el, num, sci } from "./format";
import { Link } from "./net";
import { type RenderMode, Viewport } from "./scene/viewport";
import type { Hello, PresetInfo, ServerEvent, Snapshot, Status } from "./types";
import { BuildRail } from "./ui/build";
import { Collider } from "./particles/collider";
import type { ColliderEvent } from "./particles/types";
import { renderLadder, type Workspace } from "./ui/ladder";
import { MeasureRail } from "./ui/measure";
import { RunBar } from "./ui/runbar";

const $ = (id: string) => document.getElementById(id)!;

let hello: Hello | null = null;
let status: Status | null = null;
let snap: Snapshot | null = null;

const link = new Link(onEvent, onSnapshot, (up) => {
  $("offline").hidden = up;
  if (up && document.body.dataset.workspace === "particles") collider.activate();
});
const send = (cmd: Record<string, unknown>) => link.send(cmd);

const viewport = new Viewport($("view") as HTMLCanvasElement);
const build = new BuildRail($("build"), send);
const measure = new MeasureRail($("measure"));
const runbar = new RunBar($("runbar"), send);
buildViewControls($("view-controls"));

const collider = new Collider($("p-left"), $("p-right"), $("p-bar"), $("p-view") as HTMLCanvasElement,
  $("p-labels"), $("p-legend"), $("p-progress"), $("p-empty"), send);

function setWorkspace(w: Workspace): void {
  document.body.dataset.workspace = w;
  renderLadder($("ladder"), w, setWorkspace);
  try { localStorage.setItem("workspace", w); } catch { /* storage may be unavailable */ }
  if (w === "particles") collider.activate();
}
let savedWorkspace: Workspace = "atoms";
try { if (localStorage.getItem("workspace") === "particles") savedWorkspace = "particles"; } catch { /* ignore */ }
setWorkspace(savedWorkspace);

viewport.onSelect = (i) => build.setSelected(i);
viewport.onMoveAtom = (index, pos) => send({ type: "move_atom", index, pos });
viewport.onFrame = updateScaleBar;

function currentPreset(): PresetInfo | null {
  return hello?.presets.find((p) => p.id === status?.preset) ?? null;
}

function onEvent(e: ServerEvent | ColliderEvent): void {
  if (e.type.startsWith("collider.")) {
    collider.onEvent(e as ColliderEvent);
    return;
  }
  e = e as ServerEvent;
  switch (e.type) {
    case "hello":
      hello = e;
      viewport.setElements(e.elements);
      measure.setElements(e.elements);
      build.setHello(e);
      runbar.setHello(e);
      break;
    case "status": {
      const presetChanged = status?.preset !== e.preset;
      status = e;
      if (presetChanged) viewport.resetFraming();
      if (!e.busy) $("progress").hidden = true;
      build.update(status, snap);
      runbar.update(status, snap);
      break;
    }
    case "scf_progress":
      $("progress").hidden = false;
      $("progress").textContent = measure.progress(e);
      break;
    case "verify_result": {
      const good = Math.abs(e.delta_energy) < 1e-4 && e.max_delta_force < 1e-3;
      toast(
        `${good ? "32-bit result confirmed." : "32-bit and 64-bit disagree."} ` +
          `Energy differs by ${sci(Math.abs(e.delta_energy))} Ha, forces by ${sci(e.max_delta_force)} Ha/bohr ` +
          `(checked in ${num(e.seconds, 1)} s).`,
        good ? "info" : "warn",
      );
      break;
    }
    case "spin_scan": {
      const rows = [...e.rows].sort((a, b) => a.energy - b.energy);
      const low = rows[0].energy;
      const parts = rows.map((r) => `${r.multiplicity - 1} unpaired: ${r.energy === low ? "lowest" : `+${num((r.energy - low) * 27.2114, 2)} eV`}`);
      toast(`Lowest energy with ${e.best - 1} unpaired spin${e.best === 2 ? "" : "s"}. ${parts.join(", ")}.`, "info");
      break;
    }
    case "log":
      toast(e.message, e.level === "info" ? "info" : "warn");
      break;
    case "error":
      toast(e.message, "error");
      break;
  }
}

function onSnapshot(s: Snapshot): void {
  snap = s;
  viewport.update(s);
  measure.update(s, currentPreset());
  build.update(status, s);
  runbar.update(status, s);
}

// ------------------------------------------------------------------ overlays
function toast(message: string, kind: "info" | "warn" | "error"): void {
  const t = el("div", { class: `toast ${kind}`, role: kind === "error" ? "alert" : "status" }, message);
  $("toasts").append(t);
  setTimeout(() => t.classList.add("leaving"), kind === "error" ? 9000 : 6000);
  setTimeout(() => t.remove(), kind === "error" ? 9600 : 6600);
}

function updateScaleBar(): void {
  const ppa = viewport.pixelsPerAngstrom();
  const nice = [0.1, 0.2, 0.5, 1, 2, 5, 10];
  const len = nice.find((x) => x * ppa >= 70) ?? 10;
  const bar = $("scalebar");
  (bar.querySelector(".bar") as HTMLElement).style.width = `${len * ppa}px`;
  (bar.querySelector(".label") as HTMLElement).textContent = `${len} Å`;
}

function buildViewControls(root: HTMLElement): void {
  const modes: [RenderMode, string, string][] = [
    ["glow", "Glow", "Density as light, like a long exposure"],
    ["contours", "Contours", "Stacked sheets of equal density, as in early electron-density maps"],
    ["surface", "Surface", "One surface of constant density"],
  ];
  const seg = el("div", { class: "segmented small", role: "radiogroup", "aria-label": "How to draw the electrons" });
  const iso = el("input", { type: "range", min: "0.2", max: "0.9", step: "0.01", value: "0.55", "aria-label": "Surface density" });
  const isoField = el("label", { class: "inline-field" }, "Level ", iso);
  isoField.hidden = true;
  for (const [m, label, title] of modes) {
    const b = el("button", { role: "radio", "aria-checked": String(m === "glow"), title }, label);
    b.onclick = () => {
      viewport.setRenderMode(m);
      for (const x of seg.children) x.setAttribute("aria-checked", String(x === b));
      isoField.hidden = m !== "surface";
    };
    seg.append(b);
  }
  iso.oninput = () => viewport.setIso(Number(iso.value));
  const exposure = el("input", { type: "range", min: "0.2", max: "3", step: "0.05", value: "1", "aria-label": "Brightness" });
  exposure.oninput = () => viewport.setExposure(Number(exposure.value));
  const spin = el("input", { type: "checkbox" });
  spin.onchange = () => viewport.setSpinColor(spin.checked);
  const forces = el("input", { type: "checkbox", checked: "" });
  forces.onchange = () => viewport.setForcesVisible(forces.checked);
  root.append(
    seg,
    el("label", { class: "inline-field" }, "Brightness ", exposure),
    isoField,
    el("label", { class: "check" }, spin, el("span", {}, "Colour by spin")),
    el("label", { class: "check" }, forces, el("span", {}, "Show forces")),
  );
}
if (import.meta.env.DEV) (window as unknown as Record<string, unknown>).__viewport = viewport;
