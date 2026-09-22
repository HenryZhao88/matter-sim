// Shapes of the messages the engine sends. Keep in sync with server/session.py
// and engine/simulation.py (Simulation.snapshot).

export interface ElementInfo {
  Z: number;
  symbol: string;
  name: string;
  color: string;
  available: boolean;
}

export interface Reference {
  energy_ha?: number;
  distances_A?: number[];
  angles_deg?: number[];
  binding_ev?: number;
  period_fs?: number;
  note?: string;
  method_note?: string;
}

export interface PresetInfo {
  id: string;
  name: string;
  formula: string;
  blurb: string;
  mode: Mode;
  reference: Reference;
}

export type Mode = "frozen" | "relax" | "dynamics";

export interface Hello {
  type: "hello";
  presets: PresetInfo[];
  elements: ElementInfo[];
  modes: Mode[];
  qualities: Record<string, number>;
  functionals: string[];
  backends: string[];
}

export interface Params {
  functional: string;
  quality: string;
  mode: Mode;
  backend: string;
  T_e: number;
  temperature_K: number;
  dt: number;
}

export interface Status {
  type: "status";
  running: boolean;
  busy: boolean;
  idle: boolean;
  preset: string | null;
  params: Params | null;
}

export interface AtomState {
  Z: number;
  pos: [number, number, number]; // bohr
  force: [number, number, number]; // Ha/bohr
  vel: [number, number, number];
}

export interface Meta {
  step: number;
  time_fs: number;
  mode: Mode;
  relaxed: boolean;
  params: Params & { h: number };
  backend: string;
  step_seconds: number;
  grid: { N: number; L: number; h: number; stride: number };
  system: { charge: number; multiplicity: number; n_electrons: number };
  atoms: AtomState[];
  energy_trace: (number | null)[];
  energy?: number;
  free_energy?: number;
  total_energy?: number;
  components?: Record<string, number>;
  converged?: boolean;
  scf_iterations?: number;
  scf_history?: { drho: number; energy: number }[];
  orbitals?: { up: { energy: number[]; occ: number[] }; down: { energy: number[]; occ: number[] } };
  boundary_leak?: number;
}

export interface Snapshot {
  meta: Meta;
  n: number;
  rhoMax: number;
  spinMax: number;
  rho: Uint8Array;
  spin: Int8Array;
}

export interface ScfProgress {
  type: "scf_progress";
  iter: number;
  energy: number;
  drho: number;
}

export interface VerifyResult {
  type: "verify_result";
  energy_f32: number;
  energy_f64: number;
  delta_energy: number;
  max_delta_force: number;
  seconds: number;
}

export type ServerEvent =
  | Hello
  | Status
  | ScfProgress
  | VerifyResult
  | { type: "log"; level: string; message: string }
  | { type: "error"; message: string };
