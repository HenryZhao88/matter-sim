export interface ParticleMeta {
  symbol: string;
  family: string;
  mass: number;
  charge: number;
  colour: number;
}

export interface CParticle {
  name: string;
  p: [number, number, number, number];
  origin: [number, number, number];
  parent: number | null;
  status: "final" | "decayed" | "confined" | "invisible";
  decay_point: [number, number, number] | null;
  children: number[];
}

export interface CEvent {
  beams: [string, string];
  sqrt_s: number;
  channel: [string, string] | null;
  sigma_pb?: number;
  sigma_total_pb: number;
  particles: CParticle[];
}

export interface BeamInfo {
  id: string;
  pair: [string, string];
  label: string;
  note: string;
}

export interface OutcomeRow {
  final: [string, string];
  pb: number;
  share: number;
}

export type ColliderEvent =
  | { type: "collider.hello"; beams: BeamInfo[]; energies: number[]; particles: Record<string, ParticleMeta> }
  | { type: "collider.outcomes"; beams: [string, string]; sqrt_s: number; total_pb: number; rows: OutcomeRow[] }
  | { type: "collider.events"; beams: [string, string]; sqrt_s: number; events: CEvent[] }
  | { type: "collider.scan_row"; beams: [string, string]; sqrt_s: number; total_pb: number; top: { final: [string, string]; pb: number }[] }
  | { type: "collider.scan_done"; beams: [string, string] }
  | { type: "collider.progress"; message: string }
  | { type: "collider.particle"; name: string; mass: number; lifetime_s?: number; width?: number; confined: boolean;
      decays?: { products: string[]; br: number }[] };
