import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { rich } from "../format";
import type { CEvent, CParticle, ParticleMeta } from "./types";

// A detector-style event display. Units are metres. Charged tracks curve in a solenoid
// field along the beam (z) axis, as in a real detector; that curvature is just the Lorentz
// force on each computed momentum. Nothing here decides what particles appear.
const FIELD_T = 3.8;
const TRACKER_R = 1.2;
const CALO_R = 1.8;
const MUON_R = 3.2;
const HALF_LEN = 3.0;

export const FAMILY_COLOUR: Record<string, string> = {
  lepton: "#7fd8ff",
  neutrino: "#9db8c7",
  photon: "#f6f1c8",
  quark: "#f0b455",
  gluon: "#f7d08a",
  boson: "#c9a7ff",
  higgs: "#ff8a7a",
  other: "#eaf2f5",
};

interface Track {
  line: THREE.Line;
  points: THREE.Vector3[];
  label: string;
  end: THREE.Vector3;
}

export class EventDisplay {
  private renderer: THREE.WebGLRenderer;
  private camera: THREE.PerspectiveCamera;
  private controls: OrbitControls;
  private scene = new THREE.Scene();
  private event = new THREE.Group();
  private tracks: Track[] = [];
  private t0 = 0;
  private labelsRoot: HTMLElement;
  private reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
  meta: Record<string, ParticleMeta> = {};

  constructor(private canvas: HTMLCanvasElement, labels: HTMLElement) {
    this.labelsRoot = labels;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.camera = new THREE.PerspectiveCamera(38, 1, 0.01, 100);
    this.camera.position.set(7.2, 3.6, 8.4);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.buildDetector();
    this.scene.add(this.event);
    new ResizeObserver(() => this.resize()).observe(canvas);
    this.resize();
    this.renderer.setAnimationLoop(() => this.frame());
  }

  private buildDetector(): void {
    const faint = new THREE.LineBasicMaterial({ color: 0x2a5470, transparent: true, opacity: 0.55 });
    const fainter = new THREE.LineBasicMaterial({ color: 0x2a5470, transparent: true, opacity: 0.28 });
    const ring = (r: number, z: number, mat: THREE.LineBasicMaterial) => {
      const pts = [];
      for (let i = 0; i <= 96; i++) {
        const a = (i / 96) * Math.PI * 2;
        pts.push(new THREE.Vector3(r * Math.cos(a), r * Math.sin(a), z));
      }
      this.scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), mat));
    };
    for (const [r, mat] of [[TRACKER_R, faint], [CALO_R, fainter], [MUON_R, fainter]] as const) {
      ring(r, -HALF_LEN, mat);
      ring(r, HALF_LEN, mat);
      for (let k = 0; k < 8; k++) {
        const a = (k / 8) * Math.PI * 2;
        const x = r * Math.cos(a), y = r * Math.sin(a);
        this.scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(
          [new THREE.Vector3(x, y, -HALF_LEN), new THREE.Vector3(x, y, HALF_LEN)]), mat));
      }
    }
    const beam = new THREE.LineBasicMaterial({ color: 0x9db8c7, transparent: true, opacity: 0.35 });
    this.scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(
      [new THREE.Vector3(0, 0, -4.5), new THREE.Vector3(0, 0, 4.5)]), beam));
    const dot = new THREE.Mesh(new THREE.SphereGeometry(0.03, 16, 12), new THREE.MeshBasicMaterial({ color: 0xf0b455 }));
    this.scene.add(dot);
  }

  show(ev: CEvent): void {
    this.event.clear();
    this.tracks = [];
    for (const p of ev.particles) {
      if (p.status === "decayed" && !this.flewVisibly(p)) continue;
      this.addParticle(p, ev);
    }
    const miss = this.missingMomentum(ev);
    if (miss) this.addMissing(miss);
    this.t0 = performance.now();
  }

  clear(): void {
    this.event.clear();
    this.tracks = [];
    this.labelsRoot.replaceChildren();
  }

  private flewVisibly(p: CParticle): boolean {
    if (!p.decay_point) return false;
    const d = Math.hypot(p.decay_point[0] - p.origin[0], p.decay_point[1] - p.origin[1], p.decay_point[2] - p.origin[2]);
    return d > 0.002;
  }

  private addParticle(p: CParticle, ev: CEvent): void {
    const meta = this.meta[p.name];
    const colour = new THREE.Color(FAMILY_COLOUR[meta?.family ?? "other"]);
    const origin = new THREE.Vector3(...p.origin);
    const [E, px, py, pz] = p.p;
    const pvec = new THREE.Vector3(px, py, pz);
    const pmag = pvec.length();
    const label = meta?.symbol ?? p.name;
    if (p.status === "confined") {
      // A jet would form here; draw the cone its hadrons would fill, and say so.
      const len = Math.min(CALO_R, 0.6 + 0.35 * Math.log1p(E));
      const dir = pvec.clone().normalize();
      const cone = new THREE.Mesh(
        new THREE.ConeGeometry(len * 0.16, len, 24, 1, true),
        new THREE.MeshBasicMaterial({ color: colour, transparent: true, opacity: 0.22, side: THREE.DoubleSide, depthWrite: false }),
      );
      cone.position.copy(origin.clone().addScaledVector(dir, len / 2));
      cone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.clone().negate()); // apex at the collision
      this.event.add(cone);
      const axis = this.straight(origin, dir, len, colour, 0.9);
      this.tracks.push({ ...axis, label: `${label} (confined)`, end: origin.clone().addScaledVector(dir, len) });
      return;
    }
    const q = meta?.charge ?? 0;
    const mass = meta?.mass ?? 0;
    let pts: THREE.Vector3[];
    const limitR = p.name.startsWith("mu") ? MUON_R : p.status === "invisible" ? MUON_R + 0.6 : q !== 0 ? TRACKER_R : CALO_R;
    if (p.status === "decayed" && p.decay_point) {
      const end = new THREE.Vector3(...p.decay_point);
      pts = [origin, end];
    } else if (Math.abs(q) > 1e-9) {
      pts = helix(origin, pvec, q, limitR);
    } else {
      pts = [origin, pathToCylinder(origin, pvec.clone().normalize(), limitR)];
    }
    const dashed = p.status === "invisible";
    const mat = dashed
      ? new THREE.LineDashedMaterial({ color: colour, dashSize: 0.08, gapSize: 0.07, transparent: true, opacity: 0.6 })
      : new THREE.LineBasicMaterial({ color: colour, transparent: true, opacity: 0.95 });
    const geo = new THREE.BufferGeometry().setFromPoints(pts);
    const line = new THREE.Line(geo, mat);
    if (dashed) line.computeLineDistances();
    geo.setDrawRange(0, 0);
    this.event.add(line);
    const end = pts[pts.length - 1];
    if (!dashed && q === 0 && mass === 0) {
      // calorimeter deposit sized by energy
      const box = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.08, 0.02 + 0.04 * Math.log1p(E)),
        new THREE.MeshBasicMaterial({ color: colour, transparent: true, opacity: 0.8 }));
      box.position.copy(end);
      box.lookAt(0, 0, end.z);
      this.event.add(box);
    }
    this.tracks.push({ line, points: pts, label, end });
    void ev;
    void pmag;
  }

  private straight(origin: THREE.Vector3, dir: THREE.Vector3, len: number, colour: THREE.Color, opacity: number) {
    const pts = [origin.clone(), origin.clone().addScaledVector(dir, len)];
    const geo = new THREE.BufferGeometry().setFromPoints(pts);
    const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ color: colour, transparent: true, opacity }));
    geo.setDrawRange(0, 0);
    this.event.add(line);
    return { line, points: pts };
  }

  private missingMomentum(ev: CEvent): THREE.Vector3 | null {
    let x = 0, y = 0;
    for (const p of ev.particles) if (p.status === "invisible") { x += p.p[1]; y += p.p[2]; }
    return Math.hypot(x, y) > 1 ? new THREE.Vector3(x, y, 0) : null;
  }

  private addMissing(v: THREE.Vector3): void {
    const dir = v.clone().normalize();
    const arrow = new THREE.ArrowHelper(dir, new THREE.Vector3(), CALO_R + 0.4, 0x9db8c7, 0.18, 0.09);
    (arrow.line.material as THREE.LineBasicMaterial).transparent = true;
    (arrow.line.material as THREE.LineBasicMaterial).opacity = 0.5;
    this.event.add(arrow);
    this.tracks.push({ line: arrow.line as unknown as THREE.Line, points: [], label: `missing ${v.length().toFixed(0)} GeV`,
      end: dir.clone().multiplyScalar(CALO_R + 0.5) });
  }

  private resize(): void {
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  private frame(): void {
    this.controls.update();
    const k = this.reduceMotion ? 1 : Math.min(1, (performance.now() - this.t0) / 900);
    const e = 1 - Math.pow(1 - k, 3);
    for (const t of this.tracks) {
      const n = t.points.length;
      if (n) t.line.geometry.setDrawRange(0, Math.max(2, Math.ceil(n * e)));
    }
    this.renderer.render(this.scene, this.camera);
    this.placeLabels(k);
  }

  private placeLabels(progress: number): void {
    if (!this.canvas.clientWidth) return;
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    const nodes = this.tracks.slice(0, 14).map((t) => {
      const v = t.end.clone().project(this.camera);
      const div = rich(t.label);
      div.className = "p-label";
      // keep labels inside the view even when a track leaves it
      const x = Math.min(Math.max(((v.x + 1) / 2) * w + 6, 8), w - 90);
      const y = Math.min(Math.max(((1 - v.y) / 2) * h - 8, 8), h - 40);
      div.style.transform = `translate(${x}px, ${y}px)`;
      div.style.opacity = v.z > 1 ? "0" : String(progress);
      return div;
    });
    this.labelsRoot.replaceChildren(...nodes);
  }
}

function pathToCylinder(o: THREE.Vector3, d: THREE.Vector3, R: number): THREE.Vector3 {
  const a = d.x * d.x + d.y * d.y;
  let t = 20;
  if (a > 1e-9) {
    const b = 2 * (o.x * d.x + o.y * d.y);
    const c = o.x * o.x + o.y * o.y - R * R;
    t = (-b + Math.sqrt(Math.max(b * b - 4 * a * c, 0))) / (2 * a);
  }
  if (Math.abs(d.z) > 1e-9) t = Math.min(t, (Math.sign(d.z) * HALF_LEN - o.z) / d.z);
  return o.clone().addScaledVector(d, Math.max(t, 0));
}

/** Helix of a charge q (units of e) with momentum p (GeV) in a field along z. */
function helix(o: THREE.Vector3, p: THREE.Vector3, q: number, R: number): THREE.Vector3[] {
  const pt = Math.hypot(p.x, p.y);
  const pts: THREE.Vector3[] = [o.clone()];
  if (pt < 1e-6) return [o.clone(), pathToCylinder(o, p.clone().normalize(), R)];
  const radius = pt / (0.299792458 * FIELD_T * Math.abs(q));     // metres
  const phi0 = Math.atan2(p.y, p.x);
  const sgn = -Math.sign(q);
  const dzds = p.z / pt;
  const steps = 160;
  const sMax = Math.min(Math.PI * radius, 12);
  for (let i = 1; i <= steps; i++) {
    const s = (sMax * i) / steps;                                   // transverse arc length
    const a = s / radius;
    const x = o.x + radius * (Math.sin(phi0 + sgn * a) - Math.sin(phi0)) * sgn;
    const y = o.y - radius * (Math.cos(phi0 + sgn * a) - Math.cos(phi0)) * sgn;
    const z = o.z + dzds * s;
    const v = new THREE.Vector3(x, y, z);
    if (Math.hypot(x, y) > R || Math.abs(z) > HALF_LEN) {
      pts.push(v);
      break;
    }
    pts.push(v);
  }
  return pts;
}
