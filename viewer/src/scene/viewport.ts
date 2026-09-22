import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { BOHR_A } from "../format";
import type { ElementInfo, Snapshot } from "../types";
import { volumeFragment, volumeVertex } from "./volumeShader";

export type RenderMode = "glow" | "contours" | "surface";

interface AtomVisual {
  group: THREE.Group;
  core: THREE.Mesh;
  halo: THREE.Sprite;
  hit: THREE.Mesh;
  arrow: THREE.ArrowHelper;
  from: THREE.Vector3;
  to: THREE.Vector3;
  Z: number;
}

const MODE_INDEX: Record<RenderMode, number> = { glow: 0, contours: 1, surface: 2 };

// World units are ångströms; the engine speaks bohr.
export class Viewport {
  readonly renderer: THREE.WebGLRenderer;
  readonly camera: THREE.PerspectiveCamera;
  readonly controls: OrbitControls;
  private scene = new THREE.Scene();
  private volume: THREE.Mesh;
  material: THREE.ShaderMaterial;
  private densityTex: THREE.Data3DTexture | null = null;
  private spinTex: THREE.Data3DTexture | null = null;
  private atoms: AtomVisual[] = [];
  private haloTexture = makeHaloTexture();
  private elements = new Map<number, ElementInfo>();
  private animStart = 0;
  private raycaster = new THREE.Raycaster();
  private drag: { index: number; plane: THREE.Plane; offset: THREE.Vector3 } | null = null;
  private framed = false;
  selected: number | null = null;
  showForces = true;

  onSelect: (index: number | null) => void = () => {};
  onMoveAtom: (index: number, posBohr: [number, number, number]) => void = () => {};
  onFrame: () => void = () => {};

  constructor(private canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.setClearColor(0x000000, 0);
    this.camera = new THREE.PerspectiveCamera(32, 1, 0.01, 200);
    this.camera.position.set(2.2, 1.6, 7.5);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.minDistance = 1;
    this.controls.maxDistance = 40;

    this.material = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: volumeVertex,
      fragmentShader: volumeFragment,
      side: THREE.BackSide,
      transparent: true,
      depthWrite: false,
      depthTest: false,
      blending: THREE.CustomBlending,
      blendSrc: THREE.OneFactor,
      blendDst: THREE.OneMinusSrcAlphaFactor,
      uniforms: {
        uDensity: { value: null },
        uSpin: { value: null },
        uCamLocal: { value: new THREE.Vector3() },
        uMode: { value: 0 },
        uExposure: { value: 1.0 },
        uIso: { value: 0.55 },
        uSheets: { value: 13 },
        uSpinColor: { value: false },
        uTexel: { value: 1 / 64 },
        uLogMax: { value: 8.0 },
        uGain: { value: 14.0 },
        uSpinRatio: { value: 1.0 },
      },
    });
    this.volume = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), this.material);
    this.volume.renderOrder = 0;
    this.volume.visible = false;
    this.scene.add(this.volume);

    new ResizeObserver(() => this.resize()).observe(canvas);
    this.resize();
    this.bindPointer();
    this.renderer.setAnimationLoop(() => this.frame());
  }

  setElements(list: ElementInfo[]): void {
    for (const e of list) this.elements.set(e.Z, e);
  }

  setRenderMode(mode: RenderMode): void {
    this.material.uniforms.uMode.value = MODE_INDEX[mode];
  }
  setSpinColor(on: boolean): void {
    this.material.uniforms.uSpinColor.value = on;
  }
  setExposure(x: number): void {
    this.material.uniforms.uExposure.value = x;
  }
  setIso(x: number): void {
    this.material.uniforms.uIso.value = x;
  }
  setForcesVisible(on: boolean): void {
    this.showForces = on;
    for (const a of this.atoms) a.arrow.visible = on;
  }

  /** Pixels per ångström at the orbit target (for the scale bar). */
  pixelsPerAngstrom(): number {
    const dist = this.camera.position.distanceTo(this.controls.target);
    const h = this.canvas.clientHeight;
    return h / (2 * Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2)) * dist);
  }

  update(snap: Snapshot): void {
    this.updateVolume(snap);
    this.updateAtoms(snap);
    if (!this.framed) {
      this.frameAtoms();
      this.framed = true;
    }
  }

  resetFraming(): void {
    this.framed = false;
  }

  private updateVolume(snap: Snapshot): void {
    const { n, meta } = snap;
    if (!this.densityTex || this.densityTex.image.width !== n) {
      this.densityTex?.dispose();
      this.spinTex?.dispose();
      this.densityTex = make3D(new Uint8Array(n * n * n), n);
      this.spinTex = make3D(new Uint8Array(n * n * n), n);
      this.material.uniforms.uDensity.value = this.densityTex;
      this.material.uniforms.uSpin.value = this.spinTex;
      this.material.uniforms.uTexel.value = 1 / n;
    }
    (this.densityTex.image.data as Uint8Array).set(snap.rho);
    const spin = this.spinTex!.image.data as Uint8Array;
    for (let i = 0; i < spin.length; i++) spin[i] = snap.spin[i] + 128;
    this.material.uniforms.uLogMax.value = Math.log1p(snap.rhoMax / 1e-4);
    this.material.uniforms.uSpinRatio.value = snap.spinMax / Math.max(snap.rhoMax, 1e-12);
    this.densityTex.needsUpdate = true;
    this.spinTex!.needsUpdate = true;

    // Texel k of the (block-averaged) grid is centred at (k*s + (s-1)/2 - N/2) h.
    const { N, h, stride: s } = meta.grid;
    const size = n * s * h * BOHR_A;
    const min = (-(N + 1) / 2) * h * BOHR_A;
    const centre = min + size / 2;
    this.volume.scale.setScalar(size);
    this.volume.position.setScalar(centre);
    this.volume.visible = true;
  }

  private updateAtoms(snap: Snapshot): void {
    const list = snap.meta.atoms;
    const rebuild = list.length !== this.atoms.length || list.some((a, i) => a.Z !== this.atoms[i].Z);
    if (rebuild) {
      for (const a of this.atoms) this.scene.remove(a.group);
      this.atoms = list.map((a) => this.makeAtom(a.Z));
      for (const a of this.atoms) this.scene.add(a.group);
      if (this.selected !== null && this.selected >= list.length) this.select(null);
    }
    list.forEach((a, i) => {
      const v = this.atoms[i];
      const target = new THREE.Vector3(...a.pos).multiplyScalar(BOHR_A);
      v.from.copy(rebuild ? target : v.group.position);
      v.to.copy(target);
      if (rebuild) v.group.position.copy(target);
      const f = new THREE.Vector3(...a.force);
      const mag = f.length();
      // 1 Ha/bohr ≈ 51 eV/Å; draw 0.1 Ha/bohr as ~1 Å, compressed for big forces.
      const len = Math.min(1.8, 10 * mag / (1 + 4 * mag));
      if (mag > 1e-4 && len > 0.06) {
        v.arrow.setDirection(f.normalize());
        v.arrow.setLength(len, Math.min(0.14, len * 0.35), Math.min(0.07, len * 0.18));
        v.arrow.visible = this.showForces;
      } else v.arrow.visible = false;
    });
    this.animStart = performance.now();
  }

  private makeAtom(Z: number): AtomVisual {
    const info = this.elements.get(Z);
    const color = new THREE.Color(info?.color ?? "#ffffff");
    const r = 0.022 * Math.cbrt(Z) + 0.012;
    const group = new THREE.Group();
    const core = new THREE.Mesh(
      new THREE.SphereGeometry(r, 24, 16),
      new THREE.MeshBasicMaterial({ color, depthTest: false, transparent: true }),
    );
    core.renderOrder = 2;
    const halo = new THREE.Sprite(
      new THREE.SpriteMaterial({
        map: this.haloTexture, color, blending: THREE.AdditiveBlending,
        depthTest: false, depthWrite: false, transparent: true, opacity: 0.9,
      }),
    );
    halo.scale.setScalar(r * 7);
    halo.renderOrder = 1;
    const hit = new THREE.Mesh(new THREE.SphereGeometry(0.28, 8, 6), new THREE.MeshBasicMaterial({ visible: false }));
    const arrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 0.5, 0xf0b455, 0.12, 0.06);
    (arrow.line.material as THREE.LineBasicMaterial).depthTest = false;
    (arrow.cone.material as THREE.MeshBasicMaterial).depthTest = false;
    arrow.renderOrder = 3;
    arrow.visible = false;
    group.add(halo, core, hit, arrow);
    return { group, core, halo, hit, arrow, from: new THREE.Vector3(), to: new THREE.Vector3(), Z };
  }

  private frameAtoms(): void {
    if (!this.atoms.length) return;
    const box = new THREE.Box3();
    for (const a of this.atoms) box.expandByPoint(a.to);
    const c = box.getCenter(new THREE.Vector3());
    const radius = Math.max(box.getSize(new THREE.Vector3()).length() / 2, 0.6) + 1.6;
    const dist = radius / Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2));
    const dir = new THREE.Vector3(0.28, 0.22, 1).normalize();
    this.controls.target.copy(c);
    this.camera.position.copy(c).addScaledVector(dir, dist);
  }

  select(i: number | null): void {
    this.selected = i;
    this.atoms.forEach((a, k) => {
      (a.halo.material as THREE.SpriteMaterial).opacity = k === i ? 1.0 : 0.9;
      a.halo.scale.setScalar((0.022 * Math.cbrt(a.Z) + 0.012) * (k === i ? 13 : 7));
    });
    this.onSelect(i);
  }

  // ---------------------------------------------------------------- input
  private ndc(ev: PointerEvent): THREE.Vector2 {
    const r = this.canvas.getBoundingClientRect();
    return new THREE.Vector2(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
  }

  private pick(ev: PointerEvent): number | null {
    this.raycaster.setFromCamera(this.ndc(ev), this.camera);
    const hits = this.raycaster.intersectObjects(this.atoms.map((a) => a.hit), false);
    if (!hits.length) return null;
    return this.atoms.findIndex((a) => a.hit === hits[0].object);
  }

  private bindPointer(): void {
    let downAt: { x: number; y: number } | null = null;
    this.canvas.addEventListener("pointerdown", (ev) => {
      downAt = { x: ev.clientX, y: ev.clientY };
      const i = this.pick(ev);
      if (i === null || ev.button !== 0) return;
      const normal = this.camera.getWorldDirection(new THREE.Vector3()).negate();
      const pos = this.atoms[i].group.position.clone();
      const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(normal, pos);
      this.raycaster.setFromCamera(this.ndc(ev), this.camera);
      const hitPt = this.raycaster.ray.intersectPlane(plane, new THREE.Vector3()) ?? pos.clone();
      this.drag = { index: i, plane, offset: pos.sub(hitPt) };
      this.controls.enabled = false;
      this.canvas.setPointerCapture(ev.pointerId);
      this.select(i);
    });
    this.canvas.addEventListener("pointermove", (ev) => {
      if (!this.drag) {
        this.canvas.style.cursor = this.pick(ev) !== null ? "grab" : "";
        return;
      }
      this.canvas.style.cursor = "grabbing";
      this.raycaster.setFromCamera(this.ndc(ev), this.camera);
      const p = this.raycaster.ray.intersectPlane(this.drag.plane, new THREE.Vector3());
      if (!p) return;
      const a = this.atoms[this.drag.index];
      a.group.position.copy(p.add(this.drag.offset));
      a.from.copy(a.group.position);
      a.to.copy(a.group.position);
      a.arrow.visible = false;
    });
    this.canvas.addEventListener("pointerup", (ev) => {
      if (this.drag) {
        const a = this.atoms[this.drag.index];
        const p = a.group.position.clone().divideScalar(BOHR_A);
        this.onMoveAtom(this.drag.index, [p.x, p.y, p.z]);
        this.drag = null;
        this.controls.enabled = true;
        this.canvas.style.cursor = "grab";
      } else if (downAt && Math.hypot(ev.clientX - downAt.x, ev.clientY - downAt.y) < 4) {
        this.select(this.pick(ev));
      }
      downAt = null;
    });
  }

  // ---------------------------------------------------------------- frame
  private resize(): void {
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  private frame(): void {
    this.controls.update();
    const k = Math.min(1, (performance.now() - this.animStart) / 220);
    const e = 1 - Math.pow(1 - k, 3);
    for (const [i, a] of this.atoms.entries()) {
      if (this.drag?.index === i) continue;
      a.group.position.lerpVectors(a.from, a.to, e);
    }
    this.volume.updateMatrixWorld();
    const local = this.volume.worldToLocal(this.camera.position.clone());
    this.material.uniforms.uCamLocal.value.copy(local);
    this.renderer.render(this.scene, this.camera);
    this.onFrame();
  }
}

function make3D(data: Uint8Array, n: number): THREE.Data3DTexture {
  const t = new THREE.Data3DTexture(data, n, n, n);
  t.format = THREE.RedFormat;
  t.type = THREE.UnsignedByteType;
  t.minFilter = THREE.LinearFilter;
  t.magFilter = THREE.LinearFilter;
  t.wrapS = t.wrapT = t.wrapR = THREE.ClampToEdgeWrapping;
  t.unpackAlignment = 1;
  t.needsUpdate = true;
  return t;
}

function makeHaloTexture(): THREE.Texture {
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const g = c.getContext("2d")!;
  const grd = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  grd.addColorStop(0, "rgba(255,255,255,0.95)");
  grd.addColorStop(0.18, "rgba(255,255,255,0.45)");
  grd.addColorStop(0.45, "rgba(255,255,255,0.08)");
  grd.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = grd;
  g.fillRect(0, 0, 128, 128);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}
