// Raymarching shaders for the electron density.
//
// Density arrives log-scaled in [0, 1] (see server/protocol.py). The texture is
// laid out with the engine's z index fastest, so samples use p.zyx.
//
// Modes:
//   0  glow      emission–absorption, like a long exposure on a cyanotype plate
//   1  contours  stacked iso-density sheets, after Hodgkin's perspex density maps
//   2  surface   a single translucent iso-density surface

export const volumeVertex = /* glsl */ `
out vec3 vLocal;
void main() {
  vLocal = position;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

export const volumeFragment = /* glsl */ `
precision highp float;
precision highp sampler3D;

uniform sampler3D uDensity;
uniform sampler3D uSpin;
uniform vec3 uCamLocal;
uniform int uMode;
uniform float uExposure;
uniform float uIso;
uniform float uSheets;
uniform bool uSpinColor;
uniform float uTexel;
uniform float uLogMax;
uniform float uGain;
uniform float uSpinRatio;

in vec3 vLocal;
out vec4 outColor;

const vec3 FAINT  = vec3(0.26, 0.52, 0.70);
const vec3 BRIGHT = vec3(0.93, 0.97, 0.99);
const vec3 UP     = vec3(1.00, 0.54, 0.48);
const vec3 DOWN   = vec3(0.50, 0.85, 1.00);

vec2 hitBox(vec3 o, vec3 d) {
  vec3 inv = 1.0 / d;
  vec3 a = (vec3(-0.5) - o) * inv;
  vec3 b = (vec3( 0.5) - o) * inv;
  vec3 lo = min(a, b), hi = max(a, b);
  return vec2(max(max(lo.x, lo.y), lo.z), min(min(hi.x, hi.y), hi.z));
}

float dens(vec3 p) { return texture(uDensity, (p + 0.5).zyx).r; }
// Local spin polarisation ζ = (ρ↑ − ρ↓) / ρ in [-1, 1].
float spinAt(vec3 p) {
  float s = texture(uSpin, (p + 0.5).zyx).r * 255.0 / 127.0 - 128.0 / 127.0;  // int8 / 127
  float spin = sign(s) * s * s * uSpinRatio;          // in units of ρ_max
  float q = dens(p);
  float rho = (exp(q * uLogMax) - 1.0) / (exp(uLogMax) - 1.0);
  return clamp(spin / max(rho, 1e-4), -1.0, 1.0);
}

vec3 grad(vec3 p) {
  float e = uTexel;
  return vec3(
    dens(p + vec3(e, 0, 0)) - dens(p - vec3(e, 0, 0)),
    dens(p + vec3(0, e, 0)) - dens(p - vec3(0, e, 0)),
    dens(p + vec3(0, 0, e)) - dens(p - vec3(0, 0, e))) / (2.0 * e);
}

vec3 tint(float q, float s) {
  vec3 c = mix(FAINT, BRIGHT, smoothstep(0.35, 0.97, q));
  if (uSpinColor) {
    float a = clamp(abs(s), 0.0, 1.0) * 0.85;
    c = mix(c, s > 0.0 ? UP : DOWN, a);
  }
  return c;
}

float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }

void main() {
  vec3 o = uCamLocal;
  vec3 d = normalize(vLocal - uCamLocal);
  vec2 t = hitBox(o, d);
  float t0 = max(t.x, 0.0), t1 = t.y;
  if (t1 <= t0) discard;

  vec3 col = vec3(0.0);
  float acc = 0.0;

  if (uMode == 0) {
    // Emission proportional to the real (linear) density, plus a faint
    // log-scaled tail so the outskirts stay visible; tone-mapped at the end.
    const int STEPS = 200;
    float dt = (t1 - t0) / float(STEPS);
    float tt = t0 + dt * hash(gl_FragCoord.xy);
    vec3 light = vec3(0.0);
    for (int i = 0; i < STEPS; i++) {
      vec3 p = o + d * tt;
      float q = dens(p);
      if (q > 0.05) {
        float lin = (exp(q * uLogMax) - 1.0) / (exp(uLogMax) - 1.0);
        float e = 0.9 * sqrt(lin) + 0.35 * pow(q, 5.0);
        light += tint(0.25 + 0.75 * sqrt(lin), spinAt(p)) * e * dt;
      }
      tt += dt;
    }
    light *= uExposure * uGain;
    col = 1.0 - exp(-light);
    acc = clamp(max(col.r, max(col.g, col.b)), 0.0, 1.0);
  } else if (uMode == 1) {
    int S = int(uSheets);
    bool forward = d.z > 0.0;
    for (int k = 0; k < 64; k++) {
      if (k >= S) break;
      int kk = forward ? k : S - 1 - k;
      float z = -0.5 + (float(kk) + 0.5) / uSheets;
      float tz = (z - o.z) / d.z;
      if (tz < t0 || tz > t1) continue;
      vec3 p = o + d * tz;
      float q = dens(p);
      if (q < 0.4) continue;
      vec2 g = grad(p).xy;
      float L = 6.0;
      float f = (q - 0.4) / 0.6 * L;
      float w = abs(fract(f + 0.5) - 0.5) / (L * max(length(g), 1e-3));
      float line = 1.0 - smoothstep(0.0018, 0.0042, w);
      float fill = 0.035 * smoothstep(0.45, 1.0, q);
      float a = clamp(line * (0.25 + 0.55 * q) + fill, 0.0, 1.0) * uExposure * 0.7;
      a = clamp(a, 0.0, 1.0);
      col += (1.0 - acc) * tint(q, spinAt(p)) * a;
      acc += (1.0 - acc) * a;
    }
  } else {
    const int STEPS = 256;
    float dt = (t1 - t0) / float(STEPS);
    float tt = t0 + dt * hash(gl_FragCoord.xy);
    float prev = dens(o + d * tt);
    for (int i = 0; i < STEPS; i++) {
      tt += dt;
      vec3 p = o + d * tt;
      float q = dens(p);
      if (q >= uIso && prev < uIso) {
        float lo = tt - dt, hi = tt;
        for (int j = 0; j < 6; j++) {
          float mid = 0.5 * (lo + hi);
          if (dens(o + d * mid) >= uIso) hi = mid; else lo = mid;
        }
        p = o + d * hi;
        vec3 n = -normalize(grad(p) + 1e-6);
        vec3 L = normalize(vec3(0.45, 0.75, 0.5));
        float diff = max(dot(n, L), 0.0);
        float rim = pow(1.0 - max(dot(n, -d), 0.0), 3.0);
        vec3 base = tint(0.75, spinAt(p));
        col = base * (0.28 + 0.72 * diff) + rim * 0.45 * BRIGHT;
        acc = 0.86;
        col *= acc;
        break;
      }
      prev = q;
    }
  }

  if (acc < 0.002) discard;
  outColor = vec4(col, acc);
}
`;
