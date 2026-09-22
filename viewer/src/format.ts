export const BOHR_A = 0.529177210903;
export const HARTREE_EV = 27.211386245988;

const MINUS = "−";

export function num(x: number | null | undefined, digits = 4): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return "—";
  const s = Math.abs(x).toFixed(digits);
  return (x < 0 && Number(s) !== 0 ? MINUS : "") + s;
}

export function signed(x: number, digits = 3): string {
  if (!Number.isFinite(x)) return "—";
  return (x >= 0 ? "+" : MINUS) + Math.abs(x).toFixed(digits);
}

export function sci(x: number, digits = 1): string {
  if (!Number.isFinite(x) || x === 0) return "0";
  const e = Math.floor(Math.log10(Math.abs(x)));
  const m = x / 10 ** e;
  const sup = String(e)
    .replace("-", "⁻")
    .replace(/\d/g, (d) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[Number(d)]);
  return `${m.toFixed(digits)}×10${sup}`;
}

const SUBSCRIPT = "₀₁₂₃₄₅₆₇₈₉";
export function sub(n: number): string {
  return String(n).replace(/\d/g, (d) => SUBSCRIPT[Number(d)]);
}

export function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs: Record<string, string> = {},
  ...children: (Node | string)[]
): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else e.setAttribute(k, v);
  }
  e.append(...children);
  return e;
}
