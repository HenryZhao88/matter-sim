import type { ServerEvent, Snapshot } from "./types";

// WebSocket link to the engine. Reconnects on its own.
export class Link {
  private ws: WebSocket | null = null;
  private retry = 500;

  constructor(
    private onEvent: (e: ServerEvent) => void,
    private onSnapshot: (s: Snapshot) => void,
    private onConnection: (up: boolean) => void,
  ) {
    this.connect();
  }

  send(cmd: Record<string, unknown>): void {
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(cmd));
  }

  private connect(): void {
    const url = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`;
    const ws = new WebSocket(url);
    ws.binaryType = "arraybuffer";
    ws.onopen = () => {
      this.retry = 500;
      this.onConnection(true);
    };
    ws.onmessage = (m) => {
      if (typeof m.data === "string") this.onEvent(JSON.parse(m.data));
      else this.onSnapshot(decodeSnapshot(m.data as ArrayBuffer));
    };
    ws.onclose = () => {
      this.onConnection(false);
      setTimeout(() => this.connect(), this.retry);
      this.retry = Math.min(this.retry * 1.6, 5000);
    };
    this.ws = ws;
  }
}

// Inverse of server/protocol.py encode_snapshot.
export function decodeSnapshot(buf: ArrayBuffer): Snapshot {
  const view = new DataView(buf);
  const hl = view.getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hl)));
  const off = 4 + hl + ((4 - ((4 + hl) % 4)) % 4);
  const n: number = header.n;
  const n3 = n * n * n;
  return {
    meta: header.meta,
    n,
    rhoMax: header.rho.max,
    spinMax: header.spin.max,
    rho: new Uint8Array(buf, off, n3),
    spin: new Int8Array(buf, off + n3, n3),
  };
}
