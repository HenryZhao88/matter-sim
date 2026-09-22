"""aiohttp server: serves the built viewer and the /ws engine connection."""

from __future__ import annotations

import asyncio
import json
import logging
import weakref
from pathlib import Path

from aiohttp import WSMsgType, web

from .session import Session

log = logging.getLogger("matter_sim.server")

VIEWER_DIST = Path(__file__).resolve().parent.parent / "viewer" / "dist"


def create_app(start_engine: bool = True) -> web.Application:
    app = web.Application()
    clients: weakref.WeakSet = weakref.WeakSet()
    app["clients"] = clients

    async def _send_all(payload):
        for ws in list(clients):
            if ws.closed:
                continue
            try:
                if isinstance(payload, bytes):
                    await ws.send_bytes(payload)
                else:
                    await ws.send_str(payload)
            except (ConnectionResetError, RuntimeError):
                pass

    async def on_startup(app_):
        loop = asyncio.get_running_loop()

        def pub_json(msg: dict):
            asyncio.run_coroutine_threadsafe(_send_all(json.dumps(msg)), loop)

        def pub_bytes(data: bytes):
            asyncio.run_coroutine_threadsafe(_send_all(data), loop)

        session = Session(pub_json, pub_bytes)
        app_["session"] = session
        if start_engine:
            session.start()

    async def on_cleanup(app_):
        app_["session"].stop()

    async def ws_handler(request):
        ws = web.WebSocketResponse(max_msg_size=0, heartbeat=20)
        await ws.prepare(request)
        session: Session = request.app["session"]
        clients.add(ws)
        await ws.send_str(json.dumps(session.hello()))
        await ws.send_str(json.dumps(session.status()))
        if session.latest_snapshot:
            await ws.send_bytes(session.latest_snapshot)
        async for msg in ws:
            log.debug("ws message %s %s", msg.type, str(msg.data)[:120])
            if msg.type == WSMsgType.TEXT:
                try:
                    session.submit(json.loads(msg.data))
                except json.JSONDecodeError:
                    await ws.send_str(json.dumps({"type": "error", "message": "bad JSON"}))
        clients.discard(ws)
        return ws

    async def index(_request):
        page = VIEWER_DIST / "index.html"
        if not page.exists():
            return web.Response(
                text="Viewer not built. Run:  cd viewer && npm install && npm run build",
                content_type="text/plain")
        return web.FileResponse(page)

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_handler)
    if (VIEWER_DIST / "assets").exists():
        app.router.add_static("/assets", VIEWER_DIST / "assets")
    return app


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    web.run_app(create_app(), host=host, port=port, print=None)
