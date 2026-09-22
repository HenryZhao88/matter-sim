import json

import numpy as np

from engine.scenes.presets import PRESETS, system_from_preset
from engine.simulation import Params, Simulation
from server.app import create_app
from server.protocol import decode_snapshot, encode_snapshot


def test_every_preset_builds_a_valid_system():
    for p in PRESETS:
        s = system_from_preset(p["id"])
        assert s.n_electrons >= 1


def test_snapshot_roundtrip_preserves_density():
    sim = Simulation(system_from_preset("h"), Params(quality="draft", mode="frozen"))
    sim.step()
    snap = sim.snapshot()
    out = decode_snapshot(encode_snapshot(snap))
    assert out["header"]["meta"]["atoms"][0]["Z"] == 1
    rho = snap["rho"]
    big = rho > 1e-3
    rel = np.abs(out["rho"][big] - rho[big]) / rho[big]
    assert rel.max() < 0.05


async def test_websocket_hello_then_snapshot(aiohttp_client):
    client = await aiohttp_client(create_app())
    ws = await client.ws_connect("/ws")
    hello = json.loads((await ws.receive(timeout=10)).data)
    assert hello["type"] == "hello" and hello["presets"]
    await ws.send_str(json.dumps({"type": "load_preset", "id": "h"}))
    got_binary = False
    for _ in range(200):
        msg = await ws.receive(timeout=60)
        if msg.type.name == "BINARY":
            header = decode_snapshot(msg.data)["header"]
            if header["meta"]["atoms"] and len(header["meta"]["atoms"]) == 1:
                got_binary = True
                break
    assert got_binary
    await ws.close()
