"""
SemSorter FastAPI Server
========================
Serves the web UI and bridges the Vision-Agents SDK + MuJoCo simulation.

Endpoints
---------
GET  /              → index.html
WS   /ws/video      → MJPEG frames (~10 fps) from MuJoCo renderer
WS   /ws/chat       → bidirectional: text commands → agent responses + events
GET  /api/state     → current simulation state JSON
POST /api/sort      → trigger sort_all_hazards pipeline
POST /api/command   → send a text command to the agent
POST /api/transcribe → transcribe uploaded audio via Deepgram

Run locally:
    cd SemSorter && MUJOCO_GL=egl uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import base64
import io
import json
import logging
import os
from pathlib import Path
from typing import Optional, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

# ── Local imports ─────────────────────────────────────────────────────────────
from . import agent_bridge as bridge

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s  %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="SemSorter", version="1.0")

# ── Static files ──────────────────────────────────────────────────────────────
_STATIC = Path(__file__).parent / "static"
_STATIC.mkdir(exist_ok=True)

# ── Connected WebSocket clients ───────────────────────────────────────────────
_chat_clients: Set[WebSocket] = set()
_video_clients: Set[WebSocket] = set()
_main_loop: Optional[asyncio.AbstractEventLoop] = None


async def _broadcast_chat(event: dict) -> None:
    """Push a JSON event to all connected chat WebSocket clients."""
    global _chat_clients
    payload = json.dumps(event)
    dead = set()
    for ws in list(_chat_clients):
        try:
            await ws.send_text(payload)
        except Exception:
            dead.add(ws)
    for ws in dead:
        _chat_clients.discard(ws)


def _sync_broadcast(event: dict) -> None:
    """Thread-safe push called from sync code (bridge callbacks)."""
    if _main_loop is None:
        return
    try:
        _main_loop.call_soon_threadsafe(
            asyncio.create_task, _broadcast_chat(event)
        )
    except Exception:
        logger.exception("Failed to schedule chat broadcast")


# Register the broadcast callback so agent_bridge can push quota warnings
bridge.set_notify_callback(_sync_broadcast)


# ── Startup: pre-warm simulation ──────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    global _main_loop
    _main_loop = asyncio.get_running_loop()
    logger.info("Pre-warming MuJoCo simulation…")
    await _main_loop.run_in_executor(None, bridge.get_simulation)
    logger.info("Simulation ready")


@app.on_event("shutdown")
async def shutdown():
    logger.info("Shutting down SemSorter resources…")
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, bridge.close_resources)
    logger.info("Shutdown complete")


# ── REST endpoints ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = _STATIC / "index.html"
    return HTMLResponse(html_path.read_text())


@app.get("/api/state")
async def api_state():
    loop = asyncio.get_event_loop()
    state = await loop.run_in_executor(None, bridge._state_impl)
    return JSONResponse(state)


@app.get("/health")
async def health():
    return JSONResponse({"ok": True})


@app.post("/api/sort")
async def api_sort():
    """Trigger the full detect-match-sort pipeline."""
    result = await bridge._sort_all_impl()
    await _broadcast_chat({"type": "sort_result", "data": result})
    return JSONResponse(result)


@app.post("/api/command")
async def api_command(body: dict):
    text = body.get("text", "").strip()
    if not text:
        return JSONResponse({"error": "empty command"}, status_code=400)
    response_text = await bridge.process_text_command(text)
    await _broadcast_chat({"type": "agent_response", "text": response_text})
    return JSONResponse({"response": response_text})


@app.post("/api/transcribe")
async def api_transcribe(file: UploadFile = File(...)):
    """Transcribe uploaded audio using Deepgram; returns transcript or null."""
    audio_bytes = await file.read()
    transcript = await bridge.transcribe_audio(audio_bytes, mime=file.content_type)
    return JSONResponse({"transcript": transcript})


# ── WebSocket: chat ───────────────────────────────────────────────────────────

@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket):
    await ws.accept()
    _chat_clients.add(ws)
    logger.info("Chat client connected (%d total)", len(_chat_clients))
    try:
        await ws.send_text(json.dumps({
            "type": "welcome",
            "text": "Connected to SemSorter AI. Ask me to scan or sort items!",
        }))
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                msg = {"type": "command", "text": raw}

            msg_type = msg.get("type", "command")

            if msg_type == "command":
                text = msg.get("text", "").strip()
                if text:
                    await _broadcast_chat({"type": "user_message", "text": text})
                    response = await bridge.process_text_command(text)
                    await _broadcast_chat({"type": "agent_response", "text": response})

            elif msg_type == "scan":
                result = await bridge._scan_hazards_impl()
                await _broadcast_chat({"type": "scan_result", "data": result})

            elif msg_type == "sort":
                result = await bridge._sort_all_impl()
                await _broadcast_chat({"type": "sort_result", "data": result})

            elif msg_type == "state":
                loop = asyncio.get_event_loop()
                state = await loop.run_in_executor(None, bridge._state_impl)
                await ws.send_text(json.dumps({"type": "state", "data": state}))

    except WebSocketDisconnect:
        pass
    finally:
        _chat_clients.discard(ws)
        logger.info("Chat client disconnected (%d remaining)", len(_chat_clients))


# ── WebSocket: live video stream ──────────────────────────────────────────────

def _render_frame_jpeg(quality: int = 75) -> bytes:
    """Render a MuJoCo frame and encode as JPEG bytes."""
    frame = bridge.render_frame(camera="overview")         # numpy H×W×3
    img = Image.fromarray(frame)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


@app.websocket("/ws/video")
async def ws_video(ws: WebSocket):
    await ws.accept()
    _video_clients.add(ws)
    logger.info("Video client connected")
    try:
        loop = asyncio.get_event_loop()
        while True:
            jpeg_bytes = await loop.run_in_executor(None, _render_frame_jpeg)
            b64 = base64.b64encode(jpeg_bytes).decode()
            await ws.send_text(json.dumps({"type": "frame", "data": b64}))
            await asyncio.sleep(0.1)   # ~10 fps
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning("Video stream error: %s", e)
    finally:
        _video_clients.discard(ws)
        logger.info("Video client disconnected")
