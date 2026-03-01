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
GET  /api/tts        → synthesize TTS via ElevenLabs

Run locally:
    cd SemSorter && MUJOCO_GL=egl uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import base64
import json
import logging
import os
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional, Set, Tuple

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

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
_sim_tick_task: Optional[asyncio.Task] = None
_action_lock = asyncio.Lock()
_active_action: Optional[str] = None
_frame_ready_event: Optional[asyncio.Event] = None


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

async def _run_exclusive_action(
    action: str, work: Callable[[], Awaitable[dict]]
) -> Tuple[bool, dict]:
    """Run long scan/sort actions one-at-a-time and publish lifecycle events."""
    global _active_action
    if _action_lock.locked():
        busy = _active_action or "another action"
        await _broadcast_chat({
            "type": "action_status",
            "action": action,
            "status": "rejected",
            "reason": busy,
        })
        return False, {"success": False, "error": f"{busy} already in progress"}

    await _action_lock.acquire()
    _active_action = action
    await _broadcast_chat({
        "type": "action_status",
        "action": action,
        "status": "started",
    })
    try:
        result = await work()
        await _broadcast_chat({
            "type": "action_status",
            "action": action,
            "status": "finished",
        })
        return True, result
    except Exception as exc:
        await _broadcast_chat({
            "type": "action_status",
            "action": action,
            "status": "failed",
            "reason": str(exc),
        })
        raise
    finally:
        _active_action = None
        _action_lock.release()

bridge.set_action_runner(_run_exclusive_action)


# ── Startup: pre-warm simulation ──────────────────────────────────────────────
async def _simulation_tick_loop() -> None:
    """Keep simulation progressing even when no control action is running."""
    while True:
        try:
            await bridge.run_in_sim_thread(bridge.step_simulation, 2)
        except Exception:
            logger.exception("Simulation tick loop error")
        await asyncio.sleep(0.02)


@app.on_event("startup")
async def startup():
    global _main_loop, _frame_ready_event, _sim_tick_task
    _main_loop = asyncio.get_running_loop()
    _frame_ready_event = asyncio.Event()
    bridge.set_frame_ready_event(_main_loop, _frame_ready_event)
    logger.info("Server ready. Backgrounding simulation initialization...")
    
    # Initialize the simulation asynchronously in the background so it doesn't block
    # the main event loop and cause WS timeouts on low-CPU instances like Render Free Tier.
    asyncio.create_task(bridge.run_in_sim_thread(bridge.get_simulation))
    _sim_tick_task = asyncio.create_task(_simulation_tick_loop())


@app.on_event("shutdown")
async def shutdown():
    global _sim_tick_task
    logger.info("Shutting down SemSorter resources…")
    if _sim_tick_task is not None:
        _sim_tick_task.cancel()
        try:
            await _sim_tick_task
        except asyncio.CancelledError:
            pass
        _sim_tick_task = None
    bridge.close_resources()
    logger.info("Shutdown complete")


# ── REST endpoints ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = _STATIC / "index.html"
    return HTMLResponse(html_path.read_text())


@app.get("/api/state")
async def api_state():
    state = await bridge.run_in_sim_thread(bridge._state_impl)
    state["action_busy"] = _action_lock.locked()
    state["active_action"] = _active_action
    return JSONResponse(state)


@app.get("/health")
async def health():
    return JSONResponse({"ok": True})


@app.post("/api/sort")
async def api_sort():
    """Trigger the full detect-match-sort pipeline."""
    accepted, result = await _run_exclusive_action("sort", bridge._sort_all_impl)
    if not accepted:
        return JSONResponse(result, status_code=409)
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


@app.get("/api/tts")
async def api_tts(text: str):
    """Generate audio from text using ElevenLabs TTS."""
    if not text:
        return JSONResponse({"error": "empty text"}, status_code=400)
    audio_bytes = await bridge.text_to_speech(text)
    if audio_bytes is None:
        return JSONResponse({"error": "TTS failed or quota exceeded"}, status_code=500)
    return Response(content=audio_bytes, media_type="audio/mpeg")


# ── WebSocket: chat ───────────────────────────────────────────────────────────

@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket):
    global _sim_tick_task
    await ws.accept()
    _chat_clients.add(ws)
    logger.info("Chat client connected (%d total)", len(_chat_clients))
    
    try:
        await ws.send_text(json.dumps({
            "type": "welcome",
            "text": "Connected to SemSorter AI. Ask me to scan or sort items!",
        }))
        while True:
            try:
                raw = await ws.receive_text()
            except WebSocketDisconnect:
                break
            except RuntimeError as exc:
                # Starlette can raise RuntimeError on abrupt client close.
                if "websocket is not connected" in str(exc).lower():
                    break
                raise
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                msg = {"type": "command", "text": raw}

            try:
                msg_type = msg.get("type", "command")

                if msg_type == "command":
                    text = msg.get("text", "").strip()
                    if text:
                        await _broadcast_chat({"type": "user_message", "text": text})
                        response = await bridge.process_text_command(text)
                        await _broadcast_chat({"type": "agent_response", "text": response})

                elif msg_type == "scan":
                    accepted, result = await _run_exclusive_action(
                        "scan", bridge._scan_hazards_impl
                    )
                    if accepted:
                        await _broadcast_chat({"type": "scan_result", "data": result})
                    else:
                        await ws.send_text(json.dumps({
                            "type": "system",
                            "text": result.get("error", "scan already in progress"),
                        }))

                elif msg_type == "sort":
                    accepted, result = await _run_exclusive_action(
                        "sort", bridge._sort_all_impl
                    )
                    if accepted:
                        await _broadcast_chat({"type": "sort_result", "data": result})
                    else:
                        await ws.send_text(json.dumps({
                            "type": "system",
                            "text": result.get("error", "sort already in progress"),
                        }))

                elif msg_type == "state":
                    state = await bridge.run_in_sim_thread(bridge._state_impl)
                    await ws.send_text(json.dumps({"type": "state", "data": state}))
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.exception("Chat action failed")
                try:
                    await ws.send_text(json.dumps({
                        "type": "system",
                        "text": f"Action failed: {exc}",
                    }))
                except Exception:
                    break

    except asyncio.CancelledError:
        pass
    except WebSocketDisconnect:
        pass
    finally:
        _chat_clients.discard(ws)
        logger.info("Chat client disconnected (%d remaining)", len(_chat_clients))


# ── WebSocket: live video stream ──────────────────────────────────────────────
@app.websocket("/ws/video")
async def ws_video(ws: WebSocket):
    global _sim_tick_task
    await ws.accept()
    _video_clients.add(ws)
    logger.info("Video client connected")
    
    try:
        while True:
            # Wait for new frame (event-driven; sends exactly when sim produces one)
            if _frame_ready_event is not None:
                try:
                    await asyncio.wait_for(_frame_ready_event.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    pass  # Fall through to try get_latest_frame_jpeg
                _frame_ready_event.clear()
            jpeg_bytes = bridge.get_latest_frame_jpeg()
            if not jpeg_bytes:
                await bridge.run_in_sim_thread(bridge.step_simulation, 1)
                jpeg_bytes = bridge.get_latest_frame_jpeg()
                if not jpeg_bytes:
                    await asyncio.sleep(0.05)
                    continue
            b64 = base64.b64encode(jpeg_bytes).decode()
            await ws.send_text(json.dumps({"type": "frame", "data": b64}))
    except WebSocketDisconnect:
        pass
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.warning("Video stream error: %s", e)
    finally:
        _video_clients.discard(ws)
        logger.info("Video client disconnected")
