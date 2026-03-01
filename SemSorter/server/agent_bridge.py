"""
SemSorter Agent Bridge
======================
Wraps the Vision-Agents SDK components (gemini.LLM, deepgram.STT, elevenlabs.TTS)
and the MuJoCo simulation into a single async service used by the FastAPI server.

Quota/API exhaustion is detected per-service and a UIstatus message is returned
so the frontend can display an informative banner before demo-mode engages.
"""

import asyncio
import io
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from PIL import Image

logger = logging.getLogger(__name__)

# ── Path setup ────────────────────────────────────────────────────────────────
_SERVER_DIR = Path(__file__).resolve().parent
_SEMSORTER_DIR = _SERVER_DIR.parent
_PROJECT_ROOT = _SEMSORTER_DIR.parent

sys.path.insert(0, str(_SEMSORTER_DIR / "simulation"))
sys.path.insert(0, str(_SEMSORTER_DIR / "vision"))
sys.path.insert(0, str(_PROJECT_ROOT / "Vision-Agents" / "agents-core"))
for _plugin in ("gemini", "deepgram", "elevenlabs", "getstream"):
    _plugin_path = _PROJECT_ROOT / "Vision-Agents" / "plugins" / _plugin
    if _plugin_path.exists():
        sys.path.insert(0, str(_plugin_path))

load_dotenv(_PROJECT_ROOT / ".env", override=True)

# ── Quota-tracking state ──────────────────────────────────────────────────────
_quota_exceeded: Dict[str, bool] = {
    "gemini": False,
    "deepgram": False,
    "elevenlabs": False,
}

# ── Demo-mode: build detections from current sim state (avoids ghost positions) ─

# ── Singleton resources ───────────────────────────────────────────────────────
_sim = None
_bridge = None
_llm = None
_tts = None
_notify_cb: Optional[Callable[[Dict], None]] = None  # Push events to WebSocket
_sim_lock = threading.RLock()
_sim_executor: Optional[ThreadPoolExecutor] = None
_latest_frame_jpeg: Optional[bytes] = None
_latest_frame_lock = threading.Lock()
_frame_ready_event: Optional[asyncio.Event] = None  # Signaled when new frame available
_frame_ready_loop: Optional[asyncio.AbstractEventLoop] = None
_jpeg_quality = max(
    30, min(95, int(os.environ.get("SEMSORTER_STREAM_JPEG_QUALITY", "80")))
)


def set_notify_callback(cb: Callable[[Dict], None]) -> None:
    """Register a callback that pushes quota/status events to connected WS clients."""
    global _notify_cb
    _notify_cb = cb


_action_runner: Optional[Callable[[str, Callable], Awaitable[Tuple[bool, dict]]]] = None

def set_action_runner(runner: Callable[[str, Callable], Awaitable[Tuple[bool, dict]]]) -> None:
    """Register app.py's exclusive action runner to update UI across websocket."""
    global _action_runner
    _action_runner = runner


def set_frame_ready_event(loop: asyncio.AbstractEventLoop,
                         event: asyncio.Event) -> None:
    """Register the frame-ready event for WebSocket video streaming."""
    global _frame_ready_event, _frame_ready_loop
    _frame_ready_event = event
    _frame_ready_loop = loop


async def run_in_sim_thread(func: Callable[..., Any], *args, **kwargs) -> Any:
    """Run work on the dedicated simulation thread (required for EGL context safety)."""
    global _sim_executor
    if _sim_executor is None:
        _sim_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="semsorter-sim"
        )
    loop = asyncio.get_running_loop()
    call = partial(func, *args, **kwargs)
    return await loop.run_in_executor(_sim_executor, call)


async def run_in_worker_thread(func: Callable[..., Any], *args, **kwargs) -> Any:
    """Run non-simulation work on a generic worker thread pool."""
    loop = asyncio.get_running_loop()
    call = partial(func, *args, **kwargs)
    return await loop.run_in_executor(None, call)


def _push(event: Dict) -> None:
    """Fire-and-forget push to the registered notify callback."""
    if _notify_cb:
        try:
            _notify_cb(event)
        except Exception:
            pass


def _publish_sim_event(event: Dict[str, Any]) -> None:
    """Forward simulation operation events to websocket clients."""
    _push({"type": "sim_event", "data": event})


def _check_quota_error(exc: Exception) -> Optional[str]:
    """Return service name if the exception indicates quota/auth API failures."""
    msg = str(exc).lower()
    if "reported as leaked" in msg or "api key was reported as leaked" in msg:
        return "gemini"
    if ("resource_exhausted" in msg or "429" in msg or "quota" in msg
            or "invalid api key" in msg or "unauthorized" in msg or "401" in msg
            or "permission denied" in msg or "forbidden" in msg or "403" in msg
            ):
        if "gemini" in msg or "google" in msg:
            return "gemini"
        if "deepgram" in msg:
            return "deepgram"
        if "elevenlabs" in msg or "eleven" in msg:
            return "elevenlabs"
        return "unknown"
    return None


def _mark_quota_exceeded(service: str) -> None:
    """Mark a service as quota-exceeded and push a warning to the UI."""
    if not _quota_exceeded.get(service):
        _quota_exceeded[service] = True
        _push({
            "type": "quota_warning",
            "service": service,
            "message": (
                f"⚠️ {service.title()} API quota exceeded — "
                f"switching to demo mode for this service."
            ),
        })
        logger.warning("Quota exceeded for %s — demo mode activated", service)


# ── Lazy initializers ─────────────────────────────────────────────────────────

def get_simulation():
    global _sim
    with _sim_lock:
        if _sim is None:
            os.environ.setdefault("MUJOCO_GL", "osmesa")
            from controller import SemSorterSimulation
            logger.info("Initialising MuJoCo simulation…")
            _sim = SemSorterSimulation()
            _sim.load_scene()
            _sim.step(300)
            _sim.frame_callback = _store_latest_frame
            _sim.status_callback = _publish_sim_event
            _sim._publish_frame_if_needed(force=True)
            logger.info("Simulation ready: %d items", len(_sim.items))
    return _sim


def get_bridge():
    global _bridge
    with _sim_lock:
        if _bridge is None:
            from vlm_bridge import VLMSimBridge
            _bridge = VLMSimBridge(simulation=get_simulation(), use_direct=True)
            logger.info("VLM bridge ready")
    return _bridge


def get_llm():
    """Return a configured google.generativeai chat session instead of Vision-Agents."""
    global _llm
    if _llm is None:
        import google.generativeai as genai
        import asyncio
        
        # Define native Gemini tools
        def scan_for_hazards() -> dict:
            """Scan the conveyor belt for hazardous items."""
            import app
            loop = getattr(app, "_main_loop", None) or asyncio.get_event_loop()
            return asyncio.run_coroutine_threadsafe(_trigger_scan_with_ui(), loop).result()

        def pick_and_place_item(item_name: str, bin_type: str) -> dict:
            """Pick a specific item by sim name and place it in its bin. bin_type must be 'flammable' or 'chemical'."""
            import app
            loop = getattr(app, "_main_loop", None) or asyncio.get_event_loop()
            return asyncio.run_coroutine_threadsafe(_pick_place_impl(item_name, bin_type), loop).result()

        def get_simulation_state() -> dict:
            """Get current simulation state snapshot."""
            return _state_impl()

        def sort_all_hazards() -> dict:
            """Detect ALL hazardous items and sort them automatically."""
            import app
            loop = getattr(app, "_main_loop", None) or asyncio.get_event_loop()
            return asyncio.run_coroutine_threadsafe(_trigger_sort_with_ui(), loop).result()
            
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if api_key:
            genai.configure(api_key=api_key)
        
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            tools=[scan_for_hazards, pick_and_place_item, get_simulation_state, sort_all_hazards],
            system_instruction=(
                "You are an AI assistant controlling SemSorter, a robotic waste sorting system. "
                "You can scan the conveyor belt for hazardous items and pick/place them into bins. "
                "Call the appropriate tools enthusiastically based on user requests."
            )
        )
        _llm = model.start_chat(enable_automatic_function_calling=True)
        import logging
        logging.getLogger(__name__).info("Direct Gemini LLM ready (saved ~100MB RAM)")
    return _llm


# ── Tool implementations ──────────────────────────────────────────────────────

def _build_demo_detections_from_sim() -> List[Dict]:
    """
    Build demo detections from current sim state so box_2d aligns with actual items.
    Avoids 'ghost' positions when sim has reset or items have moved.
    """
    from controller import BinType
    sim = get_simulation()
    detections = []
    for name, info in sim.items.items():
        if info.picked or info.hazard_type is None:
            continue
        pos = sim.get_item_pos(name)
        if pos is None:
            continue
        # Map 3D x to 2D box left_x (conveyor: negative x = left). Approximate.
        # Camera overview: x in [-0.8, 0.4] maps roughly to 0-960
        x_norm = (pos[0] + 0.8) / 1.2
        box_left = int(max(0, min(900, x_norm * 960)))
        box_top = 300
        box_right = box_left + 50
        box_bottom = 350
        det = {
            "name": name.replace("_", " "),
            "type": info.hazard_type.value.upper(),
            "color": "red" if info.hazard_type == BinType.FLAMMABLE else "yellow",
            "shape": "box" if "box" in name else "cylinder" if "cylinder" in name else "sphere",
            "box_2d": [box_top, box_left, box_bottom, box_right],
            "sim_item": name,
            "bin_type": info.hazard_type,
        }
        detections.append(det)
    return detections


_use_local_vlm_fallback = (
    os.environ.get("SEMSORTER_USE_LOCAL_VLM", "1").strip()
    in {"1", "true", "yes", "on"}
)


async def _scan_hazards_impl() -> Dict[str, Any]:
    if _quota_exceeded["gemini"]:
        if _use_local_vlm_fallback:
            # Local color-based detector (works offline, low accuracy)
            try:
                image = await run_in_sim_thread(_capture_hazard_frame_impl)
                detections = await run_in_worker_thread(
                    _analyze_hazard_frame_local_impl, image
                )
                matched = await run_in_sim_thread(_match_detections_impl, detections)
                return _format_scan(matched, demo=True)
            except Exception:
                await run_in_sim_thread(_unfreeze_conveyor_impl)
                matched = await run_in_sim_thread(_build_demo_detections_from_sim)
                return _format_scan(matched, demo=True)
        # No local VLM: use sim state for demo
        matched = await run_in_sim_thread(_build_demo_detections_from_sim)
        return _format_scan(matched, demo=True)

    try:
        image = await run_in_sim_thread(_capture_hazard_frame_impl)
        detections = await run_in_worker_thread(_analyze_hazard_frame_impl, image)
        matched = await run_in_sim_thread(_match_detections_impl, detections)
        return _format_scan(matched, demo=False)
    except Exception as exc:
        await run_in_sim_thread(_unfreeze_conveyor_impl)
        svc = _check_quota_error(exc)
        if svc is None:
            logger.exception("Hazard detection failed; switching to demo detections")
            svc = "gemini"
        if svc == "unknown":
            svc = "gemini"
        _mark_quota_exceeded(svc)
        matched = await run_in_sim_thread(_build_demo_detections_from_sim)
        return _format_scan(matched, demo=True)


def _format_scan(matched: List[Dict], demo: bool) -> Dict[str, Any]:
    return {
        "demo_mode": demo,
        "hazards_found": len(matched),
        "items": [
            {
                "item_name": d.get("sim_item", "unknown"),
                "bin_type": d["bin_type"].value if d.get("bin_type") else "unknown",
                "detected_name": d.get("name", "unknown"),
                "type": str(d.get("type", "")).lower(),
                "color": d.get("color", ""),
                "shape": d.get("shape", ""),
            }
            for d in matched
        ],
    }


async def _pick_place_impl(item_name: str, bin_type: str) -> Dict[str, Any]:
    from controller import BinType
    sim = get_simulation()
    type_map = {"flammable": BinType.FLAMMABLE, "chemical": BinType.CHEMICAL}
    target = type_map.get(bin_type.lower())
    if not target:
        return {"success": False, "error": f"Unknown bin: {bin_type}"}
    if item_name not in sim.items:
        return {"success": False, "error": f"Unknown item: {item_name}"}
    if sim.items[item_name].picked:
        return {"success": False, "error": f"{item_name} already sorted"}

    success = await run_in_sim_thread(_pick_place_sync, sim, item_name, target)
    return {"success": success, "item": item_name, "bin": bin_type,
            "total_sorted": sim._items_sorted}


def _state_impl() -> Dict[str, Any]:
    with _sim_lock:
        sim = get_simulation()
        state = sim.get_state()
    return {
        "time": round(state.time, 2),
        "arm_busy": state.arm_busy,
        "items_sorted": state.items_sorted,
        "ee_position": [round(x, 3) for x in state.ee_pos],
        "quota_exceeded": dict(_quota_exceeded),
        "items": [
            {"name": i["name"], "picked": i["picked"],
             "hazard_type": i.get("hazard_type")}
            for i in state.items
        ],
    }


async def _sort_all_impl() -> Dict[str, Any]:
    """Full detect → match → sort pipeline."""
    # 1. Detect
    scan_result = await _scan_hazards_impl()
    items = scan_result["items"]
    demo = scan_result["demo_mode"]

    if not items:
        return {"hazards_detected": 0, "items_matched": 0, "items_sorted": 0,
                "details": [], "demo_mode": demo}

    # 2. Sort each matched item
    details = []
    sorted_count = 0
    for item in items:
        r = await _pick_place_impl(item["item_name"], item["bin_type"])
        details.append({"item": item["item_name"], "bin": item["bin_type"],
                         "success": r.get("success", False)})
        if r.get("success"):
            sorted_count += 1

    return {"hazards_detected": len(items), "items_matched": len(items),
            "items_sorted": sorted_count, "details": details, "demo_mode": demo}


async def _trigger_scan_with_ui() -> Dict[str, Any]:
    """Helper to ensure UI is updated when scan is triggered by LLM or demo."""
    if _action_runner:
        accepted, res = await _action_runner("scan", _scan_hazards_impl)
        if accepted:
            _push({"type": "scan_result", "data": res})
        return res
    return await _scan_hazards_impl()


async def _trigger_sort_with_ui() -> Dict[str, Any]:
    """Helper to ensure UI is updated when sort is triggered by LLM or demo."""
    if _action_runner:
        accepted, res = await _action_runner("sort", _sort_all_impl)
        if accepted:
            _push({"type": "sort_result", "data": res})
        return res
    return await _sort_all_impl()


def render_frame(camera: str = "overview"):
    """
    Thread-safe simulation frame render. All sim.render_frame() calls must be
    inside _sim_lock to avoid EGL context conflicts (MuJoCo renderer is
    thread-sensitive). The video WS uses get_latest_frame_jpeg (callback cache).
    """
    with _sim_lock:
        sim = get_simulation()
        return sim.render_frame(camera=camera, width=sim._stream_width, height=sim._stream_height)


def get_latest_frame_jpeg() -> Optional[bytes]:
    """Return the latest encoded frame from simulation callbacks."""
    with _latest_frame_lock:
        return _latest_frame_jpeg


def step_simulation(steps: int = 2) -> None:
    """Advance physics for continuous conveyor motion in live video mode."""
    if steps <= 0:
        return
    with _sim_lock:
        sim = get_simulation()
        sim.step(steps)


def close_resources() -> None:
    """Best-effort shutdown for long-running server process."""
    global _bridge, _sim, _sim_executor, _latest_frame_jpeg
    with _sim_lock:
        if _bridge is not None:
            try:
                _bridge.close()
            except Exception:
                pass
            _bridge = None
        if _sim is not None and hasattr(_sim, "close"):
            try:
                _sim.close()
            except Exception:
                pass
            _sim = None
    with _latest_frame_lock:
        _latest_frame_jpeg = None
    is_sim_thread = threading.current_thread().name.startswith("semsorter-sim")
    if _sim_executor is not None and not is_sim_thread:
        try:
            _sim_executor.shutdown(wait=True, cancel_futures=True)
        except Exception:
            pass
        _sim_executor = None


# ── Text → agent response ─────────────────────────────────────────────────────

async def process_text_command(text: str) -> str:
    """
    Send a text command to the Gemini LLM (Vision-Agents SDK).
    Returns the agent's text response.
    On quota error: marks exceeded + returns a canned message.
    """
    if _quota_exceeded["gemini"]:
        return await _llm_demo_response(text)

    try:
        llm = get_llm()
        # Use the LLM's send_message method to get a response with tool-calling
        response_event = await asyncio.to_thread(llm.send_message, text)
        return response_event.text
    except Exception as exc:
        svc = _check_quota_error(exc)
        if svc:
            if svc == "unknown":
                svc = "gemini"
            _mark_quota_exceeded(svc)
            return await _llm_demo_response(text)
        logger.exception("LLM error")
        return f"Error processing command: {exc}"


def _detect_and_match_impl():
    """Run detect+match atomically to avoid simulation/render race conditions."""
    with _sim_lock:
        bridge = get_bridge()
        detections = bridge.processor.detect_hazards()
        matched = bridge.match_detections_to_items(detections)
        return detections, matched


def _capture_hazard_frame_impl():
    """Capture frame for VLM. Freezes conveyor to avoid spatial matching race."""
    with _sim_lock:
        sim = get_simulation()
        sim._arm_busy = True  # Freeze conveyor during capture->match
        bridge = get_bridge()
        return bridge.processor.capture_frame()


def _analyze_hazard_frame_impl(image):
    bridge = get_bridge()
    return bridge.processor.analyze_frame(image)


def _analyze_hazard_frame_local_impl(image):
    """Local color-based detection (no API)."""
    bridge = get_bridge()
    return bridge.processor.analyze_frame_local(image)


def _unfreeze_conveyor_impl():
    """Unfreeze conveyor after scan capture->match (or on error)."""
    with _sim_lock:
        sim = get_simulation()
        sim._arm_busy = False


def _match_detections_impl(detections: List[Dict]):
    """Match VLM detections to sim items. Unfreezes conveyor after spatial match."""
    try:
        with _sim_lock:
            bridge = get_bridge()
            return bridge.match_detections_to_items(detections)
    finally:
        _unfreeze_conveyor_impl()


def _store_latest_frame(frame) -> None:
    """
    Encode and cache the latest frame for websocket video streaming.
    Called from sim thread (frame_callback); only does CPU work (JPEG encode).
    All EGL/render_frame calls are strictly inside _sim_lock in agent_bridge.
    """
    global _latest_frame_jpeg
    try:
        img = Image.fromarray(frame)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=_jpeg_quality)
        encoded = buf.getvalue()
    except Exception:
        return
    with _latest_frame_lock:
        _latest_frame_jpeg = encoded
    # Signal WebSocket that a new frame is available (thread-safe)
    if _frame_ready_event is not None and _frame_ready_loop is not None:
        try:
            _frame_ready_loop.call_soon_threadsafe(_frame_ready_event.set)
        except Exception:
            pass
            
    # Explicit garbage collection helps keep Render 512MB RAM flat
    # during high-frequency JPEG encoding.
    import gc; gc.collect()


def _pick_place_sync(sim, item_name: str, target) -> bool:
    with _sim_lock:
        return sim.pick_and_place(item_name, target)


async def _llm_demo_response(text: str) -> str:
    """Return a plausible demo response when Gemini quota is exhausted, and trigger actions."""
    t = text.lower()
    if "scan" in t:
        asyncio.create_task(_trigger_scan_with_ui())
        return ("Scanning the conveyor belt now using local fallback. "
                "[Demo mode — Gemini quota exceeded]")
    if "sort" in t or "pick" in t or "place" in t:
        asyncio.create_task(_trigger_sort_with_ui())
        return ("Sorting all hazardous items into their respective bins. "
                "[Demo mode — Gemini quota exceeded]")
    if "state" in t or "status" in t:
        state = _state_impl()
        return (f"Simulation time: {state['time']}s. "
                f"Items sorted: {state['items_sorted']}. "
                f"Arm busy: {state['arm_busy']}. [Demo mode]")
    return "I'm SemSorter AI. Ask me to scan or sort items! [Demo mode]"


# ── TTS helper ────────────────────────────────────────────────────────────────

async def text_to_speech(text: str) -> Optional[bytes]:
    """
    Convert text to audio bytes using ElevenLabs (Vision-Agents SDK plugin).
    Returns None on quota error (frontend falls back to browser SpeechSynthesis).
    """
    if _quota_exceeded["elevenlabs"]:
        return None
    try:
        from elevenlabs.client import AsyncElevenLabs
        client = AsyncElevenLabs(api_key=os.environ.get("ELEVENLABS_API_KEY", ""))
        audio_gen = client.text_to_speech.convert(
            text=text,
            voice_id="JBFqnCBsd6RMkjVDRZzb",
            model_id="eleven_flash_v2_5",
            output_format="mp3_44100_128",
        )
        audio_bytes = b"".join([chunk async for chunk in audio_gen])
        return audio_bytes
    except Exception as exc:
        svc = _check_quota_error(exc)
        if svc == "elevenlabs" or svc == "unknown":
            _mark_quota_exceeded("elevenlabs")
        else:
            logger.exception("TTS error")
        return None


# ── STT helper (Deepgram) ─────────────────────────────────────────────────────

async def transcribe_audio(audio_bytes: bytes, mime: str = "audio/webm") -> Optional[str]:
    """
    Transcribe audio using Deepgram STT (Vision-Agents SDK plugin).
    Returns None on quota error (frontend falls back to Web Speech API result).
    """
    if _quota_exceeded["deepgram"]:
        return None
    try:
        import httpx, os
        api_key = os.environ.get("DEEPGRAM_API_KEY", "")
        if not api_key:
            return None
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.deepgram.com/v1/listen?model=nova-2",
                headers={"Authorization": f"Token {api_key}",
                         "Content-Type": mime},
                content=audio_bytes,
                timeout=10,
            )
        if resp.status_code == 429:
            _mark_quota_exceeded("deepgram")
            return None
        data = resp.json()
        return (data.get("results", {})
                    .get("channels", [{}])[0]
                    .get("alternatives", [{}])[0]
                    .get("transcript", ""))
    except Exception as exc:
        logger.warning("Deepgram STT error: %s", exc)
        return None
