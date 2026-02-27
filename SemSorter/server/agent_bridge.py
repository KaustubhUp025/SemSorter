"""
SemSorter Agent Bridge
======================
Wraps the Vision-Agents SDK components (gemini.LLM, deepgram.STT, elevenlabs.TTS)
and the MuJoCo simulation into a single async service used by the FastAPI server.

Quota/API exhaustion is detected per-service and a UIstatus message is returned
so the frontend can display an informative banner before demo-mode engages.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

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

# ── Quota-tracking state ──────────────────────────────────────────────────────
_quota_exceeded: Dict[str, bool] = {
    "gemini": False,
    "deepgram": False,
    "elevenlabs": False,
}

# ── Demo-mode pre-recorded detections ────────────────────────────────────────
_DEMO_DETECTIONS = [
    {"name": "red cylinder", "type": "FLAMMABLE", "color": "red",
     "shape": "cylinder", "box_2d": [240, 200, 290, 260]},
    {"name": "green box", "type": "FLAMMABLE", "color": "green",
     "shape": "box", "box_2d": [240, 260, 285, 310]},
    {"name": "yellow box", "type": "CHEMICAL", "color": "yellow",
     "shape": "box", "box_2d": [240, 310, 285, 360]},
    {"name": "blue box", "type": "CHEMICAL", "color": "blue",
     "shape": "box", "box_2d": [240, 370, 285, 420]},
]

# ── Singleton resources ───────────────────────────────────────────────────────
_sim = None
_bridge = None
_llm = None
_tts = None
_notify_cb: Optional[Callable[[Dict], None]] = None  # Push events to WebSocket


def set_notify_callback(cb: Callable[[Dict], None]) -> None:
    """Register a callback that pushes quota/status events to connected WS clients."""
    global _notify_cb
    _notify_cb = cb


def _push(event: Dict) -> None:
    """Fire-and-forget push to the registered notify callback."""
    if _notify_cb:
        try:
            _notify_cb(event)
        except Exception:
            pass


def _check_quota_error(exc: Exception) -> Optional[str]:
    """Return service name if the exception indicates API quota exhaustion."""
    msg = str(exc).lower()
    if "resource_exhausted" in msg or "429" in msg or "quota" in msg:
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
    if _sim is None:
        os.environ.setdefault("MUJOCO_GL", "egl")
        from controller import SemSorterSimulation
        logger.info("Initialising MuJoCo simulation…")
        _sim = SemSorterSimulation()
        _sim.load_scene()
        _sim.step(300)
        logger.info("Simulation ready: %d items", len(_sim.items))
    return _sim


def get_bridge():
    global _bridge
    if _bridge is None:
        from vlm_bridge import VLMSimBridge
        _bridge = VLMSimBridge(simulation=get_simulation(), use_direct=True)
        logger.info("VLM bridge ready")
    return _bridge


def get_llm():
    """Return a configured gemini.LLM instance from the Vision-Agents SDK."""
    global _llm
    if _llm is None:
        from vision_agents.plugins.gemini.gemini_llm import GeminiLLM as GeminiLLMCls
        _llm = GeminiLLMCls("gemini-2.0-flash")
        _register_tools(_llm)
        logger.info("Gemini LLM ready")
    return _llm


def _register_tools(llm) -> None:
    """Register simulation control tools on the LLM."""

    @llm.register_function(description="Scan the conveyor belt for hazardous items.")
    async def scan_for_hazards() -> Dict[str, Any]:
        return await _scan_hazards_impl()

    @llm.register_function(
        description="Pick a specific item by sim name and place it in its bin. "
                    "bin_type must be 'flammable' or 'chemical'.")
    async def pick_and_place_item(item_name: str, bin_type: str) -> Dict[str, Any]:
        return await _pick_place_impl(item_name, bin_type)

    @llm.register_function(description="Get current simulation state snapshot.")
    async def get_simulation_state() -> Dict[str, Any]:
        return _state_impl()

    @llm.register_function(
        description="Detect ALL hazardous items and sort them automatically.")
    async def sort_all_hazards() -> Dict[str, Any]:
        return await _sort_all_impl()


# ── Tool implementations ──────────────────────────────────────────────────────

async def _scan_hazards_impl() -> Dict[str, Any]:
    if _quota_exceeded["gemini"]:
        # Already in demo mode — return pre-recorded detections
        bridge = get_bridge()
        matched = bridge.match_detections_to_items(_DEMO_DETECTIONS)
        return _format_scan(matched, demo=True)

    try:
        bridge = get_bridge()
        loop = asyncio.get_event_loop()
        detections = await loop.run_in_executor(
            None, bridge.processor.detect_hazards)
        matched = bridge.match_detections_to_items(detections)
        return _format_scan(matched, demo=False)
    except Exception as exc:
        svc = _check_quota_error(exc)
        if svc:
            _mark_quota_exceeded(svc)
            bridge = get_bridge()
            matched = bridge.match_detections_to_items(_DEMO_DETECTIONS)
            return _format_scan(matched, demo=True)
        raise


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

    loop = asyncio.get_event_loop()
    success = await loop.run_in_executor(None, sim.pick_and_place, item_name, target)
    return {"success": success, "item": item_name, "bin": bin_type,
            "total_sorted": sim._items_sorted}


def _state_impl() -> Dict[str, Any]:
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
        # Use the LLM's chat method to get a response with tool-calling
        response = await llm.chat(text)
        return response
    except Exception as exc:
        svc = _check_quota_error(exc)
        if svc:
            _mark_quota_exceeded(svc)
            return await _llm_demo_response(text)
        logger.exception("LLM error")
        return f"Error processing command: {exc}"


async def _llm_demo_response(text: str) -> str:
    """Return a plausible demo response when Gemini quota is exhausted."""
    t = text.lower()
    if "scan" in t:
        return ("I found 4 hazardous items on the conveyor belt: "
                "2 flammable and 2 chemical. [Demo mode — Gemini quota exceeded]")
    if "sort" in t or "pick" in t or "place" in t:
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
        from vision_agents.plugins.elevenlabs.elevenlabs_tts import ElevenLabsTTS
        tts = ElevenLabsTTS(model_id="eleven_flash_v2_5")
        audio_bytes = await tts.synthesize(text)
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
