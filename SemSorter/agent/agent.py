"""
SemSorter Agent — Vision-Agents SDK Integration

This module creates a real-time AI agent using GetStream's Vision-Agents SDK.
The agent watches the MuJoCo simulation via video, listens to voice commands,
detects hazardous items using Gemini VLM, and triggers pick-and-place operations.

Usage (from the Vision-Agents directory):
    # Set env vars in .env first, then:
    uv run python ../SemSorter/SemSorter/agent/agent.py run
"""

import logging
import os
import sys
import atexit
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from vision_agents.core import Agent, AgentLauncher, Runner, User
from vision_agents.plugins import deepgram, elevenlabs, gemini, getstream

logger = logging.getLogger(__name__)

# ─── Path setup ──────────────────────────────────────────────────────────────
# Add SemSorter packages to sys.path so we can import simulation & vision
AGENT_DIR = Path(__file__).resolve().parent
SEMSORTER_DIR = AGENT_DIR.parent
PROJECT_ROOT = SEMSORTER_DIR.parent

sys.path.insert(0, str(SEMSORTER_DIR / "simulation"))
sys.path.insert(0, str(SEMSORTER_DIR / "vision"))

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

# ─── Simulation singleton ───────────────────────────────────────────────────
_simulation = None
_bridge = None


def get_simulation():
    """Lazy-initialize the MuJoCo simulation (singleton)."""
    global _simulation
    if _simulation is None:
        os.environ.setdefault("MUJOCO_GL", "egl")
        from controller import SemSorterSimulation

        logger.info("Initializing MuJoCo simulation...")
        _simulation = SemSorterSimulation()
        _simulation.load_scene()
        _simulation.step(200)  # Let physics settle
        logger.info("Simulation ready.")
    return _simulation


def get_bridge():
    """Lazy-initialize the VLM-Simulation bridge (singleton)."""
    global _bridge
    if _bridge is None:
        from vlm_bridge import VLMSimBridge

        sim = get_simulation()
        _bridge = VLMSimBridge(simulation=sim, use_direct=True)
        logger.info("VLM-Sim bridge ready.")
    return _bridge


class _EGLStderrFilter:
    """Stderr wrapper that suppresses only known EGL teardown noise."""
    _SUPPRESSED = ("EGLError", "eglDestroyContext", "eglMakeCurrent",
                   "EGL_NOT_INITIALIZED", "GLContext.__del__",
                   "Renderer.__del__", "SfuStatsReporter",
                   "Task was destroyed but it is pending")

    def __init__(self, real):
        self._real = real

    def write(self, s):
        if any(tok in s for tok in self._SUPPRESSED):
            return len(s)  # silently consume
        return self._real.write(s)

    def flush(self):
        self._real.flush()

    def __getattr__(self, name):
        return getattr(self._real, name)


def close_resources() -> None:
    """Release singleton resources on process shutdown."""
    # Only muffle known-harmless EGL teardown noise, keep real errors visible
    sys.stderr = _EGLStderrFilter(sys.stderr)

    global _bridge, _simulation
    if _bridge is not None:
        try:
            _bridge.close()
        except Exception:
            pass
        _bridge = None
    if _simulation is not None and hasattr(_simulation, "close"):
        try:
            _simulation.close()
        except Exception:
            pass
    _simulation = None


atexit.register(close_resources)


# ─── LLM Setup with Tool Registration ───────────────────────────────────────

INSTRUCTIONS = Path(AGENT_DIR / "semsorter_instructions.md").read_text()


def setup_llm(model: str = "gemini-3-flash-preview") -> gemini.LLM:
    """Create and configure the Gemini LLM with registered simulation tools."""
    llm = gemini.LLM(model)

    @llm.register_function(
        description="Scan the conveyor belt camera feed for hazardous items. "
        "Returns a list of detected hazardous items with their types and positions."
    )
    async def scan_for_hazards() -> Dict[str, Any]:
        """Capture a frame, match detections to sim items, and return actionable IDs."""
        bridge = get_bridge()
        detections = bridge.processor.detect_hazards()
        matched = bridge.match_detections_to_items(detections)
        return {
            "hazards_found": len(detections),
            "items_matched": len(matched),
            "items": [
                {
                    "item_name": d.get("sim_item", "unknown"),
                    "bin_type": d.get("bin_type").value if d.get("bin_type") else "unknown",
                    "detected_name": d.get("name", "unknown"),
                    "type": str(d.get("type", "unknown")).lower(),
                    "color": d.get("color", "unknown"),
                    "shape": d.get("shape", "unknown"),
                }
                for d in matched
            ],
        }

    @llm.register_function(
        description="Pick a specific item from the conveyor and place it in "
        "the designated hazard bin. Use item_name from scan results. "
        "bin_type must be 'flammable' or 'chemical'."
    )
    async def pick_and_place_item(item_name: str, bin_type: str) -> Dict[str, Any]:
        """Execute a pick-and-place operation for a specific item."""
        from controller import BinType

        sim = get_simulation()

        type_map = {"flammable": BinType.FLAMMABLE, "chemical": BinType.CHEMICAL}
        target_bin = type_map.get(bin_type.lower())
        if target_bin is None:
            return {"success": False, "error": f"Unknown bin type: {bin_type}"}

        if item_name not in sim.items:
            return {"success": False, "error": f"Unknown item: {item_name}"}

        if sim.items[item_name].picked:
            return {"success": False, "error": f"Item {item_name} already sorted"}

        success = sim.pick_and_place(item_name, target_bin)
        return {
            "success": success,
            "item": item_name,
            "bin": bin_type,
            "total_sorted": sim._items_sorted,
        }

    @llm.register_function(
        description="Get the current state of the simulation: items, robot position, "
        "and sorting progress."
    )
    async def get_simulation_state() -> Dict[str, Any]:
        """Return current simulation state snapshot."""
        sim = get_simulation()
        state = sim.get_state()
        return {
            "time": round(state.time, 2),
            "arm_busy": state.arm_busy,
            "gripper_open": state.gripper_open,
            "items_sorted": state.items_sorted,
            "ee_position": [round(x, 3) for x in state.ee_pos],
            "items": state.items,
        }

    @llm.register_function(
        description="Automatically scan for ALL hazardous items and sort them into "
        "the correct bins. This runs the full detect-match-sort pipeline."
    )
    async def sort_all_hazards() -> Dict[str, Any]:
        """Full automated pipeline: detect → match → pick-and-place all hazards."""
        bridge = get_bridge()
        result = bridge.detect_and_sort()
        return {
            "hazards_detected": result["detected"],
            "items_matched": result["matched"],
            "items_sorted": result["sorted"],
            "details": result["details"],
        }

    return llm


# ─── Agent Creation ──────────────────────────────────────────────────────────


async def create_agent(**kwargs) -> Agent:
    """Create the SemSorter agent with Vision-Agents SDK."""
    llm = setup_llm()

    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="SemSorter AI", id="semsorter-agent"),
        instructions=INSTRUCTIONS,
        llm=llm,
        tts=elevenlabs.TTS(model_id="eleven_flash_v2_5"),
        stt=deepgram.STT(eager_turn_detection=True),
        processors=[],
    )

    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join a GetStream video call and start the agent loop."""
    call = await agent.create_call(call_type, call_id)

    async with agent.join(call):
        # Greet the user
        await agent.simple_response(
            "Hello! I'm the SemSorter AI. I can scan the conveyor belt "
            "for hazardous items and sort them into the correct bins. "
            "Just tell me what to do!"
        )
        # Run until the call ends
        await agent.finish()


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
