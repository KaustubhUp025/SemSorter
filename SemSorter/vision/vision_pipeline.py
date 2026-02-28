"""
SemSorter Vision Pipeline — Hazard Detection Processor

Captures frames from OBS Virtual Camera or directly from the simulation,
then sends them to Gemini VLM for hazardous item detection.

Usage:
    # From OBS Virtual Camera:
    GOOGLE_API_KEY=... python3 vision_pipeline.py

    # From simulation directly (no OBS needed):
    GOOGLE_API_KEY=... python3 vision_pipeline.py --direct
"""

import os
import sys
import json
import logging
import google.generativeai as genai
from PIL import Image
from typing import List, Dict

logger = logging.getLogger(__name__)


class HazardDetectionProcessor:
    """
    Detects hazardous items in the SemSorter simulation using Gemini VLM.
    
    Supports two input modes:
    - OBS Virtual Camera: reads from /dev/videoX
    - Direct simulation rendering: calls sim.render_frame()
    """

    def __init__(self, device_id: int = 4, simulation=None):
        """
        Args:
            device_id: Video device ID for OBS Virtual Camera (e.g., 4 for /dev/video4)
            simulation: Optional SemSorterSimulation instance for direct rendering
        """
        self.device_id = device_id
        self.simulation = simulation
        self._video_cap = None  # Reusable VideoCapture
        self._gemini_model = None  # Lazy-initialized

        # System instructions to enforce structured JSON output
        self.system_instruction = (
            "You are an AI vision system for a robotic waste sorting arm. "
            "You are given an image of a conveyor belt with a robotic arm and waste bins. "
            "Your task is to identify hazardous items on the conveyor belt. "
            "Hazardous items are categorized as:\n"
            "- FLAMMABLE: Red-colored items (cylinders, boxes)\n"
            "- CHEMICAL: Yellow-colored items (boxes, spheres)\n\n"
            "Safe items are gray, white, green, or blue — IGNORE these.\n\n"
            "For each hazardous item detected, return a JSON object with:\n"
            "- 'name': descriptive name like 'red_cylinder_1' or 'yellow_box_1'\n"
            "- 'type': either 'FLAMMABLE' or 'CHEMICAL'\n"
            "- 'color': the detected color (e.g., 'red', 'yellow')\n"
            "- 'shape': the detected shape (e.g., 'cylinder', 'box', 'sphere')\n"
            "- 'box_2d': bounding box as [ymin, xmin, ymax, xmax] normalized to 0-1000 scale\n\n"
            "Return ONLY a JSON array of detected hazardous items. "
            "If no hazardous items are visible, return an empty array []."
        )

    def _get_gemini_model(self):
        """Lazy-initialize Gemini model (only when analyze_frame is called)."""
        if self._gemini_model is None:
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError(
                    "GOOGLE_API_KEY environment variable not set.\n"
                    "Get one at https://aistudio.google.com/apikey"
                )
            genai.configure(api_key=api_key)
            self._gemini_model = genai.GenerativeModel(
                model_name="gemini-3-flash-preview",
                system_instruction=self.system_instruction,
                generation_config={"response_mime_type": "application/json"}
            )
        return self._gemini_model

    def capture_frame(self) -> Image.Image:
        """
        Capture a single frame.
        Uses direct simulation rendering if available, otherwise OBS camera.
        """
        if self.simulation is not None:
            return self._capture_from_simulation()
        else:
            return self._capture_from_obs()

    def _capture_from_simulation(self) -> Image.Image:
        """Render a frame directly from the MuJoCo simulation."""
        frame = self.simulation.render_frame(camera="overview")
        return Image.fromarray(frame)

    def _capture_from_obs(self) -> Image.Image:
        """Capture a frame from the OBS Virtual Camera."""
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV is required for OBS capture mode. "
                "Install opencv-python or opencv-python-headless."
            ) from exc

        if self._video_cap is None or not self._video_cap.isOpened():
            self._video_cap = cv2.VideoCapture(self.device_id)
            if not self._video_cap.isOpened():
                raise RuntimeError(
                    f"Could not open video device /dev/video{self.device_id}. "
                    "Ensure OBS Virtual Camera is running."
                )
            # Warm up — discard stale frames
            for _ in range(5):
                self._video_cap.read()

        ret, frame = self._video_cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame from OBS Virtual Camera")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)

    def analyze_frame(self, pil_image: Image.Image) -> List[Dict]:
        """
        Send the image to Gemini VLM and parse the structured JSON response.
        
        Returns:
            List of dicts, each with keys: name, type, color, shape, box_2d
        """
        prompt = (
            "Analyze this image of a robotic sorting station. "
            "Identify all FLAMMABLE (red) and CHEMICAL (yellow) items "
            "on the conveyor belt. Return their positions as bounding boxes."
        )

        logger.info("Sending frame to Gemini VLM...")
        model = self._get_gemini_model()
        response = model.generate_content([prompt, pil_image])

        raw_text = getattr(response, "text", None)
        if not isinstance(raw_text, str) or not raw_text.strip():
            logger.error("VLM response did not contain JSON text output")
            return []

        try:
            results = json.loads(raw_text)
            if isinstance(results, dict) and "items" in results:
                results = results["items"]
            if not isinstance(results, list):
                logger.error(f"Unexpected VLM JSON shape: {type(results).__name__}")
                return []
            logger.info(f"VLM detected {len(results)} hazardous items")
            return results
        except (json.JSONDecodeError, TypeError):
            logger.error(f"Failed to parse VLM response:\n{raw_text}")
            return []

    def detect_hazards(self) -> List[Dict]:
        """
        Full pipeline: capture frame → analyze → return results.
        Convenience method combining capture_frame() and analyze_frame().
        """
        image = self.capture_frame()
        return self.analyze_frame(image)

    def close(self):
        """Release video capture resources."""
        if self._video_cap is not None:
            self._video_cap.release()
            self._video_cap = None


# ─── CLI entry point ────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="SemSorter Hazard Detection")
    parser.add_argument("--direct", action="store_true",
                        help="Use direct simulation rendering instead of OBS")
    parser.add_argument("--device", type=int, default=4,
                        help="OBS Virtual Camera device ID (default: 4)")
    parser.add_argument("--output", default="vision_debug.png",
                        help="Save captured frame to this path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    simulation = None
    if args.direct:
        # Must be set before importing MuJoCo/controller in this process.
        os.environ.setdefault("MUJOCO_GL", "egl")
        # Import and initialize simulation for direct rendering
        try:
            from ..simulation.controller import SemSorterSimulation
        except ImportError:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'simulation'))
            from controller import SemSorterSimulation
        print("Initializing simulation for direct rendering...")
        simulation = SemSorterSimulation()
        simulation.load_scene()
        simulation.step(200)  # Let physics settle

    processor = HazardDetectionProcessor(
        device_id=args.device,
        simulation=simulation
    )

    try:
        print("Capturing frame...")
        image = processor.capture_frame()
        image.save(args.output)
        print(f"Saved frame to {args.output}")

        print("Analyzing frame with Gemini VLM...")
        results = processor.analyze_frame(image)

        print("\n" + "=" * 50)
        print("  HAZARD DETECTION RESULTS")
        print("=" * 50)

        if not results:
            print("  No hazardous items detected.")
        else:
            for i, item in enumerate(results, 1):
                print(f"\n  [{i}] {item.get('name', 'unknown')}")
                print(f"      Type:  {item.get('type', '?')}")
                print(f"      Color: {item.get('color', '?')}")
                print(f"      Shape: {item.get('shape', '?')}")
                print(f"      Box:   {item.get('box_2d', '?')}")

        print("\n" + "=" * 50)
        print(f"  Total hazardous items: {len(results)}")
        print("=" * 50)

    finally:
        processor.close()
        if simulation is not None and hasattr(simulation, "close"):
            simulation.close()


if __name__ == "__main__":
    main()
