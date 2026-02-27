"""
SemSorter VLM-to-Simulation Bridge

Maps VLM hazard detections to simulation item names and orchestrates
the pick-and-place sequence. This is the glue between Phase 2 (Vision)
and Phase 1 (Simulation).

Usage:
    # End-to-end test (direct render, no OBS):
    MUJOCO_GL=egl GOOGLE_API_KEY=... python3 vlm_bridge.py --direct

    # With OBS Virtual Camera:
    GOOGLE_API_KEY=... python3 vlm_bridge.py
"""

import os
import sys
import logging
from typing import List, Dict, Optional, Tuple

try:
    from .vision_pipeline import HazardDetectionProcessor
except ImportError:
    from vision_pipeline import HazardDetectionProcessor

try:
    from ..simulation.controller import BinType, SemSorterSimulation
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'simulation'))
    from controller import BinType, SemSorterSimulation

logger = logging.getLogger(__name__)


class VLMSimBridge:
    """
    Bridge between VLM hazard detections and the simulation controller.
    
    Matching strategy:
    1. VLM detects items by color/shape → returns type (FLAMMABLE/CHEMICAL)
    2. Simulation has named items with known hazard types
    3. We match VLM detections to unpicked simulation items of the same type
    4. For multiple items of the same type, we use spatial ordering (left-to-right
       on the conveyor) to assign matches
    """

    def __init__(self, simulation, device_id: int = 4, use_direct: bool = False):
        """
        Args:
            simulation: SemSorterSimulation instance
            device_id: OBS Virtual Camera device ID
            use_direct: If True, render frames from simulation instead of OBS
        """
        self.simulation = simulation
        self.processor = HazardDetectionProcessor(
            device_id=device_id,
            simulation=simulation if use_direct else None
        )

    def get_unpicked_items_by_type(self, hazard_type: str) -> List[Tuple[str, float]]:
        """
        Get unpicked simulation items of a given hazard type,
        sorted by X position (leftmost first = highest priority on conveyor).
        
        Returns:
            List of (item_name, x_position) tuples
        """
        type_map = {
            "FLAMMABLE": BinType.FLAMMABLE,
            "CHEMICAL": BinType.CHEMICAL,
        }
        
        target_type = type_map.get(hazard_type)
        if target_type is None:
            return []
        
        items = []
        for name, info in self.simulation.items.items():
            if info.hazard_type == target_type and not info.picked:
                pos = self.simulation.get_item_pos(name)
                if pos is not None:
                    items.append((name, pos[0]))  # x_position for sorting
        
        # Sort by X (most negative = leftmost on conveyor = first to pick)
        items.sort(key=lambda x: x[1])
        return items

    def match_detections_to_items(self, detections: List[Dict]) -> List[Dict]:
        """
        Match VLM detections to simulation item names.
        
        Each detection gets an additional 'sim_item' key with the matched
        simulation item name, and 'bin_type' with the target bin.
        
        Returns:
            List of matched detections with sim_item and bin_type fields added
        """
        # Track which items have already been matched
        matched_items = set()
        results = []

        def box_left_x(det: Dict) -> float:
            box = det.get("box_2d")
            if isinstance(box, (list, tuple)) and len(box) >= 2:
                try:
                    return float(box[1])
                except (TypeError, ValueError):
                    pass
            return 1000.0
        
        # Group detections by type
        for det_type in ["FLAMMABLE", "CHEMICAL"]:
            type_detections = []
            for d in detections:
                if not isinstance(d, dict):
                    continue
                dtype = str(d.get("type", "")).strip().upper()
                if dtype == det_type:
                    type_detections.append(d)
            available_items = self.get_unpicked_items_by_type(det_type)
            
            # Sort detections by x position of bounding box (leftmost first)
            type_detections.sort(key=box_left_x)
            
            bin_type = BinType.FLAMMABLE if det_type == "FLAMMABLE" else BinType.CHEMICAL
            
            for i, detection in enumerate(type_detections):
                # Find first available item not yet matched
                sim_item = None
                for item_name, _ in available_items:
                    if item_name not in matched_items:
                        sim_item = item_name
                        matched_items.add(item_name)
                        break
                
                if sim_item:
                    detection["sim_item"] = sim_item
                    detection["bin_type"] = bin_type
                    results.append(detection)
                    logger.info(f"Matched VLM '{detection.get('name')}' → "
                               f"sim '{sim_item}' → bin '{bin_type.value}'")
                else:
                    logger.warning(f"No unmatched sim item for VLM detection: "
                                  f"{detection.get('name')} ({det_type})")
        
        return results

    def detect_and_sort(self) -> Dict:
        """
        Full pipeline: detect hazards → match to sim items → pick and place all.
        
        Returns:
            Summary dict with detection count, sort count, and details
        """
        # Step 1: Detect hazards
        logger.info("Step 1: Detecting hazards with VLM...")
        detections = self.processor.detect_hazards()
        logger.info(f"VLM found {len(detections)} hazardous items")
        
        if not detections:
            return {"detected": 0, "matched": 0, "sorted": 0, "details": []}
        
        # Step 2: Match to simulation items
        logger.info("Step 2: Matching detections to simulation items...")
        matched = self.match_detections_to_items(detections)
        logger.info(f"Matched {len(matched)} items")
        
        # Step 3: Pick and place each matched item
        logger.info("Step 3: Executing pick-and-place sequence...")
        details = []
        sorted_count = 0
        
        for match in matched:
            item_name = match["sim_item"]
            bin_type = match["bin_type"]
            vlm_name = match.get("name", "unknown")
            
            logger.info(f"Sorting: {vlm_name} ({item_name}) → {bin_type.value}")
            success = self.simulation.pick_and_place(item_name, bin_type)
            
            # Let remaining items settle after the arm moves
            self.simulation.step(200)
            
            details.append({
                "vlm_name": vlm_name,
                "sim_item": item_name,
                "target_bin": bin_type.value,
                "success": success,
            })
            
            if success:
                sorted_count += 1
        
        return {
            "detected": len(detections),
            "matched": len(matched),
            "sorted": sorted_count,
            "details": details,
        }

    def close(self):
        """Release resources."""
        self.processor.close()


# ─── CLI entry point ────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="SemSorter VLM-Sim Bridge")
    parser.add_argument("--direct", action="store_true",
                        help="Use direct simulation rendering instead of OBS")
    parser.add_argument("--device", type=int, default=4,
                        help="OBS Virtual Camera device ID (default: 4)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Initialize simulation
    print("Initializing simulation...")
    if args.direct:
        os.environ.setdefault("MUJOCO_GL", "egl")
    sim = SemSorterSimulation()
    sim.load_scene()
    sim.step(200)  # Let physics settle

    # Initialize bridge
    bridge = VLMSimBridge(
        simulation=sim,
        device_id=args.device,
        use_direct=args.direct,
    )

    try:
        # Run full detect → match → sort pipeline
        print("\n" + "=" * 60)
        print("  SemSorter: VLM-Driven Hazard Sorting")
        print("=" * 60)
        
        result = bridge.detect_and_sort()
        
        print("\n" + "=" * 60)
        print("  SORTING RESULTS")
        print("=" * 60)
        print(f"  Hazards detected by VLM:  {result['detected']}")
        print(f"  Matched to sim items:     {result['matched']}")
        print(f"  Successfully sorted:      {result['sorted']}")
        
        if result['details']:
            print("\n  Details:")
            for d in result['details']:
                status = "✅" if d['success'] else "❌"
                print(f"    {status} {d['vlm_name']} ({d['sim_item']}) → {d['target_bin']}")
        
        print("=" * 60)
        
        # Save final state
        sim.save_frame("after_sort.png")
        print(f"\nFinal scene saved to after_sort.png")

    finally:
        bridge.close()
        if hasattr(sim, "close"):
            sim.close()


if __name__ == "__main__":
    main()
