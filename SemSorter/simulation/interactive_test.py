"""
Interactive viewer for SemSorter simulation.
Runs pick-and-place in real time with the MuJoCo viewer.

Usage:
    python3 interactive_test.py
"""
import os
import time
import mujoco
import mujoco.viewer
try:
    from .controller import SemSorterSimulation, BinType
except ImportError:
    from controller import SemSorterSimulation, BinType

# How often to sync the viewer (every N physics steps)
VIEWER_SYNC_INTERVAL = 10

def main():
    print("Initializing simulation...")
    # NOTE: Do NOT set MUJOCO_GL=egl when using the interactive viewer
    if 'MUJOCO_GL' in os.environ:
        del os.environ['MUJOCO_GL']
        
    sim = SemSorterSimulation()
    sim.load_scene()
    
    print("Launching interactive viewer. Watch the arm move!")
    
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        
        # Patch mj_step to sync viewer every N steps (much faster than every step)
        original_mj_step = mujoco.mj_step
        step_counter = [0]
        
        def patched_mj_step(model, data):
            original_mj_step(model, data)
            step_counter[0] += 1
            if step_counter[0] % VIEWER_SYNC_INTERVAL == 0:
                viewer.sync()
                # Sleep only on sync frames to maintain ~real-time playback
                time.sleep(model.opt.timestep * VIEWER_SYNC_INTERVAL)
            
        mujoco.mj_step = patched_mj_step
        
        try:
            # Let the scene settle
            sim.step(200)
            
            time.sleep(2)  # Give user time to see the initial state
            print("\nStarting pick-and-place operation...")
            success = sim.pick_and_place("item_flammable_1", BinType.FLAMMABLE)

            print(f"\nDone! success={success}, items sorted: {sim._items_sorted}")
            print("\nYou can close the viewer window now, or press Ctrl+C.")
            
            # Keep viewer open until user closes it
            while viewer.is_running():
                original_mj_step(sim.model, sim.data)
                viewer.sync()
                time.sleep(0.02)  # ~50 FPS idle

        except KeyboardInterrupt:
            print("\nViewer closed.")
        finally:
            mujoco.mj_step = original_mj_step

if __name__ == "__main__":
    main()
