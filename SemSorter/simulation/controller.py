"""
SemSorter MuJoCo Simulation Controller

This module manages the Franka Panda robotic arm simulation for the SemSorter
project. It loads the Panda from mujoco_menagerie, adds conveyors, waste bins,
and hazardous items, then provides an async API for pick-and-place operations.

Usage:
    python controller.py              # Launch interactive viewer
    python controller.py --render     # Render a test frame to PNG
"""

import asyncio
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mujoco
import mujoco.viewer
import numpy as np

logger = logging.getLogger(__name__)

# ─── Path configuration ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # SemSorter/
MENAGERIE_DIR = PROJECT_ROOT / "mujoco_menagerie"
PANDA_SCENE = MENAGERIE_DIR / "franka_emika_panda" / "scene.xml"


# ─── Data types ──────────────────────────────────────────────────────────────
class BinType(str, Enum):
    FLAMMABLE = "flammable"
    CHEMICAL = "chemical"
    OUTPUT = "output"  # safe items go to output conveyor


@dataclass
class ItemInfo:
    """Metadata for a conveyor item."""
    name: str
    body_id: int
    geom_id: int
    is_hazardous: bool
    hazard_type: Optional[BinType] = None  # which bin it should go to
    picked: bool = False


@dataclass
class SimState:
    """Observable simulation state for the frontend."""
    time: float = 0.0
    ee_pos: Tuple[float, float, float] = (0, 0, 0)
    gripper_open: bool = True
    items: List[Dict] = field(default_factory=list)
    arm_busy: bool = False
    items_sorted: int = 0


# ─── Bin positions (world coordinates) ───────────────────────────────────────
BIN_POSITIONS = {
    BinType.FLAMMABLE: np.array([-0.25, -0.40, 0.35]),   # Above the red bin
    BinType.CHEMICAL:  np.array([0.25, -0.40, 0.35]),     # Above the yellow bin
    BinType.OUTPUT:    np.array([0.40, 0.0, 0.40]),       # Output conveyor
}

# ─── Panda joint configuration ──────────────────────────────────────────────
# Actuator indices (from panda.xml):
#   0-6: arm joints (actuator1-7)
#   7:   gripper (actuator8, ctrl 0=closed, 255=fully open)
GRIPPER_ACTUATOR_ID = 7
GRIPPER_OPEN = 255.0
GRIPPER_CLOSED = 0.0
NUM_ARM_JOINTS = 7
ENV_CONTACT_TYPE = 2  # Keep environment/item contacts separate from robot links.


class SemSorterSimulation:
    """
    MuJoCo simulation controller for the SemSorter pick-and-place task.

    Loads the Franka Panda from menagerie, adds the warehouse environment
    (conveyors, bins, items), and provides an async API for robot control.
    """

    def __init__(self):
        self.model: Optional[mujoco.MjModel] = None
        self.data: Optional[mujoco.MjData] = None
        self.renderer: Optional[mujoco.Renderer] = None
        self.items: Dict[str, ItemInfo] = {}
        self._arm_busy = False
        self._items_sorted = 0
        self._running = False

    # ─── Scene loading ───────────────────────────────────────────────────

    def load_scene(self) -> None:
        """Load the Panda scene from menagerie and add SemSorter objects."""
        # Load base Panda scene
        logger.info(f"Loading Panda from: {PANDA_SCENE}")
        spec = mujoco.MjSpec.from_file(str(PANDA_SCENE))

        # Modify the model name
        spec.modelname = "semsorter"

        # Set offscreen framebuffer size for rendering
        spec.visual.global_.offwidth = 1920
        spec.visual.global_.offheight = 1080

        # ─── Add additional lights ───────────────────────────────────────
        world = spec.worldbody
        light = world.add_light()
        light.pos = [0, -1, 2]
        light.dir = [0, 0.5, -0.8]
        light.diffuse = [0.4, 0.4, 0.4]

        light2 = world.add_light()
        light2.pos = [-1, -1, 2]
        light2.dir = [0.3, 0.3, -0.8]
        light2.diffuse = [0.3, 0.3, 0.3]

        # ─── Add cameras ────────────────────────────────────────────────
        cam_overview = world.add_camera()
        cam_overview.name = "overview"
        cam_overview.pos = [0, -1.4, 1.3]
        cam_overview.quat = [0.92, 0.38, 0, 0]  # Look slightly down
        cam_overview.fovy = 50

        cam_top = world.add_camera()
        cam_top.name = "topdown"
        cam_top.pos = [0, 0, 2.0]
        cam_top.quat = [0.0, 0.0, 0.0, 1.0]  # Look straight down
        cam_top.fovy = 60

        cam_side = world.add_camera()
        cam_side.name = "side"
        cam_side.pos = [1.5, 0, 0.8]
        cam_side.quat = [0.65, 0.27, 0.27, 0.65]  # Side view
        cam_side.fovy = 45

        # ─── Add conveyors ──────────────────────────────────────────────
        self._add_conveyor(spec, "input", pos=[-0.40, 0, 0])
        self._add_conveyor(spec, "output", pos=[0.40, 0, 0])

        # ─── Add waste bins ─────────────────────────────────────────────
        self._add_bin(spec, "flammable", pos=[-0.25, -0.40, 0],
                      color=[0.85, 0.15, 0.1, 0.9])
        self._add_bin(spec, "chemical", pos=[0.25, -0.40, 0],
                      color=[0.95, 0.75, 0.1, 0.9])

        # ─── Add hazardous items on input conveyor ──────────────────────
        items_spec = [
            ("item_flammable_1", [-0.50, 0.0, 0.40], "cylinder", [0.025, 0.03],
             [0.9, 0.1, 0.1, 1], True, BinType.FLAMMABLE),
            ("item_chemical_1", [-0.40, 0.05, 0.40], "box", [0.025, 0.025, 0.025],
             [0.95, 0.85, 0.1, 1], True, BinType.CHEMICAL),
            ("item_chemical_2", [-0.30, -0.03, 0.40], "sphere", [0.025],
             [0.95, 0.85, 0.1, 1], True, BinType.CHEMICAL),
            ("item_safe_1", [-0.35, -0.05, 0.40], "box", [0.03, 0.025, 0.02],
             [0.6, 0.6, 0.6, 1], False, BinType.OUTPUT),
            ("item_safe_2", [-0.55, 0.04, 0.40], "cylinder", [0.022, 0.025],
             [0.9, 0.9, 0.9, 1], False, BinType.OUTPUT),
            ("item_flammable_2", [-0.45, 0.02, 0.40], "box", [0.022, 0.022, 0.022],
             [0.9, 0.1, 0.1, 1], True, BinType.FLAMMABLE),
        ]

        for name, pos, shape, size, rgba, is_haz, haz_type in items_spec:
            self._add_item(spec, name, pos, shape, size, rgba)
            self.items[name] = ItemInfo(
                name=name, body_id=-1, geom_id=-1,
                is_hazardous=is_haz, hazard_type=haz_type if is_haz else None,
            )

        # Store desired spawn positions for post-keyframe initialization
        self._item_spawn_positions = {
            name: pos for name, pos, *_ in items_spec
        }

        # ─── Compile the model ──────────────────────────────────────────
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)

        # Keep floor contacts in the environment collision group (not robot group).
        floor_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if floor_geom_id >= 0:
            self.model.geom_contype[floor_geom_id] = ENV_CONTACT_TYPE
            self.model.geom_conaffinity[floor_geom_id] = ENV_CONTACT_TYPE

        # Resolve body/geom IDs for items
        for name in self.items:
            self.items[name].body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            geom_name = f"{name}_geom"
            self.items[name].geom_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)

        # ─── Reset to home pose ─────────────────────────────────────────
        key_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)

        # ─── Set item initial positions (keyframe only has arm joints) ──
        for name, pos in self._item_spawn_positions.items():
            jnt_name = f"{name}_jnt"
            jnt_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
            if jnt_id >= 0:
                qadr = self.model.jnt_qposadr[jnt_id]
                # freejoint qpos: [x, y, z, qw, qx, qy, qz]
                self.data.qpos[qadr:qadr+3] = pos
                self.data.qpos[qadr+3:qadr+7] = [1, 0, 0, 0]  # identity quat

        mujoco.mj_forward(self.model, self.data)

        logger.info(f"Scene compiled: {self.model.nbody} bodies, "
                     f"{self.model.njnt} joints, {self.model.nu} actuators")
        logger.info(f"Items registered: {list(self.items.keys())}")

    def _add_conveyor(self, spec: mujoco.MjSpec, name: str, pos: list) -> None:
        """Add a conveyor belt with frame and legs."""
        world = spec.worldbody
        body = world.add_body()
        body.name = f"conveyor_{name}"
        body.pos = pos

        # Belt surface
        belt = body.add_geom()
        belt.name = f"belt_{name}"
        belt.type = mujoco.mjtGeom.mjGEOM_BOX
        belt.size = [0.35, 0.12, 0.005]
        belt.pos = [0, 0, 0.35]
        belt.rgba = [0.15, 0.15, 0.15, 1]
        belt.friction = [0.8, 0.005, 0.0001]
        belt.contype = ENV_CONTACT_TYPE
        belt.conaffinity = ENV_CONTACT_TYPE

        # Side rails
        for side_name, y in [("L", 0.125), ("R", -0.125)]:
            rail = body.add_geom()
            rail.name = f"rail_{name}_{side_name}"
            rail.type = mujoco.mjtGeom.mjGEOM_BOX
            rail.size = [0.35, 0.005, 0.02]
            rail.pos = [0, y, 0.37]
            rail.rgba = [0.4, 0.4, 0.45, 1]
            rail.contype = ENV_CONTACT_TYPE
            rail.conaffinity = ENV_CONTACT_TYPE

        # Legs
        for lx, ly in [(-0.3, 0.1), (-0.3, -0.1), (0.3, 0.1), (0.3, -0.1)]:
            leg = body.add_geom()
            leg.type = mujoco.mjtGeom.mjGEOM_CYLINDER
            leg.size = [0.015, 0.175, 0]
            leg.pos = [lx, ly, 0.175]
            leg.rgba = [0.4, 0.4, 0.45, 1]
            leg.contype = ENV_CONTACT_TYPE
            leg.conaffinity = ENV_CONTACT_TYPE

    def _add_bin(self, spec: mujoco.MjSpec, name: str, pos: list,
                 color: list) -> None:
        """Add an open-top waste bin."""
        world = spec.worldbody
        body = world.add_body()
        body.name = f"bin_{name}"
        body.pos = pos

        # Walls
        wall_specs = [
            (f"bin_{name}_back",  [0, -0.095, 0.12], [0.1, 0.005, 0.12]),
            (f"bin_{name}_front", [0, 0.095, 0.12],  [0.1, 0.005, 0.12]),
            (f"bin_{name}_left",  [-0.095, 0, 0.12], [0.005, 0.1, 0.12]),
            (f"bin_{name}_right", [0.095, 0, 0.12],  [0.005, 0.1, 0.12]),
        ]
        for wname, wpos, wsize in wall_specs:
            wall = body.add_geom()
            wall.name = wname
            wall.type = mujoco.mjtGeom.mjGEOM_BOX
            wall.size = wsize
            wall.pos = wpos
            wall.rgba = color
            wall.contype = ENV_CONTACT_TYPE
            wall.conaffinity = ENV_CONTACT_TYPE

        # Bottom
        bottom = body.add_geom()
        bottom.name = f"bin_{name}_bottom"
        bottom.type = mujoco.mjtGeom.mjGEOM_BOX
        bottom.size = [0.1, 0.1, 0.005]
        bottom.pos = [0, 0, 0.005]
        bottom.rgba = [0.1, 0.1, 0.1, 1]
        bottom.contype = ENV_CONTACT_TYPE
        bottom.conaffinity = ENV_CONTACT_TYPE

    def _add_item(self, spec: mujoco.MjSpec, name: str, pos: list,
                  shape: str, size: list, rgba: list) -> None:
        """Add a free-jointed item to the world."""
        world = spec.worldbody
        body = world.add_body()
        body.name = name
        body.pos = pos

        # Free joint
        jnt = body.add_freejoint()
        jnt.name = f"{name}_jnt"

        # Geom
        geom = body.add_geom()
        geom.name = f"{name}_geom"
        shape_map = {
            "box": mujoco.mjtGeom.mjGEOM_BOX,
            "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
            "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
        }
        geom.type = shape_map[shape]
        geom.size = size + [0] * (3 - len(size))  # Pad to 3 elements
        geom.rgba = rgba
        geom.mass = 0.05
        geom.friction = [1.0, 0.005, 0.0001]
        geom.priority = 1
        geom.contype = ENV_CONTACT_TYPE
        geom.conaffinity = ENV_CONTACT_TYPE

    # ─── End-effector helpers ────────────────────────────────────────────

    def get_ee_pos(self) -> np.ndarray:
        """Get current end-effector (hand) position in world coords."""
        hand_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        return self.data.xpos[hand_id].copy()

    def get_ee_site_pos(self) -> np.ndarray:
        """Get EE position — alias."""
        return self.get_ee_pos()

    def get_item_pos(self, item_name: str) -> Optional[np.ndarray]:
        """Get position of an item by name."""
        info = self.items.get(item_name)
        if info and info.body_id >= 0:
            return self.data.xpos[info.body_id].copy()
        return None

    def _set_item_pose(self, item_name: str, pos: np.ndarray,
                       quat: Tuple[float, float, float, float] = (1, 0, 0, 0)) -> bool:
        """Directly place an item free-joint at a world pose."""
        jnt_name = f"{item_name}_jnt"
        jnt_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
        if jnt_id < 0:
            return False
        qadr = self.model.jnt_qposadr[jnt_id]
        self.data.qpos[qadr:qadr+3] = pos
        self.data.qpos[qadr+3:qadr+7] = quat
        dadr = self.model.jnt_dofadr[jnt_id]
        self.data.qvel[dadr:dadr+6] = 0.0
        return True

    # ─── IK (Solver-based) ────────────────────────────────────────────

    def reset_arm_neutral(self) -> None:
        """
        Move arm to a neutral upright pose where IK works well in all directions.
        """
        neutral_qpos = [0.0, -0.3, 0.0, -2.0, 0.0, 1.8, 0.0]
        # Set qpos directly for arm joints (first 7)
        self.data.qpos[:NUM_ARM_JOINTS] = neutral_qpos
        self.data.ctrl[:NUM_ARM_JOINTS] = neutral_qpos
        mujoco.mj_forward(self.model, self.data)

    def solve_ik(self, target_pos: np.ndarray,
                 target_quat: Optional[np.ndarray] = None,
                 max_iter: int = 300,
                 tolerance: float = 0.015,
                 step_size: float = 0.5,
                 damping: float = 0.05) -> Optional[np.ndarray]:
        """
        Pure kinematic IK solver — iterates Jacobian on qpos WITHOUT physics.
        Returns joint angles (length 7) or None if failed.
        """
        hand_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")

        # Save original qpos to restore later (critical for not corrupting physics)
        orig_qpos = self.data.qpos.copy()
        
        # Work on a copy of qpos
        qpos_arm = orig_qpos[:NUM_ARM_JOINTS].copy()

        try:
            for _ in range(max_iter):
                # Temporarily set qpos, run forward kinematics
                self.data.qpos[:NUM_ARM_JOINTS] = qpos_arm
                mujoco.mj_forward(self.model, self.data)

                current_pos = self.data.xpos[hand_id].copy()
                err_pos = target_pos - current_pos

                # Position Jacobian
                jacp = np.zeros((3, self.model.nv))
                mujoco.mj_jacBody(self.model, self.data, jacp, None, hand_id)
                J = jacp[:, :NUM_ARM_JOINTS]
                error = err_pos

                if target_quat is not None:
                    current_quat = self.data.xquat[hand_id].copy()
                    err_rot = np.zeros(3)
                    mujoco.mju_subQuat(err_rot, target_quat, current_quat)
                    
                    # Rotation Jacobian
                    jacr = np.zeros((3, self.model.nv))
                    mujoco.mj_jacBody(self.model, self.data, None, jacr, hand_id)
                    Jr = jacr[:, :NUM_ARM_JOINTS]
                    
                    # Scale rotational error so position takes priority
                    J = np.vstack([J, Jr * 0.5])
                    error = np.concatenate([error, err_rot * 0.5])

                if np.linalg.norm(error) < tolerance:
                    return qpos_arm.copy()

                # Damped least squares
                JJT = J @ J.T + damping**2 * np.eye(J.shape[0])
                dq = J.T @ np.linalg.solve(JJT, error)

                # Update with step size and clamping
                dq = np.clip(dq * step_size, -0.2, 0.2)
                qpos_arm += dq

                # Clamp to joint limits
                for j in range(NUM_ARM_JOINTS):
                    jnt_id = j  # arm joints are first 7
                    lo = self.model.jnt_range[jnt_id, 0]
                    hi = self.model.jnt_range[jnt_id, 1]
                    if lo < hi:
                        qpos_arm[j] = np.clip(qpos_arm[j], lo * 0.95, hi * 0.95)

            return None  # Did not converge
            
        finally:
            # Always restore original qpos and run forward to fix physics state
            self.data.qpos[:] = orig_qpos
            mujoco.mj_forward(self.model, self.data)

    def move_to_position(self, target_pos: np.ndarray,
                         move_steps: int = 400,
                         settle_steps: int = 100,
                         position_tolerance: float = 0.05,
                         carry_item: Optional[str] = None,
                         carry_offset: Optional[np.ndarray] = None) -> bool:
        """
        Move end-effector to target position.
        1. Solve IK kinematically
        2. Interpolate joint targets smoothly (ease-in/ease-out)
        3. Step physics to let arm move
        Returns True if IK solution found.
        """
        solution = self.solve_ik(target_pos)
        if solution is None:
            logger.warning(f"IK failed for target {target_pos}")
            return False

        current_ctrl = self.data.ctrl[:NUM_ARM_JOINTS].copy()
        
        if carry_item is not None and carry_offset is None:
            carry_offset = np.array([0.0, 0.0, -0.06])

        # Smooth interpolation to target
        for i in range(move_steps):
            alpha = (i + 1) / move_steps
            t = alpha * alpha * (3 - 2 * alpha)  # Smoothstep
            self.data.ctrl[:NUM_ARM_JOINTS] = current_ctrl * (1 - t) + solution * t
            mujoco.mj_step(self.model, self.data)
            if carry_item is not None:
                ee = self.get_ee_pos()
                self._set_item_pose(carry_item, ee + carry_offset)

        # Settle
        for _ in range(settle_steps):
            mujoco.mj_step(self.model, self.data)
            if carry_item is not None:
                ee = self.get_ee_pos()
                self._set_item_pose(carry_item, ee + carry_offset)

        if carry_item is not None:
            ee = self.get_ee_pos()
            self._set_item_pose(carry_item, ee + carry_offset)
            mujoco.mj_forward(self.model, self.data)

        final_ee = self.get_ee_pos()
        err = np.linalg.norm(target_pos - final_ee)
        if err > position_tolerance:
            logger.warning(
                f"Move failed: target {target_pos}, reached {final_ee}, err={err:.4f}")
            return False
        return True

    def set_gripper(self, open_gripper: bool) -> None:
        """Open or close the gripper."""
        self.data.ctrl[GRIPPER_ACTUATOR_ID] = (
            GRIPPER_OPEN if open_gripper else GRIPPER_CLOSED
        )

    def step(self, n: int = 1) -> None:
        """Advance the simulation by n steps."""
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    # ─── High-level pick-place operations ────────────────────────────────

    def _stabilize_unpicked_items(self, exclude: str = "") -> None:
        """Zero out velocities of all unpicked items to prevent physics drift.
        
        Called before/after each pick-and-place so that the arm doesn't
        knock neighboring items off the conveyor.
        """
        for name, info in self.items.items():
            if name == exclude or info.picked:
                continue
            jnt_name = f"{name}_jnt"
            jnt_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
            if jnt_id < 0:
                continue
            dadr = self.model.jnt_dofadr[jnt_id]
            self.data.qvel[dadr:dadr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def pick_and_place(self, item_name: str, target_bin: BinType) -> bool:
        """
        Execute a full pick-and-place sequence:
        1. Open gripper
        2. Move above item
        3. Move down to item
        4. Close gripper
        5. Move up
        6. Move above target bin
        7. Open gripper (drop)
        8. Return to neutral
        """
        info = self.items.get(item_name)
        if not info or info.picked:
            logger.warning(f"Item {item_name} not found or already picked")
            return False

        # Freeze all other items in place before we move the arm
        self._stabilize_unpicked_items(exclude=item_name)

        self._arm_busy = True
        try:
            item_pos = self.get_item_pos(item_name)
            if item_pos is None:
                logger.warning(f"Cannot get position for {item_name}")
                return False

            # Sanity check: item must be within reachable workspace
            if (abs(item_pos[0]) > 1.0 or abs(item_pos[1]) > 1.0
                    or item_pos[2] < 0.0 or item_pos[2] > 1.0):
                logger.warning(
                    f"Item {item_name} at {item_pos} is outside reachable "
                    f"workspace — it may have been displaced by physics")
                return False

            logger.info(f"Picking {item_name} at {item_pos} -> {target_bin.value}")

            # 1. Open gripper
            self.set_gripper(True)
            self.step(50)

            # 1.5 Move high to ensure we clear the scene
            safe_high = np.array([0.0, 0.0, 0.65])
            if not self.move_to_position(safe_high, move_steps=200, settle_steps=50):
                return False

            # Re-read item position after safe-high move (physics may shift items)
            item_pos = self.get_item_pos(item_name)
            if item_pos is None or item_pos[2] < 0.0 or item_pos[2] > 1.0:
                logger.warning(
                    f"Item {item_name} moved to invalid position {item_pos} "
                    f"during arm movement")
                return False

            # 2. Move above item (approach from above)
            approach_pos = item_pos.copy()
            approach_pos[2] += 0.10
            if not self.move_to_position(approach_pos):
                logger.warning(f"Failed to reach approach position for {item_name}")
                return False

            # 3. Move down to grasp
            grasp_pos = item_pos.copy()
            grasp_pos[2] += 0.03
            if not self.move_to_position(grasp_pos):
                logger.warning(f"Failed to reach grasp position for {item_name}")
                return False

            # 4. Close gripper
            self.set_gripper(False)
            self.step(120)  # allow gripper to close

            # Verify we are close enough to claim a grasp.
            ee_pos = self.get_ee_pos()
            item_now = self.get_item_pos(item_name)
            if item_now is None or np.linalg.norm(ee_pos - item_now) > 0.12:
                logger.warning(
                    f"Grasp verification failed for {item_name}: "
                    f"ee={ee_pos}, item={item_now}")
                return False

            # Kinematic carry of the item for deterministic phase testing.
            carry_offset = np.array([0.0, 0.0, -0.06])
            self._set_item_pose(item_name, ee_pos + carry_offset)
            mujoco.mj_forward(self.model, self.data)

            # 5. Lift up while carrying.
            lift_pos = grasp_pos.copy()
            lift_pos[2] += 0.22
            if not self.move_to_position(
                lift_pos, carry_item=item_name, carry_offset=carry_offset):
                return False

            # 6. Move above target bin while carrying.
            bin_pos = BIN_POSITIONS[target_bin].copy()
            if not self.move_to_position(
                bin_pos, carry_item=item_name, carry_offset=carry_offset):
                return False

            # 7. Place and release.
            drop_pos = bin_pos.copy()
            drop_pos[2] -= 0.12
            self._set_item_pose(item_name, drop_pos)
            mujoco.mj_forward(self.model, self.data)
            self.set_gripper(True)
            self.step(100)

            # Mark item as sorted only after successful place.
            info.picked = True
            self._items_sorted += 1

            # 8. Return to neutral.
            neutral = np.array([0.0, 0.0, 0.6])
            self.move_to_position(neutral)

            # Stabilize remaining items after arm movement
            self._stabilize_unpicked_items()

            logger.info(f"Successfully placed {item_name} in {target_bin.value}")
            return True
        finally:
            self._arm_busy = False

    # ─── State snapshot ──────────────────────────────────────────────────

    def get_state(self) -> SimState:
        """Get current simulation state for the frontend."""
        ee = self.get_ee_pos()
        items_info = []
        for name, info in self.items.items():
            pos = self.get_item_pos(name)
            items_info.append({
                "name": name,
                "pos": pos.tolist() if pos is not None else [0, 0, 0],
                "is_hazardous": info.is_hazardous,
                "hazard_type": info.hazard_type.value if info.hazard_type else None,
                "picked": info.picked,
            })

        return SimState(
            time=self.data.time,
            ee_pos=tuple(ee),
            gripper_open=self.data.ctrl[GRIPPER_ACTUATOR_ID] > 100,
            items=items_info,
            arm_busy=self._arm_busy,
            items_sorted=self._items_sorted,
        )

    # ─── Rendering ───────────────────────────────────────────────────────

    def render_frame(self, width: int = 1280, height: int = 720,
                     camera: str = "overview") -> np.ndarray:
        """Render a frame from the specified camera. Returns RGB array."""
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height, width)

        cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)

        self.renderer.update_scene(self.data, camera=cam_id)
        return self.renderer.render()

    def save_frame(self, path: str, camera: str = "overview") -> None:
        """Render a frame and save as PNG."""
        from PIL import Image
        frame = self.render_frame(camera=camera)
        Image.fromarray(frame).save(path)
        logger.info(f"Frame saved to {path}")

    def close(self) -> None:
        """Release renderer resources explicitly."""
        if self.renderer is not None:
            try:
                self.renderer.close()
            except Exception:
                pass  # EGL cleanup errors are harmless at shutdown
            self.renderer = None

    # ─── Interactive viewer ──────────────────────────────────────────────

    def launch_viewer(self) -> None:
        """Launch the interactive MuJoCo viewer."""
        key_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        mujoco.viewer.launch(self.model, self.data)

    # ─── Async interface for agent integration ───────────────────────────

    async def async_pick_and_place(self, item_name: str,
                                    target_bin: BinType) -> Dict:
        """Async wrapper around pick_and_place for agent integration."""
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(
            None, self.pick_and_place, item_name, target_bin
        )
        return {
            "success": success,
            "item": item_name,
            "target_bin": target_bin.value,
            "items_sorted": self._items_sorted,
        }

    async def async_get_state(self) -> Dict:
        """Async state snapshot."""
        state = self.get_state()
        return {
            "time": state.time,
            "ee_pos": list(state.ee_pos),
            "gripper_open": state.gripper_open,
            "items": state.items,
            "arm_busy": state.arm_busy,
            "items_sorted": state.items_sorted,
        }


# ─── CLI entry point ────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="SemSorter Simulation Controller")
    parser.add_argument("--render", action="store_true",
                        help="Render a test frame and save as PNG")
    parser.add_argument("--test-pick", action="store_true",
                        help="Test pick-and-place of first hazardous item")
    parser.add_argument("--output", default="test_frame.png",
                        help="Output path for rendered frame")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    sim = SemSorterSimulation()
    sim.load_scene()

    try:
        if args.render:
            sim.save_frame(args.output)
            print(f"Frame saved to {args.output}")
        elif args.test_pick:
            print("Testing pick-and-place...")
            sim.pick_and_place("item_flammable_1", BinType.FLAMMABLE)
            sim.save_frame("after_pick.png")
            print(f"Done! Items sorted: {sim._items_sorted}")
        else:
            print("Launching interactive viewer...")
            sim.launch_viewer()
    finally:
        sim.close()


if __name__ == "__main__":
    main()
