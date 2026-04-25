# go2_rec_policy/scene_cfg.py
from __future__ import annotations
import numpy as np
import torch
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg

from assets.robots import make_robot_cfg


@configclass
class Go2RecoverySceneCfg(InteractiveSceneCfg):
    """Scene for the recovery policy.

    Unlike the agile policy, recovery trains on a clean plane with no
    obstacles — the robot only needs to learn to right itself and track
    velocity commands from arbitrary (possibly fallen) initial poses.
    """

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
    )

    robot: ArticulationCfg = make_robot_cfg(0)

    contact_forces: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot_*/.*",
        update_period=0.0,
        history_length=3,
        debug_vis=False,
    )
