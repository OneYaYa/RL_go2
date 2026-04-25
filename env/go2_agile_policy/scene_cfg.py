# go1_pos_rough_env_cfg.py
from __future__ import annotations
import numpy as np
import torch
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg

from assets.robots import make_robot_cfg
from assets.objects import make_obstacle_cfg
from .sensor_cfg import _build_ray2d_sensor

@configclass
class Go2SceneCfg(InteractiveSceneCfg):

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
    )

    robot: ArticulationCfg = make_robot_cfg(0)

    obstacle_0: RigidObjectCfg = make_obstacle_cfg(0)
    obstacle_1: RigidObjectCfg = make_obstacle_cfg(1)
    obstacle_2: RigidObjectCfg = make_obstacle_cfg(2)
    obstacle_3: RigidObjectCfg = make_obstacle_cfg(3)
    obstacle_4: RigidObjectCfg = make_obstacle_cfg(4)
    obstacle_5: RigidObjectCfg = make_obstacle_cfg(5)
    obstacle_6: RigidObjectCfg = make_obstacle_cfg(6)
    obstacle_7: RigidObjectCfg = make_obstacle_cfg(7)

    ray2d = _build_ray2d_sensor()

    contact_forces: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot_*/.*",
        update_period=0.0,          # update every sim step
        history_length=3,
        debug_vis=False,
    )