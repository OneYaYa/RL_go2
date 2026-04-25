from __future__ import annotations
import numpy as np
import torch

from isaaclab.assets import ArticulationCfg, RigidObjectCfg
import isaaclab.sim as sim_utils
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
from isaaclab.actuators import IdealPDActuatorCfg

def make_robot_cfg(idx: int) -> RigidObjectCfg:
    
    return UNITREE_GO2_CFG.replace(
        prim_path="{ENV_REGEX_NS}" + f"/Robot_{idx}",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.2),
            joint_pos={
                'FL_hip_joint':    0.0,
                'RL_hip_joint':    0.0,
                'FR_hip_joint':    0.0,
                'RR_hip_joint':    0.0,
                'FL_thigh_joint':  0.8,
                'RL_thigh_joint':  0.8,
                'FR_thigh_joint':  0.8,
                'RR_thigh_joint':  0.8,
                'FL_calf_joint':  -1.5,
                'RL_calf_joint':  -1.5,
                'FR_calf_joint':  -1.5,
                'RR_calf_joint':  -1.5,
            },
        ),
        actuators={
        "legs": IdealPDActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            stiffness=30.0,
            damping=0.65,
        )
    },
    )