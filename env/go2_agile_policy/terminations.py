# go1_pos_rough_env_cfg.py
from __future__ import annotations
import numpy as np
import torch


from isaaclab.utils import configclass
from isaaclab.managers import (
    ObservationGroupCfg,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    TerminationTermCfg as DoneTerm,
    EventTermCfg as EventTerm,
    SceneEntityCfg,
)
from isaaclab.envs import ManagerBasedRLEnv

import isaaclab.envs.mdp as mdp

def base_height_too_low(
    env: ManagerBasedRLEnv,
    min_height: float = 0.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminates if robot base z position drops below min_height.
    Returns bool tensor of shape (num_envs,).
    """
    asset = env.scene[asset_cfg.name]
    base_height = asset.data.root_pos_w[:, 2]   # (num_envs,) world-frame z
    return base_height < min_height

@configclass
class TerminationsCfg:
    time_out    = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            # body_names goes INSIDE SceneEntityCfg, not as a top-level param
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["base"],       # ← here, not outside
            ),
            "threshold": 1.0,
            # NO "body_names" key here ← remove this
        },
    )
    too_low = DoneTerm(
        func=base_height_too_low,
        params={
            "min_height": 0.2,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )