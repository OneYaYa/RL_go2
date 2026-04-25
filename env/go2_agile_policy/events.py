# go1_pos_rough_env_cfg.py
from __future__ import annotations
import numpy as np
import torch
from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab.managers import (
    ObservationGroupCfg,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    TerminationTermCfg as DoneTerm,
    EventTermCfg as EventTerm,
    SceneEntityCfg,
)

import isaaclab.envs.mdp as mdp
from assets.objects import reset_obstacles_random_sparse
# from .commands import resample_position_targets
from isaaclab.envs import ManagerBasedRLEnv

def reset_feet_air_time(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
) -> None:
    """Resets feet air time buffer on episode reset."""
    if hasattr(env, "_feet_air_time"):
        env._feet_air_time[env_ids] = 0.0

@configclass
class EventCfg:
    # Friction
    randomize_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range":  (0.3, 1.25),
            "dynamic_friction_range": (0.3, 1.25),
            "restitution_range":      (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    # Base mass
    randomize_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
            "mass_distribution_params": (-1.5, 1.5),
            "operation": "add",
        },
    )
    reset_feet_air_time = EventTerm(
    func=reset_feet_air_time,
    mode="reset",
)   
    # Initial pose — yaw + xy
    reset_robot_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x":   (-0.5, 0.5),
                "y":   (-0.5, 0.5),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x":        (-0.5, 0.5),
                "y":        (-0.5, 0.5),
                "z":        (-0.5, 0.5),
                "roll":     (-0.5, 0.5),
                "pitch":    (-0.5, 0.5),
                "yaw":      (-0.5, 0.5),
            },
        },
    )
    # DOF init randomization
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "asset_cfg":    SceneEntityCfg("robot"),
            "position_range": (0.5, 1.5),   # init_dof_factor
            "velocity_range": (0.0, 0.0),
        },
    )
    reset_obstacles = EventTerm(
        func=reset_obstacles_random_sparse,
        mode="reset",                         # called on every episode reset
        params={
            "radius_range":       (0.10, 0.30),
            "pos_x_range":        (-3.0,  3.0),
            "pos_y_range":        (-3.0,  3.0),
            "height":              0.05,
            "min_obstacle_dist":   0.8,        # sparsity: obstacles stay 0.8m apart
            "min_robot_dist":      1.2,        # clearance around robot spawn
            "max_tries":           50,
            "obstacle_names": [f"obstacle_{i}" for i in range(8)],
        },
    )
    # resample_targets = EventTerm(
    #     func=resample_position_targets,
    #     mode="reset",
    #     params={
    #         "pos_1_range":    (1.5, 7.5),
    #         "pos_2_range":    (-2.0, 2.0),
    #         "heading_range":  (-0.3, 0.3),
    #         "use_polar":      False,
    #         "num_objects":    8,
    #         "object_size_list": [0.4],
    #         "obstacle_radius_margin": 0.31415926,
    #         "obstacle_radius_scale":  1.1,
    #     },
    # )
    # Push
    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(2.5, 2.5),        # push_interval_s
    #     params={
    #         "asset_cfg":    SceneEntityCfg("robot"),
    #         "velocity_range": {
    #             "x": (-0.5, 0.5), "y": (-0.5, 0.5),
    #         },
    #     },
    # )

    