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
from isaaclab.utils.noise import UniformNoiseCfg
from isaaclab.envs import ManagerBasedRLEnv
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from sensors.ray2d import obj_relative_positions, ray2d_with_illusion
from sensors.tactile import contact_forces_binary
from .sensor_cfg import _sensors_cfg

def normalized_timer_left(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """
    Remaining episode fraction in [0, 1].
    Mirrors original: self.timer_left.unsqueeze(1) / self.max_episode_length_s
    Shape: (num_envs, 1)
    """
    # episode_length_buf counts steps elapsed, max_episode_length is total steps
    steps_left = env.max_episode_length - env.episode_length_buf   # (num_envs,)
    normalized = steps_left.float() / env.max_episode_length       # (num_envs,)
    return normalized.unsqueeze(1)  

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        # ── 0:4  contact state ──────────────────────────────────────────
        contact_forces = ObsTerm(
            func=contact_forces_binary,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
                ),
                "threshold": 1.0,
            },
        )
        # ── 4:7  angular velocity ───────────────────────────────────────
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            scale=1.0,                        # obs_scales.ang_vel
            noise=UniformNoiseCfg(n_min=-0.1, n_max=0.1),  # noise_scales.ang_vel
        )
        # ── 7:10 gravity projection ─────────────────────────────────────
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),  # noise_scales.gravity
        )
        # ── 10:13 position + heading commands ───────────────────────────
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "pos_heading"},
        )
        # ── 13:14 normalised time remaining ─────────────────────────────
        timer_left = ObsTerm(
            func=normalized_timer_left,   # implement: timer_left / episode_length_s
        )
        # ── 14:26 joint positions ───────────────────────────────────────
        dof_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            scale=1.0,                        # obs_scales.dof_pos
            noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),  # noise_scales.dof_pos
        )
        # ── 26:38 joint velocities ──────────────────────────────────────
        dof_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=0.2,                        # obs_scales.dof_vel
            noise=UniformNoiseCfg(n_min=-0.3, n_max=0.3),    # noise_scales.dof_vel
        )
        # ── 38:50 last actions ──────────────────────────────────────────
        actions = ObsTerm(func=mdp.last_action)

        # ── 50:74 object relative positions (8 objects × 3) ────────────
        # Calls obj_relative_positions() from observations.py
        object_relative_pos = ObsTerm(
            func=obj_relative_positions
        )

        # ── 74:85 ray2d scan (11 rays from theta config) ────────────────
        # Calls ray2d_with_illusion() from observations.py
        # Only registered when sensor is enabled
        ray2d = ObsTerm(
            func=ray2d_with_illusion,
            scale=1.0,
            noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
            # params={
            #     "sensor_name": "ray2d",
            #     "ray2d_cfg":   _sensors_cfg.ray2d,
            # },
        )
        # if _sensors_cfg.ray2d.enable else None

        def __post_init__(self):
            self.enable_corruption = False     # noise.add_noise = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()