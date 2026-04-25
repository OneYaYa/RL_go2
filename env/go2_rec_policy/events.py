# go2_rec_policy/events.py
"""Events for the recovery policy.

Critical differences vs the agile policy:
  * Reset pose ranges cover (near-)arbitrary orientations including flipped /
    tipped-over configurations.  This is what makes the policy a *recovery*
    policy: it must learn to right itself from any state.
  * Joints are perturbed more aggressively so the robot starts in a
    sprawled configuration, not a standing one.
  * No obstacles to reset (the recovery scene is bare).
"""
from __future__ import annotations
import numpy as np
import torch

from isaaclab.utils import configclass
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    SceneEntityCfg,
)

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnv


def reset_feet_air_time(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
) -> None:
    """Resets feet-air-time buffer at the start of each episode."""
    if hasattr(env, "_feet_air_time"):
        env._feet_air_time[env_ids] = 0.0


@configclass
class EventCfg:
    # Material randomization for robustness.
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

    # ── Initial pose: FULL orientation randomization including flipped ────
    # This is the defining feature of the recovery policy.  Roll/pitch/yaw
    # span the full sphere so the robot is equally likely to start upright,
    # on its side, or fully inverted.
    reset_robot_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x":     (-0.5, 0.5),
                "y":     (-0.5, 0.5),
                "z":     (0.0, 0.0),      # sometimes spawn slightly above ground
                "roll":  (-3.14/10, 3.14/10),
                "pitch": (-3.14/10, 3.14/10),
                # "yaw":   (-3.14/10, 3.14/10),
            },
            "velocity_range": {
                "x":     (-0.5, 5.0),
                "y":     (-0.5, 0.5),
                "z":     (-0.5, 0.5),
                "roll":  (-1.5/5, 1.5/5),
                "pitch": (-1.5/5, 1.5/5),
                "yaw":   (-1.5/5, 1.5/5),
            },
        },
    )

    # ── Joints: wider init range so legs start sprawled / folded ──────────
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "asset_cfg":      SceneEntityCfg("robot"),
            "position_range": (0.75, 1.25), 
            "velocity_range": (-0.25, 0.25),
        },
    )

    # ── Occasional in-flight perturbations to keep the policy robust ──────
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 3.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {
                "x": (-0.5, 0.5), "y": (-0.5, 0.5),
            },
        },
    )
