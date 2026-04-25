# go2_rec_policy/rewards.py
"""Rewards for the recovery policy.

Shape of the reward, at a high level:

  track_lin_vel_xy   — track commanded xy body velocity (exp)
  track_ang_vel_z    — track commanded yaw rate (exp)
  base_height        — encourage the base to return to nominal standing height
  orientation        — encourage projected gravity to point down (upright)
  feet_air_time      — gait shaping once upright
  + the usual regularizers (torques, dof_vel, action_rate, ...)

There is NO position-tracking, heading-tracking, or obstacle-aware reward;
those were specific to the agile policy.
"""
from __future__ import annotations
import torch

from isaaclab.utils import configclass
from isaaclab.managers import (
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
)
from isaaclab.envs import ManagerBasedRLEnv
import isaaclab.envs.mdp as mdp


# ── Sensor entity aliases (match names defined in scene_cfg) ────────────
_thigh_sensor = SceneEntityCfg("contact_forces", body_names=[".*thigh.*"])
_calf_sensor  = SceneEntityCfg("contact_forces", body_names=[".*calf.*"])
_hip_sensor   = SceneEntityCfg("contact_forces", body_names=[".*hip.*"])
_foot_sensor  = SceneEntityCfg("contact_forces", body_names=[".*_foot"])
_body_sensor  = SceneEntityCfg("contact_forces", body_names=[".*thigh.*", ".*calf.*"])


# ──────────────────────────────────────────────────────────────────────────
# Core recovery rewards
# ──────────────────────────────────────────────────────────────────────────

def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv,
    std: float = 0.25,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """exp(-|cmd_xy - vel_xy|^2 / std^2). Shape: (N,)."""
    asset     = env.scene[asset_cfg.name]
    cmd_xy    = env.command_manager.get_command(command_name)[:, :2]
    vel_xy_b  = asset.data.root_lin_vel_b[:, :2]
    err_sq    = torch.sum((cmd_xy - vel_xy_b) ** 2, dim=-1)
    return torch.exp(-err_sq / (std ** 2))


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv,
    std: float = 0.25,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """exp(-(cmd_wz - wz)^2 / std^2). Shape: (N,)."""
    asset    = env.scene[asset_cfg.name]
    cmd_wz   = env.command_manager.get_command(command_name)[:, 2]
    wz_b     = asset.data.root_ang_vel_b[:, 2]
    return torch.exp(-((cmd_wz - wz_b) ** 2) / (std ** 2))


def base_height_reward(
    env: ManagerBasedRLEnv,
    target_height: float = 0.3,
    sigma: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gaussian reward peaked at the standing height. Drives self-righting."""
    asset  = env.scene[asset_cfg.name]
    height = asset.data.root_pos_w[:, 2]
    err    = (height - target_height).abs()
    return torch.exp(-err / sigma)


def upright_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Rewards the base for being upright.

    projected_gravity[:, 2] ∈ [-1, +1], with -1 when fully upright
    (gravity points straight down in the base frame).  We map this so that
    upright → +1 reward and fully inverted → 0.
    """
    asset      = env.scene[asset_cfg.name]
    g_body     = asset.data.projected_gravity_b       # (N, 3)
    # (1 - g_z) / 2  ∈ [0, 1],  1 when upright, 0 when upside-down
    return (1.0 - g_body[:, 2]) * 0.5


# ──────────────────────────────────────────────────────────────────────────
# Regularizers (shared with agile policy)
# ──────────────────────────────────────────────────────────────────────────

def undesired_body_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Returns 1.0 per env where *any* filtered body exceeds `threshold`."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    asset          = env.scene[asset_cfg.name]
    body_ids, _    = asset.find_bodies(sensor_cfg.body_names)
    forces         = contact_sensor.data.net_forces_w[:, body_ids, :]
    force_mag      = torch.norm(forces, dim=-1)
    return (force_mag > threshold).any(dim=-1).float()


def joint_torque_limits(
    env: ManagerBasedRLEnv,
    soft_ratio: float = 0.85,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalizes |torque| above soft_ratio * effort_limit."""
    asset         = env.scene[asset_cfg.name]
    torques       = asset.data.applied_torque
    torque_limits = asset.data.soft_joint_vel_limits
    if hasattr(asset.data, "joint_effort_limits"):
        torque_limits = asset.data.joint_effort_limits
    soft_limits = torque_limits * soft_ratio
    excess      = (torch.abs(torques) - soft_limits).clamp(min=0)
    return -excess.sum(dim=-1)


def joint_dof_vel_limits(
    env: ManagerBasedRLEnv,
    soft_ratio: float = 0.9,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset       = env.scene[asset_cfg.name]
    dof_vel     = asset.data.joint_vel
    vel_limits  = asset.data.soft_joint_vel_limits
    soft_limits = vel_limits * soft_ratio
    excess      = (torch.abs(dof_vel) - soft_limits).clamp(min=0)
    return -excess.sum(dim=-1)


def feet_air_time(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=[".*foot.*"]),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_air_time: float = 0.25,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    asset          = env.scene[asset_cfg.name]
    body_ids, _    = asset.find_bodies(sensor_cfg.body_names)

    forces       = contact_sensor.data.net_forces_w[:, body_ids, :]
    contact_filt = torch.norm(forces, dim=-1) > 1.0

    if not hasattr(env, "_feet_air_time"):
        env._feet_air_time = torch.zeros(env.num_envs, len(body_ids), device=env.device)

    first_contact = (env._feet_air_time > 0.0) & contact_filt
    env._feet_air_time += env.step_dt
    env._feet_air_time *= ~contact_filt

    return torch.sum(
        (env._feet_air_time - min_air_time) * first_contact.float(), dim=-1
    )


@configclass
class RewardsCfg:
    # ── Core recovery / tracking ────────────────────────────────────────
    track_lin_vel_xy = RewTerm(
        func=track_lin_vel_xy_exp,
        weight=1.5,
        params={"std": 0.5, "command_name": "base_velocity"},
    )
    track_ang_vel_z = RewTerm(
        func=track_ang_vel_z_exp,
        weight=0.75,
        params={"std": 0.5, "command_name": "base_velocity"},
    )
    base_height = RewTerm(
        func=base_height_reward,
        weight=5.0,
        params={"target_height": 0.3, "sigma": 0.15},
    )
    upright = RewTerm(
        func=upright_reward,
        weight=5.0,
    )
    air_time = RewTerm(
        func=feet_air_time,
        weight=1.0,
        params={
            "sensor_cfg":        _foot_sensor,
            "min_air_time":      0.25,
            "command_threshold": 0.1,
        },
    )

    # ── Penalties & regularizers ────────────────────────────────────────
    lin_vel_z       = RewTerm(func=mdp.lin_vel_z_l2,      weight=-1.0)
    ang_vel_xy      = RewTerm(func=mdp.ang_vel_xy_l2,     weight=-0.05)
    orientation     = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    dof_acc         = RewTerm(func=mdp.joint_acc_l2,      weight=-2.5e-7)
    torques         = RewTerm(func=mdp.joint_torques_l2,  weight=-0.0002)
    dof_vel_penalty = RewTerm(func=mdp.joint_vel_l2,      weight=-0.0005)
    dof_pos_limits  = RewTerm(func=mdp.joint_pos_limits,  weight=-10.0)
    action_rate     = RewTerm(func=mdp.action_rate_l2,    weight=-0.005)

    torque_limits = RewTerm(
        func=joint_torque_limits,
        weight=5.0,
        params={"soft_ratio": 0.85},
    )
    dof_vel_limits = RewTerm(
        func=joint_dof_vel_limits,
        weight=10.0,
        params={"soft_ratio": 0.9},
    )

    # NB: recovery does *not* hard-penalize thigh/calf contacts — the robot
    # often starts lying on its thigh and must be allowed to transition out
    # of that contact without a huge penalty.  We give a small penalty to
    # discourage lingering.
    thigh_contact = RewTerm(
        func=undesired_body_contact,
        weight=-1.0,
        params={"sensor_cfg": _thigh_sensor, "threshold": 1.0},
    )
    calf_contact = RewTerm(
        func=undesired_body_contact,
        weight=-1.0,
        params={"sensor_cfg": _calf_sensor, "threshold": 1.0},
    )
    hip_contact = RewTerm(
        func=undesired_body_contact,
        weight=-2.0,
        params={"sensor_cfg": _hip_sensor, "threshold": 1.0},
    )
