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

# env/go2_agile_policy/rewards.py

import torch
from isaaclab.envs import ManagerBasedRLEnv

_thigh_sensor    = SceneEntityCfg("contact_forces", body_names=[".*thigh.*"])
_calf_sensor     = SceneEntityCfg("contact_forces", body_names=[".*calf.*"])
_hip_sensor      = SceneEntityCfg("contact_forces", body_names=[".*hip.*"])
_foot_sensor = SceneEntityCfg(
    "contact_forces",
    body_names=[".*_foot"],       # regex matched against body names
)

def undesired_body_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Penalizes contact on any body except the feet.
    Returns -1.0 per env with any violating contact, else 0.0.
    Shape: (N,)
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    asset          = env.scene[asset_cfg.name]

    # Resolve body ids from name filter in sensor_cfg
    body_ids, _    = asset.find_bodies(sensor_cfg.body_names)
    forces         = contact_sensor.data.net_forces_w[:, body_ids, :]  # (N, B, 3)
    force_mag      = torch.norm(forces, dim=-1)                         # (N, B)

    # Any body in the filtered set exceeding threshold
    in_contact     = (force_mag > threshold).any(dim=-1).float()        # (N,)
    return in_contact

def reach_pos_target_soft(
    env: ManagerBasedRLEnv,
    sigma: float = 2.0,                          # position_target_sigma_soft
    command_name: str = "pos_heading",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Gaussian reward for being near the target position (wide sigma).
    Mirrors original: reach_pos_target_soft with sigma=2.0
    Shape: (num_envs,)
    """
    asset      = env.scene[asset_cfg.name]
    robot_pos  = asset.data.root_pos_w[:, :2]                        # (N, 2) world xy

    # position_targets stored in env by resample_position_targets event
    target_pos = env._position_targets[:, :2]                        # (N, 2) world xy

    dist = torch.norm(target_pos - robot_pos, dim=-1)                # (N,)
    return torch.exp(-dist / sigma)                                   # (N,) in (0, 1]


def reach_pos_target_tight(
    env: ManagerBasedRLEnv,
    sigma: float = 0.5,                          # position_target_sigma_tight
    command_name: str = "pos_heading",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Gaussian reward for being near the target position (narrow sigma).
    Mirrors original: reach_pos_target_tight with sigma=0.5
    Shape: (num_envs,)
    """
    asset      = env.scene[asset_cfg.name]
    robot_pos  = asset.data.root_pos_w[:, :2]                        # (N, 2) world xy
    target_pos = env._position_targets[:, :2]                        # (N, 2) world xy

    dist = torch.norm(target_pos - robot_pos, dim=-1)                # (N,)
    return torch.exp(-dist / sigma)                                   # (N,) in (0, 1]


def reach_heading_target(
    env: ManagerBasedRLEnv,
    sigma: float = 1.0,                          # heading_target_sigma
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Gaussian reward for matching the heading target.
    Mirrors original: reach_heading_target with sigma=1.0
    Shape: (num_envs,)
    """
    from isaaclab.utils.math import wrap_to_pi, yaw_quat
    asset      = env.scene[asset_cfg.name]
    quat       = asset.data.root_quat_w                              # (N, 4)

    # current yaw
    forward    = torch.zeros(env.num_envs, 3, device=env.device)
    forward[:, 0] = 1.0
    from isaaclab.utils.math import quat_apply
    fwd_world  = quat_apply(quat, forward)
    heading    = torch.atan2(fwd_world[:, 1], fwd_world[:, 0])      # (N,)

    # heading target stored by resample_position_targets
    hdg_target = env._heading_targets[:, 0]                         # (N,)
    hdg_err    = wrap_to_pi(hdg_target - heading)                   # (N,)

    return torch.exp(-hdg_err.abs() / sigma)                        # (N,)


def velocity_direction(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Rewards velocity pointing toward the target.
    Mirrors original: velo_dir reward.
    Shape: (num_envs,)
    """
    asset      = env.scene[asset_cfg.name]
    robot_pos  = asset.data.root_pos_w[:, :2]                       # (N, 2)
    robot_vel  = asset.data.root_lin_vel_w[:, :2]                   # (N, 2)
    target_pos = env._position_targets[:, :2]                       # (N, 2)

    to_target  = target_pos - robot_pos                             # (N, 2)
    dist       = torch.norm(to_target, dim=-1, keepdim=True).clamp(min=1e-6)
    to_target_unit = to_target / dist                               # (N, 2) unit vec

    vel_norm   = torch.norm(robot_vel, dim=-1, keepdim=True).clamp(min=1e-6)
    vel_unit   = robot_vel / vel_norm                               # (N, 2) unit vec

    # cosine similarity: 1.0 = perfect alignment, -1.0 = opposite
    alignment  = (to_target_unit * vel_unit).sum(dim=-1)            # (N,)
    return alignment.clamp(-1.0, 1.0)

def joint_torque_limits(
    env: ManagerBasedRLEnv,
    soft_ratio: float = 0.85,                    # soft_torque_limit
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Penalizes torques exceeding soft_ratio * torque limit.
    Mirrors original: torque_limits reward.
    Shape: (num_envs,)
    """
    asset         = env.scene[asset_cfg.name]
    torques       = asset.data.applied_torque                        # (N, num_dof)
    torque_limits = asset.data.soft_joint_vel_limits                 # (N, num_dof)
    # use gear_ratio-scaled effort limits if available
    if hasattr(asset.data, "joint_effort_limits"):
        torque_limits = asset.data.joint_effort_limits               # (N, num_dof)

    soft_limits   = torque_limits * soft_ratio                       # (N, num_dof)
    excess        = (torch.abs(torques) - soft_limits).clamp(min=0)  # (N, num_dof)
    return -excess.sum(dim=-1)                                        # (N,)


def joint_dof_vel_limits(
    env: ManagerBasedRLEnv,
    soft_ratio: float = 0.9,                     # soft_dof_vel_limit
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Penalizes joint velocities exceeding soft_ratio * velocity limit.
    Mirrors original: dof_vel_limits reward.
    Shape: (num_envs,)
    """
    asset      = env.scene[asset_cfg.name]
    dof_vel    = asset.data.joint_vel                                 # (N, num_dof)
    vel_limits = asset.data.soft_joint_vel_limits                    # (N, num_dof)

    soft_limits = vel_limits * soft_ratio                            # (N, num_dof)
    excess      = (torch.abs(dof_vel) - soft_limits).clamp(min=0)   # (N, num_dof)
    return -excess.sum(dim=-1)                                        # (N,)

_body_sensor   = SceneEntityCfg("contact_forces", body_names=[".*thigh.*", ".*calf.*"])

def body_collision(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Penalizes undesired body contacts (thigh, calf).
    Mirrors original: penalize_contacts_on = ["thigh", "calf"]
    Returns -1.0 per env that has any violating contact, else 0.0.
    Shape: (num_envs,)
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    net_forces     = contact_sensor.data.net_forces_w        # (N, num_bodies, 3)
    force_mag      = torch.norm(net_forces, dim=-1)           # (N, num_bodies)

    # Filter to penalized bodies via body_names in sensor_cfg
    if sensor_cfg.body_names is not None:
        body_ids, _  = env.scene[asset_cfg.name].find_bodies(sensor_cfg.body_names)
        force_mag    = force_mag[:, body_ids]                 # (N, num_penalized)

    # Any contact above threshold in the penalized bodies
    in_contact = (force_mag > threshold).any(dim=-1).float() # (N,)
    return in_contact 

def feet_air_time(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=[".*foot.*"]),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_air_time: float = 0.25,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """
    Rewards long steps — only on first contact after being airborne.
    Mirrors: _reward_feet_air_time
    Shape: (N,)
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    asset          = env.scene[asset_cfg.name]
    body_ids, _    = asset.find_bodies(sensor_cfg.body_names)

    # Current contact state (filtered)
    forces         = contact_sensor.data.net_forces_w[:, body_ids, :]   # (N, 4, 3)
    contact_filt   = torch.norm(forces, dim=-1) > 1.0                   # (N, 4) bool

    # Initialise air time buffer on first call
    if not hasattr(env, "_feet_air_time"):
        env._feet_air_time = torch.zeros(
            env.num_envs, len(body_ids), device=env.device
        )

    # First contact = was airborne last step, now touching
    first_contact  = (env._feet_air_time > 0.0) & contact_filt          # (N, 4)

    # Accumulate air time
    env._feet_air_time += env.step_dt
    env._feet_air_time *= ~contact_filt                                  # reset on contact

    # Reward only on first contact, only when moving command given
    cmd_vel        = env.command_manager.get_command("pos_heading")[:, :2]
    # moving         = torch.norm(cmd_vel, dim=-1) > command_threshold     # (N,)

    rew_air_time   = torch.sum(
        (env._feet_air_time - min_air_time) * first_contact.float(),
        dim=-1,
    )                                                                     # (N,)
    return rew_air_time #* moving.float()  
 
def fly_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Penalizes when no feet are in contact (robot is airborne).
    Mirrors original: fly reward.
    Shape: (num_envs,)
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    net_forces     = contact_sensor.data.net_forces_w        # (N, num_bodies, 3)
    force_mag      = torch.norm(net_forces, dim=-1)           # (N, num_bodies)

    if sensor_cfg.body_names is not None:
        body_ids, _  = env.scene[asset_cfg.name].find_bodies(sensor_cfg.body_names)
        force_mag    = force_mag[:, body_ids]                 # (N, num_feet)

    # No feet touching = all contacts below threshold
    any_contact = (force_mag > threshold).any(dim=-1)        # (N,) bool
    airborne    = (~any_contact).float()                      # 1.0 if flying
    return airborne                                          # 0.0 or -1.0

def no_move_penalty(
    env: ManagerBasedRLEnv,
    min_dist_threshold: float = 0.5,         # robot should be moving toward target
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Penalizes when robot is far from target but not moving.
    Mirrors original: nomove reward.
    Shape: (num_envs,)
    """
    asset      = env.scene[asset_cfg.name]
    robot_pos  = asset.data.root_pos_w[:, :2]                # (N, 2)
    robot_vel  = asset.data.root_lin_vel_w[:, :2]            # (N, 2)
    target_pos = env._position_targets[:, :2]                # (N, 2)

    dist       = torch.norm(target_pos - robot_pos, dim=-1)  # (N,)
    speed      = torch.norm(robot_vel, dim=-1)               # (N,)

    # Penalize: far from target AND not moving
    far_and_still = (dist > min_dist_threshold) & (speed < 0.1)
    return far_and_still.float()                            # 0.0 or -1.0


def stand_still_pos(
    env: ManagerBasedRLEnv,
    near_target_threshold: float = 0.3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Penalizes movement when already at target position.
    Mirrors original: stand_still_pos reward.
    Shape: (num_envs,)
    """
    asset      = env.scene[asset_cfg.name]
    robot_pos  = asset.data.root_pos_w[:, :2]                # (N, 2)
    robot_vel  = asset.data.root_lin_vel_w[:, :2]            # (N, 2)
    target_pos = env._position_targets[:, :2]                # (N, 2)

    dist  = torch.norm(target_pos - robot_pos, dim=-1)       # (N,)
    speed = torch.norm(robot_vel, dim=-1)                    # (N,)

    # At target but still moving
    at_target_moving = (dist < near_target_threshold) & (speed > 0.1)
    return at_target_moving.float()                         # 0.0 or -1.0
def base_height_reward(
    env: ManagerBasedRLEnv,
    target_height: float = 0.28,       # Go2 standing height ~0.28m
    sigma: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset  = env.scene[asset_cfg.name]
    height = asset.data.root_pos_w[:, 2]               # (N,)
    error  = (height - target_height).abs()
    return torch.exp(-error / sigma)                   # peaks at target height
@configclass
class RewardsCfg:
    # Navigation
    target_soft  = RewTerm(func=reach_pos_target_soft,   weight=0.0)
    target_tight = RewTerm(func=reach_pos_target_tight,  weight=0.0)
    heading_target   = RewTerm(func=reach_heading_target,    weight=0.0)
    velo_dir               = RewTerm(func=velocity_direction,      weight=0.0)
    # Penalties
    termination            = RewTerm(func=mdp.is_terminated,           weight=-1000.0)
    torques                = RewTerm(func=mdp.joint_torques_l2,        weight=-0.0002)
    dof_pos_limits         = RewTerm(func=mdp.joint_pos_limits,        weight=-20.0)
    dof_vel_penalty        = RewTerm(func=mdp.joint_vel_l2,            weight=-0.0005)
    torque_limits  = RewTerm(
        func=joint_torque_limits,
        weight=5.0,
        params={"soft_ratio": 0.85},
    )
    dof_vel_limits = RewTerm(
        func=joint_dof_vel_limits,
        weight=20.0,
        params={"soft_ratio": 0.9},
    )
    lin_vel_z              = RewTerm(func=mdp.lin_vel_z_l2,            weight=-2.0)
    ang_vel_xy             = RewTerm(func=mdp.ang_vel_xy_l2,           weight=-0.05)
    dof_acc                = RewTerm(func=mdp.joint_acc_l2,            weight=-2.5e-7)
    collision              = RewTerm(func=body_collision,          weight=-0.0,
                                     params={"sensor_cfg": _body_sensor,
                                       "threshold": 1.0})
    # feet_collision         = RewTerm(func=mdp.body_collision,          weight=-100.0,
    #                                  params={"body_names": ["foot"]})
    action_rate            = RewTerm(func=mdp.action_rate_l2,          weight=-0.005)
    stand_still            = RewTerm(func=stand_still_pos,         weight=-0.0)
    orientation            = RewTerm(func=mdp.flat_orientation_l2,     weight=-10.0)
    fly = RewTerm(
    func=fly_penalty,
    weight=-1.0,
    params={
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*foot.*"]),
    },
)
    nomove                 = RewTerm(func=no_move_penalty,         weight=-0.0)
    air_time  = RewTerm(
        func=feet_air_time,
        weight=5.0,
        params={
            "sensor_cfg":         _foot_sensor,
            "min_air_time":       0.25,
            "command_threshold":  0.1,
        },
    )

    base_height    = RewTerm(
        func=base_height_reward,
        weight=8.0,
        params={"target_height": 0.3, "sigma": 0.15},
    )
    thigh_contact  = RewTerm(
        func=undesired_body_contact,
        weight=-10.0,                    # same as original collision penalty
        params={
            "sensor_cfg": _thigh_sensor,
            "threshold":  1.0,
        },
    )
    calf_contact   = RewTerm(
        func=undesired_body_contact,
        weight=-10.0,
        params={
            "sensor_cfg": _calf_sensor,
            "threshold":  1.0,
        },
    )
    hip_contact    = RewTerm(
        func=undesired_body_contact,
        weight=-10.0,
        params={
            "sensor_cfg": _hip_sensor,
            "threshold":  1.0,
        },
    )


