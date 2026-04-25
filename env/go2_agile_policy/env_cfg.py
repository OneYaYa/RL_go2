from __future__ import annotations
import numpy as np
import torch
from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.managers import (
    ObservationGroupCfg,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    TerminationTermCfg as DoneTerm,
    EventTermCfg as EventTerm,
    SceneEntityCfg,
)
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.envs.mdp.actions import JointPositionActionCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

from .scene_cfg import Go2SceneCfg
from .observation_cfg import ObservationsCfg
from .rewards import RewardsCfg
from .terminations import TerminationsCfg
from .events import EventCfg
from .sensor_cfg import SensorsCfg
from .commands import CommandsCfg


@configclass
class ActionsCfg:
    """Joint position actions for 12-DOF quadruped."""

    joint_pos = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        scale=0.25,          # action_scale from your config
        use_default_offset=True,   # adds default_joint_pos as offset
    )

@configclass
class Go2PosRoughEnvCfg(ManagerBasedRLEnvCfg):

    # ── Sub-configs ────────────────────────────────────────────────────────
    scene:        Go2SceneCfg     = Go2SceneCfg(num_envs=4096, env_spacing=5.0)
    observations: ObservationsCfg = ObservationsCfg()
    commands    : CommandsCfg     = CommandsCfg()  
    rewards:      RewardsCfg      = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events:       EventCfg        = EventCfg()
    actions     : ActionsCfg      = ActionsCfg()

    # ── Sensors config exposed at env level (read by obs term functions) ───
    sensors: SensorsCfg = SensorsCfg()

    # ── Episode / control ──────────────────────────────────────────────────
    episode_length_s: float = 9.0
    decimation:       int   = 4       # control.decimation
    action_scale:     float = 0.25    # control.action_scale

    # ── Observation space size ─────────────────────────────────────────────
    # 50 base + 24 obj_relpos (8×3) + 11 ray2d = 85
    # Set to 50 if ray2d/objects disabled
    num_observations: int = 85
    num_objects = 8

    # ── Normalization ──────────────────────────────────────────────────────
    clip_observations: float = 100.0
    clip_actions:      float = 100.0

    # ── Noise master switch ────────────────────────────────────────────────
    add_noise:   bool  = False
    noise_level: float = 1.0

    def __post_init__(self):
        self.sim.dt             = 0.005              # 200 Hz physics
        self.sim.render_interval = self.decimation   # 50 Hz policy
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
        )
        self.sim.light = sim_utils.DomeLightCfg(
            intensity=2000.0,
            color=(1.0, 1.0, 1.0),
        )
        # Propagate num_envs into scene
        self.scene.num_envs = 1280

# @configclass
# class Go2PosRoughEnvCfgNoPenalty(Go2PosRoughEnvCfg):

#     @configclass
#     class _NoPenaltyEvents(EventCfg):
#         reset_robot_pose = EventTerm(
#             func=mdp.reset_root_state_uniform,
#             mode="reset",
#             params={
#                 "asset_cfg": SceneEntityCfg("robot"),
#                 "pose_range": {
#                     "x":   (-0.3, 0.3),   # narrowed from ±0.5
#                     "y":   (-0.3, 0.3),
#                     "yaw": (-3.14, 3.14),
#                 },
#                 "velocity_range": {
#                     "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5),
#                     "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5),
#                 },
#             },
#         )

#     @configclass
#     class _NoPenaltyRewards(RewardsCfg):
#         # Zero out the three penalty terms
#         collision      = RewTerm(func=mdp.body_collision,  weight=0.0,
#                                  params={"body_names": ["thigh", "calf"]})
#         feet_collision = RewTerm(func=mdp.body_collision,  weight=0.0,
#                                  params={"body_names": ["foot"]})
#         termination    = RewTerm(func=mdp.is_terminated,   weight=0.0)

#     rewards: _NoPenaltyRewards = _NoPenaltyRewards()
#     events:  _NoPenaltyEvents  = _NoPenaltyEvents()