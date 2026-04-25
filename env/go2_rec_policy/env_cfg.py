# go2_rec_policy/env_cfg.py
from __future__ import annotations

from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.envs.mdp.actions import JointPositionActionCfg
from isaaclab.envs import ManagerBasedRLEnvCfg

from .scene_cfg import Go2RecoverySceneCfg
from .observation_cfg import ObservationsCfg
from .rewards import RewardsCfg
from .terminations import TerminationsCfg
from .events import EventCfg
from .sensor_cfg import SensorsCfg
from .commands import CommandsCfg


@configclass
class ActionsCfg:
    """Joint-position actions for the 12-DoF Go2 quadruped."""

    joint_pos = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        scale=0.25,
        use_default_offset=True,
    )


@configclass
class Go2RecoveryEnvCfg(ManagerBasedRLEnvCfg):
    """Recovery policy environment: no obstacles, no perception sensors,
    velocity-tracking commands, and aggressive pose randomization so the
    policy learns to stand back up from any initial state."""

    # ── Sub-configs ────────────────────────────────────────────────────
    scene:        Go2RecoverySceneCfg = Go2RecoverySceneCfg(num_envs=4096, env_spacing=3.0)
    observations: ObservationsCfg     = ObservationsCfg()
    commands:     CommandsCfg         = CommandsCfg()
    rewards:      RewardsCfg          = RewardsCfg()
    terminations: TerminationsCfg     = TerminationsCfg()
    events:       EventCfg            = EventCfg()
    actions:      ActionsCfg          = ActionsCfg()

    # Dummy sensors cfg to keep the agile-style plumbing in place.
    sensors: SensorsCfg = SensorsCfg()

    # ── Episode / control ──────────────────────────────────────────────
    # Shorter than agile: a full recovery happens in a few seconds.
    episode_length_s: float = 5.0
    decimation:       int   = 4
    action_scale:     float = 0.25

    # ── Observation space size ─────────────────────────────────────────
    # 4 contact + 3 lin_vel + 3 ang_vel + 3 gravity + 3 cmd + 12 dof_pos
    # + 12 dof_vel + 12 last_action = 52
    num_observations: int = 52

    # ── Normalization ──────────────────────────────────────────────────
    clip_observations: float = 100.0
    clip_actions:      float = 100.0

    # ── Noise master switch ────────────────────────────────────────────
    add_noise:   bool  = False
    noise_level: float = 1.0

    def __post_init__(self):
        self.sim.dt              = 0.005             # 200 Hz physics
        self.sim.render_interval = self.decimation   # 50 Hz policy
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
        )
        self.sim.light = sim_utils.DomeLightCfg(
            intensity=2000.0,
            color=(1.0, 1.0, 1.0),
        )
        # Default number of envs at runtime (overridden by CLI).
        self.scene.num_envs = 1280
