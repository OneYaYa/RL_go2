# go2_rec_policy/observation_cfg.py
from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab.managers import (
    ObservationGroupCfg,
    ObservationTermCfg as ObsTerm,
    SceneEntityCfg,
)
from isaaclab.utils.noise import UniformNoiseCfg
import isaaclab.envs.mdp as mdp

from sensors.tactile import contact_forces_binary


@configclass
class ObservationsCfg:
    """Proprioception-only observation group for the recovery policy.

    Layout (52 dims):
      0:4   contact state (4 feet)
      4:7   base linear velocity (body frame)
      7:10  base angular velocity (body frame)
      10:13 projected gravity
      13:16 velocity commands (vx, vy, wz)
      16:28 joint positions (relative)
      28:40 joint velocities (relative)
      40:52 last action
    """

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        # Foot contact state (binary, {-1, +1})
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
        # Base linear velocity (body frame) — critical for velocity tracking.
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            scale=2.0,
            noise=UniformNoiseCfg(n_min=-0.1, n_max=0.1),
        )
        # Base angular velocity.
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            scale=0.25,
            noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
        )
        # Projected gravity — tells the policy which way is up.
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
        )
        # Velocity commands (vx, vy, wz) to track.
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        # Joint positions.
        dof_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            scale=1.0,
            noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
        )
        # Joint velocities.
        dof_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=0.05,
            noise=UniformNoiseCfg(n_min=-1.5, n_max=1.5),
        )
        # Last action.
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
