# go2_rec_policy/commands.py
"""Velocity commands for the recovery policy.

ABS recovery tracks (vx, vy, wz) body-frame velocity targets.  Unlike the
agile policy we do *not* resample commands during the episode — the
episode is short (a few seconds) and the whole point is to regain
tracking of a single, fixed command from an arbitrary initial state.
"""
from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab.envs.mdp import UniformVelocityCommandCfg


@configclass
class CommandsCfg:
    # Body-frame (vx, vy, wz) target.  Name "base_velocity" matches the
    # observation / reward hooks below.
    base_velocity: UniformVelocityCommandCfg = UniformVelocityCommandCfg(
        asset_name="robot",
        # Keep the command fixed for the whole recovery window.
        resampling_time_range=(1e9, 1e9),
        rel_standing_envs=0.25,        # 25% of envs get the "stand still" command
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.8, 1.2),
            lin_vel_y=(-0.6, 0.6),
            ang_vel_z=(-1.0, 1.0),
            heading=(-3.14, 3.14),
        ),
    )
