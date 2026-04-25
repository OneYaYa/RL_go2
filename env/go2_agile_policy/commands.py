# commands.py
from __future__ import annotations

import torch
from dataclasses import MISSING

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import wrap_to_pi, quat_apply_inverse, yaw_quat, quat_apply
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import RED_ARROW_X_MARKER_CFG, CUBOID_MARKER_CFG
import isaaclab.sim as sim_utils


##############################################################################
# 1. CommandTerm — define the class FIRST
##############################################################################
class PosHeadingCommand(CommandTerm):

    def __init__(self, cfg: "PosHeadingCommandCfg", env):
        super().__init__(cfg, env)

        # ── World-frame targets (for rewards) ──────────────────────────
        self._position_targets = torch.zeros(env.num_envs, 3, device=env.device)
        self._heading_targets  = torch.zeros(env.num_envs, 1, device=env.device)

        # ── Robot-frame command (for policy obs via self._command) ──────
        # self._command is already allocated by CommandTerm base class
        # shape: (num_envs, self.cfg.command_dim) — must match num_commands=3

        # Alias world-frame targets onto env for reward functions
        env._position_targets = self._position_targets
        env._heading_targets  = self._heading_targets

        self._target_marker:  VisualizationMarkers | None = None
        self._heading_marker: VisualizationMarkers | None = None
        self._command = torch.zeros(env.num_envs, 3, device=env.device)



    @property
    def command(self) -> torch.Tensor:
        """Robot-frame [rel_x, rel_y, heading_error] — fed to policy."""
        return self._command                      # ← base class buffer, NOT _position_targets

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        self._sample_with_rejection(env_ids)
        # _sample_with_rejection calls _update_command at the end

    def _update_command(self) -> None:
        """
        Called every step. Recomputes robot-frame command from
        fixed world-frame _position_targets as the robot moves.
        Does NOT modify _position_targets.
        """
        asset     = self._env.scene["robot"]
        base_quat = asset.data.root_quat_w                              # (N, 4)

        # Vector from robot to target in world frame
        pos_diff  = self._position_targets - asset.data.root_pos_w     # (N, 3)

        # Rotate into robot yaw frame (matches original quat_rotate_inverse)
        rel_pos   = quat_apply_inverse(yaw_quat(base_quat), pos_diff)  # (N, 3)

        # Current heading
        forward   = quat_apply(
            base_quat,
            torch.tensor([1.0, 0.0, 0.0], device=self._env.device)
                  .expand(self.num_envs, -1),
        )
        heading   = torch.atan2(forward[:, 1], forward[:, 0])          # (N,)
        hdg_err   = wrap_to_pi(self._heading_targets[:, 0] - heading)  # (N,)

        # Write into self._command (robot-frame) — NOT _position_targets
        self._command[:, 0] = rel_pos[:, 0]      # rel x in robot frame
        self._command[:, 1] = rel_pos[:, 1]      # rel y in robot frame
        self._command[:, 2] = hdg_err            # heading error

    def _update_metrics(self) -> None:
        asset     = self._env.scene["robot"]
        robot_pos = asset.data.root_pos_w[:, :2]
        # Use world-frame _position_targets, not _command
        dist = torch.norm(self._position_targets[:, :2] - robot_pos, dim=-1)
        self.metrics["dist_to_goal"]    = dist
        self.metrics["pct_within_0.5m"] = (dist < 0.5).float() * 100.0
        self.metrics["pct_within_1.0m"] = (dist < 1.0).float() * 100.0

    def _sample_with_rejection(self, env_ids: torch.Tensor) -> None:
        cfg     = self.cfg
        device  = self._env.device
        origins = self._env.scene.env_origins
        asset   = self._env.scene["robot"]

        tbd_envs = env_ids.clone()

        while len(tbd_envs) > 0:
            n = len(tbd_envs)

            _pos1 = torch.empty(n, 1, device=device).uniform_(*cfg.pos_1_range)
            _pos2 = torch.empty(n, 1, device=device).uniform_(*cfg.pos_2_range)
            _hdg  = torch.empty(n, 1, device=device).uniform_(*cfg.heading_range)

            # ── Write world-frame position ─────────────────────────────
            if cfg.use_polar:
                self._position_targets[tbd_envs, 0:1] = (
                    origins[tbd_envs, 0:1] + _pos1 * torch.cos(_pos2)
                )
                self._position_targets[tbd_envs, 1:2] = (
                    origins[tbd_envs, 1:2] + _pos1 * torch.sin(_pos2)
                )
            else:
                self._position_targets[tbd_envs, 0:1] = origins[tbd_envs, 0:1] + _pos1
                self._position_targets[tbd_envs, 1:2] = origins[tbd_envs, 1:2] + _pos2

            # Z stays as terrain height + 0.5 (for visualization only)
            self._position_targets[tbd_envs, 2] = origins[tbd_envs, 2] + 0.5

            # ── Compute heading target (world frame, stored separately) ─
            root_pos = asset.data.root_pos_w[tbd_envs]
            pos_diff = self._position_targets[tbd_envs] - root_pos    # (n, 3)

            self._heading_targets[tbd_envs] = wrap_to_pi(
                _hdg + torch.atan2(pos_diff[:, 1:2], pos_diff[:, 0:1])
            )

            # ── Reject targets inside obstacles ───────────────────────
            in_obst = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
            for obj_idx in range(cfg.num_objects):
                obj_pos   = self._env.scene[f"obstacle_{obj_idx}"].data.root_pos_w
                dist      = torch.norm(
                    self._position_targets[tbd_envs, 0:2] - obj_pos[tbd_envs, 0:2],
                    dim=-1,
                )
                obj_type  = obj_idx % max(len(cfg.object_size_list), 1)
                radius    = cfg.object_size_list[obj_type] if cfg.object_size_list else 0.4
                threshold = radius * cfg.obstacle_radius_scale + cfg.obstacle_radius_margin
                in_obst[tbd_envs] = torch.logical_or(in_obst[tbd_envs], dist < threshold)

            tbd_envs = in_obst.nonzero(as_tuple=False).flatten()

        # Final: update robot-frame command buffer from new world-frame targets
        self._update_command()


##############################################################################
# 2. Config — defined AFTER the class so class_type can reference it directly
##############################################################################

@configclass
class PosHeadingCommandCfg(CommandTermCfg):
    """Config for obstacle-aware position + heading command."""

    # ── Point directly at the class defined above ──────────────────────
    class_type: type = PosHeadingCommand          # ← no None, no post-assignment

    pos_1_range:   tuple = (1.5, 7.5)
    pos_2_range:   tuple = (-2.0, 2.0)
    heading_range: tuple = (-0.3, 0.3)
    use_polar:     bool  = False

    num_objects:            int   = 8
    object_size_list:       list  = MISSING
    obstacle_radius_margin: float = 0.31415926
    obstacle_radius_scale:  float = 1.1

    resampling_time_range:  tuple = (1e9, 1e9)
    debug_vis:              bool  = True

    marker_scale: tuple = (0.2, 0.2, 0.5)
    arrow_scale:  tuple = (0.8, 0.05, 0.05)


##############################################################################
# 3. CommandsCfg
##############################################################################

@configclass
class CommandsCfg:
    pos_heading: PosHeadingCommandCfg = PosHeadingCommandCfg(
        resampling_time_range=(1e9, 1e9),
        pos_1_range=(1.5, 7.5),
        pos_2_range=(-2.0, 2.0),
        heading_range=(-0.3, 0.3),
        use_polar=False,
        num_objects=8,
        object_size_list=[0.4],
        debug_vis=True,
    )