import numpy as np
import torch
from dataclasses import MISSING

from isaaclab.assets import ArticulationCfg, RigidObjectCfg
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv

def make_obstacle_cfg(idx: int) -> RigidObjectCfg:
    """One cfg per obstacle slot. Radius is a placeholder; overridden at reset."""
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}" + f"/Obstacle_{idx}",
        spawn=sim_utils.CylinderCfg(
            radius=0.15,                          # placeholder, randomized at reset
            height=0.4,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.1, 0.1)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.5, 0.0, 0.05)),
    )

def reset_obstacles_random_sparse(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    # Radius randomization
    radius_range: tuple[float, float] = (0.10, 0.30),
    # Spawn arena (relative to env origin)
    pos_x_range: tuple[float, float] = (-3.0, 3.0),
    pos_y_range: tuple[float, float] = (-3.0, 3.0),
    height: float = 0.05,
    # Sparsity enforcement
    min_obstacle_dist: float = 0.8,   # min dist between any two obstacles
    min_robot_dist:    float = 1.2,   # min dist from robot spawn (0,0)
    max_tries:         int   = 50,    # rejection sampling attempts per obstacle
    obstacle_names: list[str] = None,
) -> None:
    """
    Resets all obstacles with:
      - Random radius in [radius_range]
      - Random (x, y) position with rejection sampling for spatial sparsity
      - Minimum separation between obstacles
      - Minimum clearance from robot origin
    """
    if obstacle_names is None:
        obstacle_names = [f"obstacle_{i}" for i in range(8)]

    num_envs_reset = len(env_ids)
    device = env.device

    # Collect placed positions per env for sparsity checking
    # Shape: (num_envs_reset, num_placed, 2)
    placed_positions: list[torch.Tensor] = []  # grows as each obstacle is placed

    for name in obstacle_names:
        asset = env.scene[name]

        # ── Sample random radius ───────────────────────────────────────
        # radii = torch.empty(num_envs_reset, device=device).uniform_(*radius_range)

        # ── Sample positions with rejection sampling ───────────────────
        # Start with random candidates
        pos_x = torch.empty(num_envs_reset, device=device).uniform_(*pos_x_range)
        pos_y = torch.empty(num_envs_reset, device=device).uniform_(*pos_y_range)

        for _ in range(max_tries):
            candidate = torch.stack([pos_x, pos_y], dim=-1)  # (N, 2)

            # Check distance from robot origin (0, 0)
            robot_dist = torch.norm(candidate, dim=-1)        # (N,)
            too_close_robot = robot_dist < min_robot_dist     # (N,) bool

            # Check distance from all already-placed obstacles
            too_close_obs = torch.zeros(num_envs_reset, dtype=torch.bool, device=device)
            for prev_pos in placed_positions:
                # prev_pos: (num_envs_reset, 2)
                dist = torch.norm(candidate - prev_pos, dim=-1)
                too_close_obs |= dist < min_obstacle_dist

            needs_resample = too_close_robot | too_close_obs  # (N,) bool

            if not needs_resample.any():
                break

            # Resample only the failing envs
            n_resample = needs_resample.sum()
            pos_x[needs_resample] = torch.empty(n_resample, device=device).uniform_(*pos_x_range)
            pos_y[needs_resample] = torch.empty(n_resample, device=device).uniform_(*pos_y_range)

        # Record placed position for future sparsity checks
        placed_positions.append(torch.stack([pos_x, pos_y], dim=-1))

        # ── Build full root state ──────────────────────────────────────
        root_state = asset.data.default_root_state[env_ids].clone()
        # Position: x, y from sampling, z fixed
        root_state[:, 0] = pos_x + env.scene.env_origins[env_ids, 0]
        root_state[:, 1] = pos_y + env.scene.env_origins[env_ids, 1]
        root_state[:, 2] = height
        # Zero velocity
        root_state[:, 7:13] = 0.0

        asset.write_root_state_to_sim(root_state, env_ids=env_ids)


