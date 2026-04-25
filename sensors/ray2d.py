from __future__ import annotations
import torch
import numpy as np
from isaaclab.utils.math import quat_apply_inverse, yaw_quat


def obj_relative_positions(env) -> torch.Tensor:
    """Replaces the obj_relpos loop. Returns (num_envs, num_objects * 3)."""
    robot_pos = env.scene["robot"].data.root_pos_w          # (N, 3)
    robot_quat = env.scene["robot"].data.root_quat_w        # (N, 4)
    yaw_q = yaw_quat(robot_quat)

    rel_positions = []
    for obj_idx in range(env.cfg.num_objects):
        obj_pos = env.scene[f"obstacle_{obj_idx}"].data.root_pos_w  # (N, 3)
        rel = obj_pos - robot_pos
        rel_body = quat_apply_inverse(yaw_q, rel)          # into yaw frame
        rel_positions.append(rel_body)

    return torch.cat(rel_positions, dim=-1)

# def ray2d_observation(env) -> torch.Tensor:
#     """Replaces ray2d_obs tensor construction."""
#     print(env.scene.sensors.keys())
#     ray_hits = env.scene.sensors["ray2d"].data.ray_hits_w  # (N, num_rays, 3)
#     ray_origin = env.scene.sensors["ray2d"].data.pos_w.unsqueeze(1)  # (N, 1, 3)
    
#     # Euclidean distance from sensor origin to each hit point
#     distances = torch.norm(ray_hits - ray_origin, dim=-1)  # (N, num_rays)
    
#     # Clip to configured range
#     distances = distances.clamp(
#         min=env.cfg.ray2d_range[0], 
#         max=env.cfg.ray2d_range[1]
#     )
#     return distances

# def ray2d_with_illusion(env) -> torch.Tensor:
#     """
#     Ports the illusion hallucination logic.
#     Rays beyond safe_tgt_dist get randomly pulled closer (fake nearby obstacles).
#     """
#     ray2d_obs = ray2d_observation(env)   # raw distances, (N, num_rays)

#     if env.cfg.sensors.ray2d.log2:
#         ray2d_obs = torch.log2(ray2d_obs.clamp(min=1e-6))

#     if env.cfg.sensors.ray2d.illusion and env.cfg.add_noise:
#         cmd_dist = torch.norm(
#             env.command_manager.get_command("base_velocity")[:, :2], dim=-1
#         ).unsqueeze(1)                          # (N, 1)
#         safe_tgt_dist = cmd_dist + 0.35

#         beyond_safe = ray2d_obs > safe_tgt_dist  # (N, num_rays) bool mask
#         hallucination = (
#             torch.zeros_like(ray2d_obs).uniform_(0, 1)
#             * (safe_tgt_dist - ray2d_obs)        # always negative beyond safe
#         )
#         ray2d_obs = ray2d_obs + beyond_safe.float() * hallucination

#     return ray2d_obs

# sensors/ray2d.py
import torch
import numpy as np
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply_inverse, yaw_quat


def circle_ray_query(
    ray_thetas: torch.Tensor,          # (num_rays,) ray angles in robot frame
    obj_rel_pos: torch.Tensor,         # (N, 2) object position in robot yaw frame
    radius: float,
    min_dist: float,
    max_dist: float,
) -> torch.Tensor:
    """
    Analytical ray-circle intersection. Ported from original Isaac Gym code.
    Returns per-ray distance to circle. Shape (N, num_rays).
    """
    N        = obj_rel_pos.shape[0]
    num_rays = ray_thetas.shape[0]
    device   = obj_rel_pos.device

    # Ray directions in robot frame (unit vectors)
    dx = torch.cos(ray_thetas)         # (num_rays,)
    dy = torch.sin(ray_thetas)         # (num_rays,)

    # Object position relative to ray origin (robot base)
    ox = obj_rel_pos[:, 0:1]           # (N, 1)
    oy = obj_rel_pos[:, 1:2]           # (N, 1)

    # Quadratic coefficients: ||(o + t*d)||^2 = r^2
    # a*t^2 + b*t + c = 0
    a = 1.0                            # dx^2 + dy^2 = 1 (unit ray)
    b = -2.0 * (ox * dx + oy * dy)    # (N, num_rays)
    c = ox**2 + oy**2 - radius**2     # (N, 1)

    discriminant = b**2 - 4.0 * a * c  # (N, num_rays)

    # No intersection where discriminant < 0
    no_hit = discriminant < 0
    discriminant_safe = discriminant.clamp(min=0.0)

    t = (b - torch.sqrt(discriminant_safe)) / (2.0 * a)   # (N, num_rays)

    # Only forward intersections (t > 0) are valid
    t = torch.where(no_hit | (t < min_dist), torch.full_like(t, max_dist), t)
    t = t.clamp(min=min_dist, max=max_dist)

    return t   # (N, num_rays)


def build_ray_thetas(ray2d_cfg) -> torch.Tensor:
    """Builds ray angle tensor from Ray2dSensorCfg."""
    thetas = torch.arange(
        ray2d_cfg.theta_start,
        ray2d_cfg.theta_end,
        ray2d_cfg.theta_step,
        dtype=torch.float32,
    )
    return thetas


def ray2d_observation(
    env: ManagerBasedRLEnv,
    # sensor_name: str = "ray2d_scanner",
    ray2d_cfg=None,
) -> torch.Tensor:
    """
    Combines RayCaster ground hits with analytical obstacle distances.
    Returns (N, num_rays) minimum distances.
    """
    if ray2d_cfg is None:
        ray2d_cfg = env.cfg.sensors.ray2d

    device = env.device

    # ── Ground distances from RayCaster ───────────────────────────────
    # ── Obstacle distances (analytical) ───────────────────────────────
    ray_thetas = build_ray_thetas(ray2d_cfg).to(device)        # (num_rays,)

    robot      = env.scene["robot"]
    robot_pos  = robot.data.root_pos_w                         # (N, 3)
    robot_quat = robot.data.root_quat_w                        # (N, 4)
    yaw_q      = yaw_quat(robot_quat)

    for obj_idx in range(env.cfg.num_objects):
        obj_pos  = env.scene[f"obstacle_{obj_idx}"].data.root_pos_w  # (N, 3)

        # Object position in robot yaw frame
        rel_world = obj_pos - robot_pos                              # (N, 3)
        rel_body  = quat_apply_inverse(yaw_q, rel_world)           # (N, 3)

        hit_dist  = circle_ray_query(
            ray_thetas=ray_thetas,
            obj_rel_pos=rel_body[:, :2],
            radius=0.15,
            min_dist=ray2d_cfg.min_dist,
            max_dist=ray2d_cfg.max_dist,
        )
    # ── Take minimum of ground and obstacle distances ──────────────────
    return hit_dist               # (N, num_rays)


def ray2d_with_illusion(
    env: ManagerBasedRLEnv,
    ray2d_cfg=None,
) -> torch.Tensor:
    if ray2d_cfg is None:
        ray2d_cfg = env.cfg.sensors.ray2d

    distances = ray2d_observation(env, ray2d_cfg)  # (N, num_rays)

    if ray2d_cfg.log2:
        distances = torch.log2(distances.clamp(min=1e-6))

    if ray2d_cfg.illusion and env.cfg.add_noise:
        cmd_dist = torch.norm(
            env.command_manager.get_command("pos_heading")[:, :2], dim=-1
        ).unsqueeze(1)
        safe_tgt_dist = cmd_dist + 0.35
        beyond_safe   = distances > safe_tgt_dist
        hallucination = (
            torch.zeros_like(distances).uniform_(0, 1)
            * (safe_tgt_dist - distances)
        )
        distances = distances + beyond_safe.float() * hallucination

    return distances