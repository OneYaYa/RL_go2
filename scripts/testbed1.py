"""IsaacLab testbed (agile + recovery + RA) using /root/RL_finalproject/log checkpoints."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.append("./")

from isaaclab.app import AppLauncher

LOG_ROOT = Path("/home/rex/Desktop/github/finalprojectRL/RL_finalproject/log")
AGILE_ROOT = LOG_ROOT / "agile_policy"
RECOVERY_ROOT = LOG_ROOT / "recovery_policy"
RA_ROOT = LOG_ROOT / "RA"
TWIST_EPS = 0.05


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IsaacLab testbed with RA train/test.")
    parser.add_argument("--task", type=str, default="go1_pos_rough")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument(
        "--num_steps",
        type=int,
        default=None,
        help="Rollout length. If omitted, uses 300 * env.max_episode_length like testbed1.py.",
    )
    parser.add_argument("--load_run", type=str, default=None, help="Agile run hint or direct checkpoint path.")
    parser.add_argument("--agile_checkpoint", type=str, default=None)
    parser.add_argument("--recovery_checkpoint", type=str, default=None)
    parser.add_argument("--trainRA", action="store_true")
    parser.add_argument("--testRA", action="store_true")
    parser.add_argument("--ra_name", type=str, default="04_20_11-37-46_model_4000_ra.pt")
    parser.add_argument("--ra_threshold", type=float, default=-TWIST_EPS, help="Switch to recovery when RA value is above this threshold.")
    parser.add_argument("--ra_batch_size", type=int, default=20, help="RA update chunk length over the replay queue.")
    parser.add_argument("--ra_update_interval", type=int, default=20)
    parser.add_argument("--disable_noise", action="store_true")
    parser.add_argument(
        "--recovery_hold_steps",
        type=int,
        default=5,
        help="After RA switches an env to recovery, keep recovery for at least this many env steps (0 = only while RA says so).",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


PARSER = _build_parser()
ARGS = PARSER.parse_args()
ARGS.headless = False
APP_LAUNCHER = AppLauncher(ARGS)
SIMULATION_APP = APP_LAUNCHER.app

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from env.go2_agile_policy.env_cfg import Go2PosRoughEnvCfg
from env.go2_agile_policy.ppo_cfg import Go2PosRoughPPORunnerCfg
from sensors.ray2d import ray2d_with_illusion


@dataclass
class EvalStats:
    done: int = 0
    reached: int = 0
    failed: int = 0
    timed_out: int = 0
    ra_used_steps: int = 0
    collisions: int = 0
    episodic_recovery: int = 0
    episodic_recovery_success: int = 0
    episodic_recovery_fail: int = 0
    collision_when_ra_on: int = 0
    collision_when_ra_off: int = 0


def _print_metrics_block(
    episode_idx: int,
    stats: EvalStats,
    episode_max_velo_dist: float,
    episode_max_velo_dist_collision: float,
    episode_max_velo_dist_reach: float,
    episode_max_velo_dist_timeout: float,
    avg_total_velocity: float,
    avg_collision_velocity: float,
    avg_reach_velocity: float,
    avg_timeout_velocity: float,
    avg_recovery_velocity: float,
) -> None:
    print(f"========= Episode {episode_idx} =========")
    print(f"Total Episode:                         {episode_idx}")
    print(f"Total Collision:                       {stats.collisions}")
    print(f"Total Reach:                           {stats.reached}")
    print(f"Total Timeout:                         {stats.timed_out}")
    print(f"Total Collision + Reach + Timeout:      {stats.collisions + stats.reached + stats.timed_out}")
    print(f"Total Done:                            {stats.done}")
    print(f"Collision Rate:                        {stats.collisions / (stats.done + 1e-8):.2%}")
    print(f"Reach Rate:                            {stats.reached / (stats.done + 1e-8):.2%}")
    print(f"Timeout Rate:                          {stats.timed_out / (stats.done + 1e-8):.2%}")
    print(f"Average Total Velocity:                {avg_total_velocity:.2f}")
    print(f"Average Collision Velocity:            {avg_collision_velocity:.2f}")
    print(f"Average Reach Velocity:                {avg_reach_velocity:.2f}")
    print(f"Average Timeout Velocity:              {avg_timeout_velocity:.2f}")
    print(f"Average Recovery Velocity:             {avg_recovery_velocity:.2f}")
    print(f"Average in-trajectory Max Velocity:    {episode_max_velo_dist / (stats.done + 1e-8):.2f}")
    print(f"Average in-trajectory Max Velocity Collision: {episode_max_velo_dist_collision / (stats.collisions + 1e-8):.2f}")
    print(f"Average in-trajectory Max Velocity Reach: {episode_max_velo_dist_reach / (stats.reached + 1e-8):.2f}")
    print(f"Average in-trajectory Max Velocity Timeout: {episode_max_velo_dist_timeout / (stats.timed_out + 1e-8):.2f}")
    print(f"Episode that activated recovery:         {stats.episodic_recovery}")
    print(f"Episode that activated recovery - safe:  {stats.episodic_recovery_success}")
    print(f"Episode that activated recovery - collision:    {stats.episodic_recovery_fail}")
    print(f"Episode that did not activate recovery - collision: {stats.collisions - stats.episodic_recovery_fail}")
    print(f"Episodic recovery activation rate:          {stats.episodic_recovery / (stats.done + 1e-8):.2%}")
    print(f"Episodic recovery success rate (end up safe): {stats.episodic_recovery_success / (stats.episodic_recovery + 1e-8):.2%}")
    print(f"RA activation rate for collision moments: {stats.collision_when_ra_on / (stats.collisions + 1e-8):.2%}")
    print(f"RA deactivation rate for collision moments: {stats.collision_when_ra_off / (stats.collisions + 1e-8):.2%}")


def _advance_display_stats(display_stats: EvalStats, actual_stats: EvalStats) -> None:
    """Advance printed episode totals by one, even when many vector envs finish together."""
    if display_stats.done >= actual_stats.done:
        return

    display_stats.done += 1
    display_stats.failed = min(actual_stats.failed, display_stats.failed + 1)

    if display_stats.collisions < actual_stats.collisions:
        display_stats.collisions += 1
    elif display_stats.reached < actual_stats.reached:
        display_stats.reached += 1
    elif display_stats.timed_out < actual_stats.timed_out:
        display_stats.timed_out += 1

    if display_stats.episodic_recovery < actual_stats.episodic_recovery:
        display_stats.episodic_recovery += 1
    if display_stats.episodic_recovery_success < actual_stats.episodic_recovery_success:
        display_stats.episodic_recovery_success += 1
    if display_stats.episodic_recovery_fail < actual_stats.episodic_recovery_fail:
        display_stats.episodic_recovery_fail += 1
    if display_stats.collision_when_ra_on < actual_stats.collision_when_ra_on:
        display_stats.collision_when_ra_on += 1
    if display_stats.collision_when_ra_off < actual_stats.collision_when_ra_off:
        display_stats.collision_when_ra_off += 1
    display_stats.ra_used_steps = actual_stats.ra_used_steps


def _get_obstacle_positions(raw_env: ManagerBasedRLEnv) -> torch.Tensor | None:
    obstacle_positions = []
    num_objects = int(getattr(raw_env.cfg, "num_objects", 8))
    for obj_idx in range(num_objects):
        obj_name = f"obstacle_{obj_idx}"
        try:
            obstacle_positions.append(raw_env.scene[obj_name].data.root_pos_w[:, :2])
        except KeyError:
            continue
    if not obstacle_positions:
        return None
    return torch.stack(obstacle_positions, dim=-1)


def _find_contact_body_ids(contact_sensor, patterns: list[str]) -> list[int]:
    body_ids: list[int] = []
    for pattern in patterns:
        ids, _ = contact_sensor.find_bodies([pattern])
        body_ids.extend(int(body_id.item() if hasattr(body_id, "item") else body_id) for body_id in ids)
    return sorted(set(body_ids))


def _testbed1_collision(
    raw_env: ManagerBasedRLEnv,
    pre_root_pos: torch.Tensor,
    pre_obstacle_pos: torch.Tensor | None,
    pre_base_lin_vel: torch.Tensor,
    fallback_collision: torch.Tensor,
) -> torch.Tensor:
    """Match testbed1.py collision metrics using IsaacLab contact sensors."""
    if "contact_forces" not in raw_env.scene.sensors:
        return fallback_collision

    contact_sensor = raw_env.scene.sensors["contact_forces"]
    forces = contact_sensor.data.net_forces_w
    if forces.numel() == 0:
        return fallback_collision

    term_body_ids = _find_contact_body_ids(
        contact_sensor,
        ["base", "FL_thigh", "FL_calf", "FR_thigh", "FR_calf"],
    )
    if term_body_ids:
        collision = torch.any(torch.norm(forces[:, term_body_ids, :], dim=-1) > 1.0, dim=1)
    else:
        collision = fallback_collision.clone()

    foot_body_ids = _find_contact_body_ids(contact_sensor, [".*_foot"])
    if foot_body_ids:
        front_foot_ids = foot_body_ids[:2]
        foot_forces = forces[:, front_foot_ids, :]
        hor_footforce = foot_forces[:, :, 0:2].norm(dim=-1)
        ver_footforce = torch.abs(foot_forces[:, :, 2])
        foot_hor_col = torch.any(hor_footforce > 2.0 * ver_footforce + 10.0, dim=-1)
        collision = torch.logical_or(collision, foot_hor_col)

    if pre_obstacle_pos is not None:
        obj_rel_pos = pre_obstacle_pos - pre_root_pos[:, :2].unsqueeze(-1)
        min_obj_dist = obj_rel_pos.norm(dim=1)
        near_obj = torch.any(min_obj_dist < 0.95, dim=-1)
    else:
        near_obj = torch.ones(raw_env.num_envs, device=raw_env.device, dtype=torch.bool)

    near_obj = torch.logical_and(near_obj, pre_base_lin_vel[:, :2].norm(dim=-1) > 0.5)

    base_ids = _find_contact_body_ids(contact_sensor, ["base"])
    if base_ids:
        base_contact = torch.norm(forces[:, base_ids[0], :], dim=-1) > 1.0
        near_obj = torch.logical_or(near_obj, base_contact)

    return torch.logical_and(collision, near_obj)


class RANet(nn.Module):
    def __init__(self, in_dim: int = 19, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LegacyMLPPolicy(nn.Module):
    def __init__(self, actor_state_dict: dict[str, torch.Tensor]):
        super().__init__()
        mlp_keys = [(k, v) for k, v in actor_state_dict.items() if k.startswith("mlp.")]
        if not mlp_keys:
            raise ValueError("No 'mlp.*' keys in checkpoint actor_state_dict.")
        layer_ids = sorted({int(k.split(".")[1]) for k, _ in mlp_keys if len(k.split(".")) >= 3})
        modules: list[nn.Module] = []
        for idx in layer_ids:
            w_key = f"mlp.{idx}.weight"
            b_key = f"mlp.{idx}.bias"
            if w_key not in actor_state_dict or b_key not in actor_state_dict:
                continue
            out_dim, in_dim = actor_state_dict[w_key].shape
            layer = nn.Linear(in_dim, out_dim)
            layer.weight.data.copy_(actor_state_dict[w_key])
            layer.bias.data.copy_(actor_state_dict[b_key])
            modules.append(layer)
            if idx != layer_ids[-1]:
                modules.append(nn.ELU())
        self.model = nn.Sequential(*modules)
        self.input_dim = next(m for m in self.model if isinstance(m, nn.Linear)).in_features

    def forward(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        x = _fit_obs_dim(obs_tensor, self.input_dim)
        return self.model(x)


def _fit_obs_dim(obs_tensor: torch.Tensor, target_dim: int) -> torch.Tensor:
    cur_dim = obs_tensor.shape[-1]
    if cur_dim == target_dim:
        return obs_tensor
    if cur_dim > target_dim:
        return obs_tensor[:, :target_dim]
    pad = torch.zeros(obs_tensor.shape[0], target_dim - cur_dim, device=obs_tensor.device, dtype=obs_tensor.dtype)
    return torch.cat([obs_tensor, pad], dim=-1)


def _all_checkpoints(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted((p for p in root.rglob("*.pt") if p.is_file()), key=lambda p: p.stat().st_mtime, reverse=True)


def _model_step(path: Path) -> int:
    m = re.search(r"model_(\d+)\.pt$", path.name)
    return int(m.group(1)) if m else -1


def _resolve_agile_checkpoint(args) -> Path:
    if args.agile_checkpoint:
        p = Path(args.agile_checkpoint)
        if not p.is_file():
            raise FileNotFoundError(f"Agile checkpoint not found: {p}")
        return p
    if args.load_run:
        p = Path(args.load_run)
        if p.is_file():
            return p
    candidates = _all_checkpoints(AGILE_ROOT)
    if args.load_run:
        candidates = [p for p in candidates if args.load_run in str(p)]
    if not candidates:
        candidates = [p for p in _all_checkpoints(LOG_ROOT) if "recovery" not in str(p).lower() and "/RA/" not in str(p)]
    if not candidates:
        raise FileNotFoundError(f"No agile checkpoint found under {LOG_ROOT}")
    candidates.sort(key=lambda p: (_model_step(p), p.stat().st_mtime), reverse=True)
    return candidates[0]


def _resolve_recovery_checkpoint(args) -> Path | None:
    if args.recovery_checkpoint:
        p = Path(args.recovery_checkpoint)
        if not p.is_file():
            raise FileNotFoundError(f"Recovery checkpoint not found: {p}")
        return p
    candidates = _all_checkpoints(RECOVERY_ROOT)
    if not candidates:
        candidates = [p for p in _all_checkpoints(LOG_ROOT) if "recovery" in str(p).lower()]
    return candidates[0] if candidates else None


def _extract_obs_tensor(obs_obj) -> torch.Tensor:
    if isinstance(obs_obj, torch.Tensor):
        return obs_obj
    for key in ("policy", "obs", "observation"):
        if hasattr(obs_obj, "__getitem__"):
            try:
                val = obs_obj[key]
                if isinstance(val, torch.Tensor):
                    return val
            except Exception:
                pass
    raise ValueError(f"Unsupported observation type: {type(obs_obj)}")


def _build_ra_obs(raw_env: ManagerBasedRLEnv) -> torch.Tensor:
    lin = mdp.base_lin_vel(raw_env)
    ang = mdp.base_ang_vel(raw_env)
    cmd = raw_env.command_manager.get_command("pos_heading")[:, :2]
    ray = ray2d_with_illusion(raw_env)
    return torch.cat([lin, ang, cmd, ray], dim=-1)

def _build_rec_obs(raw_env: ManagerBasedRLEnv) -> torch.Tensor:
    """Match ``env/go2_rec_policy/observation_cfg.py`` PolicyCfg (52-dim, no rays).

    Scales match ObsTerm ``scale=`` there. In the agile testbed env there is no
    ``base_velocity`` command; we feed ``pos_heading`` (3,) in its place so the
    layout matches training dimensions.
    """
    foot_cfg = SceneEntityCfg(
        "contact_forces",
        body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
    )
    cs = raw_env.scene.sensors[foot_cfg.name]
    body_ids, _ = cs.find_bodies(foot_cfg.body_names)
    foot_forces = cs.data.net_forces_w[:, body_ids, :]
    foot_mag = torch.norm(foot_forces, dim=-1)
    contact = (foot_mag > 1.0).float() * 2.0 - 1.0

    lin = mdp.base_lin_vel(raw_env) * 2.0
    ang = mdp.base_ang_vel(raw_env) * 0.25
    grav = mdp.projected_gravity(raw_env)
    try:
        cmd = raw_env.command_manager.get_command("base_velocity")
    except (KeyError, AttributeError, RuntimeError):
        cmd = raw_env.command_manager.get_command("pos_heading")
    dof_pos = mdp.joint_pos_rel(raw_env)
    dof_vel = mdp.joint_vel_rel(raw_env) * 0.05
    prev_act = mdp.last_action(raw_env)

    return torch.cat([contact, lin, ang, grav, cmd, dof_pos, dof_vel, prev_act], dim=-1)


def _build_env(args) -> RslRlVecEnvWrapper:
    env_cfg = Go2PosRoughEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    if args.disable_noise:
        env_cfg.observations.policy.enable_corruption = False
    return RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=env_cfg))


def _build_agile_policy(env: RslRlVecEnvWrapper, ckpt: Path, device: str):
    runner_cfg = Go2PosRoughPPORunnerCfg()
    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=None, device=device)
    runner.load(str(ckpt))
    return runner.get_inference_policy(device=device)


def _build_recovery_policy(env: RslRlVecEnvWrapper, ckpt: Path, device: str):
    runner_cfg = Go2PosRoughPPORunnerCfg()
    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=None, device=device)
    try:
        runner.load(str(ckpt))
        infer = runner.get_inference_policy(device=device)

        def _policy(obs_obj, _obs_tensor):
            return infer(obs_obj)

        return _policy, "rsl_rl:model_state_dict"
    except Exception:
        pass

    payload = torch.load(ckpt, map_location=device, weights_only=False)
    actor_state = payload.get("actor_state_dict") if isinstance(payload, dict) else None
    if actor_state is None:
        raise ValueError(f"Unsupported recovery checkpoint format: {ckpt}")
    legacy = LegacyMLPPolicy(actor_state).to(device).eval()

    def _policy(_obs_obj, obs_tensor):
        with torch.no_grad():
            return legacy(obs_tensor)

    return _policy, f"legacy_actor_state_dict:in={legacy.input_dim}"


def _resolve_ra_path(ra_name: str) -> Path:
    p = Path(ra_name)
    if p.is_absolute():
        return p
    if not ra_name.endswith(".pt"):
        ra_name = f"{ra_name}.pt"
    return RA_ROOT / ra_name


def _load_ra(ra_name: str, device: str) -> nn.Module:
    path = _resolve_ra_path(ra_name)
    if not path.is_file():
        candidates = _all_checkpoints(RA_ROOT)
        if not candidates:
            raise FileNotFoundError(f"RA checkpoint not found: {path}")
        path = candidates[0]
    payload = torch.load(path, map_location=device, weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        ra = RANet(in_dim=int(payload.get("in_dim", 19)))
        ra.load_state_dict(payload["state_dict"])
    elif isinstance(payload, nn.Module):
        ra = payload
    else:
        raise ValueError(f"Unsupported RA checkpoint format: {path}")
    ra.to(device).eval()
    print(f"[testbed2] Loaded RA checkpoint: {path}")
    return ra


def _save_ra(ra: nn.Module, ra_name: str) -> Path:
    RA_ROOT.mkdir(parents=True, exist_ok=True)
    path = _resolve_ra_path(ra_name)
    if not path.is_absolute():
        path = RA_ROOT / path.name
    torch.save({"state_dict": ra.state_dict(), "in_dim": 19}, path)
    return path


def _run(args):
    if args.task not in {"go1_pos_rough", "go2_pos_rough"}:
        raise ValueError("Supported tasks: go1_pos_rough, go2_pos_rough.")
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    RA_ROOT.mkdir(parents=True, exist_ok=True)

    agile_ckpt = _resolve_agile_checkpoint(args)
    print(f"[testbed2] Agile policy checkpoint: {agile_ckpt}")

    env = _build_env(args)
    raw_env = env.unwrapped
    agile_policy = _build_agile_policy(env, agile_ckpt, args.device)

    recovery_policy = None
    if args.testRA:
        rec_ckpt = _resolve_recovery_checkpoint(args)
        if rec_ckpt is None:
            raise FileNotFoundError("No recovery checkpoint found /home/rex/Desktop/github/finalprojectRL/RL_finalproject/log.")
        recovery_policy, rec_info = _build_recovery_policy(env, rec_ckpt, args.device)
        print(f"[testbed2] Recovery policy checkpoint: {rec_ckpt} ({rec_info})")

    ra_net = RANet().to(args.device)
    ra_optim = torch.optim.Adam(ra_net.parameters(), lr=2e-3)
    replay_obs: list[torch.Tensor] = []
    replay_target: list[torch.Tensor] = []
    if args.testRA:
        ra_net = _load_ra(args.ra_name, args.device)

    stats = EvalStats()
    display_stats = EvalStats()
    obs, _ = env.reset()
    step_dt = float(getattr(raw_env, "step_dt", 0.02))
    robot = raw_env.scene["robot"]
    last_root_pos = robot.data.root_pos_w[:, :2].clone()
    episode_travel_dist = torch.zeros(args.num_envs, device=args.device)
    episode_time = torch.zeros(args.num_envs, device=args.device)
    episode_max_velo = torch.zeros(args.num_envs, device=args.device)
    episode_recovery_logging = torch.zeros(args.num_envs, dtype=torch.bool, device=args.device)
    recovery_hold_left = torch.zeros(args.num_envs, dtype=torch.int32, device=args.device)

    total_recovery_dist = 0.0
    total_recovery_timesteps = 0
    episode_max_velo_dist = 0.0
    episode_max_velo_dist_collision = 0.0
    episode_max_velo_dist_reach = 0.0
    episode_max_velo_dist_timeout = 0.0
    total_travel_dist = 0.0
    total_time = 0.0
    total_collision_dist = 0.0
    total_collision_time = 0.0
    total_reach_dist = 0.0
    total_reach_time = 0.0
    total_timeout_dist = 0.0
    total_timeout_time = 0.0

    # RA training queues follow the original testbed1.py structure.
    queue_len = 1001
    batch_size = max(1, min(int(args.ra_batch_size), queue_len - 1))
    hindsight = 10
    gamma = 0.999999
    best_metric = 999.0
    s_queue = torch.zeros((queue_len, args.num_envs, 19), device=args.device, dtype=torch.float)
    g_queue = torch.zeros((queue_len, args.num_envs), device=args.device, dtype=torch.float)
    g_hs_queue = g_queue.clone()
    g_hs_span = torch.zeros((2, args.num_envs), device=args.device, dtype=torch.int)
    l_queue = torch.zeros((queue_len, args.num_envs), device=args.device, dtype=torch.float)
    done_queue = torch.zeros((queue_len, args.num_envs), device=args.device, dtype=torch.bool)
    standard_raobs_init = torch.tensor(
        [[0, 0, 0, 0, 0, 0, 6.0, 0] + [0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 1.0, 0.0, 0.0, 0.0]],
        device=args.device,
        dtype=torch.float,
    )
    standard_raobs_die = torch.tensor([[5.0, 0, 0, 0, 0, 0, 6.0, 0] + [-2.5] * 11], device=args.device, dtype=torch.float)
    standard_raobs_turn = torch.tensor(
        [[0, 0, 0, 0, 0, 2.0, 0.5, 5.8] + [2.0] * 6 + [0.0] * 5],
        device=args.device,
        dtype=torch.float,
    )
    metrics_print_idx = 0

    total_steps = args.num_steps
    if total_steps is None:
        total_steps = 300 * int(raw_env.max_episode_length)
    print(f"[testbed2] train RA: {args.trainRA}; test RA: {args.testRA}; total steps: {total_steps}")
    _print_metrics_block(
        episode_idx=metrics_print_idx,
        stats=display_stats,
        episode_max_velo_dist=episode_max_velo_dist,
        episode_max_velo_dist_collision=episode_max_velo_dist_collision,
        episode_max_velo_dist_reach=episode_max_velo_dist_reach,
        episode_max_velo_dist_timeout=episode_max_velo_dist_timeout,
        avg_total_velocity=0.0,
        avg_collision_velocity=0.0,
        avg_reach_velocity=0.0,
        avg_timeout_velocity=0.0,
        avg_recovery_velocity=0.0,
    )

    with torch.inference_mode(False):
        # Main loop mirrors testbed1.py: act -> step -> calculate figures -> train/test/evaluate.
        for i in range(total_steps):
            obs_tensor = _extract_obs_tensor(obs)
            pre_root_pos = robot.data.root_pos_w[:, :2].clone()
            pre_base_lin_vel = robot.data.root_lin_vel_b[:, :2].clone()
            pre_obstacle_pos = _get_obstacle_positions(raw_env)
            if pre_obstacle_pos is not None:
                pre_obstacle_pos = pre_obstacle_pos.clone()
            if hasattr(raw_env, "_position_targets"):
                pre_target_pos = raw_env._position_targets[:, :2].clone()
            else:
                pre_target_pos = pre_root_pos + raw_env.command_manager.get_command("pos_heading")[:, :2]

            # 1) Action selection. Agile policy always runs first.
            with torch.no_grad():
                actions = agile_policy(obs).clone()
            use_recovery = torch.zeros(args.num_envs, dtype=torch.bool, device=args.device)
            ra_wants_recovery = torch.zeros(args.num_envs, dtype=torch.bool, device=args.device)

            # In testRA mode, RA can switch selected environments to recovery action before stepping.
            if args.testRA and recovery_policy is not None:
                with torch.no_grad():
                    ra_obs = _build_ra_obs(raw_env)
                    ra_score = ra_net(ra_obs).squeeze(-1)
                    ra_wants_recovery = ra_score > args.ra_threshold
                h = int(args.recovery_hold_steps)
                if h > 0:
                    # Stay in recovery until hold counter reaches 0 (armed when RA fires).
                    use_recovery = torch.logical_or(recovery_hold_left > 0, ra_wants_recovery)
                else:
                    use_recovery = ra_wants_recovery

                # print(f"The usage of recovery is {ra_score > args.ra_threshold}")
                
                # if use_recovery.any():
                #     with torch.no_grad():
                #         rec_tensor = _build_rec_obs(raw_env)
                #         if isinstance(obs, dict):
                #             rec_obs = dict(obs)
                #             rec_obs["policy"] = rec_tensor
                #         else:
                #             rec_obs = {"policy": rec_tensor}
                #         rec_actions = recovery_policy(rec_obs, rec_tensor)
                #     actions[use_recovery] = rec_actions[use_recovery]
                #     stats.ra_used_steps += int(use_recovery.sum().item())
                #     episode_recovery_logging |= use_recovery

            # 2) Step the IsaacLab environment.
            obs_next, rewards, dones, extras = env.step(actions)

            # 3) Calculate figures first: termination type, distances, velocity and RA metrics.
            dones_bool = dones.bool()
            dist = torch.norm(pre_target_pos - pre_root_pos, dim=-1)
            raw_timed_out = getattr(raw_env, "reset_time_outs", torch.zeros_like(dones_bool)).bool()
            terminated = getattr(raw_env, "reset_terminated", torch.zeros_like(dones_bool)).bool()
            contact_collision = _testbed1_collision(raw_env, pre_root_pos, pre_obstacle_pos, pre_base_lin_vel, terminated)
            collision = torch.logical_and(dones_bool, contact_collision)
            reached = torch.logical_and(dones_bool, torch.logical_and(dist < 0.65, ~collision))
            timed_out = torch.logical_and(dones_bool, torch.logical_and(raw_timed_out, torch.logical_not(reached | collision)))
            timed_out = torch.logical_or(
                timed_out,
                torch.logical_and(dones_bool, torch.logical_and(dist >= 0.65, torch.logical_not(collision))),
            )
            failed = collision

            root_pos = robot.data.root_pos_w[:, :2]
            step_dist = torch.norm(root_pos - pre_root_pos, dim=-1)
            not_in_goal = torch.logical_and(dist >= 0.65, ~dones_bool)
            episode_travel_dist += step_dist * not_in_goal.float()
            episode_time += step_dt * not_in_goal.float()
            valid_step_vel = step_dist * not_in_goal.float() / max(step_dt, 1e-8)
            episode_max_velo = torch.maximum(episode_max_velo, valid_step_vel)
            if use_recovery.any():
                total_recovery_dist += float(step_dist[use_recovery].sum().item())
                total_recovery_timesteps += int(use_recovery.sum().item())
            last_root_pos = root_pos.clone()

            done_ids = torch.where(dones)[0]
            if done_ids.numel() > 0:
                stats.done += int(done_ids.numel())
                stats.reached += int(reached.sum().item())
                stats.failed += int(failed.sum().item())
                stats.timed_out += int(timed_out.sum().item())
                stats.collisions += int(collision.sum().item())

                stats.episodic_recovery += int(episode_recovery_logging[done_ids].sum().item())
                stats.episodic_recovery_success += int(
                    torch.logical_and(episode_recovery_logging[done_ids], ~collision[done_ids]).sum().item()
                )
                stats.episodic_recovery_fail += int(
                    torch.logical_and(episode_recovery_logging[done_ids], collision[done_ids]).sum().item()
                )
                stats.collision_when_ra_on += int(torch.logical_and(collision[done_ids], use_recovery[done_ids]).sum().item())
                stats.collision_when_ra_off += int(torch.logical_and(collision[done_ids], ~use_recovery[done_ids]).sum().item())
                episode_recovery_logging[done_ids] = False
                recovery_hold_left[done_ids] = 0

                where_collision = torch.where(collision)[0]
                where_reach = torch.where(reached)[0]
                where_timeout = torch.where(timed_out)[0]
                episode_max_velo_dist += float(episode_max_velo[done_ids].sum().item())
                episode_max_velo_dist_collision += float(episode_max_velo[where_collision].sum().item())
                episode_max_velo_dist_reach += float(episode_max_velo[where_reach].sum().item())
                episode_max_velo_dist_timeout += float(episode_max_velo[where_timeout].sum().item())

                collision_dist = float(episode_travel_dist[where_collision].sum().item())
                collision_time = float(episode_time[where_collision].sum().item())
                total_collision_dist += collision_dist
                total_collision_time += collision_time
                reach_dist = float(episode_travel_dist[where_reach].sum().item())
                reach_time = float(episode_time[where_reach].sum().item())
                total_reach_dist += reach_dist
                total_reach_time += reach_time
                timeout_dist = float(episode_travel_dist[where_timeout].sum().item())
                timeout_time = float(episode_time[where_timeout].sum().item())
                total_timeout_dist += timeout_dist
                total_timeout_time += timeout_time
                total_travel_dist += collision_dist + reach_dist + timeout_dist
                total_time += collision_time + reach_time + timeout_time

                episode_travel_dist[done_ids] = 0.0
                episode_time[done_ids] = 0.0
                episode_max_velo[done_ids] = 0.0

            # 4) Decide behavior from mode after figures are available.
            if args.trainRA:
                ra_obs = _build_ra_obs(raw_env)
                # ls <= 0 means reach target; gs > 0 means failure/collision.
                gs = collision.float() * 2.0 - 1.0
                ls = torch.tanh(torch.log2(dist / 0.65 + 1e-8))

                s_queue[:-1] = s_queue[1:].clone()
                g_queue[:-1] = g_queue[1:].clone()
                l_queue[:-1] = l_queue[1:].clone()
                done_queue[:-1] = done_queue[1:].clone()
                s_queue[-1] = ra_obs.detach().clone()
                g_queue[-1] = gs.detach().clone()
                l_queue[-1] = ls.detach().clone()
                done_queue[-1] = dones.detach().clone()

                # Hindsight smoothing from testbed1.py.
                g_hs_queue[:-1] = g_hs_queue[1:].clone()
                g_hs_queue[-1] = gs.detach().clone()
                g_hs_span[:] -= 1
                g_hs_span[0][dones] = g_hs_span[1][dones].clone() + 1
                g_hs_span[1][dones] = queue_len - 1
                g_hs_span[0] = torch.maximum(g_hs_span[0], g_hs_span[1] - hindsight)
                g_hs_span = g_hs_span * (g_hs_span >= 0)
                range_tensor = torch.arange(queue_len, device=args.device).unsqueeze(1)
                mask = (range_tensor >= g_hs_span[0:1]) & (range_tensor < g_hs_span[1:2])
                new_values = gs.detach().clone().repeat(queue_len, 1)
                mask = mask & (new_values > 0)
                new_values -= (g_hs_span[1:2] - range_tensor) * 2 / hindsight * mask
                g_hs_queue[mask] = new_values[mask].clone()

                if i > queue_len and i % args.ra_update_interval == 0:
                    false_safe, false_reach, n_fail, n_reach, accu_loss = 0, 0, 0, 0, []
                    total_n_fail = torch.logical_and(g_queue[1:] > 0, done_queue[1:]).sum().item()
                    total_n_reach = torch.logical_and(l_queue[:-1] <= 0, done_queue[1:]).sum().item()
                    with torch.no_grad():
                        start_v = ra_net(standard_raobs_init).mean().item()
                        die_v = ra_net(standard_raobs_die).mean().item()
                        turn_v = ra_net(standard_raobs_turn).mean().item()
                    weight_end = 0.0
                    ra_net.train()
                    for _start in range(0, queue_len - 1, batch_size):
                        vs_old = ra_net(s_queue[_start : _start + batch_size]).squeeze(-1)
                        with torch.no_grad():
                            vs_new = (
                                ra_net(s_queue[_start + 1 : _start + batch_size + 1]).squeeze(-1)
                                * (~done_queue[_start + 1 : _start + batch_size + 1])
                                + 1.0 * done_queue[_start + 1 : _start + batch_size + 1]
                            )
                            vs_discounted_old = gamma * torch.maximum(
                                g_hs_queue[_start + 1 : _start + batch_size + 1],
                                torch.minimum(l_queue[_start : _start + batch_size], vs_new),
                            ) + (1 - gamma) * torch.maximum(
                                l_queue[_start : _start + batch_size],
                                g_hs_queue[_start + 1 : _start + batch_size + 1],
                            )
                        v_loss = 100 * torch.mean(
                            torch.square(vs_old - vs_discounted_old)
                            * (1.0 + weight_end * (done_queue[_start + 1 : _start + batch_size + 1] > 0))
                        )
                        ra_optim.zero_grad(set_to_none=True)
                        v_loss.backward()
                        torch.nn.utils.clip_grad_norm_(ra_net.parameters(), 1.0)
                        ra_optim.step()

                        vs_old_eval = vs_old.detach()
                        false_safe += torch.logical_and(
                            g_queue[_start + 1 : _start + batch_size + 1] > 0, vs_old_eval <= 0
                        ).sum().item()
                        false_reach += torch.logical_and(
                            l_queue[_start : _start + batch_size] <= 0, vs_old_eval > 0
                        ).sum().item()
                        n_fail += (g_queue[_start + 1 : _start + batch_size + 1] > 0).sum().item()
                        n_reach += (l_queue[_start : _start + batch_size] <= 0).sum().item()
                        accu_loss.append(v_loss.item())
                        del vs_old, vs_old_eval, vs_new, vs_discounted_old, v_loss
                    ra_net.eval()
                    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
                        torch.cuda.empty_cache()

                    if false_safe / (n_fail + 1e-8) < best_metric and die_v > 0.2 and start_v < -0.1 and turn_v < -0.1 and i > 3000:
                        best_metric = false_safe / (n_fail + 1e-8)
                        _save_ra(ra_net, args.ra_name)

            avg_collision_dist = total_collision_dist / (stats.collisions + 1e-8)
            avg_collision_time = total_collision_time / (stats.collisions + 1e-8)
            avg_reach_dist = total_reach_dist / (stats.reached + 1e-8)
            avg_reach_time = total_reach_time / (stats.reached + 1e-8)
            avg_timeout_dist = total_timeout_dist / (stats.timed_out + 1e-8)
            avg_timeout_time = total_timeout_time / (stats.timed_out + 1e-8)
            avg_total_dist = total_travel_dist / (stats.done + 1e-8)
            avg_total_time = total_time / (stats.done + 1e-8)
            avg_total_velocity = avg_total_dist / (avg_total_time + 1e-8)
            avg_collision_velocity = avg_collision_dist / (avg_collision_time + 1e-8)
            avg_reach_velocity = avg_reach_dist / (avg_reach_time + 1e-8)
            avg_timeout_velocity = avg_timeout_dist / (avg_timeout_time + 1e-8)
            avg_recovery_velocity = total_recovery_dist / ((total_recovery_timesteps + 1e-8) * step_dt)
            while display_stats.done < stats.done:
                _advance_display_stats(display_stats, stats)
                metrics_print_idx = display_stats.done
                done_scale = display_stats.done / (stats.done + 1e-8)
                collision_scale = display_stats.collisions / (stats.collisions + 1e-8)
                reach_scale = display_stats.reached / (stats.reached + 1e-8)
                timeout_scale = display_stats.timed_out / (stats.timed_out + 1e-8)
                _print_metrics_block(
                    episode_idx=metrics_print_idx,
                    stats=display_stats,
                    episode_max_velo_dist=episode_max_velo_dist * done_scale,
                    episode_max_velo_dist_collision=episode_max_velo_dist_collision * collision_scale,
                    episode_max_velo_dist_reach=episode_max_velo_dist_reach * reach_scale,
                    episode_max_velo_dist_timeout=episode_max_velo_dist_timeout * timeout_scale,
                    avg_total_velocity=avg_total_velocity,
                    avg_collision_velocity=avg_collision_velocity,
                    avg_reach_velocity=avg_reach_velocity,
                    avg_timeout_velocity=avg_timeout_velocity,
                    avg_recovery_velocity=avg_recovery_velocity,
                )
            obs = obs_next
            if args.testRA and recovery_policy is not None and int(args.recovery_hold_steps) > 0:
                h = int(args.recovery_hold_steps)
                hold_fill = torch.full_like(recovery_hold_left, h)
                recovery_hold_left = torch.where(ra_wants_recovery, hold_fill, recovery_hold_left)
                recovery_hold_left = torch.where(
                    use_recovery,
                    torch.clamp(recovery_hold_left - 1, min=0),
                    recovery_hold_left,
                )

    if args.trainRA:
        save_path = _save_ra(ra_net, args.ra_name)
        print(f"[testbed2] Saved RA network: {save_path}")

    total = max(stats.done, 1)
    print("\n[testbed2] Final stats")
    print(f"  done episodes: {stats.done}")
    print(f"  reach rate:    {stats.reached / total:.2%}")
    print(f"  fail rate:     {stats.failed / total:.2%}")
    print(f"  timeout rate:  {stats.timed_out / total:.2%}")
    if args.testRA:
        print(f"  RA used steps: {stats.ra_used_steps}")

    env.close()
    SIMULATION_APP.close()


if __name__ == "__main__":
    _run(ARGS)