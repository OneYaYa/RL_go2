"""Roll out a trained Go1 policy (no gradient, optional video export)."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Play Go1 position tracking policy.")
parser.add_argument("--checkpoint",  type=str,  required=True,
                    help="Path to .pt checkpoint file.")
parser.add_argument("--num_envs",    type=int,  default=16)
parser.add_argument("--num_steps",   type=int,  default=1000,
                    help="Steps to run per episode.")
parser.add_argument("--no_penalty",  action="store_true")
parser.add_argument("--save_video",  action="store_true",
                    help="Save rollout to video via Isaac Lab recorder.")
parser.add_argument("--video_dir",   type=str,  default="videos")
parser.add_argument("--device",      type=str,  default="cuda:0")
parser.add_argument("--headless",    action="store_true")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Force a window open for video capture if requested
if args.save_video:
    args.headless = False

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── Isaac / project imports ───────────────────────────────────────────────────
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from env.go2_agile_policy.env_cfg import (
    Go2PosRoughEnvCfg,
    # Go2PosRoughEnvCfgNoPenalty,
)
from env.go2_agile_policy.ppo_cfg import Go2PosRoughPPORunnerCfg


def main():
    # ── Build env config ──────────────────────────────────────────────────
    env_cfg = Go2PosRoughEnvCfg()
    env_cfg.scene.num_envs    = args.num_envs
    env_cfg.sim.device        = args.device

    # Disable randomization for clean evaluation
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.push_robot           = None
    env_cfg.events.randomize_friction   = None
    env_cfg.events.randomize_base_mass  = None

    # ── Optional video recorder ───────────────────────────────────────────
    if args.save_video:
        env_cfg.viewer = _make_video_recorder_cfg(args.video_dir)

    # ── Instantiate env ───────────────────────────────────────────────────
    env     = ManagerBasedRLEnv(cfg=env_cfg)
    env     = RslRlVecEnvWrapper(env)

    # ── Load policy ───────────────────────────────────────────────────────
    runner_cfg = Go2PosRoughPPORunnerCfg()
    runner = OnPolicyRunner(
        env,
        runner_cfg.to_dict(),
        log_dir=None,
        device=args.device,
    )
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=args.device)

    # ── Rollout loop ──────────────────────────────────────────────────────
    obs, _  = env.reset()


    print("obs min:", obs.min().item())
    print("obs max:", obs.max().item())
    print("obs mean:", obs.mean().item())
    print("obs std:", obs.std().item())



    episode_rewards   = torch.zeros(args.num_envs, device=args.device)
    episode_lengths   = torch.zeros(args.num_envs, device=args.device)
    completed_rewards = []
    completed_lengths = []

    print(f"\n[play] Rolling out {args.num_steps} steps across {args.num_envs} envs...")

    with torch.inference_mode():
        for step in range(args.num_steps):
            actions          = policy(obs)
            obs, rewards, dones, extras = env.step(actions)

            episode_rewards += rewards
            episode_lengths += 1

            # Log completed episodes
            done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            if done_ids.numel() > 0:
                completed_rewards.extend(episode_rewards[done_ids].tolist())
                completed_lengths.extend(episode_lengths[done_ids].tolist())
                episode_rewards[done_ids] = 0.0
                episode_lengths[done_ids] = 0.0

            if (step + 1) % 200 == 0:
                _print_stats(step + 1, completed_rewards, completed_lengths)

    # ── Final summary ─────────────────────────────────────────────────────
    print("\n── Final Stats ──────────────────────────────────────────")
    _print_stats(args.num_steps, completed_rewards, completed_lengths)

    env.close()
    simulation_app.close()


def _print_stats(
    step: int,
    rewards: list[float],
    lengths: list[float],
) -> None:
    if not rewards:
        print(f"  step {step:5d} | no completed episodes yet")
        return
    import statistics
    print(
        f"  step {step:5d} | episodes: {len(rewards):4d} | "
        f"reward  mean={statistics.mean(rewards):7.2f}  "
        f"std={statistics.stdev(rewards) if len(rewards) > 1 else 0.0:6.2f}  "
        f"min={min(rewards):7.2f}  max={max(rewards):7.2f} | "
        f"length mean={statistics.mean(lengths):6.1f}"
    )


def _make_video_recorder_cfg(video_dir: str):
    """Attaches Isaac Lab's built-in viewport recorder to the viewer."""
    from isaaclab.envs.ui import ViewerCfg
    from isaaclab.utils.io import VideoRecorderCfg

    os.makedirs(video_dir, exist_ok=True)
    return ViewerCfg(
        resolution=(1280, 720),
        video_recorder=VideoRecorderCfg(
            output_dir=video_dir,
            fps=50,
        ),
    )


if __name__ == "__main__":
    main()