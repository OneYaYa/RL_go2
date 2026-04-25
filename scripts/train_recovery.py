"""Train the Go2 recovery policy with RSL-RL PPO via Isaac Lab.

The recovery policy learns to right itself from arbitrary (possibly
flipped) initial poses and track a body-frame velocity command.

Use wandb to log the training progress.
wandb login

Use the following command to train the policy:
tmux new -s train    &&    tmux attach -t train
conda activate /root/autodl-tmp/conda-envs/isaaclab
cd ~/autodl-tmp/RL_finalproject
python scripts/train_recovery.py --headless --num_envs 4096 --max_iterations 4000 --run_name recovery_policy
"""

import argparse
import os
import sys
from importlib.metadata import version as _pkg_version
sys.path.append('./')

from isaaclab.app import AppLauncher

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train Go2 recovery policy.")
parser.add_argument("--num_envs",       type=int,  default=1280)
parser.add_argument("--max_iterations", type=int,  default=3000)
parser.add_argument("--run_name",       type=str,  default="")
parser.add_argument("--resume",         type=str,  default=None,
                    help="Path to checkpoint to resume from.")

# AppLauncher must consume its own args BEFORE Isaac imports.
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── Isaac / project imports (AFTER AppLauncher) ───────────────────────────────
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg
from rsl_rl.runners import OnPolicyRunner

from env.go2_rec_policy.env_cfg import Go2RecoveryEnvCfg
from env.go2_rec_policy.ppo_cfg import Go2RecoveryPPORunnerCfg


def main():
    # ── Build env config ──────────────────────────────────────────────────
    env_cfg = Go2RecoveryEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device     = args.device

    # ── Build runner config ───────────────────────────────────────────────
    runner_cfg = Go2RecoveryPPORunnerCfg()
    runner_cfg.max_iterations = args.max_iterations
    if args.run_name:
        runner_cfg.run_name = args.run_name
    if args.resume:
        runner_cfg.resume          = True
        runner_cfg.load_run        = os.path.dirname(args.resume)
        runner_cfg.load_checkpoint = os.path.basename(args.resume)

    # Migrate legacy cfg fields to whatever the installed rsl-rl expects.
    runner_cfg = handle_deprecated_rsl_rl_cfg(runner_cfg, _pkg_version("rsl-rl-lib"))

    # ── Instantiate env ───────────────────────────────────────────────────
    env = ManagerBasedRLEnv(cfg=env_cfg)
    _patch_reward_logging(env)

    env = RslRlVecEnvWrapper(env)

    # ── Runner + train ────────────────────────────────────────────────────
    runner = OnPolicyRunner(
        env,
        runner_cfg.to_dict(),
        log_dir=_make_log_dir(runner_cfg),
        device=args.device,
    )
    if args.resume:
        runner.load(args.resume)

    runner.learn(num_learning_iterations=args.max_iterations)

    env.close()
    simulation_app.close()


def _patch_reward_logging(env_unwrapped: ManagerBasedRLEnv) -> None:
    """Log weighted per-term reward values to extras['log'] for tensorboard."""
    original_compute = env_unwrapped.reward_manager.compute

    def patched_compute(dt: float) -> torch.Tensor:
        total = original_compute(dt)

        weighted_log = {}
        for name, term_cfg in zip(
            env_unwrapped.reward_manager._term_names,
            env_unwrapped.reward_manager._term_cfgs,
        ):
            try:
                raw      = term_cfg.func(env_unwrapped, **term_cfg.params)
                weighted = (raw * term_cfg.weight).mean().item()
                weighted_log[f"Rew/{name}"] = weighted
            except Exception:
                pass

        weighted_log["Rew/_total_check"] = sum(weighted_log.values())

        if "log" not in env_unwrapped.extras:
            env_unwrapped.extras["log"] = {}
        env_unwrapped.extras["log"].update(weighted_log)
        return total

    env_unwrapped.reward_manager.compute = patched_compute
    print("[train_recovery] Reward logging patched.")


def _make_log_dir(runner_cfg) -> str:
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name  = runner_cfg.run_name or timestamp
    log_dir   = os.path.join("logs", runner_cfg.experiment_name, run_name)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


if __name__ == "__main__":
    main()
