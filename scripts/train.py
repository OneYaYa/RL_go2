"""Train a policy on Go1PosRough using RSL-RL via Isaac Lab."""

import argparse
import sys
import os
sys.path.append('./')

from isaaclab.app import AppLauncher

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train Go1 position tracking.")
parser.add_argument("--num_envs",       type=int,   default=1280)
parser.add_argument("--max_iterations", type=int,   default=10000)
parser.add_argument("--run_name",       type=str,   default="")
parser.add_argument("--resume",         type=str,   default=None,
                    help="Path to checkpoint to resume from.")
parser.add_argument("--no_penalty",     action="store_true",
                    help="Use NoPenalty variant config.")
# parser.add_argument("--headless",       action="store_true")
# parser.add_argument("--device",         type=str,   default="cuda:0")

# AppLauncher must consume its own args before Isaac imports
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── Isaac / project imports (MUST come after AppLauncher) ─────────────────────
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper
)
from rsl_rl.runners import OnPolicyRunner

from env.go2_agile_policy.env_cfg import (
    Go2PosRoughEnvCfg,
    # Go2PosRoughEnvCfgNoPenalty,
)
from env.go2_agile_policy.ppo_cfg import Go2PosRoughPPORunnerCfg
def main():
    # ── Build env config ──────────────────────────────────────────────────
    env_cfg = Go2PosRoughEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device     = args.device

    # ── Build runner config ───────────────────────────────────────────────
    runner_cfg = Go2PosRoughPPORunnerCfg()
    runner_cfg.max_iterations = args.max_iterations
    if args.run_name:
        runner_cfg.run_name = args.run_name
    if args.resume:
        runner_cfg.resume          = True
        runner_cfg.load_run        = os.path.dirname(args.resume)
        runner_cfg.load_checkpoint = os.path.basename(args.resume)

    # ── Instantiate env ───────────────────────────────────────────────────
    env = ManagerBasedRLEnv(cfg=env_cfg)
    _patch_reward_logging(env)
    # _run_diagnostics(env)

    raw_env = env  # before wrapping
    print(dir(raw_env.reward_manager))
    env = RslRlVecEnvWrapper(env)

    # ═════════════════════════════════════════════════════════════════════
    # DIAGNOSTIC BLOCK — remove after confirming obs/rewards are healthy
    # ═════════════════════════════════════════════════════════════════════
    
    # ═════════════════════════════════════════════════════════════════════

    # ── Instantiate runner + train ────────────────────────────────────────
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
    """
    Monkey-patches the reward manager's compute() to also store
    weighted per-term values in extras["log"] for tensorboard.
    """
    original_compute = env_unwrapped.reward_manager.compute

    def patched_compute(dt: float) -> torch.Tensor:
        total = original_compute(dt)

        weighted_log = {}
        # for name, term_cfg in env_unwrapped.reward_manager._terms.items():
        for term_idx, (name, term_cfg) in enumerate(zip(env_unwrapped.reward_manager._term_names, env_unwrapped.reward_manager._term_cfgs)):

            try:
                raw     = term_cfg.func(env_unwrapped, **term_cfg.params)  # (N,)
                weighted = (raw * term_cfg.weight).mean().item()
                weighted_log[f"Rew/{name}"] = weighted
            except Exception:
                pass

        # Verify they sum correctly
        weighted_sum = sum(v for k, v in weighted_log.items())
        weighted_log["Rew/_total_check"] = weighted_sum

        if "log" not in env_unwrapped.extras:
            env_unwrapped.extras["log"] = {}
        env_unwrapped.extras["log"].update(weighted_log)

        return total

    env_unwrapped.reward_manager.compute = patched_compute
    print("[train] Reward logging patched — weighted terms will appear in tensorboard")


def _run_diagnostics(env) -> None:
    import torch
    print("\n" + "="*60)
    print("DIAGNOSTICS — zero-action rollout (5 steps)")
    print("="*60)

    raw_env = env.unwrapped

    # ── 1. Scene contents ─────────────────────────────────────────────
    print("\n[Scene]")
    print("  Sensors:      ", list(raw_env.scene.sensors.keys()))
    print("  Articulations:", list(raw_env.scene._articulations.keys()))
    print("  Rigid objects:", list(raw_env.scene._rigid_objects.keys()))

    # ── 2. Reset ──────────────────────────────────────────────────────
    obs_raw, _ = env.reset()
    obs = _extract_obs_tensor(obs_raw)

    print(f"\n[Observations] raw type: {type(obs_raw).__name__}")
    _print_tensor_stats(obs)

    # ── 3. Per-term breakdown ─────────────────────────────────────────
    print(f"\n[Per-term observation breakdown]")
    try:
        term_obs   = raw_env.observation_manager.compute()
        policy_obs = term_obs.get("policy", term_obs)

        if isinstance(policy_obs, dict):
            items = policy_obs.items()
        elif hasattr(policy_obs, "items"):        # TensorDict
            items = policy_obs.items()
        else:
            items = [("policy", policy_obs)]

        for name, tensor in items:
            if not isinstance(tensor, torch.Tensor):
                continue
            flag = ""
            if tensor.isnan().any():              flag = "  ✗ NaN"
            elif tensor.isinf().any():            flag = "  ✗ Inf"
            elif tensor.abs().max().item() > 100: flag = "  ⚠ large"
            elif tensor.std().item() > 5.0:       flag = "  ⚠ high std"
            print(
                f"  {name:30s} shape={str(tuple(tensor.shape)):20s} "
                f"min={tensor.min().item():8.3f}  "
                f"max={tensor.max().item():8.3f}  "
                f"std={tensor.std().item():7.3f}"
                f"{flag}"
            )
    except Exception as e:
        print(f"  Could not compute per-term obs: {e}")

    # ── 4. Reward term breakdown ──────────────────────────────────────
    print(f"\n[Reward terms — zero actions, 5 steps]")
    zero_actions = torch.zeros(
        env.num_envs,
        raw_env.action_manager.total_action_dim,
        device=raw_env.device,
    )

    reward_totals = {}
    for _ in range(5):
        obs_raw, rewards, dones, extras = env.step(zero_actions)

        # Extract scalar reward
        if isinstance(rewards, torch.Tensor):
            r_mean = rewards.mean().item()
        elif hasattr(rewards, "mean"):
            r_mean = float(rewards.mean())
        else:
            r_mean = float(rewards)

        reward_totals.setdefault("_total", 0.0)
        reward_totals["_total"] += r_mean

        if "log" in extras:
            for key, val in extras["log"].items():
                reward_totals.setdefault(key, 0.0)
                if isinstance(val, torch.Tensor):
                    reward_totals[key] += val.mean().item()
                elif hasattr(val, "__float__"):
                    reward_totals[key] += float(val)

    print(f"  {'Term':40s} {'Avg/step':>12s}")
    print(f"  {'-'*55}")
    for name, total in sorted(reward_totals.items()):
        avg  = total / 5
        flag = "  ⚠ large" if abs(avg) > 50 else ""
        print(f"  {name:40s} {avg:12.4f}{flag}")

    # ── 5. Action manager ─────────────────────────────────────────────
    print(f"\n[Action manager]")
    for term_idx, (name, term_cfg) in enumerate(zip(raw_env.reward_manager._term_names, raw_env.reward_manager._term_cfgs)):

        scale = getattr(term_cfg, "scale", "N/A")
        print(f"  {name}: scale={scale}")

    # ── 6. Summary ────────────────────────────────────────────────────
    print(f"\n[Health summary]")
    obs_nan = obs.isnan().any().item()
    obs_inf = obs.isinf().any().item()
    obs_std = obs.std().item()
    obs_ok  = not obs_nan and not obs_inf and obs_std < 5.0
    print(f"  Obs NaN:  {'✗ YES' if obs_nan else '✓ none'}")
    print(f"  Obs Inf:  {'✗ YES' if obs_inf else '✓ none'}")
    print(f"  Obs std:  {obs_std:.3f}  {'⚠ high' if obs_std > 5 else '✓ ok'}")
    print(f"  Overall:  {'✓ healthy' if obs_ok else '✗ issues detected'}")
    print("="*60 + "\n")


def _extract_obs_tensor(obs_raw) -> "torch.Tensor":
    """Handles tensor, dict, and TensorDict returned by different wrappers."""
    import torch
    if isinstance(obs_raw, torch.Tensor):
        return obs_raw
    # TensorDict or dict — try common policy keys
    for key in ("policy", "obs", "observation"):
        if hasattr(obs_raw, "__getitem__"):
            try:
                val = obs_raw[key]
                if isinstance(val, torch.Tensor):
                    return val
            except (KeyError, IndexError):
                continue
    # Last resort: stack all tensors found
    tensors = []
    if hasattr(obs_raw, "values"):
        for v in obs_raw.values():
            if isinstance(v, torch.Tensor):
                tensors.append(v.reshape(v.shape[0], -1))
    if tensors:
        return torch.cat(tensors, dim=-1)
    raise ValueError(f"Cannot extract tensor from obs type: {type(obs_raw)}")


def _print_tensor_stats(obs: "torch.Tensor") -> None:
    print(f"  Shape:  {obs.shape}")
    print(f"  dtype:  {obs.dtype}")
    print(f"  min:    {obs.min().item():.4f}")
    print(f"  max:    {obs.max().item():.4f}")
    print(f"  mean:   {obs.mean().item():.4f}")
    print(f"  std:    {obs.std().item():.4f}")
    print(f"  NaNs:   {obs.isnan().sum().item()}")
    print(f"  Infs:   {obs.isinf().sum().item()}")
    if obs.std().item() > 5.0:
        print("  ⚠ WARNING: obs std > 5 — likely a scaling issue")
    if obs.abs().max().item() > 100.0:
        print("  ⚠ WARNING: obs values outside ±100 — check scaling")
    if obs.isnan().any():
        print("  ✗ ERROR: NaN in observations")
    if obs.isinf().any():
        print("  ✗ ERROR: Inf in observations")


def _make_log_dir(runner_cfg) -> str:
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name  = runner_cfg.run_name or timestamp
    log_dir   = os.path.join("logs", runner_cfg.experiment_name, run_name)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


if __name__ == "__main__":
    main()