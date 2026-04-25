# go2_rec_policy/ppo_cfg.py
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)
from isaaclab.utils import configclass


@configclass
class Go2RecoveryPPORunnerCfg(RslRlOnPolicyRunnerCfg):

    # ── Top-level runner ──────────────────────────────────────────────
    seed:                 int  = 1
    runner_class_name:    str  = "OnPolicyRunner"
    num_steps_per_env:    int  = 48        # episode is shorter → fewer steps
    max_iterations:       int  = 5000
    save_interval:        int  = 1000
    experiment_name:      str  = "go2_recovery"
    run_name:             str  = ""
    resume:               bool = False
    logger:               str = "wandb"
    wandb_project:        str = "go2_recovery"


    # ── Observation groups fed to actor / critic ──────────────────────
    obs_groups: dict = {
        "actor":  ["policy"],
        "critic": ["policy"],
    }

    # ── Policy network ────────────────────────────────────────────────
    # Smaller MLP than the agile policy: obs dim is ~half the size and
    # the behavior is lower-frequency (stand up then stabilize).
    policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
        class_name         = "ActorCritic",
        init_noise_std     = 1.0,
        actor_hidden_dims  = [256, 128, 64],
        critic_hidden_dims = [256, 128, 64],
        activation         = "elu",
    )

    # ── PPO algorithm ─────────────────────────────────────────────────
    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        class_name             = "PPO",
        value_loss_coef        = 1.0,
        use_clipped_value_loss = True,
        clip_param             = 0.2,
        entropy_coef           = 0.01,   # higher entropy → wider exploration
        num_learning_epochs    = 5,
        num_mini_batches       = 4,
        learning_rate          = 1.0e-3,
        schedule               = "adaptive",
        gamma                  = 0.99,
        lam                    = 0.95,
        desired_kl             = 0.01,
        max_grad_norm          = 1.0,
    )
