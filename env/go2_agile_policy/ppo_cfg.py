# envs/go1_pos_rough/runner_cfg.py
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab.utils import configclass


@configclass
class Go2PosRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):

    # ── Top-level runner ──────────────────────────────────────────────────
    seed:                 int  = 1
    runner_class_name:    str  = "OnPolicyRunner"
    num_steps_per_env:    int  = 96
    max_iterations:       int  = 6000
    save_interval:        int  = 200
    experiment_name:      str  = "test"
    run_name:             str  = ""
    resume:               bool = False

    # ── Observation groups fed to actor / critic ──────────────────────────
    obs_groups: dict = {
        "actor":  ["policy"],
        "critic": ["policy"],
    }

    # ── Policy network ────────────────────────────────────────────────────
    policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
        class_name         = "ActorCritic",
        init_noise_std     = 1.0,
        actor_hidden_dims  = [512, 256, 128],
        critic_hidden_dims = [512, 256, 128],
        activation         = "elu",
    )

    # ── PPO algorithm ─────────────────────────────────────────────────────
    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        class_name             = "PPO",
        value_loss_coef        = 1.0,
        use_clipped_value_loss = True,
        clip_param             = 0.2,
        entropy_coef           = 0.003,
        num_learning_epochs    = 8,
        num_mini_batches       = 8,
        learning_rate          = 0.001,
        schedule               = "adaptive",
        gamma                  = 0.99,
        lam                    = 0.95,
        desired_kl             = 0.01,
        max_grad_norm          = 1.0,
    )