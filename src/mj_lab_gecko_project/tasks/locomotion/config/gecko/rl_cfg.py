"""PPO configuration for the gecko locomotion task (Tian et al. 2026, Table II)."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def gecko_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Paper: 3e8 steps, 8192 envs, unroll 20, gamma 0.97, lr 3e-4, entropy 1e-2,
  4 updates per batch, grad clip 1.0, obs normalization, MLP (512, 256, 128)."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=1.0e-2,
      num_learning_epochs=4,
      num_mini_batches=4,
      learning_rate=3.0e-4,
      schedule="fixed",
      gamma=0.97,
      lam=0.95,
      max_grad_norm=1.0,
    ),
    experiment_name="gecko_locomotion",
    logger="tensorboard",
    save_interval=100,
    num_steps_per_env=20,
    # 3e8 steps / (8192 envs * 20 steps) ~= 1830 iterations.
    max_iterations=2000,
  )
