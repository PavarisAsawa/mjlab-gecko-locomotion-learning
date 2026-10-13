"""PPO configurations for the gecko locomotion task (Tian et al. 2026, Table II)."""

from dataclasses import dataclass

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

from mj_lab_gecko_project.rl.es_models import hidden_sizes, load_es_cfg


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


##
# Case 4: Hebbian network trained by PPO.
##


@dataclass(kw_only=True)
class HebbianModelCfg(RslRlModelCfg):
  """Actor config for mj_lab_gecko_project.rl.hebbian_ppo:HebbianModel."""

  class_name: str = "mj_lab_gecko_project.rl.hebbian_ppo:HebbianModel"
  activation: str = "tanh"
  init_noise: float = 0.04
  norm_mode: str = "var"
  std_init_param: float = 0.025


def gecko_hebbian_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Same PPO settings as the MLP baseline; the actor is the Hebbian network from
  es_cfg.yaml (same sizes and plasticity settings as the ES Hebbian case)."""
  es = load_es_cfg()
  cfg = gecko_ppo_runner_cfg()
  cfg.actor = HebbianModelCfg(
    hidden_dims=tuple(hidden_sizes(es["HEBB_ARCHITECTURE"])),
    obs_normalization=True,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
    init_noise=es["HEBB_init_wnoise"],
    norm_mode=es["HEBB_norm"],
    std_init_param=es["HEBB_std_init_param"],
  )
  cfg.experiment_name = "gecko_locomotion_hebb_ppo"
  # The plastic weights are stored per env and step, so train with 1024 envs
  # (8 GB GPU). 6000 iterations x 1024 envs x 20 steps = the MLP-PPO step budget.
  cfg.max_iterations = 6000
  cfg.save_interval = 500
  return cfg
