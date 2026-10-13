from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import gecko_flat_env_cfg
from .rl_cfg import gecko_hebbian_ppo_runner_cfg, gecko_ppo_runner_cfg

# Case 1: MLP + PPO.
register_mjlab_task(
  task_id="Gecko-Locomotion-Flat",
  env_cfg=gecko_flat_env_cfg(),
  play_env_cfg=gecko_flat_env_cfg(play=True),
  rl_cfg=gecko_ppo_runner_cfg(),
  # Also exports <run_dir>/<run_name>.onnx (with obs/action metadata) on every save.
  runner_cls=VelocityOnPolicyRunner,
)

# Case 4: Hebbian network + PPO. Same environment, different actor.
# (Cases 2 and 3, ES, are trained by scripts/train_es.py on the same environment.)
register_mjlab_task(
  task_id="Gecko-Locomotion-Flat-HebbPPO",
  env_cfg=gecko_flat_env_cfg(),
  play_env_cfg=gecko_flat_env_cfg(play=True),
  rl_cfg=gecko_hebbian_ppo_runner_cfg(),
  runner_cls=MjlabOnPolicyRunner,  # No ONNX export for the Hebbian model.
)
