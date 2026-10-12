from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import gecko_flat_env_cfg
from .rl_cfg import gecko_ppo_runner_cfg

register_mjlab_task(
  task_id="Gecko-Locomotion-Flat",
  env_cfg=gecko_flat_env_cfg(),
  play_env_cfg=gecko_flat_env_cfg(play=True),
  rl_cfg=gecko_ppo_runner_cfg(),
  # Also exports <run_dir>/<run_name>.onnx (with obs/action metadata) on every save.
  runner_cls=VelocityOnPolicyRunner,
)
