"""Train the gecko with OpenES (cases 2 and 3) on the Gecko-Locomotion-Flat environment.

Each of the POPSIZE parallel envs evaluates one population member per generation:
  ask() -> one parameter set per env -> reset all envs -> run EPISODE_LENGTH_TRAIN
  steps -> fitness = summed task reward (an env stops counting when it falls)
  -> tell(fitness).
The reward is exactly the PPO task reward (Gecko-Locomotion-Flat), so returns are
comparable. Antithetic pairs (member i and i + POPSIZE/2) get the same velocity
command and gait phase, which reduces fitness noise between the +eps/-eps pair.

Settings come from tasks/locomotion/config/gecko/es_cfg.yaml (ES_params, EPOCHS,
SAVE_EVERY, EPISODE_LENGTH_TRAIN, model, *_ARCHITECTURE, HEBB_*); CLI flags override.

Usage:
  uv run python scripts/train_es.py --model ff     # case 2: MLP + OpenES
  uv run python scripts/train_es.py --model hebb   # case 3: Hebbian + OpenES
  uv run tensorboard --logdir logs/es
"""

import argparse
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402

from mj_lab_gecko_project.rl.es_models import (  # noqa: E402
  ES_CFG_PATH,
  RunningObsNorm,
  build_es_model,
  load_es_cfg,
)
from mj_lab_gecko_project.rl.evolution_strategies import OpenES  # noqa: E402
from mj_lab_gecko_project.tasks.locomotion.config.gecko.env_cfgs import (  # noqa: E402
  gecko_flat_env_cfg,
)

LOG_ROOT = Path("logs/es")


def make_env(num_envs: int, device: str, seed: int) -> ManagerBasedRlEnv:
  cfg = gecko_flat_env_cfg()
  cfg.scene.num_envs = num_envs
  cfg.seed = seed
  # One velocity command per evaluation episode (no mid-episode resampling).
  cfg.commands["twist"].resampling_time_range = (1e9, 1e9)
  cfg.commands["twist"].debug_vis = False
  return ManagerBasedRlEnv(cfg=cfg, device=device)


def pair_antithetic_conditions(env: ManagerBasedRlEnv) -> None:
  """Give member i + P/2 the same command and gait phase as member i."""
  half = env.num_envs // 2
  twist = env.command_manager.get_term("twist")
  gait = env.command_manager.get_term("gait")
  twist.vel_command_b[half:] = twist.vel_command_b[:half]
  twist.is_standing_env[half:] = twist.is_standing_env[:half]
  gait.clock[half:] = gait.clock[:half]


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--config", type=Path, default=ES_CFG_PATH)
  ap.add_argument("--model", choices=["ff", "hebb"], help="default: 'model' in es_cfg.yaml")
  ap.add_argument("--popsize", type=int, help="default: ES_params.POPSIZE")
  ap.add_argument("--generations", type=int, help="default: EPOCHS")
  ap.add_argument("--episode-steps", type=int, help="default: EPISODE_LENGTH_TRAIN (50 Hz steps)")
  ap.add_argument("--save-every", type=int, help="default: SAVE_EVERY")
  ap.add_argument("--seed", type=int, help="default: seed (-1 = random)")
  args = ap.parse_args()

  cfg = load_es_cfg(args.config)
  es_params = cfg["ES_params"]
  model_name = args.model or cfg["model"]
  popsize = args.popsize or es_params["POPSIZE"]
  generations = args.generations or cfg["EPOCHS"]
  steps = args.episode_steps or cfg["EPISODE_LENGTH_TRAIN"]
  save_every = args.save_every or cfg["SAVE_EVERY"]
  seed = args.seed if args.seed is not None else cfg["seed"]
  if seed is None or seed < 0:
    seed = int(np.random.SeedSequence().entropy % 2**31)
  np.random.seed(seed)
  torch.manual_seed(seed)
  device = "cuda:0"

  env = make_env(popsize, device, seed)
  obs = env.reset()[0]["actor"]
  obs_dim, act_dim = obs.shape[1], env.action_manager.total_action_dim
  model = build_es_model(model_name, popsize, obs_dim, act_dim, cfg)
  obs_norm = RunningObsNorm(obs_dim, device)
  es = OpenES(
    model.num_params,
    sigma_init=es_params["sigma_init"],
    sigma_decay=es_params["sigma_decay"],
    sigma_limit=es_params["sigma_limit"],
    learning_rate=es_params["learning_rate"],
    learning_rate_decay=es_params["learning_rate_decay"],
    learning_rate_limit=es_params["learning_rate_limit"],
    popsize=popsize,
    antithetic=es_params["antithetic"],
    rank_fitness=es_params["rank_fitness"],
  )
  if model_name == "hebb":  # Start from HebbianNet's own N(0, HEBB_std_init_param) init.
    es.set_mu(model.net.get_a_model_params().astype(np.float64))

  run_dir = LOG_ROOT / f"gecko_locomotion_es_{model_name}" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  (run_dir / "params").mkdir(parents=True)
  shutil.copy(args.config, run_dir / "params" / "es_cfg.yaml")
  (run_dir / "params" / "args.txt").write_text(
    f"model={model_name} popsize={popsize} generations={generations} "
    f"episode_steps={steps} seed={seed} num_params={model.num_params} sizes={model.sizes}\n"
  )
  writer = SummaryWriter(str(run_dir))
  rm = env.reward_manager
  term_names = list(rm._term_names)
  dt = env.step_dt
  print(f"[ES] {model_name}: {model.num_params} params/member, popsize {popsize}, "
        f"{generations} generations x {steps} steps, seed {seed}\n[ES] logging to {run_dir}")

  def save(gen: int, env_steps: int) -> None:
    torch.save(
      {
        "model": model_name, "sizes": model.sizes, "num_params": model.num_params,
        "mu": es.mu.copy(), "best_mu": np.array(es.best_mu), "obs_norm": obs_norm.state_dict(),
        "generation": gen, "env_steps": env_steps, "es_cfg": cfg,
        # Small optimizer state only. Pickling the OpenES object would also store its
        # population buffers (popsize x num_params float64, GBs for the Hebbian net).
        "es_state": {
          "sigma": es.sigma, "learning_rate": es.learning_rate,
          "adam_m": es.optimizer.m, "adam_v": es.optimizer.v, "adam_t": es.optimizer.t,
        },
      },
      run_dir / f"es_{gen}.pt",
    )

  env_steps = 0
  for gen in range(generations):
    t0 = time.perf_counter()
    model.set_params(es.ask())
    with torch.inference_mode():
      env.reset()
      pair_antithetic_conditions(env)
      obs = env.observation_manager.compute()["actor"]  # Includes the paired commands.
      model.reset()
      alive = torch.ones(popsize, dtype=torch.bool, device=device)
      fitness = torch.zeros(popsize, device=device)
      term_sums = torch.zeros(popsize, len(term_names), device=device)
      length = torch.zeros(popsize, device=device)
      for _ in range(steps):
        obs_norm.update(obs)
        actions = model(obs_norm(obs))
        obs_dict, reward, terminated, truncated, _ = env.step(actions)
        obs = obs_dict["actor"]
        live = alive.float()
        fitness += reward * live
        term_sums += rm._step_reward * dt * live.unsqueeze(1)
        length += live
        alive &= ~(terminated | truncated)
    es.tell(fitness.cpu().numpy())
    env_steps += popsize * steps
    gen_time = time.perf_counter() - t0

    f = fitness.cpu().numpy()
    seconds = steps * dt
    writer.add_scalar("Fitness/mean", f.mean(), gen)
    writer.add_scalar("Fitness/max", f.max(), gen)
    writer.add_scalar("Fitness/min", f.min(), gen)
    writer.add_scalar("Fitness/std", f.std(), gen)
    writer.add_scalar("ES/sigma", es.sigma, gen)
    writer.add_scalar("ES/learning_rate", es.learning_rate, gen)
    writer.add_scalar("ES/update_ratio", es.update_ratio, gen)
    # Same units as PPO's Episode_Reward/*: reward per second of episode.
    for i, name in enumerate(term_names):
      writer.add_scalar(f"Episode_Reward/{name}", term_sums[:, i].mean().item() / seconds, gen)
    writer.add_scalar("Episode/survived_fraction", alive.float().mean().item(), gen)
    writer.add_scalar("Episode/mean_length_s", length.mean().item() * dt, gen)
    writer.add_scalar("Perf/env_steps", env_steps, gen)
    writer.add_scalar("Perf/steps_per_second", popsize * steps / gen_time, gen)
    writer.add_scalar("Perf/generation_time_s", gen_time, gen)
    print(f"[ES] gen {gen:4d} | fitness mean {f.mean():7.3f} max {f.max():7.3f} | "
          f"survived {alive.float().mean().item():.2f} | sigma {es.sigma:.4f} | {gen_time:5.1f}s", flush=True)
    if (gen + 1) % save_every == 0:
      save(gen, env_steps)

  save(generations - 1, env_steps)
  writer.close()
  env.close()


if __name__ == "__main__":
  main()
