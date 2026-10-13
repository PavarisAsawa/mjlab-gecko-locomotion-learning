"""Population policies for evolution strategies (one network per parallel env).

Each env in the batch evaluates one ES population member, so every model here
holds ``popsize`` independent parameter sets and runs them in one batched pass.

  PopulationMLP     feed-forward tanh MLP, parameters = weights + biases.
  PopulationHebbian wraps HebbianNet (hebbian_neural_net.py); ES evolves the
                    ABCD + learning-rate coefficients, the weights themselves start
                    random every episode and change by the Hebbian rule.

Shared interface (used by scripts/train_es.py and scripts/compare_policies.py):
  num_params                   parameters of ONE member
  set_params(flat)             (popsize, num_params) array, one row per member
  reset(env_ids)               start of an episode for these envs
  __call__(obs) -> actions     (popsize, obs_dim) -> (popsize, act_dim) in [-1, 1]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml

from .hebbian_neural_net import HebbianNet

ES_CFG_PATH = (
  Path(__file__).resolve().parents[1] / "tasks/locomotion/config/gecko/es_cfg.yaml"
)


def load_es_cfg(path: str | Path = ES_CFG_PATH) -> dict:
  with open(path) as f:
    return yaml.safe_load(f)


def hidden_sizes(arch: list[int]) -> list[int]:
  """es_cfg.yaml architectures are [input, hidden..., output]; input/output come
  from the environment, so only the hidden sizes are used."""
  return list(arch[1:-1])


class RunningObsNorm:
  """Running mean/std of observations (Welford), shared by the whole population.

  Updated from rollout observations each generation, as in Salimans et al. 2017.
  """

  def __init__(self, dim: int, device: str, clip: float = 5.0) -> None:
    self.mean = torch.zeros(dim, device=device, dtype=torch.float64)
    self.var = torch.ones(dim, device=device, dtype=torch.float64)
    self.count = 1e-4
    self.clip = clip

  def update(self, x: torch.Tensor) -> None:
    x = x.reshape(-1, x.shape[-1]).double()
    batch_mean, batch_var, n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
    delta = batch_mean - self.mean
    total = self.count + n
    self.mean += delta * n / total
    m2 = self.var * self.count + batch_var * n + delta.square() * self.count * n / total
    self.var = m2 / total
    self.count = total

  def __call__(self, x: torch.Tensor) -> torch.Tensor:
    y = (x - self.mean.float()) / torch.sqrt(self.var.float() + 1e-8)
    return torch.clamp(y, -self.clip, self.clip)

  def state_dict(self) -> dict:
    return {"mean": self.mean.cpu(), "var": self.var.cpu(), "count": self.count}

  def load_state_dict(self, state: dict) -> None:
    self.mean = state["mean"].to(self.mean.device)
    self.var = state["var"].to(self.var.device)
    self.count = state["count"]


class PopulationMLP:
  """tanh MLP with per-member weights and biases (no learning within an episode)."""

  def __init__(self, popsize: int, sizes: list[int], device: str = "cuda") -> None:
    self.sizes = sizes
    self.popsize = popsize
    self.device = device
    self._shapes = [(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
    self.num_params = sum(a * b + b for a, b in self._shapes)
    self.weights = [torch.zeros(popsize, a, b, device=device) for a, b in self._shapes]
    self.biases = [torch.zeros(popsize, b, device=device) for _, b in self._shapes]

  def set_params(self, flat: np.ndarray) -> None:
    flat_t = torch.as_tensor(flat, dtype=torch.float32, device=self.device)
    m = 0
    for i, (a, b) in enumerate(self._shapes):
      self.weights[i] = flat_t[:, m : m + a * b].reshape(-1, a, b)
      m += a * b
      self.biases[i] = flat_t[:, m : m + b]
      m += b

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    pass

  @torch.no_grad()
  def __call__(self, obs: torch.Tensor) -> torch.Tensor:
    x = obs
    for w, b in zip(self.weights, self.biases):
      x = torch.tanh(torch.einsum("pi,pio->po", x, w) + b)
    return x


class PopulationHebbian:
  """Adapter around HebbianNet: float32 params, per-episode random weights."""

  def __init__(
    self,
    popsize: int,
    sizes: list[int],
    init_noise: float = 0.04,
    norm_mode: str = "var",
    std_init_param: float = 0.025,
  ) -> None:
    self.sizes = sizes
    self.popsize = popsize
    self.init_noise = init_noise
    self.net = HebbianNet(popsize, sizes, init_noise, norm_mode, std_init_param)
    self.num_params = int(self.net.get_n_params_a_model())

  def set_params(self, flat: np.ndarray) -> None:
    # float32: HebbianNet keeps the dtype it is given, and float64 is very slow on GPU.
    self.net.set_models_params(np.ascontiguousarray(flat, dtype=np.float32))

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    """Re-draw the plastic weights U(-init_noise, init_noise), as at construction."""
    for i, w in enumerate(self.net.weights):
      fresh = torch.empty_like(w, dtype=torch.float32).uniform_(-self.init_noise, self.init_noise)
      if env_ids is None:
        self.net.weights[i] = fresh
      else:
        w = w.float().clone()
        w[env_ids] = fresh[env_ids]
        self.net.weights[i] = w

  def __call__(self, obs: torch.Tensor) -> torch.Tensor:
    return self.net.forward(obs.float())


def build_es_model(model: str, popsize: int, obs_dim: int, act_dim: int, cfg: dict):
  """model: 'ff' (MLP) or 'hebb' (Hebbian), sizes from es_cfg.yaml."""
  if model == "ff":
    sizes = [obs_dim, *hidden_sizes(cfg["FF_ARCHITECTURE"]), act_dim]
    return PopulationMLP(popsize, sizes)
  if model == "hebb":
    sizes = [obs_dim, *hidden_sizes(cfg["HEBB_ARCHITECTURE"]), act_dim]
    return PopulationHebbian(
      popsize,
      sizes,
      init_noise=cfg["HEBB_init_wnoise"],
      norm_mode=cfg["HEBB_norm"],
      std_init_param=cfg["HEBB_std_init_param"],
    )
  raise ValueError(f"Unsupported ES model {model!r}; use 'ff' or 'hebb'.")
