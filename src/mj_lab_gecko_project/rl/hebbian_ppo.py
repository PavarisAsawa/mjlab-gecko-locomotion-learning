"""Hebbian network as a recurrent rsl-rl model, so PPO can train it (case 4).

Same network and plasticity rule as HebbianNet (hebbian_neural_net.py):
  post = tanh(pre @ W)
  W   <- norm(W + lr * (A * pre*post + B * pre + C * post + D))
but A, B, C, D, lr are nn.Parameters trained by gradient descent and shared by
all envs (with ES every population member has its own).

The plastic weights W are the recurrent hidden state: they start random
(U(-init_noise, init_noise)) every episode, are stored at each rollout step by
rsl-rl's RolloutStorage as a (1, num_envs, num_weights) tensor, and PPO
backpropagates through the Hebbian updates along each trajectory (truncated to
the rollout length). Each step is gradient-checkpointed to bound GPU memory.

Use as the actor: RslRlModelCfg(class_name="mj_lab_gecko_project.rl.hebbian_ppo:HebbianModel").
"""

from __future__ import annotations

import math

import torch
from tensordict import TensorDict
from torch import nn
from torch.utils.checkpoint import checkpoint

from rsl_rl.models import MLPModel
from rsl_rl.utils import unpad_trajectories

from .hebbian_neural_net import clip_weights, max_norm, var_norm

_NORMS = {"var": var_norm, "max": max_norm, "clip": clip_weights}


class HebbianModel(MLPModel):
  is_recurrent: bool = True

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (128, 64),
    activation: str = "tanh",  # Unused: the Hebbian network is tanh by definition.
    obs_normalization: bool = False,
    distribution_cfg: dict | None = None,
    init_noise: float = 0.04,
    norm_mode: str = "var",
    std_init_param: float = 0.025,
  ) -> None:
    del activation
    # The parent builds obs selection, normalizer and distribution (+ a throwaway MLP).
    super().__init__(
      obs, obs_groups, obs_set, output_dim, [8], "elu", obs_normalization, distribution_cfg
    )
    out_dim = self.distribution.input_dim if self.distribution is not None else output_dim
    if not isinstance(out_dim, int):
      raise ValueError("HebbianModel needs a distribution with an int input_dim "
                       "(e.g. GaussianDistribution with std_type='scalar').")
    self.mlp = nn.Identity()  # The Hebbian output is the distribution input.

    self.sizes = [self.obs_dim, *hidden_dims, out_dim]
    self._shapes = [(self.sizes[i], self.sizes[i + 1]) for i in range(len(self.sizes) - 1)]
    self._numel = [a * b for a, b in self._shapes]
    self.num_weights = sum(self._numel)
    self.init_noise = init_noise
    self.weight_norm = _NORMS[norm_mode]

    def coeffs() -> nn.ParameterList:
      return nn.ParameterList(
        nn.Parameter(torch.randn(a, b) * std_init_param) for a, b in self._shapes
      )

    self.A, self.B, self.C, self.D, self.lr = coeffs(), coeffs(), coeffs(), coeffs(), coeffs()
    self._num_envs = obs.batch_size[0]
    self.hidden_state: torch.Tensor | None = None

  # Hebbian dynamics.

  def _random_weights(self, n: int) -> torch.Tensor:
    device = self.A[0].device
    return torch.empty(1, n, self.num_weights, device=device).uniform_(
      -self.init_noise, self.init_noise
    )

  def _step(self, pre: torch.Tensor, flat_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One forward pass + Hebbian update. pre: (B, in), flat_w: (B, num_weights)."""
    new_w = []
    for i, w in enumerate(torch.split(flat_w, self._numel, dim=1)):
      w = w.reshape(-1, *self._shapes[i])
      pre_i = pre.unsqueeze(2)  # (B, in, 1)
      # = einsum("bi,bio->bo"). Written elementwise because the einsum's backward
      # (a batched outer product) dispatches to a Triton kernel that needs a C
      # compiler at runtime.
      post = torch.tanh((pre_i * w).sum(dim=1))
      post_j = post.unsqueeze(1)  # (B, 1, out)
      dw = self.lr[i] * (self.A[i] * pre_i * post_j + self.B[i] * pre_i + self.C[i] * post_j + self.D[i])
      new_w.append(self.weight_norm(w + dw).flatten(1))
      pre = post
    return pre, torch.cat(new_w, dim=1)

  def get_latent(
    self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state=None
  ) -> torch.Tensor:
    x = super().get_latent(obs)
    if masks is None:  # Rollout / inference: one step for every env.
      if self.hidden_state is None:
        self.hidden_state = self._random_weights(x.shape[0])
      out, w = self._step(x, self.hidden_state[0])
      self.hidden_state = w.unsqueeze(0)
      return out
    # PPO update: padded trajectories x (T, B, D), hidden_state = weights at their start.
    w = (hidden_state[0] if isinstance(hidden_state, (tuple, list)) else hidden_state)[0]
    outs = []
    for t in range(x.shape[0]):
      out, w = checkpoint(self._step, x[t], w, use_reentrant=False)
      outs.append(out)
    return unpad_trajectories(torch.stack(outs), masks)

  # Recurrent-model interface used by rsl-rl.

  def reset(self, dones: torch.Tensor | None = None, hidden_state=None) -> None:
    if dones is None:
      self.hidden_state = hidden_state
    elif self.hidden_state is not None:
      done = dones.bool().flatten()
      if done.any():
        self.hidden_state[:, done] = self._random_weights(int(done.sum()))

  def get_hidden_state(self) -> torch.Tensor:
    if self.hidden_state is None:
      self.hidden_state = self._random_weights(self._num_envs)
    return self.hidden_state

  def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
    if self.hidden_state is not None:
      self.hidden_state = self.hidden_state.detach()

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    raise NotImplementedError("ONNX export of the Hebbian model is not implemented.")

  def as_jit(self) -> nn.Module:
    raise NotImplementedError("JIT export of the Hebbian model is not implemented.")

  @property
  def num_coefficients(self) -> int:
    return 5 * self.num_weights


def count_params(sizes: list[int]) -> int:
  """Trainable coefficients of a Hebbian network with these layer sizes."""
  return 5 * sum(math.prod(s) for s in zip(sizes[:-1], sizes[1:]))
