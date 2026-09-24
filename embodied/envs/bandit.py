import functools

import embodied
import numpy as np


class Bandit(embodied.Env):
  """K-armed bandit that resamples reward mapping every episode.

  Task string format: ``variant_k`` where variant is one of
  ``ber``, ``lip``, ``pol``, ``gaus`` and k is the number of arms.
  For example ``ber_10`` gives a 10-arm Bernoulli bandit.
  If only a variant is given (e.g. ``ber``), k defaults to 11.

  Variants:
    ber   – Bernoulli: each arm has a random success probability.
    lip   – Lipschitz: arm means are a random walk with uniform slopes.
    pol   – Polynomial: arm means follow a random degree-3 polynomial.
    gaus  – Gaussian (RBF RFF): arm means sampled from a GP via random
            Fourier features.
  """

  VARIANTS = ('ber', 'lip', 'pol', 'gaus')

  def __init__(self, task, length=100, reward_noise=0.05, seed=None):
    parts = task.split('_')
    self._variant = parts[0]
    self._k = int(parts[1]) if len(parts) > 1 else 11
    assert self._variant in self.VARIANTS, (
        f'Unknown variant {self._variant!r}, choose from {self.VARIANTS}')

    self._length = length
    self._reward_noise = reward_noise
    self._step = 0
    self._done = True
    self._rng = np.random.RandomState(seed)
    self._objective = None
    self._prev_reward = 0.0

  # -- spaces ----------------------------------------------------------------

  @functools.cached_property
  def obs_space(self):
    return {
        'reward': embodied.Space(np.float32),
        'is_first': embodied.Space(bool),
        'is_last': embodied.Space(bool),
        'is_terminal': embodied.Space(bool),
        'observation': embodied.Space(np.float32, (1,), 0.0, 1.0),
        'prev_reward': embodied.Space(np.float32, (1,)),
    }

  @functools.cached_property
  def act_space(self):
    return {
        'reset': embodied.Space(bool),
        'action': embodied.Space(np.int32, (), 0, self._k),
    }

  # -- step ------------------------------------------------------------------

  def step(self, action):
    action = action.copy()
    reset = action.pop('reset')
    if reset or self._done:
      return self._reset()
    return self._step_env(action['action'])

  def _reset(self):
    self._step = 0
    self._done = False
    self._prev_reward = 0.0
    self._objective = self._sample_bandit()
    return {
        'observation': np.zeros((1,), dtype=np.float32),
        'prev_reward': np.array([0.0], dtype=np.float32),
        'reward': np.float32(0.0),
        'is_first': True,
        'is_last': False,
        'is_terminal': False,
    }

  def _step_env(self, action):
    action = int(action)
    reward = np.float32(self._objective(action, noise=1.0))
    prev_reward = np.array([self._prev_reward], dtype=np.float32)
    self._prev_reward = float(reward)
    self._step += 1
    done = self._step >= self._length
    self._done = done
    return {
        'observation': np.zeros((1,), dtype=np.float32),
        'prev_reward': prev_reward,
        'reward': reward,
        'is_first': False,
        'is_last': done,
        'is_terminal': False,
    }

  # -- bandit sampling -------------------------------------------------------

  def _sample_bandit(self):
    rng = self._rng
    k = self._k
    noise_std = self._reward_noise
    variant = self._variant

    if variant == 'ber':
      probs = rng.uniform(0, 1, size=k)
      def reward_fn(a, noise=0.0):
        if noise == 0:
          return probs[a]
        return float(rng.binomial(1, probs[a]))
      return reward_fn

    if variant == 'lip':
      slopes = rng.uniform(-1, 1, size=k - 1)
      mu = np.empty(k)
      mu[0] = 0.0
      for i in range(1, k):
        mu[i] = mu[i - 1] + slopes[i - 1]
      def reward_fn(a, noise=0.0):
        return mu[a] + noise * rng.normal(0.0, noise_std)
      return reward_fn

    if variant == 'pol':
      coeffs = rng.normal(0.0, 1.0, size=4)  # degree 3
      x = np.linspace(0.0, 1.0, k)
      mu = np.polyval(coeffs[::-1], x)
      span = mu.max() - mu.min()
      if span > 1e-8:
        mu = (mu - mu.min()) / span
      def reward_fn(a, noise=0.0):
        return mu[a] + noise * rng.normal(0.0, noise_std)
      return reward_fn

    if variant == 'gaus':
      rff_fn = self._sample_rff_function(rng, k)
      def reward_fn(a, noise=0.0):
        return rff_fn(a / k) + noise * rng.normal(0.0, noise_std)
      return reward_fn

  @staticmethod
  def _sample_rff_function(rng, k, num_features=500, length_scale=0.2):
    omega = rng.normal(scale=1.0 / length_scale, size=(num_features, 1))
    b = rng.uniform(0, 2 * np.pi, size=num_features)
    w = rng.normal(size=num_features)

    def f(x):
      x = np.atleast_2d(x)
      proj = x @ omega.T
      phi = np.sqrt(2.0 / num_features) * np.cos(proj + b)
      return float(phi @ w)

    return f
