import functools
import importlib
import inspect
import json
from collections.abc import Mapping

import embodied
import numpy as np
from embodied.envs._jax_image_env_mixin import JaxImageEnvMixin


class Craftax(JaxImageEnvMixin, embodied.Env):

  _render_obs_keys = ('image', 'pixels')

  def __init__(
      self, task, obs_key='image', act_key='action', seed=None,
      size=None, resize='pillow', render_size=None,
      cpu_in_workers=True, num_envs=0, log_image=False,
      achievement_reward_weights=None, **kwargs):
    self._task = task
    self._obs_key = obs_key
    self._act_key = act_key
    self._size = tuple(size) if size is not None else None
    self._resize = resize
    self._render_size = render_size
    self._cpu_in_workers = bool(cpu_in_workers)
    self._batched = bool(num_envs)
    self.num_envs = int(num_envs or 0)
    self._log_image = bool(log_image)
    self._done = True
    self._last_info = {}
    self._last_obs = None
    self._last_state = None
    self._last_achievements = None

    self._configure_jax_backend()
    self._jax = importlib.import_module('jax')
    with self._allow_host_to_device():
      self._rng = self._jax.random.PRNGKey(0 if seed is None else int(seed))

    factory = self._resolve_factory()
    self._env, self._params = self._create_env(factory, task, kwargs)
    self._renderer = self._resolve_renderer()
    self._configure_reward_recompute(achievement_reward_weights)
    if self._batched:
      self._init_batched_jax_mode()

  @property
  def info(self):
    return self._last_info

  @functools.cached_property
  def obs_space(self):
    spec = self._get_obs_spec()
    if spec is None:
      probe = self._reset_env()
      spec = probe['obs']
      self._done = True
      self._last_obs = None
      self._last_state = None
      self._last_achievements = None
    if isinstance(spec, Mapping) or hasattr(spec, 'spaces'):
      spaces = self._flatten_spaces(spec)
      spaces = {k: self._convert_space(v) for k, v in spaces.items()}
      spaces = {self._safe_obs_key(k): v for k, v in spaces.items()}
    else:
      space = self._convert_space(spec)
      if self._is_symbolic_array_obs_shape(space.shape):
        size = int(np.prod(space.shape, dtype=np.int64))
        dtype = self._infer_symbolic_dtype(self._probe_symbolic_obs(), space.dtype)
        spaces = {self._obs_key: embodied.Space(dtype, (size,))}
      else:
        spaces = {self._obs_key: space}
    if self._size:
      spaces = {
          k: self._resize_space(v) if self._looks_like_image_space(v) else v
          for k, v in spaces.items()}
    spaces = self._compact_image_spaces(spaces)
    if self._log_image:
      if self._size:
        shape = self._size + (3,)
      else:
        sample = np.asarray(self._probe_log_image())
        shape = sample.shape[1:] if (
            self._batched and sample.ndim == 4 and sample.shape[0] == self.num_envs
        ) else sample.shape
      spaces['log_image'] = embodied.Space(np.uint8, shape)
    return {
        **spaces,
        'reward': embodied.Space(np.float32),
        'is_first': embodied.Space(bool),
        'is_last': embodied.Space(bool),
        'is_terminal': embodied.Space(bool),
    }

  def _is_symbolic_array_obs_shape(self, shape):
    shape = tuple(shape)
    if not shape:
      return True
    if len(shape) >= 4 and self._batched and shape[0] == self.num_envs:
      shape = shape[1:]
    return not (len(shape) == 3 and shape[-1] in (1, 3, 4))

  def _probe_symbolic_obs(self):
    try:
      probe = self._reset_env()
      return np.asarray(probe['obs'])
    except Exception:
      return None
    finally:
      self._done = True
      self._last_obs = None
      self._last_state = None
      self._last_achievements = None
      self._last_info = {}

  def _probe_log_image(self):
    if self._batched and self._last_state is not None:
      return self._render_log_image()
    try:
      probe = self._reset_env()
      return self._render_frame(probe['state'], probe['obs'], self._params)
    finally:
      self._done = True
      self._last_obs = None
      self._last_state = None
      self._last_achievements = None
      self._last_info = {}

  def _infer_symbolic_dtype(self, probe, default_dtype):
    dtype = np.dtype(default_dtype)
    if probe is None:
      return dtype
    arr = np.asarray(probe)
    if arr.size == 0:
      return dtype
    if np.issubdtype(arr.dtype, np.floating):
      if not np.isfinite(arr).all():
        return dtype
      rounded = np.rint(arr)
      if not np.allclose(arr, rounded):
        return dtype
      mn = int(rounded.min())
      mx = int(rounded.max())
      if mn >= 0 and mx <= np.iinfo(np.uint8).max:
        return np.dtype(np.uint8)
      if mn >= 0 and mx <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
      if mn >= np.iinfo(np.int16).min and mx <= np.iinfo(np.int16).max:
        return np.dtype(np.int16)
      return np.dtype(np.int32)
    return arr.dtype

  @functools.cached_property
  def act_space(self):
    spec = self._get_act_spec()
    if isinstance(spec, Mapping) or hasattr(spec, 'spaces'):
      spaces = self._flatten_spaces(spec)
      spaces = {k: self._convert_space(v) for k, v in spaces.items()}
    else:
      spaces = {self._act_key: self._convert_space(spec)}
    spaces['reset'] = embodied.Space(bool)
    return spaces

  def step(self, action):
    if self._batched:
      return self._step_batched(action)
    if action['reset'] or self._done:
      self._done = False
      result = self._reset_env()
      self._last_state = result['state']
      self._last_info = result.get('info', {}) or {}
      self._last_obs = result['obs']
      self._set_last_achievements(self._last_state)
      return self._maybe_add_log_image(
          self._format_obs(result['obs'], 0.0, is_first=True))

    act = self._prepare_action(action)
    prev_achievements = self._last_achievements
    result = self._step_env(act)
    self._last_state = result['state']
    self._last_info = result.get('info', {}) or {}
    self._last_obs = result['obs']
    self._set_last_achievements(self._last_state)
    self._done = bool(result['done'])
    info = self._last_info
    is_terminal = bool(self._done)
    if 'discount' in info:
      try:
        is_terminal = bool(np.asarray(info['discount']).item() == 0)
      except Exception:
        pass
    reward = self._recompute_reward(result['reward'], prev_achievements, self._last_state)
    return self._maybe_add_log_image(self._format_obs(result['obs'], reward, is_last=self._done, is_terminal=is_terminal))

  def render(self):
    self._ensure_last_state()
    render_params = getattr(self, '_batch_params', None) if self._batched else self._params
    render_state = self._batch_item(self._last_state, 0) if self._batched else self._last_state
    render_obs = self._batch_item(self._last_obs, 0) if self._batched else self._last_obs
    render_params = self._batch_item(render_params, 0) if self._batched else render_params
    return self._render_frame(render_state, render_obs, render_params)

  def _first_batch_item(self, tree):
    return self._batch_item(tree, 0)

  def _batch_item(self, tree, index):
    if not self._batched:
      return tree
    try:
      return tree[index]
    except Exception:
      pass
    tree_util = getattr(self._jax, 'tree_util', None)
    if tree_util is None:
      return tree

    def _slice_leaf(x):
      try:
        arr = np.asarray(x)
      except Exception:
        return x
      if arr.ndim >= 1 and arr.shape[0] == self.num_envs:
        try:
          return x[index]
        except Exception:
          return arr[index]
      return x

    with self._allow_host_to_device():
      return tree_util.tree_map(_slice_leaf, tree)

  def close(self):
    if hasattr(self._env, 'close'):
      self._env.close()

  def _init_batched_jax_mode(self):
    self._jnp = importlib.import_module('jax.numpy')
    self._batch_params = self._hostify_tree(self._params)
    # Probe reset ordering once outside JIT.
    self._rng, key = self._jax.random.split(self._rng)
    reset_fn = getattr(self._env, 'reset', None) or getattr(self._env, 'reset_env', None)
    step_fn = getattr(self._env, 'step', None) or getattr(self._env, 'step_env', None)
    raw_reset = self._call_maybe_params_with(reset_fn, self._batch_params, key)
    first, second = raw_reset[:2]
    obs, state = self._disambiguate_reset(first, second)
    self._reset_obs_first = (obs is first)
    with self._allow_host_to_device():
      dummy_action = self._jnp.zeros(self.act_space[self._act_key].shape, self.act_space[self._act_key].dtype)
    self._rng, key = self._jax.random.split(self._rng)
    raw_step = self._call_maybe_params_with(step_fn, self._batch_params, key, state, dummy_action)
    if not isinstance(raw_step, tuple) or len(raw_step) not in (4, 5):
      raise TypeError(f'Unsupported Craftax step signature for batched mode: {type(raw_step)}')
    self._step_tuple_len = len(raw_step)

    self._jit_reset_batched = self._jax.jit(
        self._jax.vmap(self._reset_one_batched, in_axes=(None, 0)))
    self._jit_transition_batched = self._jax.jit(
        self._jax.vmap(self._transition_one_batched, in_axes=(None, 0, 0, 0, 0)))
    self._rng, rng = self._jax.random.split(self._rng)
    rngs = self._jax.random.split(rng, self.num_envs)
    with self._allow_host_to_device():
      self._states, _ = self._jit_reset_batched(self._batch_params, rngs)
    self._done_vec = np.ones((self.num_envs,), bool)
    self._set_last_achievements(self._states)

    # Warmup compile.
    with self._allow_host_to_device():
      dummy_reset = self._jnp.ones((self.num_envs,), dtype=bool)
      dummy_action = self._jnp.zeros(
          (self.num_envs,) + self.act_space[self._act_key].shape,
          dtype=self.act_space[self._act_key].dtype)
    self._rng, rng = self._jax.random.split(self._rng)
    rngs = self._jax.random.split(rng, self.num_envs)
    with self._allow_host_to_device():
      states, obs, rew, done, first = self._jit_transition_batched(
          self._batch_params, dummy_reset, rngs, self._states, dummy_action)
    self._jax.block_until_ready(rew)
    self._states = states
    self._last_obs = obs
    self._last_state = states

  def _reset_one_batched(self, params, key):
    fn = getattr(self._env, 'reset', None) or getattr(self._env, 'reset_env', None)
    result = self._call_maybe_params_with(fn, params, key)
    first, second = result[:2]
    if self._reset_obs_first:
      obs, state = first, second
    else:
      obs, state = second, first
    return state, obs

  def _step_one_batched(self, params, key, state, action):
    fn = getattr(self._env, 'step', None) or getattr(self._env, 'step_env', None)
    result = self._call_maybe_params_with(fn, params, key, state, action)
    if self._step_tuple_len == 5:
      obs, next_state, reward, done, _ = result
    else:
      obs, next_state, reward, done = result
    reward = self._jnp.asarray(reward, self._jnp.float32)
    done = self._jnp.asarray(done, bool)
    return next_state, obs, reward, done

  def _transition_one_batched(self, params, has_reset, key, state, action):
    def _do_reset(_):
      next_state, obs = self._reset_one_batched(params, key)
      return (
          next_state,
          obs,
          self._jnp.asarray(0.0, self._jnp.float32),
          self._jnp.asarray(False, bool),
          self._jnp.asarray(True, bool),
      )
    def _do_step(_):
      next_state, obs, reward, done = self._step_one_batched(params, key, state, action)
      return (next_state, obs, reward, done, self._jnp.asarray(False, bool))
    return self._jax.lax.cond(has_reset, _do_reset, _do_step, operand=None)

  def _step_batched(self, action):
    reset_mask = np.asarray(action['reset'], bool) | self._done_vec
    prev_achievements = self._last_achievements
    self._rng, rng = self._jax.random.split(self._rng)
    rngs = self._jax.random.split(rng, self.num_envs)
    act_dtype = self.act_space[self._act_key].dtype
    with self._allow_host_to_device():
      acts = self._jnp.asarray(action[self._act_key], dtype=act_dtype)
      reset_mask_jax = self._jnp.asarray(reset_mask, bool)
    with self._allow_host_to_device():
      states, obs, reward, done, is_first = self._jit_transition_batched(
          self._batch_params, reset_mask_jax, rngs, self._states, acts)
      reward = np.asarray(reward, np.float32)
      is_first = np.asarray(is_first, bool)
      reward = self._recompute_reward(
          reward, prev_achievements, states, reset_mask=is_first)
      self._states = states
      self._last_state = states
      self._last_obs = obs
      self._set_last_achievements(states)
      self._done_vec = np.asarray(done, bool)
      self._last_info = {}
      return self._maybe_add_log_image(self._format_obs_batched(
          obs,
          reward,
          is_first,
          self._done_vec,
          self._done_vec))

  def _configure_reward_recompute(self, reward_weights):
    self._base_achievement_reward_weights = None
    self._achievement_reward_weights = None
    if reward_weights in (None, '', {}, ()):
      return
    task_lower = str(self._task).lower()
    module_name = (
        'craftax.craftax_classic.constants'
        if 'classic' in task_lower else
        'craftax.craftax.constants')
    module = importlib.import_module(module_name)
    achievement_enum = getattr(module, 'Achievement')
    reward_map = getattr(module, 'ACHIEVEMENT_REWARD_MAP', None)
    if reward_map is None:
      size = 1 + max(int(achievement.value) for achievement in achievement_enum)
      reward_map = np.ones((size,), np.float32)
    reward_map = np.asarray(reward_map, np.float32)
    name_to_index = {
        achievement.name.lower(): int(achievement.value)
        for achievement in achievement_enum
    }
    custom_weights = reward_map.copy()
    for key, value in self._parse_reward_weights(reward_weights).items():
      index = self._resolve_achievement_index(key, name_to_index, len(reward_map))
      custom_weights[index] = np.float32(value)
    self._base_achievement_reward_weights = reward_map
    self._achievement_reward_weights = custom_weights

  def _parse_reward_weights(self, reward_weights):
    if isinstance(reward_weights, Mapping):
      return dict(reward_weights)
    if not isinstance(reward_weights, str):
      raise TypeError(
          'Craftax achievement reward weights must be a mapping or string.')
    reward_weights = reward_weights.strip()
    if not reward_weights:
      return {}
    try:
      parsed = json.loads(reward_weights)
    except json.JSONDecodeError:
      parsed = None
    if isinstance(parsed, Mapping):
      return dict(parsed)
    pairs = {}
    for item in reward_weights.split(','):
      item = item.strip()
      if not item:
        continue
      if '=' not in item:
        raise ValueError(
            'Craftax achievement reward weights must use JSON or '
            '`name=value,name=value` format.')
      name, value = item.split('=', 1)
      pairs[name.strip()] = float(value.strip())
    return pairs

  def _normalize_achievement_name(self, name):
    name = str(name).strip()
    if name.lower().startswith('achievements/'):
      name = name.split('/', 1)[1]
    return name.replace('-', '_').replace(' ', '_').lower()

  def _resolve_achievement_index(self, key, name_to_index, size):
    if isinstance(key, (int, np.integer)):
      index = int(key)
    else:
      text = str(key).strip()
      if text.isdigit():
        index = int(text)
      else:
        name = self._normalize_achievement_name(text)
        if name not in name_to_index:
          available = ', '.join(sorted(name_to_index))
          raise KeyError(
              f'Unknown Craftax achievement weight "{key}". '
              f'Expected an integer achievement id in [0, {size - 1}] or one of: {available}')
        return name_to_index[name]
    if not 0 <= index < size:
      raise KeyError(
          f'Craftax achievement id {index} is out of range [0, {size - 1}].')
    return index

  def _extract_achievements(self, state):
    if self._achievement_reward_weights is None or state is None:
      return None
    achievements = getattr(state, 'achievements', None)
    if achievements is not None:
      return np.asarray(achievements, np.float32)
    if isinstance(state, Mapping):
      if 'achievements' in state:
        return np.asarray(state['achievements'], np.float32)
      if 'state' in state:
        return self._extract_achievements(state['state'])
    inner_state = getattr(state, 'state', None)
    if inner_state is not None and inner_state is not state:
      return self._extract_achievements(inner_state)
    raise AttributeError('Craftax state does not expose achievements.')

  def _set_last_achievements(self, state):
    self._last_achievements = self._extract_achievements(state)

  def _recompute_reward(self, reward, prev_achievements, next_state, reset_mask=None):
    if self._achievement_reward_weights is None or prev_achievements is None:
      return reward
    next_achievements = self._extract_achievements(next_state)
    delta = next_achievements - np.asarray(prev_achievements, np.float32)
    base_reward = (delta * self._base_achievement_reward_weights).sum(-1)
    custom_reward = (delta * self._achievement_reward_weights).sum(-1)
    reward = np.asarray(reward, np.float32)
    shaped = reward - np.asarray(base_reward, np.float32) + np.asarray(
        custom_reward, np.float32)
    if reset_mask is not None:
      shaped = np.where(np.asarray(reset_mask, bool), reward, shaped)
    shaped = np.asarray(shaped, np.float32)
    return shaped.item() if shaped.ndim == 0 else shaped

  def _maybe_add_log_image(self, obs):
    if not self._log_image:
      return obs
    obs = dict(obs)
    obs['log_image'] = self._render_log_image()
    return obs

  def _ensure_last_state(self):
    if self._last_state is not None:
      return
    if self._batched:
      act_space = self.act_space[self._act_key]
      shape = (self.num_envs,) + tuple(act_space.shape)
      batched_zero = np.zeros(shape, act_space.dtype)
      self.step({'reset': np.ones((self.num_envs,), bool), self._act_key: batched_zero})
    else:
      self.step({'reset': True, self._act_key: self._zero_action()})

  def _render_log_image(self):
    self._ensure_last_state()
    if not self._batched:
      return self._render_frame(self._last_state, self._last_obs, self._params)
    render_params = getattr(self, '_batch_params', None)
    images = [
        self._render_frame(
            self._batch_item(self._last_state, i),
            self._batch_item(self._last_obs, i),
            self._batch_item(render_params, i))
        for i in range(self.num_envs)
    ]
    return np.stack(images, axis=0)

  def _render_frame(self, render_state, render_obs, render_params):
    image = self._image_from_obs(render_obs)
    if image is not None:
      return image
    if self._renderer is None:
      raise NotImplementedError(
          'Craftax rendering is unavailable for this task/version. '
          'Use a pixels task or install a Craftax build with a renderer.')
    render_state = self._hostify_tree(render_state)
    render_params = self._hostify_tree(render_params)
    render_obs = self._hostify_tree(render_obs)
    inner_state = getattr(render_state, 'state', None)
    if inner_state is not None:
      render_state = inner_state

    try:
      sig = inspect.signature(self._renderer)
      param_names = tuple(sig.parameters.keys())
    except (TypeError, ValueError):
      param_names = ()
    last_error = None
    if 'block_pixel_size' in param_names:
      block_sizes = []
      if self._render_size is not None:
        block_sizes.append(int(self._render_size))
      module = inspect.getmodule(self._renderer)
      textures = getattr(module, 'TEXTURES', None)
      if isinstance(textures, dict):
        for key in sorted(textures.keys()):
          if isinstance(key, (int, np.integer)) and int(key) not in block_sizes:
            block_sizes.append(int(key))
      block_sizes.extend([16, 8, 4, 2, 1])
      seen = set()
      for block_size in block_sizes:
        if block_size in seen:
          continue
        seen.add(block_size)
        kwargs = {'block_pixel_size': int(block_size)}
        if 'params' in param_names:
          kwargs['params'] = render_params
        try:
          image = self._renderer(render_state, **kwargs)
          if image is not None:
            return self._finalize_image(image)
        except Exception as e:
          last_error = e
        try:
          image = self._renderer(render_state, int(block_size))
          if image is not None:
            return self._finalize_image(image)
        except Exception as e:
          last_error = e
          continue

    for args in (
        (render_state, render_params),
        (render_params, render_state),
        (render_state,),
    ):
      try:
        image = self._renderer(*args)
        if image is not None:
          return self._finalize_image(image)
      except Exception as e:
        last_error = e
        continue
    if last_error is not None:
      raise RuntimeError(
          f'Could not call Craftax renderer with known signatures. Last error: {type(last_error).__name__}: {last_error}')
    raise RuntimeError('Could not call Craftax renderer with known signatures.')

  def _image_from_obs(self, obs):
    if obs is None:
      return None
    if isinstance(obs, Mapping):
      flat = self._flatten_values(obs)
      for key in (self._obs_key, 'image', 'pixels'):
        image = self._maybe_image(flat.get(key))
        if image is not None:
          return image
      for value in flat.values():
        image = self._maybe_image(value)
        if image is not None:
          return image
      return None
    return self._maybe_image(obs)

  def _maybe_image(self, value):
    if value is None:
      return None
    image = np.asarray(value)
    if not self._looks_like_image(image):
      return None
    return self._finalize_image(image)

  def _finalize_image(self, image):
    image = np.asarray(image)
    if (
        self._size and self._looks_like_image(image) and
        image.shape[:2] != self._size):
      image = self._resize_image(image)
    if image.dtype != np.uint8:
      image = image.astype(np.uint8)
    return image

  def _get_obs_spec(self):
    fn = getattr(self._env, 'observation_space', None)
    if fn is None:
      return None
    return self._call_maybe_params(fn)

  def _get_act_spec(self):
    fn = getattr(self._env, 'action_space', None)
    if fn is not None:
      return self._call_maybe_params(fn)
    if hasattr(self._env, 'num_actions'):
      num_actions = getattr(self._env, 'num_actions')
      if callable(num_actions):
        num_actions = self._call_maybe_params(num_actions)
      class _Discrete:
        n = int(num_actions)
      return _Discrete()
    raise AttributeError('Craftax env does not expose action_space() or num_actions.')

  def _prepare_action(self, action):
    act = action
    if isinstance(self.act_space[self._act_key], embodied.Space):
      if self.act_space[self._act_key].discrete:
        act = np.asarray(action[self._act_key]).astype(np.int32).item()
      else:
        act = np.asarray(action[self._act_key])
    return act

  def _reset_env(self):
    self._rng, key = self._jax.random.split(self._rng)
    fn = getattr(self._env, 'reset', None) or getattr(self._env, 'reset_env', None)
    if fn is None:
      raise AttributeError('Craftax env does not expose reset() or reset_env().')
    result = self._call_maybe_params(fn, key)
    if not isinstance(result, tuple) or len(result) < 2:
      raise TypeError(f'Unexpected Craftax reset return: {type(result)}')
    first, second = result[:2]
    obs, state = self._disambiguate_reset(first, second)
    info = result[2] if len(result) > 2 and isinstance(result[2], Mapping) else {}
    return {'obs': obs, 'state': state, 'info': info}

  def _step_env(self, action):
    self._rng, key = self._jax.random.split(self._rng)
    fn = getattr(self._env, 'step', None) or getattr(self._env, 'step_env', None)
    if fn is None:
      raise AttributeError('Craftax env does not expose step() or step_env().')
    result = self._call_maybe_params(fn, key, self._last_state, action)
    if not isinstance(result, tuple):
      raise TypeError(f'Unexpected Craftax step return: {type(result)}')
    if len(result) == 5:
      obs, state, reward, done, info = result
    elif len(result) == 4:
      obs, state, reward, done = result
      info = {}
    else:
      raise TypeError(f'Unexpected Craftax step return length: {len(result)}')
    return {
        'obs': obs,
        'state': state,
        'reward': np.asarray(reward).astype(np.float32).item(),
        'done': bool(np.asarray(done).item()),
        'info': dict(info) if isinstance(info, Mapping) else {},
    }

  def _resolve_factory(self):
    candidates = (
        ('craftax.craftax_env', 'make_craftax_env_from_name'),
        ('craftax', 'make_craftax_env_from_name'),
    )
    for module_name, fn_name in candidates:
      try:
        module = importlib.import_module(module_name)
      except ImportError:
        continue
      if hasattr(module, fn_name):
        return getattr(module, fn_name)
    raise ImportError(
        'Could not find Craftax env factory. Expected '
        '`craftax.craftax_env.make_craftax_env_from_name`.')

  def _create_env(self, factory, task, kwargs):
    kwargs = dict(kwargs)
    if 'auto_reset' not in kwargs:
      kwargs['auto_reset'] = False
    result = self._call_filtered(factory, task, **kwargs)
    if isinstance(result, tuple) and len(result) == 2:
      env, params = result
    else:
      env, params = result, None
    if params is None:
      default = getattr(env, 'default_params', None)
      if callable(default):
        params = default()
      else:
        params = default
    return env, params

  def _resolve_renderer(self):
    env_module = getattr(self._env.__class__, '__module__', '')
    preferred_modules = []
    task_lower = str(self._task).lower()
    if env_module:
      preferred_modules.append(env_module)
    if 'classic' in task_lower:
      preferred_modules.extend((
          'craftax.craftax_classic.renderer',
          'craftax.craftax.renderer',
          'craftax.renderer',
      ))
    else:
      preferred_modules.extend((
          'craftax.craftax.renderer',
          'craftax.craftax_classic.renderer',
          'craftax.renderer',
      ))
    deduped = []
    for name in preferred_modules:
      if name and name not in deduped:
        deduped.append(name)
    preferred_modules = deduped

    for module_name in preferred_modules:
      try:
        module = importlib.import_module(module_name)
      except ImportError:
        continue
      for fn_name in ('render_craftax_pixels', 'render_pixels'):
        if hasattr(module, fn_name):
          fn = getattr(module, fn_name)
          if self._render_size is None:
            return fn
          return lambda *args, _fn=fn, **kw: _fn(*args, block_pixel_size=self._render_size, **kw)
    # Generic `render` often refers to jitted env methods with static args and can
    # fail when called from wrappers. Use it only as a final fallback.
    for module_name in preferred_modules:
      try:
        module = importlib.import_module(module_name)
      except ImportError:
        continue
      if hasattr(module, 'render'):
        return getattr(module, 'render')
    if hasattr(self._env, 'render'):
      return self._env.render
    return None

  def _call_maybe_params(self, fn, *args):
    return self._call_maybe_params_with(fn, self._params, *args)

  def _call_maybe_params_with(self, fn, params, *args):
    with self._allow_host_to_device():
      if params is None:
        return fn(*args)
      try:
        sig = inspect.signature(fn)
      except (TypeError, ValueError):
        try:
          return fn(*args, params)
        except TypeError:
          return fn(*args)
      kinds = {p.kind for p in sig.parameters.values()}
      if inspect.Parameter.VAR_POSITIONAL in kinds:
        return fn(*args, params)
      positional = [
          p for p in sig.parameters.values()
          if p.kind in (
              inspect.Parameter.POSITIONAL_ONLY,
              inspect.Parameter.POSITIONAL_OR_KEYWORD)]
      if len(positional) >= len(args) + 1:
        return fn(*args, params)
      return fn(*args)

  def _disambiguate_reset(self, first, second):
    if self._looks_like_obs(first) and not self._looks_like_obs(second):
      return first, second
    if self._looks_like_obs(second) and not self._looks_like_obs(first):
      return second, first
    return first, second

  def _looks_like_obs(self, value):
    if isinstance(value, Mapping):
      return True
    value = np.asarray(value)
    return value.ndim >= 1
