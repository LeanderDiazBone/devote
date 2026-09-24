import functools
import importlib
import inspect
from collections.abc import Mapping
import math as pymath

import embodied
import numpy as np
from embodied.envs._jax_image_env_mixin import JaxImageEnvMixin


class XLandMiniGrid(JaxImageEnvMixin, embodied.Env):

  def __init__(
      self, task, obs_key='image', act_key='action', seed=None,
      image=True, size=(64, 64), resize='pillow',
      benchmark=None, benchmark_name=None, sample_ruleset=False, ruleset=None,
      cpu_in_workers=True, num_envs=0, **kwargs):
    self._task = task
    self._obs_key = obs_key
    self._act_key = act_key
    self._image = bool(image)
    self._size = tuple(size) if size is not None else None
    self._resize = resize
    self._done = True
    self._batched = bool(num_envs)
    self.num_envs = int(num_envs or 0)
    self._last_info = {}
    self._last_obs = None
    self._last_timestep = None
    self._last_params = None
    self._cpu_in_workers = bool(cpu_in_workers)
    self._batched_params_per_env = False

    self._configure_jax_backend()
    self._jax = importlib.import_module('jax')
    self._xminigrid = importlib.import_module('xminigrid')
    with self._allow_host_to_device():
      self._rng = self._jax.random.PRNGKey(0 if seed is None else int(seed))

    self._env, self._base_params = self._create_env(task, kwargs)
    if self._image:
      self._env = self._maybe_wrap_image(self._env)
    self._benchmark = self._load_benchmark(benchmark or benchmark_name)
    self._sample_ruleset = bool(sample_ruleset or self._benchmark is not None)
    self._fixed_ruleset = ruleset
    self._renderer = getattr(self._env, 'render', None)
    if self._batched:
      self._init_batched_jax_mode()

  def _is_symbolic_array_obs_shape(self, shape):
    if self._image:
      return False
    shape = tuple(shape)
    if not shape:
      return True
    if len(shape) >= 4 and self._batched and shape[0] == self.num_envs:
      shape = shape[1:]
    if len(shape) == 3 and shape[-1] in (1, 3, 4):
      return False
    return True

  def _symbolic_obs_key(self):
    return 'state' if self._obs_key == 'image' else self._obs_key

  def _is_xland_task(self, task):
    return str(task).startswith('XLand-MiniGrid')

  def _flatten_symbolic_obs(self, arr):
    arr = np.asarray(arr)
    if self._batched and arr.ndim >= 2 and arr.shape[0] == self.num_envs:
      return arr.reshape((arr.shape[0], int(pymath.prod(arr.shape[1:]))))
    return arr.reshape((int(pymath.prod(arr.shape)),))

  def _probe_symbolic_obs(self):
    try:
      probe = self._reset_env()
      return np.asarray(probe['obs'])
    except Exception:
      return None
    finally:
      self._done = True
      self._last_obs = None
      self._last_timestep = None
      self._last_params = None
      self._last_info = {}

  def _infer_symbolic_dtype(self, probe, default_dtype):
    del default_dtype
    if probe is None:
      return np.dtype(np.float16)
    arr = np.asarray(probe)
    if arr.size == 0:
      return np.dtype(np.float16)
    if np.issubdtype(arr.dtype, np.number):
      finite = np.isfinite(arr)
      if finite.all():
        maxabs = np.abs(arr).max() if arr.size else 0.0
        if maxabs <= np.finfo(np.float16).max:
          return np.dtype(np.float16)
    return np.dtype(np.float32)

  def _first_batch_item(self, tree):
    if not self._batched:
      return tree
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
          return x[0]
        except Exception:
          return arr[0]
      return x

    with self._allow_host_to_device():
      return tree_util.tree_map(_slice_leaf, tree)

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
      self._last_timestep = None
      self._last_params = None
    if isinstance(spec, Mapping) or hasattr(spec, 'spaces'):
      spaces = self._flatten_spaces(spec)
      spaces = {k: self._convert_space(v) for k, v in spaces.items()}
      spaces = {self._safe_obs_key(k): v for k, v in spaces.items()}
    else:
      space = self._convert_space(spec)
      if self._is_symbolic_array_obs_shape(space.shape):
        size = int(np.prod(space.shape, dtype=np.int64))
        dtype = self._infer_symbolic_dtype(self._probe_symbolic_obs(), space.dtype)
        spaces = {self._symbolic_obs_key(): embodied.Space(dtype, (size,))}
      else:
        spaces = {self._obs_key: space}
    if self._size:
      spaces = {
          k: self._resize_space(v) if self._looks_like_image_space(v) else v
          for k, v in spaces.items()}
    spaces = self._compact_image_spaces(spaces)
    return {
        **spaces,
        'reward': embodied.Space(np.float32),
        'is_first': embodied.Space(bool),
        'is_last': embodied.Space(bool),
        'is_terminal': embodied.Space(bool),
    }

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
      self._last_timestep = result['timestep']
      self._last_params = result['params']
      self._last_info = result.get('info', {}) or {}
      self._last_obs = result['obs']
      return self._format_obs(result['obs'], 0.0, is_first=True)

    act = self._prepare_action(action)
    result = self._step_env(act)
    self._last_timestep = result['timestep']
    self._last_params = result['params']
    self._last_info = result.get('info', {}) or {}
    self._last_obs = result['obs']
    self._done = bool(result['done'])
    is_terminal = bool(self._done)
    discount = self._last_info.get('discount', None)
    if discount is not None:
      try:
        is_terminal = bool(np.asarray(discount).item() == 0)
      except Exception:
        pass
    return self._format_obs(
        result['obs'], result['reward'],
        is_last=self._done,
        is_terminal=is_terminal)

  def render(self):
    if self._batched and self._last_params is None and hasattr(self, '_batch_params'):
      self._last_params = self._batch_params
    if (not self._batched) and (self._last_timestep is None or self._last_params is None):
      result = self._reset_env()
      self._last_timestep = result['timestep']
      self._last_params = result['params']
      self._last_info = result.get('info', {}) or {}
      self._last_obs = result['obs']
      self._done = True

    if self._batched and self._last_obs is not None:
      obs = self._last_obs
      if isinstance(obs, Mapping):
        flat = self._flatten_values(obs)
        for key in (self._obs_key, 'image', 'pixels', 'rgb'):
          if key in flat:
            image = np.asarray(flat[key])[0]
            if self._looks_like_image(image):
              return self._resize_image(image) if self._size else image
      else:
        image = np.asarray(obs)[0]
        if self._looks_like_image(image):
          return self._resize_image(image) if self._size else image
    image = self._render_from_obs()
    if image is not None:
      return image
    if self._renderer is None:
      raise NotImplementedError(
          'XLand-MiniGrid rendering is unavailable. Enable image observations '
          'or use an xminigrid build exposing env.render().')
    render_params = self._last_params
    render_timestep = self._last_timestep
    if self._batched:
      render_params = self._first_batch_item(render_params)
      render_timestep = self._first_batch_item(render_timestep)
    state = getattr(render_timestep, 'state', None)
    for args in (
        (render_params, render_timestep),
        (render_params, state),
        (render_params,),
    ):
      try:
        image = self._renderer(*args)
        if image is not None:
          image = np.asarray(image)
          return self._resize_image(image) if self._size and self._looks_like_image(image) else image
      except TypeError:
        continue
    raise RuntimeError('Could not call xminigrid render() with known signatures.')

  def _flatten_obs_values(self, obs):
    if not isinstance(obs, Mapping):
      arr = np.asarray(obs)
      if self._is_symbolic_array_obs_shape(arr.shape):
        return {self._symbolic_obs_key(): self._flatten_symbolic_obs(arr)}
    return super()._flatten_obs_values(obs)

  def close(self):
    if hasattr(self._env, 'close'):
      self._env.close()

  def _init_batched_jax_mode(self):
    has_fast_jax = all(
        callable(getattr(self._jax, name, None))
        for name in ('jit', 'vmap'))
    if (self._sample_ruleset and self._fixed_ruleset is None and
        self._benchmark is not None and not has_fast_jax):
      self._init_batched_loop_mode()
      return
    self._jnp = importlib.import_module('jax.numpy')
    self._reset_fn = getattr(self._env, 'reset')
    self._step_fn = getattr(self._env, 'step')

    sampled_benchmark = (
        self._sample_ruleset and
        self._fixed_ruleset is None and
        self._benchmark is not None)
    if sampled_benchmark:
      self._batched_params_per_env = True
      self._jit_reset_batched = self._jax.jit(
          self._jax.vmap(self._reset_sampled_one_batched, in_axes=0))
      self._jit_transition_batched = self._jax.jit(
          self._jax.vmap(
              self._transition_sampled_one_batched,
              in_axes=(0, 0, 0, 0, 0)))
    else:
      self._batch_params = self._hostify_tree(self._resolve_params_for_reset())
      self._jit_reset_batched = self._jax.jit(
          self._jax.vmap(self._reset_one_batched, in_axes=(None, 0)))
      self._jit_transition_batched = self._jax.jit(
          self._jax.vmap(self._transition_one_batched, in_axes=(None, 0, 0, 0, 0)))

    self._rng, rng = self._jax.random.split(self._rng)
    rngs = self._jax.random.split(rng, self.num_envs)
    with self._allow_host_to_device():
      if self._batched_params_per_env:
        self._batch_params, self._timesteps = self._jit_reset_batched(rngs)
      else:
        self._timesteps = self._jit_reset_batched(self._batch_params, rngs)
    self._done_vec = np.ones((self.num_envs,), bool)
    # Warmup compile on representative shapes.
    with self._allow_host_to_device():
      dummy_reset = self._jnp.ones((self.num_envs,), dtype=bool)
      dummy_action = self._jnp.zeros((self.num_envs,) + self.act_space[self._act_key].shape,
                                     dtype=self.act_space[self._act_key].dtype)
    _, rng = self._jax.random.split(self._rng)
    rngs = self._jax.random.split(rng, self.num_envs)
    with self._allow_host_to_device():
      if self._batched_params_per_env:
        _, ts, _ = self._jit_transition_batched(
            self._batch_params, dummy_reset, rngs, self._timesteps, dummy_action)
      else:
        ts, _ = self._jit_transition_batched(
            self._batch_params, dummy_reset, rngs, self._timesteps, dummy_action)
    self._jax.block_until_ready(getattr(ts, 'reward', 0))
    self._last_timestep = self._timesteps
    self._last_params = self._batch_params
    self._batched_loop = False

  def _init_batched_loop_mode(self):
    self._batched_loop = True
    self._params_list = [
        self._hostify_tree(self._resolve_params_for_reset())
        for _ in range(self.num_envs)
    ]
    self._timesteps = [None] * self.num_envs
    self._done_vec = np.ones((self.num_envs,), bool)
    self._last_timestep = None
    self._last_params = None
    self._last_obs = None

  def _stack_batched_leaves(self, items):
    first = items[0]
    if isinstance(first, Mapping):
      return {
          key: self._stack_batched_leaves([item[key] for item in items])
          for key in first
      }
    return np.stack([np.asarray(item) for item in items], axis=0)

  def _reset_one_batched(self, params, rng):
    return self._call_xminigrid_reset(self._reset_fn, params, rng)

  def _sample_params_for_batched_reset(self, rng):
    params = self._base_params
    reset_rng = rng
    if self._fixed_ruleset is not None:
      return self._replace_param(params, ruleset=self._fixed_ruleset), reset_rng
    if self._sample_ruleset and self._benchmark is not None:
      rule_rng, reset_rng = self._jax.random.split(rng)
      sampler = getattr(self._benchmark, 'sample_ruleset', None)
      if sampler is None:
        raise AttributeError('Benchmark does not expose sample_ruleset().')
      ruleset = sampler(rule_rng)
      params = self._replace_param(params, ruleset=ruleset)
    return params, reset_rng

  def _reset_sampled_one_batched(self, rng):
    params, reset_rng = self._sample_params_for_batched_reset(rng)
    timestep = self._call_xminigrid_reset(self._reset_fn, params, reset_rng)
    return params, timestep

  def _step_one_batched(self, params, timestep, action):
    return self._call_xminigrid_step(self._step_fn, params, timestep, action)

  def _transition_one_batched(self, params, has_reset, rng, timestep, action):
    def _do_reset(_):
      return self._reset_one_batched(params, rng), self._jnp.bool_(True)
    def _do_step(_):
      return self._step_one_batched(params, timestep, action), self._jnp.bool_(False)
    return self._jax.lax.cond(has_reset, _do_reset, _do_step, operand=None)

  def _transition_sampled_one_batched(self, params, has_reset, rng, timestep, action):
    def _do_reset(_):
      params, reset_rng = self._sample_params_for_batched_reset(rng)
      timestep = self._reset_one_batched(params, reset_rng)
      return params, timestep, self._jnp.bool_(True)

    def _do_step(_):
      return params, self._step_one_batched(params, timestep, action), self._jnp.bool_(False)

    return self._jax.lax.cond(has_reset, _do_reset, _do_step, operand=None)

  def _step_batched(self, action):
    if getattr(self, '_batched_loop', False):
      return self._step_batched_loop(action)
    resets = np.asarray(action['reset'], bool)
    reset_mask = resets | self._done_vec
    self._rng, rng = self._jax.random.split(self._rng)
    rngs = self._jax.random.split(rng, self.num_envs)
    with self._allow_host_to_device():
      acts = self._jnp.asarray(action[self._act_key], dtype=self.act_space[self._act_key].dtype)
      reset_mask_jax = self._jnp.asarray(reset_mask, bool)
    with self._allow_host_to_device():
      if self._batched_params_per_env:
        self._batch_params, self._timesteps, is_first = self._jit_transition_batched(
            self._batch_params, reset_mask_jax, rngs, self._timesteps, acts)
      else:
        self._timesteps, is_first = self._jit_transition_batched(
            self._batch_params, reset_mask_jax, rngs, self._timesteps, acts)
      parsed = self._parse_timestep_batched(self._timesteps)
      self._done_vec = np.asarray(parsed['done'], bool)
      self._last_timestep = self._timesteps
      self._last_params = self._batch_params
      self._last_obs = parsed['obs']
      self._last_info = {'discount': np.asarray(parsed['discount'])} if parsed['discount'] is not None else {}
      return self._format_obs_batched(
          parsed['obs'], parsed['reward'],
          is_first=np.asarray(is_first, bool),
          is_last=self._done_vec,
          is_terminal=np.asarray(parsed['is_terminal'], bool))

  def _step_batched_loop(self, action):
    resets = np.asarray(action['reset'], bool)
    reset_mask = resets | self._done_vec
    act_values = np.asarray(action[self._act_key])

    transitions = []
    infos = []
    done_vec = np.zeros((self.num_envs,), bool)

    for index in range(self.num_envs):
      if reset_mask[index]:
        params = self._hostify_tree(self._resolve_params_for_reset())
        self._params_list[index] = params
        self._rng, key = self._jax.random.split(self._rng)
        timestep = self._call_xminigrid_reset(getattr(self._env, 'reset'), params, key)
        self._timesteps[index] = timestep
        parsed = self._parse_timestep(timestep)
        infos.append(parsed['info'])
        done_vec[index] = False
        transitions.append(self._format_obs(parsed['obs'], 0.0, is_first=True))
      else:
        params = self._params_list[index]
        act = act_values[index]
        if self.act_space[self._act_key].discrete:
          act = np.asarray(act).astype(np.int32).item()
        timestep = self._call_xminigrid_step(
            getattr(self._env, 'step'), params, self._timesteps[index], act)
        self._timesteps[index] = timestep
        parsed = self._parse_timestep(timestep)
        infos.append(parsed['info'])
        discount = parsed['info'].get('discount', None)
        if discount is not None:
          try:
            is_terminal = bool(np.asarray(discount).item() == 0)
          except Exception:
            is_terminal = bool(parsed['done'])
        else:
          is_terminal = bool(parsed['done'])
        done_vec[index] = bool(parsed['done'])
        transitions.append(self._format_obs(
            parsed['obs'], parsed['reward'],
            is_last=bool(parsed['done']),
            is_terminal=is_terminal))

    self._done_vec = done_vec
    self._last_timestep = list(self._timesteps)
    self._last_params = list(self._params_list)
    self._last_obs = self._stack_batched_leaves([
        {k: v for k, v in tran.items() if k not in ('reward', 'is_first', 'is_last', 'is_terminal')}
        for tran in transitions
    ])
    discounts = [info.get('discount', None) for info in infos]
    if any(discount is not None for discount in discounts):
      self._last_info = {
          'discount': np.asarray([
              1.0 if discount is None else np.asarray(discount).item()
              for discount in discounts
          ], np.float32)
      }
    else:
      self._last_info = {}
    return self._stack_batched_leaves(transitions)

  def _create_env(self, task, kwargs):
    kwargs = dict(kwargs)
    make = getattr(self._xminigrid, 'make')
    result = self._call_filtered(make, task, **kwargs)
    if not (isinstance(result, tuple) and len(result) == 2):
      raise TypeError(
          'xminigrid.make() must return (env, env_params); '
          f'got {type(result)}')
    return result

  def _maybe_wrap_image(self, env):
    candidates = (
      ('xminigrid.experimental.img_obs', 'RGBImgObservationWrapper'),
      ('xminigrid.experimental.img_obs', 'RGBImgObsWrapper'),
    )
    for module_name, cls_name in candidates:
      try:
        module = importlib.import_module(module_name)
      except ImportError:
        continue
      if hasattr(module, cls_name):
        wrapper = getattr(module, cls_name)
        try:
          return wrapper(env)
        except TypeError:
          return self._call_filtered(wrapper, env)
    return env

  def _load_benchmark(self, name):
    if not name:
      return None
    fn = getattr(self._xminigrid, 'load_benchmark', None)
    if fn is None:
      raise AttributeError('xminigrid.load_benchmark() not found.')
    try:
      return fn(name)
    except TypeError:
      return self._call_filtered(fn, name=name)

  def _resolve_params_for_reset(self):
    params = self._base_params
    if self._fixed_ruleset is None and not self._sample_ruleset:
      return params
    if self._benchmark is None and self._fixed_ruleset is None:
      return params
    ruleset = self._fixed_ruleset
    if ruleset is None:
      self._rng, rule_key = self._jax.random.split(self._rng)
      sampler = getattr(self._benchmark, 'sample_ruleset', None)
      if sampler is None:
        raise AttributeError('Benchmark does not expose sample_ruleset().')
      ruleset = sampler(rule_key)
    return self._replace_param(params, ruleset=ruleset)

  def _replace_param(self, params, **updates):
    # Filter out fields not present on the params dataclass (e.g. MiniGrid
    # EnvParams lacks 'ruleset' which is XLand-MiniGrid specific).
    fields = getattr(params, '__dataclass_fields__', None)
    if fields is not None:
      updates = {k: v for k, v in updates.items() if k in fields}
    if not updates:
      return params
    if hasattr(params, 'replace'):
      return params.replace(**updates)
    if hasattr(params, '_replace'):
      return params._replace(**updates)
    raise AttributeError('Env params object does not support replace().')

  def _get_obs_spec(self):
    fn = getattr(self._env, 'observation_space', None)
    if fn is not None:
      return self._call_with_optional_params(fn, self._base_params)
    fn = getattr(self._env, 'observation_shape', None)
    if fn is not None:
      shape = self._call_with_optional_params(fn, self._base_params)
      dtype = getattr(self._env, 'observation_dtype', None)
      if callable(dtype):
        dtype = self._call_with_optional_params(dtype, self._base_params)
      dtype = np.float32 if dtype is None else dtype
      return _ArraySpec(shape=shape, dtype=dtype)
    return None

  def _get_act_spec(self):
    fn = getattr(self._env, 'action_space', None)
    if fn is not None:
      return self._call_with_optional_params(fn, self._base_params)
    for name in ('num_actions', 'n_actions'):
      if hasattr(self._env, name):
        value = getattr(self._env, name)
        if callable(value):
          value = self._call_with_optional_params(value, self._base_params)
        return _DiscreteSpec(int(value))
    raise AttributeError('xminigrid env does not expose action_space() or num_actions.')

  def _prepare_action(self, action):
    space = self.act_space[self._act_key]
    if space.discrete:
      return np.asarray(action[self._act_key]).astype(np.int32).item()
    return np.asarray(action[self._act_key])

  def _reset_env(self):
    params = self._resolve_params_for_reset()
    self._rng, key = self._jax.random.split(self._rng)
    reset = getattr(self._env, 'reset', None)
    if reset is None:
      raise AttributeError('xminigrid env does not expose reset().')
    timestep = self._call_xminigrid_reset(reset, params, key)
    parsed = self._parse_timestep(timestep)
    return {
        'timestep': timestep,
        'params': params,
        'obs': parsed['obs'],
        'reward': parsed['reward'],
        'done': parsed['done'],
        'info': parsed['info'],
    }

  def _step_env(self, action):
    step = getattr(self._env, 'step', None)
    if step is None:
      raise AttributeError('xminigrid env does not expose step().')
    params = self._last_params if self._last_params is not None else self._base_params
    timestep = self._call_xminigrid_step(step, params, self._last_timestep, action)
    parsed = self._parse_timestep(timestep)
    return {
        'timestep': timestep,
        'params': params,
        'obs': parsed['obs'],
        'reward': parsed['reward'],
        'done': parsed['done'],
        'info': parsed['info'],
    }

  def _call_xminigrid_reset(self, fn, params, key):
    with self._allow_host_to_device():
      for call in (
          lambda: fn(params, key),
          lambda: fn(env_params=params, key=key),
          lambda: fn(key, params),
          lambda: fn(key=key, env_params=params),
          lambda: fn(key=key, params=params),
      ):
        try:
          return call()
        except TypeError:
          continue
    raise RuntimeError('Could not call xminigrid reset() with known signatures.')

  def _call_xminigrid_step(self, fn, params, timestep, action):
    with self._allow_host_to_device():
      for call in (
          lambda: fn(params, timestep, action=action),
          lambda: fn(env_params=params, timestep=timestep, action=action),
          lambda: fn(params, timestep, action),
          lambda: fn(timestep=timestep, env_params=params, action=action),
      ):
        try:
          return call()
        except TypeError:
          continue
    raise RuntimeError('Could not call xminigrid step() with known signatures.')

  def _parse_timestep(self, timestep):
    obs = getattr(timestep, 'observation', None)
    if obs is None and isinstance(timestep, Mapping):
      obs = timestep.get('observation')
    reward = getattr(timestep, 'reward', 0.0)
    discount = getattr(timestep, 'discount', None)
    step_type = getattr(timestep, 'step_type', None)
    info = {}
    if discount is not None:
      info['discount'] = np.asarray(discount)
    if step_type is not None:
      info['step_type'] = step_type
    return {
        'obs': obs,
        'reward': np.asarray(reward).astype(np.float32).item(),
        'done': self._is_last_timestep(timestep, discount, step_type),
        'info': info,
    }

  def _parse_timestep_batched(self, timestep):
    obs = getattr(timestep, 'observation', None)
    if obs is None and isinstance(timestep, Mapping):
      obs = timestep.get('observation')
    reward = np.asarray(getattr(timestep, 'reward', 0.0), np.float32)
    reward = reward.reshape((self.num_envs,) + reward.shape[1:]) if reward.ndim > 1 else reward.reshape(self.num_envs)
    discount = getattr(timestep, 'discount', None)
    discount_np = None if discount is None else np.asarray(discount)
    done = None
    last_fn = getattr(timestep, 'last', None)
    if callable(last_fn):
      try:
        done = np.asarray(last_fn(), bool).reshape(self.num_envs)
      except Exception:
        done = None
    if done is None:
      done_field = getattr(timestep, 'done', None)
      if done_field is not None:
        try:
          done = np.asarray(done_field, bool).reshape(self.num_envs)
        except Exception:
          done = None
    if done is None:
      step_type = getattr(timestep, 'step_type', None)
      if step_type is not None:
        for candidate in (
            step_type,
            getattr(step_type, 'value', None),
            getattr(step_type, 'values', None),
        ):
          if candidate is None:
            continue
          try:
            values = np.asarray(candidate).reshape(self.num_envs)
          except Exception:
            continue
          if values.dtype.kind in 'OUS':
            # Some xminigrid builds expose enum names instead of integer codes.
            done = np.asarray([str(x).upper().endswith('LAST') for x in values], bool)
          else:
            done = (values.astype(np.int32) == 2)
          break
    if done is None:
      if discount_np is not None:
        try:
          done = (np.asarray(discount_np).reshape(self.num_envs) == 0)
        except Exception:
          done = None
    if done is None:
      done = np.zeros((self.num_envs,), bool)
    if discount_np is not None:
      try:
        is_terminal = (np.asarray(discount_np).reshape(self.num_envs) == 0)
      except Exception:
        is_terminal = done.copy()
    else:
      is_terminal = done.copy()
    return {
        'obs': obs,
        'reward': reward.astype(np.float32),
        'discount': None if discount_np is None else np.asarray(discount_np),
        'done': done.astype(bool),
        'is_terminal': is_terminal.astype(bool),
    }

  def _is_last_timestep(self, timestep, discount=None, step_type=None):
    for obj in (timestep, step_type):
      if obj is None:
        continue
      for name in ('last', 'is_last'):
        fn = getattr(obj, name, None)
        if callable(fn):
          try:
            return bool(np.asarray(fn()).item())
          except Exception:
            pass
    if step_type is not None:
      name = getattr(step_type, 'name', None)
      if isinstance(name, str):
        return name.upper() == 'LAST'
      try:
        value = int(np.asarray(step_type).item())
        if value in (2,):
          return True
      except Exception:
        pass
    if discount is not None:
      try:
        return bool(np.asarray(discount).item() == 0)
      except Exception:
        pass
    done = getattr(timestep, 'done', None)
    if done is not None:
      try:
        return bool(np.asarray(done).item())
      except Exception:
        pass
    return False

  def _call_with_optional_params(self, fn, params):
    try:
      sig = inspect.signature(fn)
    except (TypeError, ValueError):
      try:
        return fn(params)
      except TypeError:
        return fn()
    kinds = {p.kind for p in sig.parameters.values()}
    if inspect.Parameter.VAR_POSITIONAL in kinds:
      return fn(params)
    positional = [
        p for p in sig.parameters.values()
        if p.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    if len(positional) >= 1:
      return fn(params)
    return fn()

class _ArraySpec:

  def __init__(self, shape, dtype):
    self.shape = tuple(shape)
    self.dtype = np.dtype(dtype)


class _DiscreteSpec:

  def __init__(self, n):
    self.n = int(n)
