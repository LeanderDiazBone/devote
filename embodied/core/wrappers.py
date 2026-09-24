import functools
import time

import numpy as np

from . import base
from . import space as spacelib
from .tolerance_reward import ToleranceReward


class TimeLimit(base.Wrapper):

    def __init__(self, env, duration, reset=True):
        super().__init__(env)
        self._duration = duration
        self._reset = reset
        self._step = 0
        self._done = False

    def step(self, action):
        if action['reset'] or self._done:
            self._step = 0
            self._done = False
            if self._reset:
                action.update(reset=True)
                return self.env.step(action)
            else:
                action.update(reset=False)
                obs = self.env.step(action)
                obs['is_first'] = True
                return obs
        self._step += 1
        obs = self.env.step(action)
        if self._duration and self._step >= self._duration:
            obs['is_last'] = True
        self._done = obs['is_last']
        return obs


class ActionCost(base.Wrapper):
    def __init__(self, env, action_cost: float = 0.0, key='action', use_tolerance_reward: bool = False):
        super().__init__(env)
        self.action_cost = action_cost
        self._act_key = key
        self._use_tolerance_reward = use_tolerance_reward
        self.tolerance_reward = ToleranceReward(bounds=(0.0, 0.1), value_at_margin=0.0, sigmoid='gaussian', margin=0.1)

    def step(self, action):
        obs = self.env.step(action)
        if self._use_tolerance_reward:
            action_reward = self.tolerance_reward(np.linalg.norm(action[self._act_key])) * self.action_cost
            reward = obs['reward'] + action_reward
        else:
            action_cost = np.linalg.norm(action[self._act_key], axis=-1) * self.action_cost
            reward = obs['reward'] - action_cost
        obs['reward'] = reward.astype(obs['reward'].dtype)
        return obs


class DynamicsComplexity(base.Wrapper):

    _SKIP_KEYS = frozenset(('reward', 'is_first', 'is_last', 'is_terminal'))

    def __init__(
            self, env, dims=0, key='state', source_keys='auto',
            mode='append', seed=0, features=128, amplitude=1.0,
            length_scale=1.0, input_scale=1.0, noise_std=0.0):
        super().__init__(env)
        self._dims = int(dims)
        self._key = key
        self._mode = mode
        self._features = max(1, int(features))
        self._amplitude = float(amplitude)
        self._length_scale = max(float(length_scale), 1e-6)
        self._input_scale = max(float(input_scale), 1e-6)
        self._noise_std = float(noise_std)
        self._rng = np.random.RandomState(int(seed))

        spaces = self.env.obs_space.copy()
        self._source_keys = self._resolve_source_keys(spaces, source_keys)
        self._source_spaces = {key: spaces[key] for key in self._source_keys}
        self._input_dim = sum(
            int(np.prod(space.shape)) for space in self._source_spaces.values())
        if self._input_dim <= 0:
            raise ValueError('DynamicsComplexity needs at least one source dimension.')
        self._weights = self._rng.normal(
            size=(self._features, self._input_dim)).astype(np.float32)
        self._weights /= self._length_scale
        self._bias = self._rng.uniform(
            0.0, 2.0 * np.pi, size=(self._features,)).astype(np.float32)
        self._coeff = self._rng.normal(
            size=(self._features, self._dims)).astype(np.float32)

        self._obs_space = spaces
        if self._mode == 'append':
            if self._key not in spaces:
                raise KeyError(
                    f"DynamicsComplexity append target '{self._key}' is missing.")
            target = spaces[self._key]
            if len(target.shape) != 1 or not np.issubdtype(target.dtype, np.floating):
                raise ValueError(
                    f"DynamicsComplexity append target '{self._key}' must be a "
                    f'1D floating observation, got {target}.')
            self._target_dim = int(np.prod(target.shape))
            low = np.concatenate([
                target.low.reshape(-1),
                -np.inf * np.ones((self._dims,), np.float32),
            ])
            high = np.concatenate([
                target.high.reshape(-1),
                np.inf * np.ones((self._dims,), np.float32),
            ])
            self._obs_space[self._key] = spacelib.Space(
                np.float32, (self._target_dim + self._dims,), low, high)
        elif self._mode == 'create':
            if self._key in spaces:
                raise KeyError(
                    f"DynamicsComplexity create target '{self._key}' already exists.")
            self._target_dim = 0
            self._obs_space[self._key] = spacelib.Space(
                np.float32, (self._dims,), -np.inf, np.inf)
        else:
            raise ValueError(
                f"Unknown DynamicsComplexity mode '{self._mode}'. "
                "Use 'append' or 'create'.")

    @functools.cached_property
    def obs_space(self):
        return self._obs_space

    def step(self, action):
        obs = self.env.step(action).copy()
        return self._augment_obs(obs, noise=True)

    def get_state_table(self, *args, **kwargs):
        if not hasattr(self.env, 'get_state_table'):
            return None
        table = self.env.get_state_table(*args, **kwargs)
        if table is None:
            return None
        table = table.copy()
        table['obs'] = self._augment_obs(table['obs'].copy(), noise=False)
        names = dict(table.get('obs_col_names', {}))
        extra_names = [f'{self._key}_dc_{i}' for i in range(self._dims)]
        if self._mode == 'append':
            base = names.get(
                self._key, [f'{self._key}_{i}' for i in range(self._target_dim)])
            names[self._key] = list(base) + extra_names
        else:
            names[self._key] = extra_names
        table['obs_col_names'] = names
        return table

    def get_action_table(self, *args, **kwargs):
        if not hasattr(self.env, 'get_action_table'):
            return None
        table = self.env.get_action_table(*args, **kwargs)
        if table is None:
            return None
        table = table.copy()
        table['obs'] = self._augment_obs(table['obs'].copy(), noise=False)
        names = dict(table.get('obs_col_names', {}))
        extra_names = [f'{self._key}_dc_{i}' for i in range(self._dims)]
        if self._mode == 'append':
            base = names.get(
                self._key, [f'{self._key}_{i}' for i in range(self._target_dim)])
            names[self._key] = list(base) + extra_names
        else:
            names[self._key] = extra_names
        table['obs_col_names'] = names
        return table

    def _resolve_source_keys(self, spaces, source_keys):
        if source_keys is None:
            source_keys = 'auto'
        if isinstance(source_keys, str):
            source_keys = source_keys.strip()
            if source_keys in ('', 'auto'):
                if self._is_source_space(self._key, spaces.get(self._key)):
                    return [self._key]
                keys = sorted(
                    key for key, space in spaces.items()
                    if self._is_source_space(key, space))
                if not keys:
                    raise ValueError(
                        'DynamicsComplexity could not find any 1D floating '
                        'observation keys to use as sources.')
                return keys
            keys = [key.strip() for key in source_keys.split(',') if key.strip()]
        else:
            keys = list(source_keys)
        for key in keys:
            if key not in spaces:
                raise KeyError(f"DynamicsComplexity source key '{key}' is missing.")
            if not self._is_source_space(key, spaces[key]):
                raise ValueError(
                    f"DynamicsComplexity source key '{key}' must be a 1D "
                    f'floating observation, got {spaces[key]}.')
        return keys

    def _is_source_space(self, key, space):
        return (
            space is not None and
            key not in self._SKIP_KEYS and
            not key.startswith('log_') and
            len(space.shape) == 1 and
            np.issubdtype(space.dtype, np.floating))

    def _augment_obs(self, obs, noise):
        source = self._source_vector(obs)
        extra = self._features_from_source(source)
        if noise and self._noise_std:
            noise = self._rng.normal(size=extra.shape).astype(np.float32)
            extra = extra + self._noise_std * noise
        extra = extra * self._output_scale(obs, extra.shape)
        extra = extra.astype(np.float32)
        if self._mode == 'append':
            base = np.asarray(obs[self._key], np.float32)
            obs[self._key] = np.concatenate([base, extra], axis=-1).astype(np.float32)
        else:
            obs[self._key] = extra
        return obs

    def _source_vector(self, obs):
        parts = []
        for key in self._source_keys:
            value = np.asarray(obs[key], np.float32)
            rank = len(self._source_spaces[key].shape)
            leading = value.shape[:-rank] if rank else value.shape
            parts.append(value.reshape(leading + (-1,)))
        return np.concatenate(parts, axis=-1) / self._input_scale

    def _features_from_source(self, source):
        proj = np.einsum('...i,fi->...f', source, self._weights)
        basis = np.sin(proj + self._bias)
        raw = np.einsum('...f,fd->...d', basis, self._coeff)
        raw *= np.sqrt(2.0 / self._features)
        return self._amplitude * np.tanh(raw)

    def _output_scale(self, obs, extra_shape):
        try:
            scale_fn = getattr(self.env, 'dynamics_complexity_scale')
        except (AttributeError, ValueError):
            return np.float32(1.0)
        scale = np.asarray(scale_fn(obs), np.float32)
        if not np.isfinite(scale).all():
            raise ValueError('DynamicsComplexity output scale must be finite.')
        leading = extra_shape[:-1]
        if scale.shape == ():
            return scale
        if scale.shape == leading:
            return scale[..., None]
        if scale.shape == leading + (1,):
            return scale
        if scale.size == 1:
            return scale.reshape((1,) * len(leading) + (1,))
        raise ValueError('DynamicsComplexity output scale must be scalar or match the ' f'observation batch shape {leading}, got {scale.shape}.')


class ActionRepeat(base.Wrapper):

    def __init__(self, env, repeat):
        super().__init__(env)
        self._repeat = repeat

    def step(self, action):
        if action['reset']:
            return self.env.step(action)
        reward = 0.0
        for _ in range(self._repeat):
            obs = self.env.step(action)
            reward += obs['reward']
            if obs['is_last'] or obs['is_terminal']:
                break
        obs['reward'] = np.float32(reward)
        return obs


class ClipAction(base.Wrapper):

    def __init__(self, env, key='action', low=-1, high=1):
        super().__init__(env)
        self._key = key
        self._low = low
        self._high = high

    def step(self, action):
        clipped = np.clip(action[self._key], self._low, self._high)
        return self.env.step({**action, self._key: clipped})


class NormalizeAction(base.Wrapper):

    def __init__(self, env, key='action'):
        super().__init__(env)
        self._key = key
        self._space = env.act_space[key]
        self._mask = np.isfinite(self._space.low) & np.isfinite(self._space.high)
        self._low = np.where(self._mask, self._space.low, -1)
        self._high = np.where(self._mask, self._space.high, 1)

    @functools.cached_property
    def act_space(self):
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        space = spacelib.Space(np.float32, self._space.shape, low, high)
        return {**self.env.act_space, self._key: space}

    def step(self, action):
        orig = (action[self._key] + 1) / 2 * (self._high - self._low) + self._low
        orig = np.where(self._mask, orig, action[self._key])
        return self.env.step({**action, self._key: orig})


class ExpandScalars(base.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self._obs_expanded = []
        self._obs_space = {}
        for key, space in self.env.obs_space.items():
            if space.shape == () and key != 'reward' and not space.discrete:
                space = spacelib.Space(space.dtype, (1,), space.low, space.high)
                self._obs_expanded.append(key)
            self._obs_space[key] = space
        self._act_expanded = []
        self._act_space = {}
        for key, space in self.env.act_space.items():
            if space.shape == () and not space.discrete:
                space = spacelib.Space(space.dtype, (1,), space.low, space.high)
                self._act_expanded.append(key)
            self._act_space[key] = space

    @functools.cached_property
    def obs_space(self):
        return self._obs_space

    @functools.cached_property
    def act_space(self):
        return self._act_space

    def step(self, action):
        action = {
            key: np.squeeze(value, 0) if key in self._act_expanded else value
            for key, value in action.items()}
        obs = self.env.step(action)
        obs = {
            key: np.expand_dims(value, 0) if key in self._obs_expanded else value
            for key, value in obs.items()}
        return obs


class FrameStack(base.Wrapper):

    def __init__(self, env, key='image', length=3):
        super().__init__(env)
        self._key = key
        self._length = int(length)
        if self._length < 1:
            raise ValueError(f'FrameStack length must be positive, got {length}.')
        if key not in env.obs_space:
            raise KeyError(f"FrameStack observation key '{key}' is missing.")
        space = env.obs_space[key]
        if len(space.shape) != 3:
            raise ValueError(
                f"FrameStack expects an HWC observation for '{key}', got {space}.")
        self._shape = space.shape
        shape = space.shape[:-1] + (space.shape[-1] * self._length,)
        low = np.concatenate([space.low] * self._length, axis=-1)
        high = np.concatenate([space.high] * self._length, axis=-1)
        self._obs_space = env.obs_space.copy()
        self._obs_space[key] = spacelib.Space(space.dtype, shape, low, high)
        self._frames = []

    @functools.cached_property
    def obs_space(self):
        return self._obs_space

    def step(self, action):
        obs = self.env.step(action).copy()
        frame = np.asarray(obs[self._key])
        if frame.shape != self._shape:
            raise ValueError(
                f"FrameStack expected '{self._key}' shape {self._shape}, "
                f'got {frame.shape}.')
        is_first = np.asarray(obs['is_first'])
        if is_first.shape:
            raise ValueError(
                f'FrameStack expects an unbatched environment, got is_first '
                f'shape {is_first.shape}.')
        frame = frame.copy()
        if bool(is_first) or not self._frames:
            self._frames = [frame.copy() for _ in range(self._length)]
        else:
            self._frames = self._frames[1:] + [frame]
        obs[self._key] = np.concatenate(self._frames, axis=-1)
        return obs


class FlattenTwoDimObs(base.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self._keys = []
        self._obs_space = {}
        for key, space in self.env.obs_space.items():
            if len(space.shape) == 2:
                space = spacelib.Space(
                    space.dtype,
                    (int(np.prod(space.shape)),),
                    space.low.flatten(),
                    space.high.flatten())
                self._keys.append(key)
            self._obs_space[key] = space

    @functools.cached_property
    def obs_space(self):
        return self._obs_space

    def step(self, action):
        obs = self.env.step(action).copy()
        for key in self._keys:
            obs[key] = obs[key].flatten()
        return obs


class FlattenTwoDimActions(base.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self._origs = {}
        self._act_space = {}
        for key, space in self.env.act_space.items():
            if len(space.shape) == 2:
                space = spacelib.Space(
                    space.dtype,
                    (int(np.prod(space.shape)),),
                    space.low.flatten(),
                    space.high.flatten())
                self._origs[key] = space.shape
            self._act_space[key] = space

    @functools.cached_property
    def act_space(self):
        return self._act_space

    def step(self, action):
        action = action.copy()
        for key, shape in self._origs.items():
            action[key] = action[key].reshape(shape)
        return self.env.step(action)


class ForceDtypes(base.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self._obs_space, _, self._obs_outer = self._convert(env.obs_space)
        self._act_space, self._act_inner, _ = self._convert(env.act_space)

    @property
    def obs_space(self):
        return self._obs_space

    @property
    def act_space(self):
        return self._act_space

    def step(self, action):
        action = action.copy()
        for key, dtype in self._act_inner.items():
            action[key] = np.asarray(action[key], dtype)
        obs = self.env.step(action)
        for key, dtype in self._obs_outer.items():
            obs[key] = np.asarray(obs[key], dtype)
        return obs

    def _convert(self, spaces):
        results, befores, afters = {}, {}, {}
        for key, space in spaces.items():
            before = after = space.dtype
            if np.issubdtype(before, np.floating):
                after = np.float32
            elif np.issubdtype(before, np.uint8):
                after = np.uint8
            elif np.issubdtype(before, np.integer):
                after = np.int32
            befores[key] = before
            afters[key] = after
            results[key] = spacelib.Space(after, space.shape, space.low, space.high)
        return results, befores, afters


class CheckSpaces(base.Wrapper):

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        for key, value in action.items():
            self._check(value, self.env.act_space[key], key)
        obs = self.env.step(action)
        for key, value in obs.items():
            self._check(value, self.env.obs_space[key], key)
        return obs

    def _check(self, value, space, key):
        if not isinstance(value, (
                np.ndarray, np.generic, list, tuple, int, float, bool)):
            raise TypeError(f'Invalid type {type(value)} for key {key}.')
        if value in space:
            return
        dtype = np.array(value).dtype
        shape = np.array(value).shape
        lowest, highest = np.min(value), np.max(value)
        raise ValueError(
            f"Value for '{key}' with dtype {dtype}, shape {shape}, "
            f"lowest {lowest}, highest {highest} is not in {space}.")


class DiscretizeAction(base.Wrapper):

    def __init__(self, env, key='action', bins=5):
        super().__init__(env)
        self._shape = env.act_space[key].shape
        self._dims = int(np.prod(self._shape))
        self._values = np.linspace(-1, 1, bins, dtype=np.float32)
        axes = np.meshgrid(*([self._values] * self._dims), indexing='xy')
        self._grid = np.stack([x.reshape(-1) for x in axes], axis=-1)
        self._key = key

    @functools.cached_property
    def act_space(self):
        space = spacelib.Space(np.int32, (), 0, len(self._grid))
        return {**self.env.act_space, self._key: space}

    def step(self, action):
        indices = np.asarray(action[self._key], np.int32)
        continuous = self._grid[indices].reshape(indices.shape + self._shape)
        return self.env.step({**action, self._key: continuous})

    def get_action_table(self, *args, **kwargs):
        if not hasattr(self.env, 'get_action_table'):
            return None
        kwargs = dict(kwargs)
        if not args:
            try:
                if 'action_points' not in kwargs and 'action_points_per_axis' not in kwargs:
                    kwargs['action_points'] = len(self._values)
                table = self.env.get_action_table(*args, **kwargs)
            except TypeError:
                if kwargs.pop('action_points', None) is None:
                    raise
                kwargs.setdefault('action_points_per_axis', len(self._values))
                table = self.env.get_action_table(*args, **kwargs)
        else:
            table = self.env.get_action_table(*args, **kwargs)
        if table is None:
            return None
        table = table.copy()
        action_values = {
            key: np.asarray(value, np.float32)
            for key, value in table['actions'].items()}
        values = action_values[self._key].reshape(len(action_values[self._key]), -1)
        ids = np.square(values[:, None] - self._grid[None]).sum(-1).argmin(-1)
        table['action_values'] = action_values
        table['actions'] = {self._key: ids.astype(np.int32)}
        return table


class ResizeImage(base.Wrapper):

    def __init__(self, env, size=(64, 64)):
        super().__init__(env)
        self._size = size
        self._keys = [
            k for k, v in env.obs_space.items()
            if len(v.shape) > 1 and v.shape[:2] != size]
        print(f'Resizing keys {",".join(self._keys)} to {self._size}.')
        if self._keys:
            from PIL import Image
            self._Image = Image

    @functools.cached_property
    def obs_space(self):
        spaces = self.env.obs_space
        for key in self._keys:
            shape = self._size + spaces[key].shape[2:]
            spaces[key] = spacelib.Space(np.uint8, shape)
        return spaces

    def step(self, action):
        obs = self.env.step(action)
        for key in self._keys:
            obs[key] = self._resize(obs[key])
        return obs

    def _resize(self, image):
        image = self._Image.fromarray(image)
        image = image.resize(self._size, self._Image.NEAREST)
        image = np.array(image)
        return image


class RenderImage(base.Wrapper):

    def __init__(self, env, key='image'):
        super().__init__(env)
        self._key = key
        self._num_envs = int(getattr(env, 'num_envs', 0) or 0)
        sample = np.asarray(self.env.render())
        if self._num_envs and sample.ndim == 4 and sample.shape[0] == self._num_envs:
            self._shape = sample.shape[1:]
        else:
            self._shape = sample.shape

    @functools.cached_property
    def obs_space(self):
        spaces = self.env.obs_space
        spaces[self._key] = spacelib.Space(np.uint8, self._shape)
        return spaces

    def step(self, action):
        obs = self.env.step(action)
        image = np.asarray(self.env.render())
        if self._num_envs:
            if image.ndim == len(self._shape):
                image = np.repeat(image[None], self._num_envs, axis=0)
            elif not (image.ndim == len(self._shape) + 1 and image.shape[0] == self._num_envs):
                raise ValueError(
                    f"RenderImage expected batched render with leading dim {self._num_envs} "
                    f"or single image shape {self._shape}, got {image.shape}")
        obs[self._key] = image
        return obs


class BackwardReturn(base.Wrapper):

    def __init__(self, env, horizon):
        super().__init__(env)
        self._discount = 1 - 1 / horizon
        self._bwreturn = 0.0

    @functools.cached_property
    def obs_space(self):
        return {
            **self.env.obs_space,
            'bwreturn': spacelib.Space(np.float32),
        }

    def step(self, action):
        obs = self.env.step(action)
        self._bwreturn *= (1 - obs['is_first']) * self._discount
        self._bwreturn += obs['reward']
        obs['bwreturn'] = np.float32(self._bwreturn)
        return obs


class RestartOnException(base.Wrapper):

    def __init__(
            self, ctor, exceptions=(Exception,), window=300, maxfails=2, wait=20):
        if not isinstance(exceptions, (tuple, list)):
            exceptions = [exceptions]
        self._ctor = ctor
        self._exceptions = tuple(exceptions)
        self._window = window
        self._maxfails = maxfails
        self._wait = wait
        self._last = time.time()
        self._fails = 0
        super().__init__(self._ctor())

    def step(self, action):
        try:
            return self.env.step(action)
        except self._exceptions as e:
            if time.time() > self._last + self._window:
                self._last = time.time()
                self._fails = 1
            else:
                self._fails += 1
            if self._fails > self._maxfails:
                raise RuntimeError('The env crashed too many times.')
            message = f'Restarting env after crash with {type(e).__name__}: {e}'
            print(message, flush=True)
            time.sleep(self._wait)
            self.env = self._ctor()
            action['reset'] = np.ones_like(action['reset'])
            return self.env.step(action)
