import contextlib
import inspect
import os
from collections.abc import Mapping

import embodied
import numpy as np


class JaxImageEnvMixin:
  """Shared helpers for JAX-backed image env wrappers.

  The wrappers keep env-specific reset/step parsing logic, while this mixin
  centralizes transfer-guard handling, nested space flattening, image resizing,
  and observation formatting.
  """

  _render_obs_keys = ('image', 'pixels', 'rgb')

  def _configure_jax_backend(self):
    if not getattr(self, '_cpu_in_workers', False):
      return
    try:
      import multiprocessing as mp
      if mp.current_process().name == 'MainProcess':
        return
    except Exception:
      return
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

  def _allow_host_to_device(self):
    stack = contextlib.ExitStack()
    entered = False
    for name in (
        'transfer_guard',
        'transfer_guard_host_to_device',
        'transfer_guard_device_to_host',
        'transfer_guard_device_to_device',
    ):
      guard = getattr(self._jax, name, None)
      if not callable(guard):
        continue
      try:
        stack.enter_context(guard('allow'))
        entered = True
      except Exception:
        continue
    if entered:
      return stack
    stack.close()
    return contextlib.nullcontext()

  def _hostify_tree(self, tree):
    tree_util = getattr(self._jax, 'tree_util', None)
    if tree_util is None or tree is None:
      return tree
    with self._allow_host_to_device():
      return tree_util.tree_map(self._hostify_leaf, tree)

  def _hostify_leaf(self, value):
    jax_array = getattr(self._jax, 'Array', None)
    if jax_array is not None and isinstance(value, jax_array):
      return np.asarray(value)
    module = type(value).__module__
    if module.startswith(('jax.', 'jaxlib.')):
      return np.asarray(value)
    return value

  def _call_filtered(self, fn, *args, **kwargs):
    try:
      sig = inspect.signature(fn)
    except (TypeError, ValueError):
      return fn(*args, **kwargs)
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
      return fn(*args, **kwargs)
    filtered = {k: v for k, v in kwargs.items() if k in params}
    return fn(*args, **filtered)

  def _flatten_spaces(self, nest, prefix=None):
    if hasattr(nest, 'spaces') and not isinstance(nest, dict):
      nest = nest.spaces
    result = {}
    for key, value in dict(nest).items():
      key = f'{prefix}/{key}' if prefix else str(key)
      if hasattr(value, 'spaces') or isinstance(value, Mapping):
        result.update(self._flatten_spaces(value, key))
      else:
        result[key] = value
    return result

  def _flatten_values(self, nest, prefix=None):
    result = {}
    for key, value in dict(nest).items():
      key = f'{prefix}/{key}' if prefix else str(key)
      if isinstance(value, Mapping):
        result.update(self._flatten_values(value, key))
      else:
        result[key] = value
    return result

  def _safe_obs_key(self, key):
    if key in ('reward', 'is_first', 'is_last', 'is_terminal'):
      return f'obs_{key}'
    return key

  def _convert_space(self, space):
    if hasattr(space, 'n'):
      return embodied.Space(np.int32, (), 0, int(space.n))
    if hasattr(space, 'num_values'):
      return embodied.Space(np.int32, (), 0, int(space.num_values))
    if hasattr(space, 'nvec'):
      nvec = np.asarray(space.nvec, np.int32)
      return embodied.Space(np.int32, nvec.shape, 0, nvec)
    if hasattr(space, 'shape') and hasattr(space, 'dtype'):
      low = getattr(space, 'low', None)
      high = getattr(space, 'high', None)
      shape = tuple(space.shape) if hasattr(space.shape, '__len__') else (space.shape,)
      if shape == (None,):
        shape = ()
      return embodied.Space(space.dtype, shape, low, high)
    if isinstance(space, (tuple, list)):
      return embodied.Space(np.float32, tuple(space))
    array = np.asarray(space)
    return embodied.Space(array.dtype, array.shape)

  def _format_obs(self, obs, reward, is_first=False, is_last=False, is_terminal=False):
    flat = self._flatten_obs_values(obs)
    flat = self._maybe_resize_obs(flat)
    flat = self._coerce_to_obs_space(flat)
    return {
        **flat,
        'reward': np.float32(reward),
        'is_first': bool(is_first),
        'is_last': bool(is_last),
        'is_terminal': bool(is_terminal),
    }

  def _format_obs_batched(self, obs, reward, is_first, is_last, is_terminal):
    flat = self._flatten_obs_values(obs)
    flat = self._maybe_resize_obs_batched(flat)
    flat = self._coerce_to_obs_space(flat)
    return {
        **flat,
        'reward': np.asarray(reward, np.float32),
        'is_first': np.asarray(is_first, bool),
        'is_last': np.asarray(is_last, bool),
        'is_terminal': np.asarray(is_terminal, bool),
    }

  def _flatten_obs_values(self, obs):
    if isinstance(obs, Mapping):
      flat = self._flatten_values(obs)
      return {self._safe_obs_key(k): np.asarray(v) for k, v in flat.items()}
    return {self._obs_key: np.asarray(obs)}

  def _maybe_resize_obs(self, flat):
    if not getattr(self, '_size', None):
      return flat
    return {
        k: self._resize_image(v) if self._looks_like_image(v) else v
        for k, v in flat.items()}

  def _maybe_resize_obs_batched(self, flat):
    if not getattr(self, '_size', None):
      return flat
    return {
        k: self._resize_image_batch(v) if self._looks_like_batched_image(v) else v
        for k, v in flat.items()}

  def _render_from_obs(self):
    if self._last_obs is None:
      return None
    if isinstance(self._last_obs, Mapping):
      flat = self._flatten_values(self._last_obs)
      keys = (self._obs_key,) + tuple(self._render_obs_keys)
      for key in keys:
        if key in flat:
          image = np.asarray(flat[key])
          if self._looks_like_image(image):
            return self._resize_image(image) if self._size else image
      for value in flat.values():
        image = np.asarray(value)
        if self._looks_like_image(image):
          return self._resize_image(image) if self._size else image
      return None
    image = np.asarray(self._last_obs)
    if not self._looks_like_image(image):
      return None
    return self._resize_image(image) if self._size else image

  def _zero_action(self):
    space = self.act_space[self._act_key]
    return np.zeros(space.shape, space.dtype) if space.shape else space.dtype.type(0)

  def _looks_like_image(self, value):
    if not np.issubdtype(value.dtype, np.number):
      return False
    return value.ndim == 3 and value.shape[-1] in (1, 3, 4)

  def _looks_like_image_space(self, space):
    if not np.issubdtype(space.dtype, np.number):
      return False
    return len(space.shape) == 3 and space.shape[-1] in (1, 3, 4)

  def _looks_like_batched_image(self, value):
    return (
        np.issubdtype(value.dtype, np.number) and
        value.ndim == 4 and
        value.shape[-1] in (1, 3, 4))

  def _resize_space(self, space):
    if space.shape[:2] == self._size:
      return space
    shape = self._size + (space.shape[-1],)
    low = getattr(space, 'low', None)
    high = getattr(space, 'high', None)
    if low is not None:
      low = np.asarray(low)
      low = low.min() if low.size else None
    if high is not None:
      high = np.asarray(high)
      high = high.max() if high.size else None
    return embodied.Space(space.dtype, shape, low, high)

  def _compact_image_space(self, space):
    if not self._looks_like_image_space(space):
      return space
    if not np.issubdtype(space.dtype, np.floating):
      return space
    low = np.asarray(getattr(space, 'low', ()))
    high = np.asarray(getattr(space, 'high', ()))
    finite_low = low.size and np.isfinite(low).all()
    finite_high = high.size and np.isfinite(high).all()
    if finite_low and finite_high:
      if low.min() < -1e-6 or high.max() > 255.0 + 1e-6:
        return space
    return embodied.Space(np.uint8, space.shape, 0, 255)

  def _compact_image_spaces(self, spaces):
    return {
        k: self._compact_image_space(v) if self._looks_like_image_space(v) else v
        for k, v in spaces.items()}

  def _resize_image(self, image):
    if image.shape[:2] == self._size:
      return image
    # PIL often changes float RGB arrays to uint8; keep float paths stable.
    if self._resize == 'pillow' and np.issubdtype(image.dtype, np.floating):
      return self._resize_nearest(image)
    if self._resize == 'nearest':
      return self._resize_nearest(image)
    if self._resize == 'opencv':
      import cv2
      return cv2.resize(image, self._size, interpolation=cv2.INTER_AREA)
    if self._resize == 'pillow':
      from PIL import Image
      try:
        pil_image = Image.fromarray(image)
      except Exception:
        return self._resize_nearest(image)
      out = pil_image.resize(self._size, Image.BILINEAR)
      out = np.array(out)
      if out.ndim == 2:
        out = out[:, :, None]
      return out.astype(image.dtype, copy=False)
    raise ValueError(f'Unknown resize backend: {self._resize}')

  def _resize_nearest(self, image):
    h, w = image.shape[:2]
    th, tw = self._size
    ys = np.clip(np.round(np.linspace(0, h - 1, th)).astype(np.int32), 0, h - 1)
    xs = np.clip(np.round(np.linspace(0, w - 1, tw)).astype(np.int32), 0, w - 1)
    out = image[ys][:, xs]
    if out.ndim == 2:
      out = out[:, :, None]
    return out

  def _resize_image_batch(self, images):
    if images.shape[1:3] == self._size:
      return images
    return np.stack([self._resize_image(np.asarray(img)) for img in images], axis=0)

  def _coerce_to_obs_space(self, flat):
    spaces = self.__dict__.get('obs_space', None)
    if not spaces:
      return flat
    out = {}
    for key, value in flat.items():
      if key not in spaces:
        out[key] = value
        continue
      target = spaces[key]
      arr = np.asarray(value)
      if arr.dtype != target.dtype:
        if (
            target.dtype == np.uint8 and
            np.issubdtype(arr.dtype, np.floating) and
            (self._looks_like_image(arr) or self._looks_like_batched_image(arr))):
          arrf = arr.astype(np.float32)
          if arrf.size:
            scale = 255.0 if (
                np.nanmin(arrf) >= -1e-6 and
                np.nanmax(arrf) <= 1.0 + 1e-6) else 1.0
          else:
            scale = 1.0
          arrf = np.nan_to_num(
              arrf, nan=0.0, posinf=255.0 / scale, neginf=0.0)
          arr = np.clip(arrf * scale, 0, 255).astype(target.dtype)
        elif (
            np.issubdtype(target.dtype, np.floating) and
            arr.dtype == np.uint8 and
            (self._looks_like_image(arr) or self._looks_like_batched_image(arr))):
          arr = arr.astype(target.dtype) / 255.0
        else:
          arr = arr.astype(target.dtype)
      out[key] = arr
    return out
