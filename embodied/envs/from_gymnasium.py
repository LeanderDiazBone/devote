import functools
import os
from collections import deque
from typing import Dict, Optional

import embodied
import gymnasium as gym
import numpy as np


class FromGymnasium(embodied.Env):
    # Same grayscale weights as the Atari class
    WEIGHTS = np.array([0.299, 0.587, 1 - (0.299 + 0.587)])

    def __init__(self, env,
                 info_dict: Optional[Dict] = None,
                 obs_key='image', 
                 act_key='action',
                 size=(64, 64), 
                 gray=False,
                 pooling=1, 
                 aggregate='max', 
                 resize='pillow',
                 **kwargs):
        
        if isinstance(env, str):
            # We use rgb_array to get pixels for our manual processing
            self._env = gym.make(env, render_mode='rgb_array', **kwargs)
        else:
            self._env = env

        self._obs_key = obs_key
        self._act_key = act_key
        self._size = size
        self._gray = gray
        self._pooling = pooling
        self._aggregate = aggregate
        self._resize = resize
        self._info = None
        self._done = True
        self.info_dict = info_dict or {}

        # Buffer for frame pooling (handles flickering/temporal consistency)
        self._buffers = deque(maxlen=self._pooling)

    @functools.cached_property
    def obs_space(self):
        # We manually define the image space based on the size/gray settings
        shape = (*self._size, 1 if self._gray else 3)
        spaces = {self._obs_key: embodied.Space(np.uint8, shape)}
        
        # Add metadata spaces
        infos = {k: embodied.Space(np.float32) for k in self.info_dict.keys()}
        return {
            **spaces,
            'reward': embodied.Space(np.float32),
            'is_first': embodied.Space(bool),
            'is_last': embodied.Space(bool),
            'is_terminal': embodied.Space(bool),
            **infos,
        }

    @functools.cached_property
    def act_space(self):
        # Determine base action space
        if hasattr(self._env.action_space, 'n'):
            space = embodied.Space(np.int32, (), 0, self._env.action_space.n)
        else:
            space = embodied.Space(
                self._env.action_space.dtype, 
                self._env.action_space.shape, 
                self._env.action_space.low, 
                self._env.action_space.high)
        
        return {self._act_key: space, 'reset': embodied.Space(bool)}

    def step(self, action):
        if action['reset'] or self._done:
            self._done = False
            obs, self._info = self._env.reset()
            # Clear and fill buffers with the initial frame
            initial_frame = self._env.render()
            for _ in range(self._pooling):
                self._buffers.append(initial_frame)
            return self._obs(0.0, is_first=True)

        # Standard step
        act = action[self._act_key]
        obs, reward, terminate, truncate, self._info = self._env.step(act)
        
        # Add current frame to pooling buffer
        self._buffers.append(self._env.render())
        
        self._done = terminate or truncate
        return self._obs(
            reward, 
            is_last=bool(self._done), 
            is_terminal=terminate)

    def _obs(self, reward, is_first=False, is_last=False, is_terminal=False):
        # 1. Aggregate frames (Pooling)
        if self._aggregate == 'max':
            image = np.amax(self._buffers, 0)
        else:
            image = np.mean(self._buffers, 0).astype(np.uint8)

        # 2. Resize
        if self._resize == 'opencv':
            import cv2
            image = cv2.resize(image, self._size, interpolation=cv2.INTER_AREA)
        elif self._resize == 'pillow':
            from PIL import Image
            image = Image.fromarray(image)
            image = image.resize(self._size, Image.BILINEAR)
            image = np.array(image)

        # 3. Grayscale
        if self._gray:
            image = (image * self.WEIGHTS).sum(-1).astype(image.dtype)[:, :, None]

        # 4. Pack observations
        obs = {self._obs_key: image}
        infos = {k: np.float32(self._info.get(k, 0.0)) for k in self.info_dict.keys()}
        
        obs.update(
            **infos,
            reward=np.float32(reward),
            is_first=is_first,
            is_last=is_last,
            is_terminal=is_terminal)
        return obs

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()