import functools

import embodied
from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE)
import numpy as np
import gymnasium as gym
from typing import Optional, Dict
import os
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit


class FromMetaWorld(embodied.Env):

    def __init__(self, env, obs_key='state', act_key='action', max_episode_steps: int = 200, **kwargs):
        if isinstance(env, str):
            self.constructor = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env]
            if 'seed' in kwargs:
                self._env = self.constructor(seed=kwargs['seed'])
        else:
            assert not kwargs, kwargs
            self._env = env
        self._env.observation_space = spaces.Box(low=-np.ones_like(self._env.observation_space.low) * np.inf,
                                                 high=np.ones_like(self._env.observation_space.low) * np.inf,
                                                 shape=self._env.observation_space.low.shape,
                                                 dtype=self._env.observation_space.low.dtype)
        self._env = TimeLimit(self._env, max_episode_steps=max_episode_steps)
        self._obs_dict = hasattr(self._env.observation_space, 'spaces')
        self._act_dict = hasattr(self._env.action_space, 'spaces')
        self._obs_key = obs_key
        self._act_key = act_key
        self._done = True
        self._info = None
        self.info_dict = {'success': int}

    @property
    def env(self):
        return self._env

    @property
    def info(self):
        return self._info

    @functools.cached_property
    def obs_space(self):
        spaces = {self._obs_key: self._env.observation_space}
        spaces = {k: self._convert(v) for k, v in spaces.items()}
        infos = {k: embodied.Space(np.float32) for k, dtype in self.info_dict.items()}
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
        if self._act_dict:
            spaces = self._flatten(self._env.action_space.spaces)
        else:
            spaces = {self._act_key: self._env.action_space}
        spaces = {k: self._convert(v) for k, v in spaces.items()}
        spaces['reset'] = embodied.Space(bool)
        return spaces

    def step(self, action):
        if action['reset'] or self._done:
            self._done = False
            obs, info = self._env.reset()
            if 'success' not in info:
                info['success'] = 0
            self._info = info
            return self._obs(obs, 0.0, is_first=True)
        if self._act_dict:
            action = self._unflatten(action)
        else:
            action = action[self._act_key]
        obs, reward, terminate, truncate, self._info = self._env.step(action)
        self._done = terminate or truncate
        return self._obs(
            obs, reward,
            is_last=bool(self._done),
            is_terminal=terminate)

    def _obs(
            self, obs, reward, is_first=False, is_last=False, is_terminal=False):
        if not self._obs_dict:
            obs = {self._obs_key: obs}
        obs = self._flatten(obs)
        obs = {k: np.asarray(v) for k, v in obs.items()}
        infos = {key: np.float32(self.info[key]) for key in self.info_dict.keys()}
        obs = obs | infos
        obs.update(
            reward=np.float32(reward),
            is_first=is_first,
            is_last=is_last,
            is_terminal=is_terminal)
        return obs

    def render(self):
        image = self._env.render('rgb_array')
        assert image is not None
        return image

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass

    def _flatten(self, nest, prefix=None):
        result = {}
        for key, value in nest.items():
            key = prefix + '/' + key if prefix else key
            if isinstance(value, gym.spaces.Dict):
                value = value.spaces
            if isinstance(value, dict):
                result.update(self._flatten(value, key))
            else:
                result[key] = value
        return result

    def _unflatten(self, flat):
        result = {}
        for key, value in flat.items():
            parts = key.split('/')
            node = result
            for part in parts[:-1]:
                if part not in node:
                    node[part] = {}
                node = node[part]
            node[parts[-1]] = value
        return result

    def _convert(self, space):
        if hasattr(space, 'n'):
            return embodied.Space(np.int32, (), 0, space.n)
        return embodied.Space(space.dtype, space.shape, space.low, space.high)


class MetaWorld(embodied.Env):
    def __init__(
            self,
            env,
            repeat=1,
            *args, **kwargs):
        if 'MUJOCO_GL' not in os.environ:
            os.environ['MUJOCO_GL'] = 'egl'
        self._env = FromMetaWorld(env, *args, **kwargs)
        self._env = embodied.wrappers.ExpandScalars(self._env)
        self._env = embodied.wrappers.ActionRepeat(self._env, repeat)

    @functools.cached_property
    def obs_space(self):
        return self._env.obs_space.copy()

    @functools.cached_property
    def act_space(self):
        return self._env.act_space.copy()

    def step(self, action):
        for key, space in self.act_space.items():
            if not space.discrete:
                assert np.isfinite(action[key]).all(), (key, action[key])
        obs = self._env.step(action)
        for key, space in self.obs_space.items():
            if np.issubdtype(space.dtype, np.floating):
                assert np.isfinite(obs[key]).all(), (key, obs[key])
        return obs
