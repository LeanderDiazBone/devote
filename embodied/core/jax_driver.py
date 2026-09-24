import numpy as np

from . import timer


class JaxDriver:

    def __init__(self, env, **kwargs):
        self.kwargs = kwargs
        self.length = env.num_envs

        self.batched_env = env
        self.act_space = self.batched_env.act_space
        self.callbacks = []
        self.acts = None
        self.carry = None
        self.reset()

    def reset(self, init_policy=None):
        self.acts = {k: np.zeros((self.length,) + v.shape, v.dtype) for k, v in self.act_space.items()}
        self.acts['reset'] = np.ones(self.length, bool)
        self.carry = init_policy and init_policy(self.length)

    def reset_carry(self, init_policy):
        # Reinitialize the policy carry without touching ``self.acts``.
        # Use this when the agent's parameters changed (e.g. a soft reset)
        # but the env state should keep running -- calling ``reset`` instead
        # would set ``acts['reset'] = True`` and force every env to reset
        # on the next step, silently truncating in-flight episodes.
        self.carry = init_policy(self.length)

    def close(self):
        self.batched_env.close()

    def on_step(self, callback):
        self.callbacks.append(callback)

    def __call__(self, policy, steps=0, episodes=0):
        step, episode = 0, 0
        while step < steps or episode < episodes:
            step, episode = self._step(policy, step, episode)

    def _step(self, policy, step, episode):
        acts = self.acts
        assert all(len(x) == self.length for x in acts.values())
        assert all(isinstance(v, np.ndarray) for v in acts.values())

        # step the envs
        with timer.section('env_step'):
            obs = self.batched_env.step(acts)
        assert all(len(x) == self.length for x in obs.values()), obs

        # call the policy (obs should already be numpy)
        acts, outs, self.carry = policy(obs, self.carry, **self.kwargs)
        assert all(k not in acts for k in outs), (list(outs.keys()), list(acts.keys()))

        # mask the actions of the done envs
        if obs['is_last'].any():
            mask = ~obs['is_last']
            acts = {k: self._mask(v, mask) for k, v in acts.items()}
        acts['reset'] = obs['is_last'].copy()
        self.acts = acts
        trans = {**obs, **acts, **outs}
        # callbacks for each env
        with timer.section('driver_callbacks'):
            for i in range(self.length):
                trn = {k: v[i] for k, v in trans.items()}
                [fn(trn, i, **self.kwargs) for fn in self.callbacks]
        step += len(obs['is_first'])
        episode += obs['is_last'].sum()
        return step, episode

    def _mask(self, value, mask):
        while mask.ndim < value.ndim:
            mask = mask[..., None]
        return value * mask.astype(value.dtype)