"""Env-side helpers for policy-evaluation experiments on DMC.

Provides:
  * `make_eval_dmc(task, action_repeat, seed)` — a bare DMC env (no batching,
    proprio obs) usable for MC rollouts.
  * `set_physics(env, qpos, qvel)` — overwrite the underlying dm_control state
    and recompute observables so the next obs reflects the new state.
  * `obs_at(env, qpos, qvel)` — set state, run physics.forward(), and return
    the resulting observation dict (without stepping).
  * `mc_return(env, qpos, qvel, init_action, policy_fn, discount, max_steps)` —
    one MC rollout starting at (qpos, qvel), applying `init_action` first,
    then `policy_fn(obs)` for each subsequent step. Returns the discounted
    return.

These are independent of the JAX agent and can be unit-tested standalone.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from embodied.envs.dmc import DMC


def make_eval_dmc(task: str, action_repeat: int = 1, seed: Optional[int] = None,
                  max_step_limit: int = 0, render: bool = True) -> DMC:
    """Build a non-batched, proprio-only DMC env for the given task string.

    `task` follows the launcher convention: 'dmc_cartpole_swingup', etc.

    `max_step_limit` (env frames, not agent steps) overrides dm_control's
    internal time-limit so MC rollouts can run past the default episode end.
    0 keeps the dm_control default.

    `render=False` builds the DMC with `image=False, log_image=False`, which
    DMC honors by skipping the EGL render call inside `step`. Saves a few ms
    per step and avoids the EGL-context-per-thread conflict when env.step is
    called concurrently from a ThreadPoolExecutor.
    """
    assert task.startswith('dmc_'), task
    env_name = task[len('dmc_'):]
    return DMC(env_name, repeat=action_repeat, image=False, log_image=render,
               seed=seed, max_step_limit=max_step_limit)


def _raw_dmenv(env: DMC):
    return env._dmenv


def set_physics(env: DMC, qpos: np.ndarray, qvel: np.ndarray) -> None:
    """Overwrite qpos/qvel on the underlying dm_control env and recompute."""
    dmenv = _raw_dmenv(env)
    dmenv.physics.data.qpos[:] = np.asarray(qpos, dtype=np.float64)
    dmenv.physics.data.qvel[:] = np.asarray(qvel, dtype=np.float64)
    dmenv.physics.forward()


def _reset_dmc(env: DMC):
    """Drive the embodied wrapper through one reset step so internal flags
    (`is_first`, episode bookkeeping) are consistent. Returns the reset obs."""
    act_space = env.act_space
    reset_action = {k: np.zeros(s.shape, s.dtype) for k, s in act_space.items() if k != 'reset'}
    reset_action['reset'] = np.bool_(True)
    return env.step(reset_action)


def obs_at(env: DMC, qpos: np.ndarray, qvel: np.ndarray,
           natural_reset: bool = False) -> dict:
    """Return the observation dict at physics state (qpos, qvel) without
    stepping the env. Useful for Q(s, a) lookups on the sweep grid.

    If `natural_reset=True`, ignore qpos/qvel and return the obs from
    dm_control's natural init (matching MC's behaviour for natural_reset
    anchors). The init is stochastic, so this is a single sample from the
    init distribution that MC averages over."""
    reset_obs = _reset_dmc(env)
    if natural_reset:
        obs = {k: np.asarray(v) for k, v in reset_obs.items()
               if k not in ('reward', 'is_first', 'is_last', 'is_terminal')}
    else:
        set_physics(env, qpos, qvel)
        # Re-derive the obs from the dm_control time_step machinery. We grab the
        # raw observation through from_dm's internal env to skip a re-step.
        dm_obs = env._dmenv.task.get_observation(env._dmenv.physics)
        obs = {k.replace('/', '_'): np.atleast_1d(np.asarray(v))
               for k, v in dm_obs.items()}
        if 'reward' in obs:
            obs['obs_reward'] = obs.pop('reward')
    obs['reward'] = np.float32(0.0)
    obs['is_first'] = np.bool_(True)
    obs['is_last'] = np.bool_(False)
    obs['is_terminal'] = np.bool_(False)
    return obs


def mc_return(
    env: DMC,
    qpos: np.ndarray,
    qvel: np.ndarray,
    init_action: np.ndarray,
    policy_fn: Callable[[dict], np.ndarray],
    discount: float = 0.99,
    max_steps: int = 1000,
    action_key: str = 'action',
) -> float:
    """Roll out from (qpos, qvel) with `init_action` applied at t=0, then the
    policy thereafter, returning the discounted return.

    `policy_fn(obs)` must return an action array matching `act_space[action_key]`.
    """
    _reset_dmc(env)
    set_physics(env, qpos, qvel)

    act_space = env.act_space
    discounted = 0.0
    gamma = 1.0

    action = {k: np.zeros(s.shape, s.dtype) for k, s in act_space.items() if k != 'reset'}
    action[action_key] = np.asarray(init_action, dtype=np.float32)
    action['reset'] = np.bool_(False)

    for t in range(max_steps):
        tran = env.step(action)
        if t > 0:
            discounted += gamma * float(tran['reward'])
            gamma *= discount
        else:
            # First step is the (s, a)-conditioned step; reward is for that
            # transition. Use undiscounted weight for it.
            discounted += float(tran['reward'])
            gamma = discount
        if tran['is_last']:
            break
        act = policy_fn(tran)
        action = {action_key: np.asarray(act, dtype=np.float32), 'reset': np.bool_(False)}
    return discounted
