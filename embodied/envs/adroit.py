from embodied.envs.from_gymnasium import Gymnasium
from typing import Tuple
from gymnasium.wrappers import TimeLimit


class AdroitEnv(Gymnasium):
    def __init__(self, env_name: str = 'pen', sparse_reward: bool = True, size: Tuple = (64, 64),
                 time_limit: int =  200,
                 *args, **kwargs):
        assert env_name in ['pen', 'hammer', 'door']
        reward_type = 'sparse' if sparse_reward else 'dense'
        env = None
        if env_name == 'pen':
            from adroit_envs.adroit_pen import AdroitHandPenEnv
            env = AdroitHandPenEnv(reward_type=reward_type, render_mode='rgb_array',
                                   height=size[0],
                                   width=size[0],
                                   )
        elif env_name == 'hammer':
            from adroit_envs.adroit_hammer import AdroitHandHammerEnv
            env = AdroitHandHammerEnv(reward_type=reward_type, render_mode='rgb_array',
                                      height=size[0],
                                      width=size[0],
                                      )
        elif env_name == 'door':
            from adroit_envs.adroit_hand import AdroitHandDoorEnv
            env = AdroitHandDoorEnv(reward_type=reward_type, render_mode='rgb_array',
                                    height=size[0],
                                    width=size[0],
                                    )
        env = TimeLimit(env, max_episode_steps=time_limit)
        super().__init__(env=env, info_dict={'success': bool}, *args, **kwargs)
