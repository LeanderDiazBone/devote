from embodied.envs.from_gymnasium import Gymnasium
from typing import Tuple
from gymnasium.wrappers import TimeLimit


class InsertionEnv(Gymnasium):
    def __init__(self,
                 env_name: str = 'tactile_insertion',
                 no_gripping: bool = False,
                 no_rotation: bool = True,
                 start_grasped: bool = False,
                 state_type: str = 'vision_and_touch',
                 camera_idx: int = 0,
                 symlog_tactile: bool = True,
                 env_id: int = -1,
                 im_size: int = 64,
                 tactile_shape: Tuple = (16, 16),
                 time_limit: int = 1_000,
                 num_init_grasp_steps: int = 0,
                 multi_obj: bool = True,
                 *args, **kwargs):
        assert env_name in ['tactile_insertion', 'tactile_exploration', 'tactile_hand_exploration']
        if env_name in ['tactile_insertion', 'tactile_exploration']:
            if env_name == 'tactile_insertion':
                from tactile_envs.envs.insertion import InsertionEnv as Insertion
                env_cls = Insertion
                env_kwargs = {}
            else:
                from tactile_envs.envs.exploration import ExplorationEnv
                env_cls = ExplorationEnv
                env_kwargs = {'multi_obj': multi_obj}

            env = env_cls(
                no_gripping=no_gripping,
                no_rotation=no_rotation,
                start_grasped=start_grasped,
                state_type=state_type,
                camera_idx=camera_idx,
                symlog_tactile=symlog_tactile,
                env_id=env_id,
                im_size=im_size,
                tactile_shape=tactile_shape,
                num_init_grasp_steps=num_init_grasp_steps,
                **env_kwargs,
            )
        else:
            from tactile_envs.envs.hand_exploration import HandExplorationEnv
            assert not start_grasped, "hand environment only works without grasping"
            if state_type == 'vision_and_touch':
                env = HandExplorationEnv(
                    start_grasped=start_grasped,
                    state_type='vision',
                    camera_idx=camera_idx,
                    env_id=env_id,
                    im_size=im_size,
                )
                from tactile_envs.utils.add_tactile import AddTactile
                env = AddTactile(env, use_symlog=symlog_tactile)
            else:
                env = HandExplorationEnv(
                    start_grasped=start_grasped,
                    state_type=state_type,
                    camera_idx=camera_idx,
                    env_id=env_id,
                    im_size=im_size,
                )

        env = TimeLimit(env, max_episode_steps=time_limit)
        super().__init__(env=env, info_dict={'is_success': int, 'grasped': int},
                         add_render_wrapper=False,
                         *args, **kwargs)
