import pathlib
import warnings
from functools import partial as bind
import embodied
from experiments import make_agent, make_replay, make_logger, make_env, Logger, hash_dict
import numpy as np
import os
import sys
import argparse
import re

directory = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(directory.parent))
sys.path.insert(0, str(directory.parent.parent))
__package__ = directory.name

warnings.filterwarnings('ignore', '.*box bound precision lowered.*')
warnings.filterwarnings('ignore', '.*using stateful random seeds*')
warnings.filterwarnings('ignore', '.*is a deprecated alias for.*')
warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

def _resolve_craftax_task(task: str, use_image: bool) -> str:
    if not isinstance(task, str) or not task.startswith('craftax_'):
        return task
    target = 'Pixels' if use_image else 'Symbolic'
    return re.sub(r'(-v\d+)', rf'-{target}\1', task)


def _to_wandb_config(value):
    if isinstance(value, dict):
        return {k: _to_wandb_config(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_to_wandb_config(v) for v in value]
    return value


def dict_to_sys_argv(d):
    argv = []
    for key, value in d.items():
        if isinstance(value, list):
            argv.append(f"--{key}")
            for val in value:
                argv.append(f"{str(val)}")
        else:
            argv.append(f"--{key}")
            argv.append(f"{str(value)}")
    return argv


def experiment(
        script: str = 'train_eval',
        policy_eval_mc_episodes: int = 128,
        policy_eval_mc_max_steps: int = 1000,
        policy_eval_mc_max_step_limit: int = 0,
        env_dmc_max_step_limit: int = 0,
        env_dmc_distract_difficulty: str = '',
        env_dmc_distract_dataset_path: str = '',
        env_dmc_distract_dynamic: bool = False,
        env_dmc_distract_videos: str = 'train',
        env_mjp_length: int = 0,
        env_mjp_action_scale: float = 0.0,
        policy_eval_every: int = 5000,
        policy_eval_discount: float = 0.99,
        policy_eval_visit_steps: int = 0,
        policy_eval_corrector: str = 'fresh_train',
        policy_eval_checkpoint_config: str = '',
        policy_eval_diagnostics_every: int = 0,
        policy_eval_diagnostics_points: int = 24,
        policy_eval_diagnostics_horizons: str = '64,200,2499',
        seed: int = 0,
        alg: str = 'MultiMex',
        entity: str = 'sukhijab',
        project_name: str = 'Test',
        logs_dir: str = './logs/{timestamp}',
        total_steps: int = 1e6,
        expl_until: int = 0,
        num_envs: int = 1,
        num_envs_eval: int = 4,
        train_ratio: int = 512,
        eval_every: float = 120,
        train_rollout_steps: int = 10,
        config_class: str = 'dmc',
        task: str = 'dmc_cartpole_swingup_sparse',
        debug: bool = True,
        action_cost: float = 0.0,
        init_temp: float = 1.0,
        train_temp: bool = True,
        use_dyn_disg: bool = True,
        alternate_exploration: bool = False,
        log_video: bool = False,
        action_repeat: int = 0,
        outscale: float = 0.0,
        use_tolerance_reward: bool = False,
        constraint_weight: float = 1.0,
        temp_wd: float = 0.1,
        beta: float = 0.0,
        beta_learn: bool = False,
        beta_init: float = 1.0,
        beta_target_kl: float = 0.05,
        beta_pessimism: float = 0.0,
        beta_log_min: float = -10.0,
        beta_log_max: float = 7.5,
        exp_name: str = None,
        group_name: str = "default",
        replay_critic_bootstrap: str = "imag",
        num_heads: int=2,
        boot_prob: float=1.0,
        target_mode: str ='random_min',
        random_target_k: int=2,
        value_prior_scale: float=0.0,
        reward_prior_scale: float=0.0,
        visual_prior: str='none',
        visual_prior_dim: int=8,
        pessimism: float=-1.0,
        td_lambda: float=0.0,
        td_lambda_horizon: int=0,
        td_target_mode: str='joint',
        bellman_z_from_target: int=0,
        residual_transform: float=0.0,
        residual_transform_reduction: float=10.0,
        target_entropy_scale_disc=0.75,
        target_entropy_scale_cont=0.5,
        actor_dist_cont: str='normal',  # normal | trunc_normal | squashed_normal
        init_alpha: float=1.0,
        critic_dist: str = "symexp_twohot",
        critic_bins: int = 255,
        residual_bootstrap_dist: str = "mse",
        actor_layers: int = -1,
        actor_units: int = -1,
        critic_layers: int = -1,
        critic_units: int = -1,
        loss_scales_critic: float = 1.0,
        loss_scales_exp_actor: float = 1.0,
        loss_scales_actor: float = 1.0,
        loss_scales_alpha: float = 1.0,
        loss_scales_temp: float = 1.0,
        critic_outscale: float=0.0,
        batch_size: int=16,
        policy_mode: str="actor",
        actor_obj_norm_mode: str="std",
        replay_context: int=1,
        contdisc: int=1,
        add_global_position: bool = False,
        use_jax_driver: bool = False,
        use_image: bool = False,
        frame_stack: int = 1,
        achievement_reward_weights: str = '',
        log_images: bool = False,
        benchmark_id: str = "",
        sample_ruleset: bool = False,
        batch_length: int = 65,
        batch_length_eval: int = 33,
        actor_heads: int = 1,
        eval_trajectory_dir: bool = False,
        explore_trajectory_dir: bool = False,
        big_report_every: int = 0,
        save_every: int = -1,
        replay_size: int = 1e6,
        exp_obj: str = "ucb",
        normalize_exp_mean: int = 0,
        sampling_temp: float = 1.0,
        num_samples: int = 4,
        prior_layers: int = 0,
        prior_units: int = 0,
        prior_use_rff: int = 0,
        prior_length_scale: float = 1.0,
        prior_inputs: int = '[deter,stoch]',
        prior_act: str = 'silu',
        prior_outact: str = 'tanh',
        prior_action_independent: int = 0,
        prior_corrector: int = 0,
        prior_epistemic: int = 0,
        prior_epistemic_dim: int = 8,
        prior_epistemic_std: float = 1.0,
        prior_epistemic_samples: int = 1,
        intrinsic_mode: str = "model",
        model_size: str = None,
        enumerate_actions: bool = False,
        cand_grid_points: int = 0,
        cand_policy: str = "softmax",
        greedy_eps: float = 0.1,
        discretize: int = 0,
        enc_typ: str = "simple",
        dyn_typ: str = "rssm",
        freeze_wm: bool = False,
        freeze_corrector: bool = False,
        freeze_all: bool = False,
        load_keys: str = '',
        from_checkpoint: str = '',
        use_exp_critic: int = 1,
        use_entropy_backup_sac: int = 1,
        test_mode: int = 1,
        dynamics_complexity_dims: int = 0,
        dynamics_complexity_seed: int = 0,
        dynamics_complexity_noise_std: float = 0.0,
        dynamics_complexity_length_scale: float = 1.0,
        dynamics_complexity_features: int = 128,
        dynamics_complexity_amplitude: float = 1.0,
        dynamics_complexity_input_scale: float = 1.0,
        dynamics_complexity_key: str = 'state',
        dynamics_complexity_source_keys: str = 'auto',
        dynamics_complexity_mode: str = 'append',
        ac_inputs: str = 'wm',
        image_shift_pad: int = 0,
        drq_num_views: int = 1,
        bro_actor: int = 0,
        bro_critic: int = 0,
        slow_critic_fraction: float = 0.02,
        actor_lr: float = 4e-5,
        critic_lr: float = 3e-4,
        encoder_lr: float = 4e-5,
        critic_wd: float = 1e-2,
        actor_wd: int = 0,
        opt_par_noise_scale: float = 0.0,
        discount: float = -1.0,
        actor_update_delay: int = 1,
        actor_update_warmup: int = 0,
        actor_kl_reg: float = 0.0,
        exp_alpha_scale: float = 1.0,
        pc_target_actor: str = 'actor',
        td_target_actor: str = 'actor',
        reset_every: int = 0,
        reset_max_step: int = 0,
        reset_keys: str = '',
        reset_mode: str = 'hard',
        reset_soft_alpha: float = 0.5,
        reset_num_layers: int = 0,
        log_coverage: bool = False,
        log_coverage_heatmap: bool = False,
        coverage_bins_per_cell: int = 4,
        reward_scale: float = 1.0,
        offline_replay: bool = False,
        dual_batch_training: bool = False,
        corrector_update_every: int = 1,
        ac_update_every: int = 1,
        replay_fracs_uniform: float = 1.0,
        replay_fracs_priority: float = 0.0,
        replay_fracs_recency: float = 0.0,
        replay_res_fracs_uniform: float = 1.0,
        replay_res_fracs_priority: float = 0.0,
        replay_res_fracs_recency: float = 0.0,
        replay_priosignal_ac: str = 'none',
        replay_priosignal_res: str = 'none',
        ):
    

    from sys import platform
    if platform == 'darwin':
        import os
        os.environ['MUJOCO_GL'] = 'glfw'

    task_suite = task.split('_', 1)[0]
    if task_suite == 'craftax':
        task = _resolve_craftax_task(task, bool(use_image))

    config = dict(
        alg=alg,
        total_steps=total_steps,
        num_envs=num_envs,
        train_ratio=train_ratio,
        env_class=config_class,
        task=task,
        action_cost=action_cost,
        alternate_exploration=alternate_exploration,
        init_temp=init_temp,
        train_temp=train_temp,
        use_dyn_disg=use_dyn_disg,
        action_repeat=action_repeat,
        outscale=outscale,
        use_tolerance_reward=use_tolerance_reward,
        constraint_weight=constraint_weight,
        temp_wd=temp_wd,
        beta=beta,
        add_global_position=add_global_position,
    )
    if exp_name is None:
        exp_name = alg + "_" + config_class + "_" + task
    task_suite = task.split('_', 1)[0]
    should_log_images = bool(log_images)
    supports_independent_log_images = task_suite in (
        'mjp', 'xminigrid', 'xlandminigrid', 'craftax', 'ogbench')
    should_log_videos = bool(should_log_images or (log_video and seed == 0))

    wandb_dict = dict(project=project_name, exp_name=exp_name, entity=entity, dir=logs_dir, group=group_name)

    sqrt2 = float(np.sqrt(2.0))
    ac_overrides = {}
    if bro_actor:
        ac_overrides.update({
            'actor.residual': True,
            'actor.norm': 'layer',
            'actor.act': 'relu',
            'actor.winit': 'ortho',
            'actor.hidden_outscale': sqrt2,
            'actor.norm_eps': 1e-6,
        })
    if bro_critic:
        ac_overrides.update({
            'critic.residual': True,
            'critic.norm': 'layer',
            'critic.act': 'relu',
            'critic.winit': 'ortho',
            'critic.hidden_outscale': sqrt2,
            'critic.norm_eps': 1e-6,
        })

    args_dict = {
        'seed': seed,
        'logdir': logs_dir,
        'replay.size': int(min(total_steps, int(replay_size))),
        'run.script': script,
        'run.policy_eval_mc_episodes': int(policy_eval_mc_episodes),
        'run.policy_eval_mc_max_steps': int(policy_eval_mc_max_steps),
        'run.policy_eval_mc_max_step_limit': int(policy_eval_mc_max_step_limit),
        'env.dmc.max_step_limit': int(env_dmc_max_step_limit),
        'env.dmc.distract_difficulty': str(env_dmc_distract_difficulty),
        'env.dmc.distract_dataset_path': str(env_dmc_distract_dataset_path),
        'env.dmc.distract_dynamic': bool(env_dmc_distract_dynamic),
        'env.dmc.distract_videos': str(env_dmc_distract_videos),
        'run.policy_eval_every': int(policy_eval_every),
        'run.policy_eval_discount': float(policy_eval_discount),
        'run.policy_eval_visit_steps': int(policy_eval_visit_steps),
        'run.policy_eval_corrector': policy_eval_corrector,
        'run.policy_eval_checkpoint_config': policy_eval_checkpoint_config,
        'run.policy_eval_diagnostics_every': int(policy_eval_diagnostics_every),
        'run.policy_eval_diagnostics_points': int(policy_eval_diagnostics_points),
        'run.policy_eval_diagnostics_horizons': policy_eval_diagnostics_horizons,
        'run.steps': int(total_steps),
        'run.expl_until': int(expl_until),
        'run.num_envs': int(num_envs),
        'run.train_ratio': train_ratio,
        'run.num_envs_eval': int(num_envs_eval),
        'run.eval_every': float(eval_every),
        'run.train_rollout_steps': int(train_rollout_steps),
        'run.use_jax_driver': use_jax_driver,
        'task': task,
        'wrapper.action_cost': action_cost,
        'wrapper.frame_stack': int(frame_stack),
        'int_rew_model.model.outscale': outscale,
        'env.dmc.repeat': action_repeat,
        'wrapper.use_tolerance_reward': use_tolerance_reward,
        'beta': beta,
        'beta_learn': beta_learn,
        'beta_init': beta_init,
        'beta_target_kl': beta_target_kl,
        'beta_pessimism': beta_pessimism,
        'beta_log_min': beta_log_min,
        'beta_log_max': beta_log_max,
        'use_exp_critic': bool(use_exp_critic),
        'use_entropy_backup_sac': use_entropy_backup_sac,
        'replay_critic_bootstrap': replay_critic_bootstrap,
        'ens_heads': num_heads,
        'boot_prob': boot_prob,
        'sac.target_mode': target_mode,
        'sac.random_target_k': random_target_k,
        'value_prior_scale': value_prior_scale,
        'reward_prior_scale': reward_prior_scale,
        'visual_prior': visual_prior,
        'visual_prior_dim': visual_prior_dim,
        'pessimism': pessimism,
        'td_lambda': td_lambda,
        'td_lambda_horizon': td_lambda_horizon,
        'td_target_mode': td_target_mode,
        'bellman_z_from_target': bool(bellman_z_from_target),
        'residual_transform': residual_transform,
        'residual_transform_reduction': residual_transform_reduction,
        'sac.target_entropy_scale_disc': target_entropy_scale_disc,
        'sac.target_entropy_scale_cont': target_entropy_scale_cont,
        'actor_dist_cont': actor_dist_cont,
        'sac.init_alpha': init_alpha,
        'temp.init_temp': init_temp,
        'critic.dist': critic_dist,
        'critic.bins': critic_bins,
        'residual_bootstrap_dist': residual_bootstrap_dist,
        'loss_scales.critic': loss_scales_critic,
        'loss_scales.exp_actor': loss_scales_exp_actor,
        'loss_scales.actor': loss_scales_actor,
        'loss_scales.alpha': loss_scales_alpha,
        'loss_scales.temp': loss_scales_temp,
        'critic.outscale': critic_outscale,
        **({'actor.layers': actor_layers} if actor_layers > 0 else {}),
        **({'actor.units': actor_units} if actor_units > 0 else {}),
        **({'critic.layers': critic_layers} if critic_layers > 0 else {}),
        **({'critic.units': critic_units} if critic_units > 0 else {}),
        'batch_size': batch_size,
        'policy_mode': policy_mode,
        'sac.actor_obj_norm_mode': actor_obj_norm_mode,
        'replay_context': replay_context,
        'contdisc': contdisc,
        'env.loconav.add_global_position': add_global_position,
        'wrapper.log_images': (should_log_images and supports_independent_log_images),
        'report_videos': should_log_videos,
        'batch_length': batch_length,
        'batch_length_eval': batch_length_eval,
        'actor_ens_heads': actor_heads,
        'run.eval_trajectory_dir': eval_trajectory_dir,
        'run.explore_trajectory_dir': explore_trajectory_dir,
        'run.big_report_every': int(big_report_every),
        'log_metrics_table': int(big_report_every) > 0,
        'run.save_every': save_every,
        'exp_obj': exp_obj,
        'normalize_exp_mean': bool(normalize_exp_mean),
        'sampling_temp': sampling_temp,
        'num_samples': num_samples,
        'critic_prior.layers': prior_layers,
        'critic_prior.units': prior_units,
        'critic_prior.use_rff': bool(prior_use_rff),
        'critic_prior.length_scale': prior_length_scale,
        'critic_prior.inputs': prior_inputs,
        'critic_prior.act': prior_act,
        'critic_prior.outact': prior_outact,
        'critic_prior.action_independent': bool(prior_action_independent),
        'critic_prior.corrector': bool(prior_corrector),
        'critic_prior.epistemic': bool(prior_epistemic),
        'critic_prior.epistemic_dim': prior_epistemic_dim,
        'critic_prior.epistemic_std': prior_epistemic_std,
        'critic_prior.epistemic_samples': prior_epistemic_samples,
        'intrinsic_mode': intrinsic_mode,
        'enumerate_actions': bool(enumerate_actions),
        'cand_grid_points': int(cand_grid_points),
        'cand_policy': cand_policy,
        'greedy_eps': greedy_eps,
        'wrapper.discretize': int(discretize),
        'enc.typ': enc_typ,
        'dyn.typ': dyn_typ,
        'freeze_wm': freeze_wm,
        'freeze_corrector': bool(freeze_corrector),
        'freeze_all': freeze_all,
        'run.load_keys': load_keys,
        'run.from_checkpoint': from_checkpoint,
        'run.test_mode': bool(test_mode),
        'wrapper.dynamics_complexity.dims': int(dynamics_complexity_dims),
        'wrapper.dynamics_complexity.seed': int(dynamics_complexity_seed),
        'wrapper.dynamics_complexity.noise_std': float(dynamics_complexity_noise_std),
        'wrapper.dynamics_complexity.length_scale': float(dynamics_complexity_length_scale),
        'wrapper.dynamics_complexity.features': int(dynamics_complexity_features),
        'wrapper.dynamics_complexity.amplitude': float(dynamics_complexity_amplitude),
        'wrapper.dynamics_complexity.input_scale': float(dynamics_complexity_input_scale),
        'wrapper.dynamics_complexity.key': dynamics_complexity_key,
        'wrapper.dynamics_complexity.source_keys': dynamics_complexity_source_keys,
        'wrapper.dynamics_complexity.mode': dynamics_complexity_mode,
        'ac_inputs': ac_inputs,
        'image_shift_pad': image_shift_pad,
        'drq_num_views': int(drq_num_views),
        'slow_critic_fraction': slow_critic_fraction,
        'lrs.actor_sac': actor_lr,
        'lrs.exp_actor_sac': actor_lr,
        'lrs.q': critic_lr,
        'lrs.q2': critic_lr,
        'lrs.enc': encoder_lr,
        'opt.wd': float(critic_wd),
        'opt.wd_pattern': (
            r'/(q|critic|intr_q|actor|actor_sac|exp_actor_sac)/.*/kernel$'
            if actor_wd else r'/(q|critic|intr_q)/.*/kernel$'),
        'opt.par_noise_scale': float(opt_par_noise_scale),
        'opt.par_noise_seed': int(seed),
        'discount': discount,
        'actor_update_delay': actor_update_delay,
        'actor_update_warmup': actor_update_warmup,
        'actor_kl_reg': actor_kl_reg,
        'exp_alpha_scale': exp_alpha_scale,
        'loss_scales.exp_alpha': exp_alpha_scale,
        'pc_target_actor': pc_target_actor,
        'td_target_actor': td_target_actor,
        'run.reset_every': int(reset_every),
        'run.reset_max_step': int(reset_max_step),
        'run.reset_keys': reset_keys,
        'run.reset_mode': reset_mode,
        'run.reset_soft_alpha': float(reset_soft_alpha),
        'run.reset_num_layers': int(reset_num_layers),
        'run.log_coverage': bool(log_coverage),
        'run.log_coverage_heatmap': bool(log_coverage_heatmap),
        'run.coverage_bins_per_cell': int(coverage_bins_per_cell),
        'env.dmc.reward_scale': float(reward_scale),
        'env.ogbench.reward_scale': float(reward_scale),
        'env.mjp.reward_scale': float(reward_scale),
        'env.bsuite.reward_scale': float(reward_scale),
        'run.offline_replay': bool(offline_replay),
        # Dual-batch training and per-sampler prioritized replay.
        'dual_batch_training': bool(dual_batch_training),
        'corrector_update_every': int(corrector_update_every),
        'ac_update_every': int(ac_update_every),
        'replay.fracs.uniform': float(replay_fracs_uniform),
        'replay.fracs.priority': float(replay_fracs_priority),
        'replay.fracs.recency': float(replay_fracs_recency),
        'replay.res_fracs.uniform': float(replay_res_fracs_uniform),
        'replay.res_fracs.priority': float(replay_res_fracs_priority),
        'replay.res_fracs.recency': float(replay_res_fracs_recency),
        'replay.priosignal_ac': str(replay_priosignal_ac),
        'replay.priosignal_res': str(replay_priosignal_res),
        **ac_overrides,
        # 'run.driver_parallel': False,
    }
    if task_suite == 'dmc':
        args_dict['env.dmc.image'] = bool(use_image)
        args_dict['run.log_keys_video'] = ['image' if use_image else 'log_image'] if should_log_images else ['nothing']
        args_dict['enc.spaces'] = 'image' if use_image else '.*'
        args_dict['dec.spaces'] = 'image' if use_image else '.*'
    elif task_suite == 'mjp':
        # For MJP, use a single config and switch between full state and full image
        args_dict['env.mjp.image'] = bool(use_image)
        args_dict['env.mjp.log_image'] = should_log_images
        args_dict['run.log_keys_video'] = ['image' if use_image else 'log_image'] if should_log_images else ['nothing']
        args_dict['enc.spaces'] = 'image' if use_image else '.*'
        args_dict['dec.spaces'] = 'image' if use_image else '.*'
        if env_mjp_length > 0:
            args_dict['env.mjp.length'] = int(env_mjp_length)
        # 0.0 keeps the env's own default (panda: 0.05); any non-zero value
        # is forwarded to PandaEnv as a config_override in BatchedMujocoPlayground.
        if env_mjp_action_scale != 0.0:
            args_dict['env.mjp.action_scale'] = float(env_mjp_action_scale)
    elif task_suite in ('xminigrid', 'xlandminigrid'):
        args_dict[f'env.{task_suite}.image'] = bool(use_image)
        if benchmark_id:
            args_dict[f'env.{task_suite}.benchmark'] = benchmark_id
            args_dict[f'env.{task_suite}.sample_ruleset'] = True
        elif sample_ruleset:
            args_dict[f'env.{task_suite}.sample_ruleset'] = True
        args_dict['enc.spaces'] = 'image' if use_image else '.*'
        args_dict['dec.spaces'] = 'image' if use_image else '.*'
        args_dict['run.log_keys_video'] = ['image' if use_image else 'log_image'] if should_log_images else ['nothing']
    elif task_suite == 'craftax':
        args_dict['enc.spaces'] = 'image' if use_image else '.*'
        args_dict['dec.spaces'] = 'image' if use_image else '.*'
        args_dict['env.craftax.achievement_reward_weights'] = achievement_reward_weights
        args_dict['run.log_keys_video'] = ['image' if use_image else 'log_image'] if should_log_images else ['nothing']
    elif task_suite == 'ogbench':
        ogbench_task = task.split('_', 1)[1]
        use_ogbench_image = (
            ogbench_task.startswith('visual-') or
            ogbench_task.startswith('powderworld-'))
        obs_key = 'image' if use_ogbench_image else 'state'
        args_dict['enc.spaces'] = obs_key
        args_dict['dec.spaces'] = obs_key
        args_dict['run.log_keys_video'] = [
            'image' if use_ogbench_image else 'log_image'
        ] if should_log_images else ['nothing']

    if debug:
        args_dict['jax.platform'] = 'cpu'
        args_dict['jax.transfer_guard'] = False
        args_dict['jax.debug'] = True
        args_dict['run.driver_parallel'] = False
        # args_dict['configs'] = [config_class, 'debug']

    embodied.print(r"---  __  __  __  __ ---")
    embodied.print(r"--- |  \/  ||  \/  |---")
    embodied.print(r"--- | |\/| || |\/| |---")
    embodied.print(r"--- | |  | || |  | |---")
    embodied.print(r"--- |_|  |_||_|  |_|---")

    import ruamel.yaml as yaml

    if script == 'policy_eval':
        if policy_eval_corrector not in ('load_freeze', 'load_train', 'fresh_train'):
            raise ValueError(f'Unknown policy_eval_corrector: {policy_eval_corrector}')
        args_dict['freeze_corrector'] = policy_eval_corrector == 'load_freeze'
    other = dict_to_sys_argv(args_dict)

    # parsed, other = embodied.Flags(configs=['defaults']).parse_known(other)
    exp_config = yaml.YAML(typ='safe').load((embodied.Path(__file__).parent / 'configs.yaml').read())
    # extract default config
    config = embodied.Config(exp_config['defaults'])

    # update default config with parameters specific to the config_class
    config = config.update(exp_config[config_class])
    config = embodied.Flags(config).parse(other)
    config = config.update(
        logdir=config.logdir.format(timestamp=embodied.timestamp()),
        replay_length=config.replay_length or config.batch_length,
        replay_length_eval=config.replay_length_eval or config.batch_length_eval)
    if model_size:
        config = config.update(exp_config[model_size])
    # Explicit architecture flags are more specific than the broad
    # ``.*.units`` patterns in model-size blocks, so apply them last.
    config = config.update({
        **({'actor.layers': actor_layers} if actor_layers > 0 else {}),
        **({'actor.units': actor_units} if actor_units > 0 else {}),
        **({'critic.layers': critic_layers} if critic_layers > 0 else {}),
        **({'critic.units': critic_units} if critic_units > 0 else {}),
    })
    # convert config to dict and delete keys that are not required
    config = config.flat
    # only log the zeroth seed and nothing to the list (trainer does not accept an empty list)
    if not should_log_videos:
        config['run.log_keys_video'] = ['nothing']

    if not use_dyn_disg:
        if 'int_rew_model.model.dist.dyn' in config:
            config.pop('int_rew_model.model.dist.dyn')
        if 'int_rew_model.model.dist.rew' in config:
            config.pop('int_rew_model.model.dist.rew')

    # convert updated config back to embodied Config class
    config = embodied.Config(**config)
    wandb_dict['config'] = _to_wandb_config(dict(config))

    # define args and run the experiment
    args = embodied.Config(
        **config.run,
        logdir=config.logdir,
        task=config.task,
        seed=config.seed,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        batch_length_eval=config.batch_length_eval,
        replay_length=config.replay_length,
        replay_length_eval=config.replay_length_eval,
        replay_context=config.replay_context,
        alternate_exploration=alternate_exploration,
        dual_batch_training=config.dual_batch_training,
        # policy_eval needs action_repeat on `args`; config.env.dmc.repeat
        # is the source of truth (set in args_dict above).
        action_repeat=int(config.env.dmc.repeat),
    )
    print('Run script:', args.script)
    print('Logdir:', args.logdir)

    logdir = embodied.Path(args.logdir)
    if not args.script.endswith(('_env', '_replay')):
        logdir.mkdir()
        config.save(logdir / 'config.yaml')

    def init():
        embodied.timer.global_timer.enabled = args.timer

    embodied.distr.Process.initializers.append(init)
    init()

    env_ctor = bind(make_env, config)
    train_env_ctor = env_ctor
    eval_env_ctor = env_ctor
    if should_log_images and not use_image:
        train_env_ctor = bind(make_env, config, wrapper_log_images=False)
        eval_env_ctor = bind(make_env, config, wrapper_log_images=True)

    def make_reporter(*args, **kwargs):
        # Load reporting after the run loop starts process-backed environments.
        from optimistic_curiosity.report import Reporter
        return Reporter(*args, **kwargs)

    if args.script == 'train':
        embodied.run.train(
            bind(lambda cfg: make_agent(config=cfg, alg=alg), config),
            bind(make_replay, config, 'replay'),
            env_ctor,
            bind(lambda cfg: make_logger(config=cfg, wandb_config=wandb_dict), config), args)

    elif args.script == 'train_eval':
        embodied.run.train_eval(
            bind(lambda cfg: make_agent(config=cfg, alg=alg), config),
            bind(make_replay, config, 'replay'),
            bind(make_replay, config, 'eval_replay', is_eval=True),
            bind(make_replay, config, 'explore_replay', is_eval=True),
            train_env_ctor,
            eval_env_ctor,
            bind(lambda cfg: make_logger(config=cfg, wandb_config=wandb_dict), config), args, make_reporter=make_reporter)

    elif args.script == 'train_holdout':
        assert config.eval_dir
        embodied.run.train_holdout(
            bind(make_agent, config),
            bind(make_replay, config, 'replay'),
            bind(make_replay, config, config.eval_dir),
            bind(make_env, config),
            bind(make_logger, config), args)

    elif args.script == 'eval_only':
        embodied.run.eval_only(
            bind(make_agent, config),
            eval_env_ctor,
            bind(make_logger, config), args)

    elif args.script == 'policy_eval':
        embodied.run.policy_eval(
            bind(lambda cfg: make_agent(config=cfg, alg=alg), config),
            bind(make_replay, config, 'replay'),
            env_ctor,
            bind(lambda cfg: make_logger(config=cfg, wandb_config=wandb_dict), config),
            args, make_reporter=make_reporter)

    elif args.script == 'parallel':
        embodied.run.parallel.combined(
            bind(make_agent, config),
            bind(make_replay, config, 'replay', rate_limit=True),
            env_ctor,
            bind(make_logger, config), args)

    elif args.script == 'parallel_env':
        envid = args.env_replica
        if envid < 0:
            envid = int(os.environ['JOB_COMPLETION_INDEX'])
        embodied.run.parallel.env(
            env_ctor, envid, args, False)

    elif args.script == 'parallel_replay':
        embodied.run.parallel.replay(
            bind(make_replay, config, 'replay', rate_limit=True), args)

    elif args.script == 'parallel_with_eval':
        embodied.run.parallel_with_eval.combined(
            bind(make_agent, config),
            bind(make_replay, config, 'replay', rate_limit=True),
            bind(make_replay, config, 'replay_eval', is_eval=True),
            train_env_ctor,
            eval_env_ctor,
            bind(make_logger, config), args)

    elif args.script == 'parallel_with_eval_env':
        envid = args.env_replica
        if envid < 0:
            envid = int(os.environ['JOB_COMPLETION_INDEX'])
        is_eval = envid >= args.num_envs
        env_ctor = eval_env_ctor if is_eval else train_env_ctor
        embodied.run.parallel_with_eval.parallel_env(
            env_ctor, envid, args, True, is_eval)

    elif args.script == 'parallel_with_eval_replay':
        embodied.run.parallel_with_eval.parallel_replay(
            bind(make_replay, config, 'replay', rate_limit=True),
            bind(make_replay, config, 'replay_eval', is_eval=True), args)

    else:
        raise NotImplementedError(args.script)


def main(args):
    """"""
    from pprint import pprint
    print(args)
    """ generate experiment hash and set up redirect of output streams """
    exp_hash = hash_dict(args.__dict__)
    if args.exp_result_folder is not None:
        os.makedirs(args.exp_result_folder, exist_ok=True)
        log_file_path = os.path.join(args.exp_result_folder, '%s.log ' % exp_hash)
        logger = Logger(log_file_path)
        sys.stdout = logger
        sys.stderr = logger

    pprint(args.__dict__)
    print('\n ------------------------------------ \n')

    """ Experiment core """
    np.random.seed(args.seed)

    run_name = f'{args.exp_name or args.alg + "_" + args.config_class + "_" + args.task}_seed_{args.seed}_{{timestamp}}'
    experiment(
        script=args.script,
        policy_eval_mc_episodes=args.policy_eval_mc_episodes,
        policy_eval_mc_max_steps=args.policy_eval_mc_max_steps,
        policy_eval_mc_max_step_limit=args.policy_eval_mc_max_step_limit,
        env_dmc_max_step_limit=args.env_dmc_max_step_limit,
        env_dmc_distract_difficulty=args.env_dmc_distract_difficulty,
        env_dmc_distract_dataset_path=args.env_dmc_distract_dataset_path,
        env_dmc_distract_dynamic=bool(args.env_dmc_distract_dynamic),
        env_dmc_distract_videos=args.env_dmc_distract_videos,
        env_mjp_length=args.env_mjp_length,
        env_mjp_action_scale=args.env_mjp_action_scale,
        policy_eval_every=args.policy_eval_every,
        policy_eval_discount=args.policy_eval_discount,
        policy_eval_visit_steps=args.policy_eval_visit_steps,
        policy_eval_corrector=args.policy_eval_corrector,
        policy_eval_checkpoint_config=args.policy_eval_checkpoint_config,
        policy_eval_diagnostics_every=args.policy_eval_diagnostics_every,
        policy_eval_diagnostics_points=args.policy_eval_diagnostics_points,
        policy_eval_diagnostics_horizons=args.policy_eval_diagnostics_horizons,
        logs_dir=os.path.join(args.logs_dir, args.group_name, run_name),
        entity=args.entity,
        project_name=args.project_name,
        alg=args.alg,
        total_steps=args.total_steps,
        expl_until=args.expl_until,
        num_envs=args.num_envs,
        num_envs_eval=args.num_envs_eval,
        train_ratio=args.train_ratio,
        eval_every=args.eval_every,
        train_rollout_steps=args.train_rollout_steps,
        config_class=args.config_class,
        task=args.task,
        debug=bool(args.debug),
        seed=args.seed,
        action_cost=args.action_cost,
        init_temp=args.init_temp,
        use_dyn_disg=bool(args.use_dyn_disg),
        train_temp=bool(args.train_temp),
        action_repeat=args.action_repeat,
        outscale=args.outscale,
        use_tolerance_reward=bool(args.use_tolerance_reward),
        constraint_weight=args.constraint_weight,
        temp_wd=args.temp_wd,
        beta = args.beta,
        beta_learn=bool(args.beta_learn),
        beta_init=args.beta_init,
        beta_target_kl=args.beta_target_kl,
        beta_pessimism=args.beta_pessimism,
        beta_log_min=args.beta_log_min,
        beta_log_max=args.beta_log_max,
        exp_name = args.exp_name,
        group_name = args.group_name,
        replay_critic_bootstrap = args.replay_critic_bootstrap,
        num_heads=args.num_heads,
        boot_prob=args.boot_prob,
        target_mode=args.target_mode,
        random_target_k=args.random_target_k,
        value_prior_scale=args.value_prior_scale,
        reward_prior_scale=args.reward_prior_scale,
        visual_prior=args.visual_prior,
        visual_prior_dim=args.visual_prior_dim,
        pessimism=args.pessimism,
        td_lambda=args.td_lambda,
        td_lambda_horizon=args.td_lambda_horizon,
        td_target_mode=args.td_target_mode,
        bellman_z_from_target=args.bellman_z_from_target,
        residual_transform=args.residual_transform,
        residual_transform_reduction=args.residual_transform_reduction,
        target_entropy_scale_disc=args.target_entropy_scale_disc,
        target_entropy_scale_cont=args.target_entropy_scale_cont,
        actor_dist_cont=args.actor_dist_cont,
        init_alpha=args.init_alpha,
        critic_dist = args.critic_dist,
        critic_bins = args.critic_bins,
        residual_bootstrap_dist = args.residual_bootstrap_dist,
        actor_layers = args.actor_layers,
        actor_units = args.actor_units,
        critic_layers = args.critic_layers,
        critic_units = args.critic_units,
        loss_scales_critic = args.loss_scales_critic,
        loss_scales_exp_actor = args.loss_scales_exp_actor,
        loss_scales_actor = args.loss_scales_actor,
        loss_scales_alpha = args.loss_scales_alpha,
        loss_scales_temp = args.loss_scales_temp,
        critic_outscale = args.critic_outscale,
        batch_size = args.batch_size,
        policy_mode = args.policy_mode,
        actor_obj_norm_mode = args.actor_obj_norm_mode,
        replay_context = args.replay_context,
        contdisc = args.contdisc,
        add_global_position=bool(args.add_global_position),
        use_jax_driver=bool(args.use_jax_driver),
        use_image=bool(args.use_image),
        frame_stack=args.frame_stack,
        log_images=bool(args.log_images),
        benchmark_id=args.benchmark_id,
        sample_ruleset=bool(args.sample_ruleset),
        batch_length=args.batch_length,
        batch_length_eval=args.batch_length_eval,
        actor_heads=args.actor_heads,
        eval_trajectory_dir=bool(args.eval_trajectory_dir),
        explore_trajectory_dir=bool(args.explore_trajectory_dir),
        big_report_every=args.big_report_every,
        save_every=args.save_every,
        replay_size=args.replay_size,
        exp_obj=args.exp_obj,
        normalize_exp_mean=args.normalize_exp_mean,
        sampling_temp=args.sampling_temp,
        num_samples=args.num_samples,
        prior_layers=args.prior_layers,
        prior_units=args.prior_units,
        prior_use_rff=args.prior_use_rff,
        prior_length_scale=args.prior_length_scale,
        intrinsic_mode=args.intrinsic_mode,
        model_size=args.model_size,
        enumerate_actions=bool(args.enumerate_actions),
        cand_grid_points=args.cand_grid_points,
        cand_policy=args.cand_policy,
        greedy_eps=args.greedy_eps,
        discretize=args.discretize,
        enc_typ=args.enc_typ,
        dyn_typ=args.dyn_typ,
        prior_inputs=args.prior_inputs,
        freeze_wm=bool(args.freeze_wm),
        freeze_corrector=bool(args.freeze_corrector),
        freeze_all=bool(args.freeze_all),
        load_keys=args.load_keys,
        from_checkpoint=args.from_checkpoint,
        use_exp_critic=args.use_exp_critic,
        use_entropy_backup_sac=args.use_entropy_backup_sac,
        prior_act=args.prior_act,
        prior_outact=args.prior_outact,
        prior_action_independent=args.prior_action_independent,
        prior_corrector=args.prior_corrector,
        prior_epistemic=args.prior_epistemic,
        prior_epistemic_dim=args.prior_epistemic_dim,
        prior_epistemic_std=args.prior_epistemic_std,
        prior_epistemic_samples=args.prior_epistemic_samples,
        achievement_reward_weights=args.achievement_reward_weights,
        test_mode=args.test_mode,
        dynamics_complexity_dims=args.dynamics_complexity_dims,
        dynamics_complexity_seed=args.dynamics_complexity_seed,
        dynamics_complexity_noise_std=args.dynamics_complexity_noise_std,
        dynamics_complexity_length_scale=args.dynamics_complexity_length_scale,
        dynamics_complexity_features=args.dynamics_complexity_features,
        dynamics_complexity_amplitude=args.dynamics_complexity_amplitude,
        dynamics_complexity_input_scale=args.dynamics_complexity_input_scale,
        dynamics_complexity_key=args.dynamics_complexity_key,
        dynamics_complexity_source_keys=args.dynamics_complexity_source_keys,
        dynamics_complexity_mode=args.dynamics_complexity_mode,
        ac_inputs=args.ac_inputs,
        image_shift_pad=args.image_shift_pad,
        drq_num_views=args.drq_num_views,
        bro_actor=args.bro_actor,
        bro_critic=args.bro_critic,
        slow_critic_fraction=args.slow_critic_fraction,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        encoder_lr=args.encoder_lr,
        critic_wd=args.critic_wd,
        actor_wd=args.actor_wd,
        opt_par_noise_scale=args.opt_par_noise_scale,
        discount=args.discount,
        actor_update_delay=args.actor_update_delay,
        actor_update_warmup=args.actor_update_warmup,
        actor_kl_reg=args.actor_kl_reg,
        exp_alpha_scale=args.exp_alpha_scale,
        pc_target_actor=args.pc_target_actor,
        td_target_actor=args.td_target_actor,
        reset_every=args.reset_every,
        reset_max_step=args.reset_max_step,
        reset_keys=args.reset_keys,
        reset_mode=args.reset_mode,
        reset_soft_alpha=args.reset_soft_alpha,
        reset_num_layers=args.reset_num_layers,
        log_coverage=bool(args.log_coverage),
        log_coverage_heatmap=bool(args.log_coverage_heatmap),
        coverage_bins_per_cell=int(args.coverage_bins_per_cell),
        reward_scale=float(args.reward_scale),
        offline_replay=bool(args.offline_replay),
        dual_batch_training=bool(args.dual_batch_training),
        corrector_update_every=args.corrector_update_every,
        ac_update_every=args.ac_update_every,
        replay_fracs_uniform=args.replay_fracs_uniform,
        replay_fracs_priority=args.replay_fracs_priority,
        replay_fracs_recency=args.replay_fracs_recency,
        replay_res_fracs_uniform=args.replay_res_fracs_uniform,
        replay_res_fracs_priority=args.replay_res_fracs_priority,
        replay_res_fracs_recency=args.replay_res_fracs_recency,
        replay_priosignal_ac=args.replay_priosignal_ac,
        replay_priosignal_res=args.replay_priosignal_res,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MTTest')

    # general experiment args
    parser.add_argument('--logs_dir', type=str, default='./logs/')
    parser.add_argument('--entity', type=str, default='sukhijab')
    parser.add_argument('--project_name', type=str, default='ManipulatorImgTest')
    parser.add_argument('--alg', type=str, default='DreamerUCB')
    parser.add_argument('--total_steps', type=int, default=1_000_000)
    parser.add_argument('--expl_until', type=int, default=0)
    parser.add_argument('--num_envs', type=int, default=8)
    parser.add_argument('--num_envs_eval', type=int, default=4)
    parser.add_argument('--train_ratio', type=int, default=512)
    parser.add_argument('--eval_every', type=float, default=120)
    parser.add_argument('--train_rollout_steps', type=int, default=10)
    parser.add_argument('--config_class', type=str, default='dmc')
    parser.add_argument('--task', type=str, default='dmc_finger_spin')
    parser.add_argument('--debug', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--action_cost', type=float, default=0.0)
    parser.add_argument('--init_temp', type=float, default=1.0)
    parser.add_argument('--train_temp', type=int, default=1)
    parser.add_argument('--use_dyn_disg', type=int, default=1)
    parser.add_argument('--exp_result_folder', type=str, default=None)
    #parser.add_argument('--num_layers', type=int, default=2)
    #parser.add_argument('--num_units', type=int, default=256)
    #parser.add_argument('--pooling_output', type=int, default=32)
    #parser.add_argument('--num_ensembles', type=int, default=5)
    parser.add_argument('--action_repeat', type=int, default=0)
    parser.add_argument('--outscale', type=float, default=1.0)
    parser.add_argument('--use_tolerance_reward', type=int, default=0)
    parser.add_argument('--constraint_weight', type=float, default=1.0)
    parser.add_argument('--temp_wd', type=float, default=0.001)
    parser.add_argument('--beta', type=float, default=0.0)
    parser.add_argument('--beta_learn', type=int, default=0)
    parser.add_argument('--beta_init', type=float, default=1.0)
    parser.add_argument('--beta_target_kl', type=float, default=0.05)
    parser.add_argument('--beta_pessimism', type=float, default=0.0)
    parser.add_argument('--beta_log_min', type=float, default=-10.0)
    parser.add_argument('--beta_log_max', type=float, default=7.5)
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--group_name', type=str, default="default")
    parser.add_argument('--replay_critic_bootstrap', type=str, default='imag')
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--boot_prob', type=float, default=1.0)
    parser.add_argument('--target_mode', type=str, default='random_min')
    parser.add_argument('--random_target_k', type=int, default=2)
    parser.add_argument('--value_prior_scale', type=float, default=0.0)
    parser.add_argument('--reward_prior_scale', type=float, default=0.0)
    parser.add_argument('--visual_prior', type=str, default='none')  # none | resnet18 | dinov2[:path] | vc1[-base|-large] | vc1:<path>
    parser.add_argument('--visual_prior_dim', type=int, default=8)
    parser.add_argument('--pessimism', type=float, default=-1.0)
    parser.add_argument('--td_lambda', type=float, default=0.0)
    parser.add_argument('--td_lambda_horizon', type=int, default=0)
    parser.add_argument('--td_target_mode', type=str, default='joint')  # joint | separate | q_full
    parser.add_argument('--bellman_z_from_target', type=int, default=0)  # OFU only; see configs.yaml
    parser.add_argument('--residual_transform', type=float, default=0.0)  # q_full/rnd: downshift subtracted from (rb+c+p)' inside the rb target
    parser.add_argument('--residual_transform_reduction', type=float, default=10.0)  # R for UCB σ_rb = Std(exp((1-γ)·log(R)·rb))
    parser.add_argument('--target_entropy_scale_disc', type=float, default=0.75)
    parser.add_argument('--target_entropy_scale_cont', type=float, default=0.5)
    parser.add_argument('--actor_dist_cont', type=str, default='normal')  # normal | trunc_normal | squashed_normal
    parser.add_argument('--init_alpha', type=float, default=0.1)
    parser.add_argument('--critic_dist', type=str, default="symexp_twohot")
    parser.add_argument('--critic_bins', type=int, default=255)
    parser.add_argument('--residual_bootstrap_dist', type=str, default="mse")
    parser.add_argument('--actor_layers', type=int, default=-1)
    parser.add_argument('--actor_units', type=int, default=-1)
    parser.add_argument('--critic_layers', type=int, default=-1)
    parser.add_argument('--critic_units', type=int, default=-1)
    parser.add_argument('--critic_outscale', type=float, default=0.0)
    parser.add_argument('--loss_scales_critic', type=float, default=1.0)
    parser.add_argument('--loss_scales_exp_actor', type=float, default=1.0)
    parser.add_argument('--loss_scales_actor', type=float, default=1.0)
    parser.add_argument('--loss_scales_alpha', type=float, default=1.0)
    parser.add_argument('--loss_scales_temp', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--policy_mode', type=str, default="actor")
    parser.add_argument('--actor_obj_norm_mode', type=str, default="std")
    parser.add_argument('--replay_context', type=int, default=1)
    parser.add_argument('--contdisc', type=int, default=1)
    parser.add_argument('--add_global_position', type=int, default=0)
    parser.add_argument('--use_jax_driver', type=int, default=0)
    parser.add_argument('--use_image', type=int, default=0)
    parser.add_argument('--frame_stack', type=int, default=1)
    parser.add_argument('--achievement_reward_weights', type=str, default='')
    parser.add_argument('--log_images', type=int, default=0)
    parser.add_argument('--benchmark_id', type=str, default="")
    parser.add_argument('--sample_ruleset', type=int, default=0)
    parser.add_argument('--batch_length', type=int, default=65)
    parser.add_argument('--batch_length_eval', type=int, default=33)
    parser.add_argument('--actor_heads', type=int, default=1)
    parser.add_argument('--eval_trajectory_dir', type=int, default=0)
    parser.add_argument('--explore_trajectory_dir', type=int, default=0)
    parser.add_argument('--big_report_every', type=int, default=0)
    parser.add_argument('--save_every', type=int, default=-1)
    parser.add_argument('--replay_size', type=int, default=1e6)
    parser.add_argument('--exp_obj', type=str, default='ucb')            # ucb | sombrl
    parser.add_argument('--normalize_exp_mean', type=int, default=0)     # observer: divide extrinsic mean_term in exp_score by exp_retnorm/retnorm percentile span
    parser.add_argument('--sampling_temp', type=float, default=1.0)
    parser.add_argument('--num_samples', type=int, default=4)
    parser.add_argument('--prior_layers', type=int, default=0)
    parser.add_argument('--prior_units', type=int, default=0)
    parser.add_argument('--prior_use_rff', type=int, default=0)
    parser.add_argument('--prior_length_scale', type=float, default=1.0)
    parser.add_argument('--prior_act', type=str, default='silu')
    parser.add_argument('--prior_outact', type=str, default='tanh')
    parser.add_argument('--prior_action_independent', type=int, default=0)
    parser.add_argument('--prior_corrector', type=int, default=0)
    parser.add_argument('--prior_epistemic', type=int, default=0)
    parser.add_argument('--prior_epistemic_dim', type=int, default=8)
    parser.add_argument('--prior_epistemic_std', type=float, default=1.0)
    parser.add_argument('--prior_epistemic_samples', type=int, default=1)
    parser.add_argument('--intrinsic_mode', type=str, default='model')   # model | dynamics
    parser.add_argument('--model_size', type=str, default=None)
    parser.add_argument('--enumerate_actions', type=int, default=0)
    parser.add_argument('--cand_grid_points', type=int, default=0)
    parser.add_argument('--cand_policy', type=str, default='softmax')    # softmax | epsilon_greedy
    parser.add_argument('--greedy_eps', type=float, default=0.1)
    parser.add_argument('--discretize', type=int, default=0)
    parser.add_argument('--enc_typ', type=str, default='simple', choices=['simple', 'drq'])
    parser.add_argument('--dyn_typ', type=str, default='rssm', choices=['rssm'])
    parser.add_argument('--prior_inputs', type=str, default='deter,stoch')
    parser.add_argument('--freeze_wm', type=int, default=0)
    parser.add_argument('--freeze_corrector', type=int, default=0)
    parser.add_argument('--freeze_all', type=int, default=0)
    parser.add_argument('--load_keys', type=str, default='')
    parser.add_argument('--from_checkpoint', type=str, default='')
    parser.add_argument('--script', type=str, default='train_eval')
    parser.add_argument('--policy_eval_mc_episodes', type=int, default=128)
    parser.add_argument('--policy_eval_mc_max_steps', type=int, default=1000)
    parser.add_argument('--policy_eval_mc_max_step_limit', type=int, default=0)
    parser.add_argument('--policy_eval_visit_steps', type=int, default=0)
    parser.add_argument('--policy_eval_corrector', default='fresh_train',
                        choices=['load_freeze', 'load_train', 'fresh_train'])
    parser.add_argument('--policy_eval_checkpoint_config', default='')
    parser.add_argument('--policy_eval_diagnostics_every', type=int, default=0)
    parser.add_argument('--policy_eval_diagnostics_points', type=int, default=24)
    parser.add_argument('--policy_eval_diagnostics_horizons', default='64,200,2499')
    parser.add_argument('--env_dmc_max_step_limit', type=int, default=0)
    parser.add_argument('--env_dmc_distract_difficulty', type=str, default='')
    parser.add_argument('--env_dmc_distract_dataset_path', type=str, default='')
    parser.add_argument('--env_dmc_distract_dynamic', type=int, default=0)
    parser.add_argument('--env_dmc_distract_videos', type=str, default='train')
    parser.add_argument('--env_mjp_length', type=int, default=0)
    parser.add_argument('--env_mjp_action_scale', type=float, default=0.0)
    parser.add_argument('--policy_eval_every', type=int, default=5000)
    parser.add_argument('--policy_eval_discount', type=float, default=0.99)
    parser.add_argument('--use_exp_critic', type=int, default=1)
    parser.add_argument('--use_entropy_backup_sac', type=int, default=1)
    parser.add_argument('--test_mode', type=int, default=0)
    parser.add_argument('--dynamics_complexity_dims', type=int, default=0)
    parser.add_argument('--dynamics_complexity_seed', type=int, default=0)
    parser.add_argument('--dynamics_complexity_noise_std', type=float, default=0.0)
    parser.add_argument('--dynamics_complexity_length_scale', type=float, default=1.0)
    parser.add_argument('--dynamics_complexity_features', type=int, default=128)
    parser.add_argument('--dynamics_complexity_amplitude', type=float, default=1.0)
    parser.add_argument('--dynamics_complexity_input_scale', type=float, default=1.0)
    parser.add_argument('--dynamics_complexity_key', type=str, default='state')
    parser.add_argument('--dynamics_complexity_source_keys', type=str, default='auto')
    parser.add_argument('--dynamics_complexity_mode', type=str, default='append',choices=['append', 'create'])
    parser.add_argument('--ac_inputs', type=str, default='wm')
    parser.add_argument('--image_shift_pad', type=int, default=0)
    parser.add_argument('--drq_num_views', type=int, default=1)
    parser.add_argument('--bro_actor', type=int, default=0)
    parser.add_argument('--bro_critic', type=int, default=0)
    parser.add_argument('--slow_critic_fraction', type=float, default=0.02)
    parser.add_argument('--actor_lr', type=float, default=4e-5)
    parser.add_argument('--critic_lr', type=float, default=3e-4)
    parser.add_argument('--encoder_lr', type=float, default=4e-5)
    parser.add_argument('--critic_wd', type=float, default=1e-2)
    parser.add_argument('--actor_wd', type=int, default=0)
    parser.add_argument('--opt_par_noise_scale', type=float, default=0.0)
    parser.add_argument('--discount', type=float, default=-1.0)
    parser.add_argument('--actor_update_delay', type=int, default=1)
    parser.add_argument('--actor_update_warmup', type=int, default=0)
    parser.add_argument('--actor_kl_reg', type=float, default=0.0)
    parser.add_argument('--exp_alpha_scale', type=float, default=1.0)
    parser.add_argument('--pc_target_actor', type=str, default='actor')  # actor | exp_actor
    parser.add_argument('--td_target_actor', type=str, default='actor')  # actor | exp_actor
    parser.add_argument('--reset_every', type=int, default=0)
    parser.add_argument('--reset_max_step', type=int, default=0)
    parser.add_argument('--reset_keys', type=str, default='')
    parser.add_argument('--reset_mode', type=str, default='hard', choices=['hard', 'soft'])
    parser.add_argument('--reset_soft_alpha', type=float, default=0.5)
    parser.add_argument('--reset_num_layers', type=int, default=0)
    parser.add_argument('--log_coverage', type=int, default=0)
    parser.add_argument('--log_coverage_heatmap', type=int, default=0)
    parser.add_argument('--coverage_bins_per_cell', type=int, default=4)
    parser.add_argument('--reward_scale', type=float, default=1.0)
    parser.add_argument('--offline_replay', type=int, default=0)
    # Dual-batch training and per-sampler prioritized replay
    parser.add_argument('--dual_batch_training', type=int, default=0)
    parser.add_argument('--corrector_update_every', type=int, default=1)
    parser.add_argument('--ac_update_every', type=int, default=1)
    parser.add_argument('--replay_fracs_uniform', type=float, default=1.0)
    parser.add_argument('--replay_fracs_priority', type=float, default=0.0)
    parser.add_argument('--replay_fracs_recency', type=float, default=0.0)
    parser.add_argument('--replay_res_fracs_uniform', type=float, default=1.0)
    parser.add_argument('--replay_res_fracs_priority', type=float, default=0.0)
    parser.add_argument('--replay_res_fracs_recency', type=float, default=0.0)
    parser.add_argument('--replay_priosignal_ac', type=str, default='none')   # none | td | rb_td | critic
    parser.add_argument('--replay_priosignal_res', type=str, default='none')  # none | pc
    args = parser.parse_args()
    main(args)
