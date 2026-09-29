import hashlib
import importlib
import json
import sys
from typing import Dict, Optional

import numpy as np

import embodied
from embodied import wrappers


class Logger:
    """Write console output to both the console and a log file."""

    def __init__(self, filename, stream=sys.stdout):
        self.stream = stream
        self.file = open(filename, 'a')

    def write(self, message):
        self.stream.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyArrayEncoder, self).default(obj)


def hash_dict(d: Dict) -> str:
    dhash = hashlib.md5()
    dhash.update(json.dumps(d, sort_keys=True, cls=NumpyArrayEncoder, indent=4).encode())
    return dhash.hexdigest()


def make_agent(config, alg='MultiMex'):
    env = make_env(config, 0, wrapper_log_images=False)
    suite = config.task.split('_', 1)[0]
    if suite in ('craftax', 'xminigrid', 'xlandminigrid'):
        image_keys = [k for k, v in env.obs_space.items() if len(v.shape) == 3]
        print(f'{suite} obs spaces:', {
            k: (env.obs_space[k].shape, str(env.obs_space[k].dtype))
            for k in image_keys
        }, flush=True)
    if config.random_agent:
        agent = embodied.RandomAgent(env.obs_space, env.act_space)
    else:
        def build_default(module_name):
            agt = importlib.import_module(module_name)
            return agt.Agent(env.obs_space, env.act_space, config)

        registry = {
            'Dreamer': lambda: build_default('devote.dreamer_agent'),
            'DreamerOld': lambda: build_default('devote.dreamer_agent_old'),
            'Observer': lambda: build_default('devote.observer_agent'),
            'ObserverOld': lambda: build_default('devote.observer_agent_old'),
            'DreamerLegacy': lambda: build_default('dreamerv3.agent'),
            'SOMBRLLegacy': lambda: build_default('multimex.dreamerucb.agent'),
        }
        builder = registry.get(alg)
        if builder is None:
            raise NotImplementedError(f'Unknown algorithm: {alg}')
        agent = builder()

    env.close()
    return agent


def make_logger(config, wandb_config: Optional[Dict] = None):
    step = embodied.Counter()
    logdir = config.logdir
    exp_name = wandb_config["exp_name"]
    del wandb_config["exp_name"]
    multiplier = config.env.get(config.task.split('_')[0], {}).get('repeat', 1)
    if wandb_config is not None:
        logger = embodied.Logger(step, [
            embodied.logger.TerminalOutput(config.filter, 'Agent'),
            embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
            embodied.logger.JSONLOutput(logdir, 'scores.jsonl', 'episode/score'),
            embodied.logger.WandBOutput(name=exp_name, **wandb_config),
        ], multiplier)
    else:
        logger = embodied.Logger(step, [
            embodied.logger.TerminalOutput(config.filter, 'Agent'),
            embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
            embodied.logger.JSONLOutput(logdir, 'scores.jsonl', 'episode/score'),
        ], multiplier)
    return logger


def make_replay(config, directory=None, is_eval=False, rate_limit=False):
    directory = directory and embodied.Path(config.logdir) / directory
    size = int(config.replay.size / 10 if is_eval else config.replay.size)
    length = config.replay_length_eval if is_eval else config.replay_length
    kwargs = {}
    kwargs['online'] = config.replay.online
    kwargs['save_to_disk'] = config.replay.save_to_disk
    if rate_limit and config.run.train_ratio > 0:
        kwargs['samples_per_insert'] = config.run.train_ratio / (
                length - config.replay_context)
        kwargs['tolerance'] = 5 * config.batch_size
        kwargs['min_size'] = min(
            max(config.batch_size, config.run.train_fill), size)
    selectors = embodied.replay.selectors

    def _build_selector(fracs, prio):
        if fracs.uniform >= 1:
            return selectors.Uniform()
        assert config.jax.compute_dtype in ('bfloat16', 'float32'), (
            'Gradient scaling for low-precision training can produce invalid loss '
            'outputs that are incompatible with prioritized replay.')
        recency = 1.0 / np.arange(1, size + 1) ** config.replay.recexp
        return selectors.Mixture(dict(
            uniform=selectors.Uniform(),
            priority=selectors.Prioritized(**prio),
            recency=selectors.Recency(recency),
        ), fracs)

    if not is_eval:
        ac_sel = _build_selector(config.replay.fracs, config.replay.prio)
        if config.dual_batch_training:
            # The res (corrector) sampler is independent: its own fracs/prio mix
            # over the same chunk storage.
            res_sel = _build_selector(config.replay.res_fracs, config.replay.res_prio)
            kwargs['selectors_map'] = {'ac': ac_sel, 'res': res_sel}
        else:
            kwargs['selector'] = ac_sel
    kwargs['chunksize'] = config.replay.chunksize
    replay = embodied.replay.Replay(length, size, directory, **kwargs)
    return replay


def make_env(config, index, **overrides):
    suite, task = config.task.split('_', 1)
    requested_num_envs = overrides.get('num_envs', None)
    wrapper_log_images = overrides.pop('wrapper_log_images', None)
    enable_log_images = getattr(config.wrapper, 'log_images', False) if wrapper_log_images is None else bool(wrapper_log_images)
    if suite == 'memmaze':
        from embodied.envs import from_gym
        import memory_maze  # noqa
    if suite == 'tactile':
        import tactile_envs
    if suite == 'humanoid':
        import humanoid_bench
    ctor = {
        'dummy': 'embodied.envs.dummy:Dummy',
        'gym': 'embodied.envs.from_gym:FromGym',
        'gymnasium': 'embodied.envs.from_gymnasium:FromGymnasium',
        'ogbench': 'embodied.envs.ogbench:OGBench',
        'metaworld': 'embodied.envs.from_metaworld:MetaWorld',
        'tactile': 'embodied.envs.from_gymnasium:FromGymnasium',
        "humanoid": "embodied.envs.from_gymnasium:FromGymnasium",
        "insertion": 'embodied.envs.insertion:InsertionEnv',
        "adroit": 'embodied.envs.adroit:AdroitEnv',
        'dm': 'embodied.envs.from_dmenv:FromDM',
        'crafter': 'embodied.envs.crafter:Crafter',
        'craftax': 'embodied.envs.craftax:Craftax',
        'xminigrid': 'embodied.envs.xland_minigrid:XLandMiniGrid',
        'xlandminigrid': 'embodied.envs.xland_minigrid:XLandMiniGrid',
        'dmc': 'embodied.envs.dmc:DMC',
        'atari': 'embodied.envs.atari:Atari',
        'atari100k': 'embodied.envs.atari:Atari',
        'dmlab': 'embodied.envs.dmlab:DMLab',
        'minecraft': 'embodied.envs.minecraft:Minecraft',
        'loconav': 'embodied.envs.loconav:LocoNav',
        'pinpad': 'embodied.envs.pinpad:PinPad',
        'mjp': 'embodied.envs.mujoco_playground:BatchedMujocoPlayground',
        'langroom': 'embodied.envs.langroom:LangRoom',
        'procgen': 'embodied.envs.procgen:ProcGen',
        'bsuite': 'embodied.envs.bsuite:BSuite',
        'bandit': 'embodied.envs.bandit:Bandit',
        'memmaze': lambda task, **kw: from_gym.FromGym(f'MemoryMaze-{task}-ExtraObs-v0', **kw),
    }[suite]
    if isinstance(ctor, str):
        module, cls = ctor.split(':')
        module = importlib.import_module(module)
        ctor = getattr(module, cls)
    kwargs = config.env.get(suite, {})
    kwargs.update(overrides)
    if kwargs.pop('use_seed', False):
        kwargs['seed'] = hash((config.seed, index)) % (2 ** 32 - 1)
    if kwargs.pop('use_logdir', False):
        kwargs['logdir'] = embodied.Path(config.logdir) / f'env{index}'
    if suite in ["mjp"]:
        kwargs.setdefault("num_envs", config.run.num_envs)
    if suite in ["craftax", "xminigrid", "xlandminigrid"] and requested_num_envs is not None:
        kwargs.setdefault("num_envs", requested_num_envs)
    if suite == 'craftax' and enable_log_images and 'log_image' not in kwargs:
        kwargs['log_image'] = ('Pixels' not in task)
    # For envs whose own step() renders a log_image, gate that rendering on
    # enable_log_images so train workers don't pay the per-step GL cost.
    # The eval/explore ctors are built with wrapper_log_images=True (see
    # exp.py), so they still render for report videos.
    if suite in ('dmc', 'mjp'):
        kwargs['log_image'] = enable_log_images
    env = ctor(task, **kwargs)
    return wrap_env(env, config, log_images=enable_log_images)


def wrap_env(env, config, log_images=None):
    args = config.wrapper
    if log_images is None:
        log_images = getattr(args, 'log_images', False)
    if log_images:
        obs_keys = set(env.obs_space.keys())
        image_space = env.obs_space.get('image', None)
        image_key_is_render_like = bool(
            image_space is not None and
            hasattr(image_space, 'shape') and
            len(image_space.shape) == 3 and
            image_space.shape[-1] in (1, 3, 4)
        )
        if 'log_image' not in obs_keys and (not image_key_is_render_like) and hasattr(env, 'render'):
            try:
                env = wrappers.RenderImage(env, key='log_image')
            except Exception as e:
                print(f'RenderImage wrapper failed ({e}), skipping log_image')

    for name, space in env.act_space.items():
        if name == 'reset':
            continue
        elif not space.discrete:
            env = wrappers.NormalizeAction(env, name)
            if args.action_cost:
                if hasattr(args, 'use_tolerance_reward'):
                    use_tolerance_reward = args.use_tolerance_reward
                else:
                    use_tolerance_reward = False
                env = wrappers.ActionCost(env,
                                          action_cost=args.action_cost,
                                          key=name,
                                          use_tolerance_reward=use_tolerance_reward)
            if args.discretize:
                env = wrappers.DiscretizeAction(env, name, args.discretize)

    env = wrappers.ExpandScalars(env)
    frame_stack = int(getattr(args, 'frame_stack', 1))
    if frame_stack < 1:
        raise ValueError(f'wrapper.frame_stack must be positive, got {frame_stack}.')
    if frame_stack > 1:
        env = wrappers.FrameStack(env, key='image', length=frame_stack)
    if hasattr(args, 'dynamics_complexity') and args.dynamics_complexity.dims:
        dc = args.dynamics_complexity
        env = wrappers.DynamicsComplexity(
            env,
            dims=dc.dims,
            key=dc.key,
            source_keys=dc.source_keys,
            mode=dc.mode,
            seed=dc.seed,
            features=dc.features,
            amplitude=dc.amplitude,
            length_scale=dc.length_scale,
            input_scale=dc.input_scale,
            noise_std=dc.noise_std)
    if args.length:
        env = wrappers.TimeLimit(env, args.length, args.reset)
    if args.checks:
        env = wrappers.CheckSpaces(env)

    for name, space in env.act_space.items():
        if not space.discrete:
            env = wrappers.ClipAction(env, name)
    return env