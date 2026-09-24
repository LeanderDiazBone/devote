import re
from collections import defaultdict
from functools import partial as bind

import embodied
import numpy as np


def train(make_agent, make_replay, make_env, make_logger, args):
    agent = make_agent()
    replay = make_replay()
    logger = make_logger()

    logdir = embodied.Path(args.logdir)
    logdir.mkdir()
    print('Logdir', logdir)
    step = logger.step
    usage = embodied.Usage(**args.usage)
    agg = embodied.Agg()
    epstats = embodied.Agg()
    episodes = defaultdict(embodied.Agg)
    policy_fps = embodied.FPS()
    train_fps = embodied.FPS()

    expl_switches = lambda step: False
    if hasattr(args, 'expl_switches'):
        if hasattr(args, 'alternate_exploration'):
            if args.alternate_exploration:
                steps_scale, step_horizon, frequency_scale = args.expl_switches.steps_scale, \
                    args.expl_switches.steps_horizon, args.expl_switches.frequency_scale
                expl_freqs = []
                total_steps = args.steps
                size_per_step = int(total_steps * steps_scale)
                total_splits = int(total_steps // size_per_step)
                for i in range(1, total_splits + 1):
                    expl_freqs.append(
                        [i * size_per_step, frequency_scale * i]
                    )
                expl_switches = embodied.when.ConditionalEvery(expl_freqs=expl_freqs, step_horizon=step_horizon)

    batch_steps = args.batch_size * (args.batch_length - args.replay_context)
    should_expl_till = embodied.when.Until(args.expl_until)
    should_train = embodied.when.Ratio(args.train_ratio / batch_steps)
    should_log = embodied.when.Clock(args.log_every)
    should_eval = embodied.when.Clock(args.eval_every)
    should_save = embodied.when.Clock(args.save_every)
    reset_every = int(getattr(args, 'reset_every', 0) or 0)
    reset_max_step = int(getattr(args, 'reset_max_step', 0) or 0)
    _reset_trigger = embodied.when.Every(reset_every, initial=False) if reset_every > 0 else (lambda s: False)
    should_reset = (lambda s: _reset_trigger(s) and (reset_max_step <= 0 or int(s) <= reset_max_step))
    reset_keys = getattr(args, 'reset_keys', '') or ''
    reset_mode = getattr(args, 'reset_mode', 'hard') or 'hard'
    reset_soft_alpha = float(getattr(args, 'reset_soft_alpha', 1.0))
    reset_num_layers = int(getattr(args, 'reset_num_layers', 0) or 0)

    @embodied.timer.section('log_step')
    def log_step(tran, worker):

        episode = episodes[worker]
        episode.add('score', tran['reward'], agg='sum')
        episode.add('length', 1, agg='sum')
        episode.add('rewards', tran['reward'], agg='stack')

        if tran['is_first']:
            episode.reset()

        if worker < args.log_video_streams:
            for key in args.log_keys_video:
                if key in tran:
                    episode.add(f'policy_{key}', tran[key], agg='stack')
        for key, value in tran.items():
            if re.match(args.log_keys_sum, key):
                episode.add(key, value, agg='sum')
            if re.match(args.log_keys_avg, key):
                episode.add(key, value, agg='avg')
            if re.match(args.log_keys_max, key):
                episode.add(key, value, agg='max')

        if tran['is_last']:
            result = episode.result()
            logger.add({
                'score': result.pop('score'),
                'length': result.pop('length'),
            }, prefix='episode')
            keys = result.keys()
            keys_to_remove = []
            for key in keys:
                if re.match(args.log_keys_sum, key):
                    res = result[key]
                    if isinstance(res, np.ndarray):
                        res = res.mean().item()
                    logger.add({key: res,}, prefix='episode/sum')
                    keys_to_remove.append(key)
                if re.match(args.log_keys_avg, key):
                    res = result[key]
                    if isinstance(res, np.ndarray):
                        res = res.mean().item()
                    logger.add({key: res,}, prefix='episode/avg')
                    keys_to_remove.append(key)
                if re.match(args.log_keys_max, key):
                    res = result[key]
                    if isinstance(res, np.ndarray):
                        res = res.mean().item()
                    logger.add({key: res,}, prefix='episode/max')
                    keys_to_remove.append(key)
            [result.pop(key) for key in keys_to_remove]
            rew = result.pop('rewards')
            if len(rew) > 1:
                result['reward_rate'] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
            epstats.add(result)

    if getattr(args, 'use_jax_driver', False):
        print('Using JaxDriver!')
        env = make_env(0, num_envs=args.num_envs)
        driver = embodied.JaxDriver(env)
    else:
        fns = [bind(make_env, i) for i in range(args.num_envs)]
        driver = embodied.Driver(fns, args.driver_parallel)
    driver.on_step(lambda tran, _: step.increment())
    driver.on_step(lambda tran, _: policy_fps.step())
    driver.on_step(replay.add)
    driver.on_step(log_step)

    dataset_train = iter(agent.dataset(bind(replay.dataset, args.batch_size, args.batch_length, 'ac')))
    dataset_train_res = iter(agent.dataset(bind(replay.dataset, args.batch_size, args.batch_length, 'res'))) if args.dual_batch_training else None
    dataset_report = iter(agent.dataset(bind(replay.dataset, args.batch_size, args.batch_length_eval, 'ac')))
    carry = [agent.init_train(args.batch_size)]
    carry_report = agent.init_report(args.batch_size)
    sync_agent_step = lambda: agent.set_global_step(step)

    def train_step(tran, worker):
        if len(replay) < args.batch_size or step < args.train_fill:
            return
        repeats = should_train(step)
        if repeats or worker == args.num_envs - 1:
            sync_agent_step()
        for _ in range(repeats):
            with embodied.timer.section('dataset_next'):
                batch = next(dataset_train)
                if args.dual_batch_training:
                    batch_res = next(dataset_train_res)
                    batch_res.pop('seed', None)
                    batch = {**batch, 'res': batch_res}
            outs, carry[0], mets = agent.train(batch, carry[0])
            train_fps.step(batch_steps)
            if 'replay' in outs:
                replay.update(outs['replay'])
            agg.add(mets, prefix='train')

    driver.on_step(train_step)

    autoresume = getattr(args, 'autoresume', True)
    checkpoint = embodied.Checkpoint(logdir / 'checkpoint.ckpt')
    checkpoint.step = step
    checkpoint.agent = agent
    checkpoint.replay = replay
    if args.from_checkpoint:
        checkpoint.load(args.from_checkpoint)
        # See what properties of the agents we should retain from the checkpoint.
        if hasattr(args, 'reset_step'):
            if args.reset_step:
                checkpoint.step = step
        if hasattr(args, 'reset_agent'):
            if args.reset_agent:
                checkpoint.agent = agent
        if hasattr(args, 'reset_replay'):
            if args.reset_replay:
                checkpoint.replay = replay

    if not args.from_checkpoint:
        if autoresume:
            checkpoint.load_or_save()
        else:
            checkpoint.save()
    should_save(step)  # Register that we just saved.
    sync_agent_step()

    print('Start training loop')
    policy = lambda *args: agent.policy(*args, mode='explore' if should_expl_till(step) or expl_switches(step) else 'train')
    driver.reset(agent.init_policy)
    while step < args.steps:

        driver(policy, steps=10)

        if should_reset(step):
            agent.reset_params(reset_keys, mode=reset_mode, alpha=reset_soft_alpha, num_layers=reset_num_layers)
            carry[0] = agent.init_train(args.batch_size)
            carry_report = agent.init_report(args.batch_size)
            # Reinitialize the policy carry to match the freshly-reset network,
            # but leave the env state alone -- a full driver reset would queue
            # acts['reset']=True on every env and silently truncate in-flight
            # episodes, which previously suppressed train_episode/train_epstats
            # for envs whose episode length exceeded reset_every / num_envs.
            driver.reset_carry(agent.init_policy)
            sync_agent_step()

        if should_eval(step) and len(replay):
            sync_agent_step()
            mets, _ = agent.report(next(dataset_report), carry_report)
            logger.add(mets, prefix='report')

        if should_log(step):
            logger.add(agg.result())
            logger.add(epstats.result(), prefix='epstats')
            logger.add(embodied.timer.stats(), prefix='timer')
            logger.add(replay.stats(), prefix='replay')
            logger.add(usage.stats(), prefix='usage')
            logger.add({'fps/policy': policy_fps.result()})
            logger.add({'fps/train': train_fps.result()})
            logger.add({'exploration_phase': int(should_expl_till(step) or expl_switches(step))})
            if isinstance(expl_switches, embodied.when.ConditionalEvery):
                logger.add({'explore': expl_switches.expl_steps}, prefix='steps')
            logger.write()

        if should_save(step):
            checkpoint.save()

    logger.close()
