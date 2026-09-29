from functools import partial as bind

import embodied


def train_eval(
    make_agent, make_train_replay, make_eval_replay, make_explore_replay,
    make_train_env, make_eval_env, make_logger, args, make_reporter=None):

  # Start process-backed environments before agent construction initializes
  # JAX and loads pretrained visual representations. Otherwise forked DMC
  # workers inherit that large native process state and can exhaust host RAM.
  train_driver = None
  eval_driver = None
  explore_driver = None
  if not getattr(args, 'use_jax_driver', False):
    fns = [bind(make_train_env, i) for i in range(args.num_envs)]
    train_driver = embodied.Driver(fns, args.driver_parallel)
    fns = [bind(make_eval_env, i) for i in range(args.num_envs_eval)]
    eval_driver = embodied.Driver(fns, args.driver_parallel)
    if getattr(args, 'explore_trajectory_dir', False):
      fns = [bind(make_eval_env, i) for i in range(args.num_envs_eval)]
      explore_driver = embodied.Driver(fns, args.driver_parallel)

  agent = make_agent()
  train_replay = make_train_replay()
  eval_replay = make_eval_replay()
  explore_replay = make_explore_replay() if getattr(args, 'explore_trajectory_dir', False) else None
  logger = make_logger()
  if make_reporter is None:
    # Compatibility for existing launchers; experiments injects the factory.
    from devote.report import Reporter as make_reporter
  reporter = make_reporter(agent, logger, args)

  logdir = embodied.Path(args.logdir)
  logdir.mkdir()
  print('Logdir', logdir)
  step = logger.step
  usage = embodied.Usage(**args.usage)
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
  should_expl = embodied.when.Until(args.expl_until)
  should_train = embodied.when.Ratio(args.train_ratio / batch_steps)
  should_log = embodied.when.Clock(args.log_every)
  should_save = embodied.when.Clock(args.save_every)
  should_eval = embodied.when.Clock(args.eval_every)
  reset_every = int(getattr(args, 'reset_every', 0) or 0)
  reset_max_step = int(getattr(args, 'reset_max_step', 0) or 0)
  _reset_trigger = embodied.when.Every(reset_every, initial=False) if reset_every > 0 else (lambda s: False)
  should_reset = (lambda s: _reset_trigger(s) and (reset_max_step <= 0 or int(s) <= reset_max_step))
  reset_keys = getattr(args, 'reset_keys', '') or ''
  reset_mode = getattr(args, 'reset_mode', 'hard') or 'hard'
  reset_soft_alpha = float(getattr(args, 'reset_soft_alpha', 1.0))
  reset_num_layers = int(getattr(args, 'reset_num_layers', 0) or 0)
  train_rollout_steps = int(getattr(args, 'train_rollout_steps', 10) or 10)

  if getattr(args, 'use_jax_driver', False):
    print('Using JaxDriver!')
    train_env = make_train_env(0, num_envs=args.num_envs)
    train_driver = embodied.JaxDriver(train_env)
  else:
    assert train_driver is not None
  train_driver.on_step(lambda tran, _: step.increment())
  train_driver.on_step(lambda tran, _: policy_fps.step())
  train_driver.on_step(bind(reporter.log_step, mode='train'))

  reporter.init_coverage(
      make_train_env, train_env if getattr(args, 'use_jax_driver', False) else None)
  reporter.attach_coverage(train_driver, count=True)

  # Coverage annotation must happen before insertion so report batches retain
  # the true-state bin even when the original log_coverage_* key is stripped.
  if not getattr(args, 'offline_replay', False):
    train_driver.on_step(train_replay.add)

  if getattr(args, 'use_jax_driver', False):
    eval_env = make_eval_env(0, num_envs=args.num_envs_eval)
    eval_driver = embodied.JaxDriver(eval_env)
  else:
    assert eval_driver is not None
  reporter.attach_coverage(eval_driver)
  eval_driver.on_step(eval_replay.add)
  eval_driver.on_step(bind(reporter.log_step, mode='eval'))
  eval_driver.on_step(lambda tran, _: policy_fps.step())

  reporter.attach_trajectories(eval_driver, 'eval')
  if getattr(args, 'explore_trajectory_dir', False):
    if getattr(args, 'use_jax_driver', False):
      explore_env = make_eval_env(0, num_envs=args.num_envs_eval)
      explore_driver = embodied.JaxDriver(explore_env)
    else:
      assert explore_driver is not None
    explore_driver.on_step(lambda tran, _: policy_fps.step())
    reporter.attach_trajectories(explore_driver, 'explore')
    explore_driver.on_step(bind(reporter.log_step, mode='explore'))
    reporter.attach_coverage(explore_driver)
    if explore_replay is not None:
      explore_driver.on_step(explore_replay.add)

  if args.dual_batch_training:
    # Two background sampler threads feed independent queues, so the ac and
    # res batches are produced in parallel. A combiner generator merges them
    # before `agent.dataset` does the (single) device_put.
    ac_pre = embodied.Prefetch(bind(train_replay.dataset, args.batch_size, args.batch_length, 'ac'))
    res_pre = embodied.Prefetch(bind(train_replay.dataset, args.batch_size, args.batch_length, 'res'))
    def _dual_source():
      while True:
        ac = next(ac_pre)
        res = next(res_pre)
        res.pop('seed', None)
        yield {**ac, 'res': res}
    dataset_train = agent.dataset(_dual_source)
  else:
    dataset_train = agent.dataset(bind(train_replay.dataset, args.batch_size, args.batch_length, 'ac'))
  reporter.init_replays(train=train_replay, eval=eval_replay, explore=explore_replay)
  carry = [agent.init_train(args.batch_size)]
  reporter.reset_carry()
  sync_agent_step = lambda: agent.set_global_step(step)

  def train_step(tran, worker):
    if len(train_replay) < args.batch_size or step < args.train_fill:
      return
    repeats = should_train(step)
    if repeats or worker == args.num_envs - 1:
      sync_agent_step()
    for _ in range(repeats):
      with embodied.timer.section('dataset_next'):
        batch = next(dataset_train)
      outs, carry[0], mets = agent.train(batch, carry[0])
      train_fps.step(batch_steps)
      if 'replay' in outs:
        train_replay.update(outs['replay'])
      reporter.add_train_metrics(mets)
  train_driver.on_step(train_step)

  checkpoint = embodied.Checkpoint(logdir / 'checkpoint.ckpt')
  checkpoint.step = step
  checkpoint.agent = agent
  checkpoint.train_replay = train_replay
  checkpoint.eval_replay = eval_replay
  if explore_replay is not None:
    checkpoint.explore_replay = explore_replay
  if args.from_checkpoint:
    load_keys = getattr(args, 'load_keys', None) or None
    ckpt_keys = ['agent']
    if getattr(args, 'offline_replay', False):
      ckpt_keys.append('train_replay')
    checkpoint.load(args.from_checkpoint, keys=ckpt_keys, load_keys=load_keys)
  checkpoint.load_or_save()
  should_save(step)  # Register that we just saved.
  sync_agent_step()

  print('Start training loop')
  train_policy = lambda *args: agent.policy(*args, mode='explore' if should_expl(step) or expl_switches(step) else 'train')
  eval_policy = lambda *args: agent.policy(*args, mode='eval')
  explore_policy = lambda *args: agent.policy(*args, mode='explore')
  train_driver.reset(agent.init_policy)
  
  
  if getattr(args, 'test_mode', False): #  
    print('Test mode: one train episode + one eval episode')
    eval_driver.on_step(lambda tran, _: step.increment())

    for i in range(1000):
      train_driver.reset(agent.init_policy)
      train_driver(train_policy, episodes=16)

      sync_agent_step()
      eval_driver.reset(agent.init_policy)
      eval_driver(eval_policy, episodes=16)

      reporter.flush_train()
      reporter.flush_episodes('train')
      reporter.flush_episodes('eval')
      logger.write()
  else:
    while step < args.steps:

      if should_eval(step):
        print('Start evaluation')
        sync_agent_step()
        eval_driver.reset(agent.init_policy)
        with embodied.timer.section('eval_rollout'):
          eval_driver(eval_policy, episodes=args.eval_eps)
        if explore_driver is not None:
          explore_driver.reset(agent.init_policy)
          with embodied.timer.section('explore_rollout'):
            explore_driver(explore_policy, episodes=args.eval_eps)
          reporter.flush_episodes('explore')
        reporter.flush_episodes('eval')
        for mode in ('train', 'eval', 'explore'):
          reporter.report_replay(mode)

      with embodied.timer.section('train_rollout'):
        train_driver(train_policy, steps=train_rollout_steps)

      if should_reset(step):
        agent.reset_params(reset_keys, mode=reset_mode, alpha=reset_soft_alpha, num_layers=reset_num_layers)
        carry[0] = agent.init_train(args.batch_size)
        reporter.reset_carry()
        # Reinitialize the policy carry to match the freshly-reset network,
        # but leave the env state alone -- a full driver reset would queue
        # acts['reset']=True on every env and silently truncate in-flight
        # episodes, which previously suppressed train_episode/train_epstats
        # for envs whose episode length exceeded reset_every / num_envs.
        train_driver.reset_carry(agent.init_policy)
        sync_agent_step()

      if should_log(step):
        policy_fps_value = policy_fps.result()
        reporter.flush_train()
        reporter.flush_episodes('train')
        logger.add(embodied.timer.stats(), prefix='timer')
        logger.add(train_replay.stats(), prefix='replay')
        logger.add(usage.stats(), prefix='usage')
        logger.add({'fps/policy': policy_fps_value})
        logger.add({'fps/train': train_fps.result()})
        logger.add({'exploration_phase': int(should_expl(step) or expl_switches(step))})
        reporter.log_coverage()
        logger.write()

      if should_save(step):
        checkpoint.save()
        checkpoint.save(logdir / f'checkpoint/checkpoint_{int(step)}.ckpt')

    logger.close()
