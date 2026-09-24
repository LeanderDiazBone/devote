import os
import re
import threading

import chex
import embodied
import jax
import jax.numpy as jnp
import numpy as np

from . import jaxutils
from . import ninjax as nj


def Wrapper(agent_cls):
  class Agent(JAXAgent):
    configs = agent_cls.configs
    inner = agent_cls
    def __init__(self, *args, **kwargs):
      super().__init__(agent_cls, *args, **kwargs)
  return Agent


class JAXAgent(embodied.Agent):

  def __init__(self, agent_cls, obs_space, act_space, config):
    print('Observation space')
    [embodied.print(f'  {k:<16} {v}') for k, v in obs_space.items()]
    print('Action space')
    [embodied.print(f'  {k:<16} {v}') for k, v in act_space.items()]

    self.obs_space = obs_space
    self.act_space = act_space
    self.config = config
    self.jaxcfg = config.jax
    self.logdir = embodied.Path(config.logdir)
    self._setup()
    self.agent = agent_cls(obs_space, act_space, config, name='agent')
    self.rng = np.random.default_rng(config.seed)
    self.spaces = {**obs_space, **act_space, **self.agent.aux_spaces}
    self.keys = [k for k in self.spaces if (
        not k.startswith('_') and not k.startswith('log_') and k != 'reset')]

    available = jax.devices(self.jaxcfg.platform)
    embodied.print(f'JAX devices ({jax.local_device_count()}):', available)
    if self.jaxcfg.assert_num_devices > 0:
      assert len(available) == self.jaxcfg.assert_num_devices, (
          available, len(available), self.jaxcfg.assert_num_devices)

    policy_devices = [available[i] for i in self.jaxcfg.policy_devices]
    train_devices = [available[i] for i in self.jaxcfg.train_devices]
    print('Policy devices:', ', '.join([str(x) for x in policy_devices]))
    print('Train devices: ', ', '.join([str(x) for x in train_devices]))

    self.policy_mesh = jax.sharding.Mesh(policy_devices, 'i')
    self.policy_sharded = jax.sharding.NamedSharding(
        self.policy_mesh, jax.sharding.PartitionSpec('i'))
    self.policy_mirrored = jax.sharding.NamedSharding(
        self.policy_mesh, jax.sharding.PartitionSpec())

    self.train_mesh = jax.sharding.Mesh(train_devices, 'i')
    self.train_sharded = jax.sharding.NamedSharding(
        self.train_mesh, jax.sharding.PartitionSpec('i'))
    self.train_mirrored = jax.sharding.NamedSharding(
        self.train_mesh, jax.sharding.PartitionSpec())

    self.pending_outs = None
    self.pending_mets = None
    self.pending_sync = None

    self._transform()
    self.policy_lock = threading.Lock()
    self.train_lock = threading.Lock()
    self.params = self._init_params(obs_space, act_space)
    self.updates = embodied.Counter()

    pattern = re.compile(self.agent.policy_keys)
    self.policy_keys = [k for k in self.params.keys() if pattern.search(k)]
    assert self.policy_keys, (list(self.params.keys()), self.agent.policy_keys)
    self.should_sync = embodied.when.Every(self.jaxcfg.sync_every)
    self.policy_params = jax.device_put(
        {k: jax.tree.map(lambda v: v.copy(), self.params[k]) for k in self.policy_keys},
        self.policy_mirrored)

    self._lower_train()
    self._lower_report()
    self._train = self._train.compile()
    self._report = self._report.compile()
    self._stack = jax.jit(lambda xs: jax.tree.map(
        jnp.stack, xs, is_leaf=lambda x: isinstance(x, list)))
    self._split = jax.jit(lambda xs: jax.tree.map(
        lambda x: [y[0] for y in jnp.split(x, len(x))], xs))
    print('Done compiling train and report!')

  def init_policy(self, batch_size):
    seed = self._next_seeds(self.policy_sharded)
    batch_size //= len(self.policy_mesh.devices)
    carry = self._init_policy(self.policy_params, seed, batch_size)
    if self.jaxcfg.fetch_policy_carry:
      carry = self._take_outs(fetch_async(carry))
    else:
      carry = self._split(carry)
    return carry

  def init_train(self, batch_size):
    seed = self._next_seeds(self.train_sharded)
    batch_size //= len(self.train_mesh.devices)
    carry = self._init_train(self.params, seed, batch_size)
    return carry

  def init_report(self, batch_size):
    seed = self._next_seeds(self.train_sharded)
    batch_size //= len(self.train_mesh.devices)
    carry = self._init_report(self.params, seed, batch_size)
    return carry

  @embodied.timer.section('jaxagent_policy')
  def policy(self, obs, carry, mode='train'):
    obs = self._filter_data(obs)

    with embodied.timer.section('prepare_carry'):
      if self.jaxcfg.fetch_policy_carry:
        carry = jax.tree.map(
            np.stack, carry, is_leaf=lambda x: isinstance(x, list))
      else:
        with self.policy_lock:
          carry = self._stack(carry)

    with embodied.timer.section('check_inputs'):
      for key, space in self.obs_space.items():
        if key in self.keys:
          assert np.isfinite(obs[key]).all(), (obs[key], key, space)
      if self.jaxcfg.fetch_policy_carry:
        for keypath, value in jax.tree_util.tree_leaves_with_path(carry):
          assert np.isfinite(value).all(), (value, keypath)

    with embodied.timer.section('upload_inputs'):
      with self.policy_lock:
        obs, carry = jax.device_put((obs, carry), self.policy_sharded)
        seed = self._next_seeds(self.policy_sharded)

    with embodied.timer.section('jit_policy'):
      with self.policy_lock:
        acts, outs, carry = self._policy(
            self.policy_params, obs, carry, seed, mode)

    with embodied.timer.section('swap_params'):
      with self.policy_lock:
        if self.pending_sync:
          old = self.policy_params
          self.policy_params = self.pending_sync
          jax.tree.map(lambda x: x.delete(), old)
          self.pending_sync = None

    with embodied.timer.section('fetch_outputs'):
      if self.jaxcfg.fetch_policy_carry:
        acts, outs, carry = self._take_outs(fetch_async((acts, outs, carry)))
      else:
        carry = self._split(carry)
        acts, outs = self._take_outs(fetch_async((acts, outs)))

    with embodied.timer.section('check_outputs'):
      finite = outs.pop('finite', {})
      for key, (isfinite, _, _) in finite.items():
        assert isfinite.all(), str(finite)
      for key, space in self.act_space.items():
        if key == 'reset':
          continue
        elif space.discrete:
          assert (acts[key] >= 0).all(), (acts[key], key, space)
        else:
          assert np.isfinite(acts[key]).all(), (acts[key], key, space)

    return acts, outs, carry

  @embodied.timer.section('jaxagent_train')
  def train(self, data, carry):
    seed = data['seed']
    res_batch = data.pop('res', None) if isinstance(data.get('res'), dict) else None
    ac_data = self._filter_data(data)
    res_data = self._filter_data(res_batch) if res_batch is not None else ac_data
    data = {'ac': ac_data, 'res': res_data}
    allo = {k: v for k, v in self.params.items() if k in self.policy_keys}
    dona = {k: v for k, v in self.params.items() if k not in self.policy_keys}
    with embodied.timer.section('jit_train'):
      with self.train_lock:
        self.params, outs, carry, mets = self._train(
            allo, dona, data, carry, seed)
    self.updates.increment()

    if self.should_sync(self.updates) and not self.pending_sync:
      self.pending_sync = jax.device_put(
          {k: allo[k] for k in self.policy_keys}, self.policy_mirrored)
    else:
      jax.tree.map(lambda x: x.delete(), allo)

    return_outs = {}
    if self.pending_outs:
      return_outs = self._take_outs(self.pending_outs)
    self.pending_outs = fetch_async(outs)

    return_mets = {}
    if self.pending_mets:
      return_mets = self._take_mets(self.pending_mets)
    self.pending_mets = fetch_async(mets)

    if self.jaxcfg.profiler:
      outdir, copyto = self.logdir, None
      if str(outdir).startswith(('gs://', '/gcs/')):
        copyto = outdir
        outdir = embodied.Path('/tmp/profiler')
        outdir.mkdir()
      if self.updates == 100:
        embodied.print(f'Start JAX profiler: {str(outdir)}', color='yellow')
        jax.profiler.start_trace(str(outdir))
      if self.updates == 120:
        from embodied.core import path as pathlib
        embodied.print('Stop JAX profiler', color='yellow')
        jax.profiler.stop_trace()
        if copyto:
          pathlib.GFilePath(outdir).copy(copyto)
          print(f'Copied profiler result {outdir} to {copyto}')

    return return_outs, carry, return_mets

  @embodied.timer.section('jaxagent_report')
  def report(self, data, carry):
    seed = data['seed']
    data = self._filter_data(data)
    with embodied.timer.section('jit_report'):
      with self.train_lock:
        mets, carry = self._report(self.params, data, carry, seed)
        mets = self._take_mets(fetch_async(mets))
    return mets, carry

  def _replace_scalar_state(self, tree, key, value, sharding):
    if key not in tree:
      return tree
    updated = dict(tree)
    old = updated[key]
    updated[key] = jax.device_put(value, sharding)
    old.delete()
    return updated

  def set_global_step(self, step):
    if not hasattr(self.agent, 'set_global_step'):
      return
    key = next(
        (k for k in self.params if k.endswith('/global_step/value')),
        None)
    if key is None:
      return
    value = np.asarray(int(step), np.int32)
    with self.train_lock:
      with self.policy_lock:
        self.params = self._replace_scalar_state(
            self.params, key, value, self.train_mirrored)
        self.policy_params = self._replace_scalar_state(
            self.policy_params, key, value, self.policy_mirrored)
        if self.pending_sync:
          self.pending_sync = self._replace_scalar_state(
              self.pending_sync, key, value, self.policy_mirrored)

  def dataset(self, generator):
    def transform(data):
      return {
          **jax.device_put(data, self.train_sharded),
          'seed': self._next_seeds(self.train_sharded)}
    return embodied.Prefetch(generator, transform)

  @embodied.timer.section('jaxagent_save')
  def save(self):
    with self.train_lock:
      return jax.device_get(self.params)

  @embodied.timer.section('jaxagent_load')
  def load(self, state, load_keys=None):
    with self.train_lock:
      with self.policy_lock:
        if load_keys:
          import re
          pattern = re.compile(load_keys)
          matched = {k: v for k, v in state.items() if pattern.search(k) and k in self.params}
          skipped = [k for k in state if not k.startswith('_') and (not pattern.search(k) or k not in self.params)]
          if skipped:
            print(f'Partial load: skipped {len(skipped)} keys, loaded {len(matched)} keys.')
          for k, v in matched.items():
            chex.assert_trees_all_equal_shapes(self.params[k], v)
          merged = {**jax.device_get(self.params), **matched}
        else:
          chex.assert_trees_all_equal_shapes(self.params, state)
          merged = state
        jax.tree.map(lambda x: x.delete(), self.params)
        jax.tree.map(lambda x: x.delete(), self.policy_params)
        self.params = jax.device_put(merged, self.train_mirrored)
        self.policy_params = jax.device_put(
            {k: jax.tree.map(lambda v: v.copy(), self.params[k]) for k in self.policy_keys},
            self.policy_mirrored)

  def _setup(self):
    try:
      import tensorflow as tf
      tf.config.set_visible_devices([], 'GPU')
      tf.config.set_visible_devices([], 'TPU')
    except Exception as e:
      print('Could not disable TensorFlow devices:', e)
    if not self.jaxcfg.prealloc:
      os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    xla_flags = []
    if self.jaxcfg.logical_cpus:
      count = self.jaxcfg.logical_cpus
      xla_flags.append(f'--xla_force_host_platform_device_count={count}')
    if self.jaxcfg.nvidia_flags:
      xla_flags.append('--xla_gpu_enable_latency_hiding_scheduler=true')
      xla_flags.append('--xla_gpu_enable_async_all_gather=true')
      xla_flags.append('--xla_gpu_enable_async_reduce_scatter=true')
      xla_flags.append('--xla_gpu_enable_triton_gemm=false')
      os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
      os.environ['NCCL_IB_SL'] = '1'
      os.environ['NCCL_NVLS_ENABLE'] = '0'
      os.environ['CUDA_MODULE_LOADING'] = 'EAGER'
    if self.jaxcfg.xla_dump:
      outdir = embodied.Path(self.config.logdir) / 'xla_dump'
      outdir.mkdir()
      xla_flags.append(f'--xla_dump_to={outdir}')
      xla_flags.append('--xla_dump_hlo_as_long_text')
    if xla_flags:
      os.environ['XLA_FLAGS'] = ' '.join(xla_flags)
    jax.config.update('jax_platform_name', self.jaxcfg.platform)
    jax.config.update('jax_disable_jit', not self.jaxcfg.jit)
    if self.jaxcfg.transfer_guard:
      jax.config.update('jax_transfer_guard', 'disallow')
    if self.jaxcfg.platform == 'cpu':
      jax.config.update('jax_disable_most_optimizations', self.jaxcfg.debug)
    jaxutils.COMPUTE_DTYPE = getattr(jnp, self.jaxcfg.compute_dtype)
    jaxutils.PARAM_DTYPE = getattr(jnp, self.jaxcfg.param_dtype)

  def _transform(self):

    def init_policy(params, seed, batch_size):
      pure = nj.pure(self.agent.init_policy)
      return pure(params, batch_size, seed=seed)[1]

    def policy(params, obs, carry, seed, mode):
      pure = nj.pure(self.agent.policy)
      return pure(params, obs, carry, mode, seed=seed)[1]

    def init_train(params, seed, batch_size):
      pure = nj.pure(self.agent.init_train)
      return pure(params, batch_size, seed=seed)[1]

    def train(alloc, donated, data, carry, seed):
      pure = nj.pure(self.agent.train)
      combined = {**alloc, **donated}
      params, (outs, carry, mets) = pure(combined, data, carry, seed=seed)
      mets = {k: v[None] for k, v in mets.items()}
      return params, outs, carry, mets

    def init_report(params, seed, batch_size):
      pure = nj.pure(self.agent.init_report)
      return pure(params, batch_size, seed=seed)[1]

    def report(params, data, carry, seed):
      pure = nj.pure(self.agent.report)
      _, (mets, carry) = pure(params, data, carry, seed=seed)
      mets = {k: v[None] for k, v in mets.items()}
      return mets, carry

    from jax.experimental.shard_map import shard_map
    s = jax.sharding.PartitionSpec('i')  # sharded
    m = jax.sharding.PartitionSpec()     # mirrored
    if len(self.policy_mesh.devices) > 1:
      init_policy = lambda params, seed, batch_size, fn=init_policy: shard_map(
          lambda params, seed: fn(params, seed, batch_size),
          self.policy_mesh, (m, s), s, check_rep=False)(params, seed)
      policy = lambda params, obs, carry, seed, mode, fn=policy: shard_map(
          lambda params, obs, carry, seed: fn(params, obs, carry, seed, mode),
          self.policy_mesh, (m, s, s, s), s, check_rep=False)(
              params, obs, carry, seed)
    if len(self.train_mesh.devices) > 1:
      init_train = lambda params, seed, batch_size, fn=init_train: shard_map(
          lambda params, seed: fn(params, seed, batch_size),
          self.train_mesh, (m, s), s, check_rep=False)(params, seed)
      train = shard_map(
          train, self.train_mesh,
          (m, m, s, s, s), (m, s, s, m), check_rep=False)
      init_report = lambda params, seed, batch_size, fn=init_report: shard_map(
          lambda params, seed: fn(params, seed, batch_size),
          self.train_mesh, (m, s), s, check_rep=False)(params, seed)
      report = shard_map(
          report, self.train_mesh,
          (m, s, s, s), (m, s), check_rep=False)

    ps, pm = self.policy_sharded, self.policy_mirrored
    self._init_policy = jax.jit(init_policy, (pm, ps), ps, static_argnames=['batch_size'])
    self._policy = jax.jit(policy, (pm, ps, ps, ps), ps, static_argnames=['mode'])

    ts, tm = self.train_sharded, self.train_mirrored
    self._init_train = jax.jit(init_train, (tm, ts), ts, static_argnames=['batch_size'])
    self._train = jax.jit(train, (tm, tm, ts, ts, ts), (tm, ts, ts, tm), donate_argnums=[1])
    self._init_report = jax.jit(init_report, (tm, ts), ts, static_argnames=['batch_size'])
    self._report = jax.jit(report, (tm, ts, ts, ts), (tm, ts))

  def _take_mets(self, mets):
    mets = jax.tree.map(lambda x: x.__array__(), mets)
    mets = {k: v[0] for k, v in mets.items()}
    mets = jax.tree.map(
        lambda x: np.float32(x) if x.dtype == jnp.bfloat16 else x, mets)
    return mets

  def _take_outs(self, outs):
    outs = jax.tree.map(lambda x: x.__array__(), outs)
    outs = jax.tree.map(
        lambda x: np.float32(x) if x.dtype == jnp.bfloat16 else x, outs)
    return outs

  def _init_params(self, obs_space, act_space, seed=None):
    B, T = self.config.batch_size, self.config.batch_length
    if seed is None:
      seed = self.config.seed
    seed = jax.device_put(np.array([int(seed), 0], np.uint32))
    data = jax.device_put(self._dummy_batch(self.spaces, (B, T)))
    params = nj.init(self.agent.init_train, static_argnums=[1])(
        {}, B, seed=seed)
    _, carry = jax.jit(nj.pure(self.agent.init_train), static_argnums=[1])(
        params, B, seed=seed)
    # BaseAgent.train expects a dict-of-batches; wrap the init dummy accordingly.
    # Third-party pretrained modules can hold parameters as device arrays before
    # Ninjax registers them as agent state. During this one-time jitted
    # initialization, JAX may materialize those closed-over arrays on the host
    # as compilation constants. Allow that initialization transfer explicitly;
    # the global transfer guard remains enforced for policy and training calls.
    with jax.transfer_guard('allow'):
      params = nj.init(self.agent.train)(
          params, {'ac': data, 'res': data}, carry, seed=seed)
    return jax.device_put(params, self.train_mirrored)

  @embodied.timer.section('jaxagent_reset_params')
  def reset_params(self, keys=None, mode='hard', alpha=1.0, num_layers=0):
    """Re-initialize parameters in place. Preserves global_step.

    keys:        regex of param names to reset. If empty/None, resets all
                 params except the global step counter.
    mode:        'hard' overwrites matched params with a fresh init.
                 'soft' mixes old and fresh: new = (1 - alpha) * old + alpha * fresh
                 (soft weight-decay + perturb reset). alpha=1 reduces to hard.
    alpha:       soft mixing fraction in [0, 1] (only used for mode='soft').
    num_layers:  if > 0 and `keys` is non-empty, restricts the reset to the
                 last `num_layers` layers of each module that `keys` selects.
                 Ordering within a module is `stem` -> `h{i}(a|b)?` (ascending i,
                 a before b) -> any other sub-name (output heads). 0 = no
                 filtering, every matched parameter is reset.

    After the per-key reset, any SlowUpdater target (e.g. q_target) whose
    source (e.g. q) was modified is forced to match the new source values,
    and its update counter is reset to 0 so the next call lands on the
    `need_init` (mix=1) branch.
    """
    assert mode in ('hard', 'soft'), mode
    alpha = float(alpha)
    num_layers = int(num_layers)
    if mode == 'soft' and alpha == 0.0:
      return
    seed = int(self.rng.integers(0, np.iinfo(np.uint32).max))
    fresh = jax.device_get(self._init_params(self.obs_space, self.act_space, seed=seed))
    if keys:
      pattern = re.compile(keys)
      matched = [k for k in fresh if pattern.search(k)]
    else:
      pattern = None
      matched = [k for k in fresh if not k.endswith('/global_step/value')]
    if num_layers > 0:
      if pattern is None:
        print(f'reset_params: num_layers={num_layers} requires non-empty `keys`; ignoring.')
      else:
        matched = self._filter_last_n_layers(matched, pattern, num_layers)
    if not matched:
      print(f'reset_params: no keys matched pattern {keys!r}; nothing reset.')
      return

    old = jax.device_get(self.params)
    if mode == 'soft':
      def mix(o, f):
        return ((1.0 - alpha) * o.astype(np.float32) + alpha * f.astype(np.float32)).astype(f.dtype)
      replace = {k: (mix(old[k], fresh[k]) if k in old else fresh[k]) for k in matched}
    else:
      replace = {k: fresh[k] for k in matched}

    # Sync SlowUpdater destinations to new source values when the source was
    # modified, and reset the updater's `updates` counter so the next slow-update
    # call enters the `need_init` branch (mix=1) — preserving target == source.
    sync, counter_resets = {}, {}
    for src_name, dst_name, upd_name in self._slow_updater_pairs():
      src_token = f'/{src_name}/'
      dst_token = f'/{dst_name}/'
      hit = False
      for k, v in replace.items():
        if src_token in k:
          dst_k = k.replace(src_token, dst_token)
          if dst_k in old:
            sync[dst_k] = v.astype(old[dst_k].dtype)
            hit = True
      if hit:
        counter_key = next((k for k in old if f'/{upd_name}/updates/value' in k), None)
        if counter_key is not None:
          counter_resets[counter_key] = np.zeros_like(old[counter_key])
    replace.update(sync)
    replace.update(counter_resets)

    tag = f'soft alpha={alpha:.3g}' if mode == 'soft' else 'hard'
    if num_layers > 0:
      tag += f' last-{num_layers}-layers'
    print(f'reset_params [{tag}]: resetting {len(replace)}/{len(self.params)} params'
          + (f' matching {keys!r}' if keys else ''))
    with self.train_lock:
      with self.policy_lock:
        merged = {**jax.device_get(self.params), **replace}
        jax.tree.map(lambda x: x.delete(), self.params)
        jax.tree.map(lambda x: x.delete(), self.policy_params)
        self.params = jax.device_put(merged, self.train_mirrored)
        self.policy_params = jax.device_put(
            {k: jax.tree.map(lambda v: v.copy(), self.params[k]) for k in self.policy_keys},
            self.policy_mirrored)
        self.pending_sync = None

  def _slow_updater_pairs(self):
    """Return [(src_name, dst_name, updater_name)] for every SlowUpdater
    attached as an attribute of the inner agent."""
    pairs = []
    for attr in vars(self.agent).values():
      if isinstance(attr, jaxutils.SlowUpdater):
        pairs.append((attr.src.name, attr.dst.name, attr.name))
    return pairs

  @staticmethod
  def _layer_order(name):
    """Sort key for layer sub-names within an MLP-like module.
    `stem` < `h0` / `h0a` < `h0b` < `h1` / `h1a` < `h1b` < ... < any other name."""
    if name == 'stem':
      return (0, 0, 0)
    m = re.match(r'h(\d+)([ab]?)$', name)
    if m:
      sub = {'': 0, 'a': 0, 'b': 1}[m.group(2)]
      return (1, int(m.group(1)), sub)
    return (2, 0, 0)

  def _filter_last_n_layers(self, matched, pattern, num):
    """Keep only keys belonging to the last `num` layers of each module that
    `pattern` selects. A 'module' is the prefix up through pattern's match;
    the segment immediately after is treated as the layer name."""
    groups = {}                                         # prefix -> [(key, layer_seg)]
    for k in matched:
      m = pattern.search(k)
      if m is None:
        continue
      prefix, suffix = k[:m.end()], k[m.end():]
      seg = suffix.split('/', 1)[0]
      groups.setdefault(prefix, []).append((k, seg))
    keep = set()
    for entries in groups.values():
      segs = sorted({s for _, s in entries}, key=self._layer_order)
      last = set(segs[-num:])
      keep.update(k for k, s in entries if s in last)
    return [k for k in matched if k in keep]

  def _next_seeds(self, sharding):
    shape = [2 * x for x in sharding.mesh.devices.shape]
    seeds = self.rng.integers(0, np.iinfo(np.uint32).max, shape, np.uint32)
    return jax.device_put(seeds, sharding)

  def _filter_data(self, data):
    return {k: v for k, v in data.items() if k in self.keys}

  def _dummy_batch(self, spaces, batch_dims):
    spaces = [(k, v) for k, v in spaces.items()]
    data = {k: np.zeros(v.shape, v.dtype) for k, v in spaces}
    data = self._filter_data(data)
    for dim in reversed(batch_dims):
      data = {k: np.repeat(v[None], dim, axis=0) for k, v in data.items()}
    return data

  def _lower_train(self):
    B = self.config.batch_size
    T = self.config.batch_length
    data = self._dummy_batch(self.spaces, (B, T))
    data = jax.device_put(data, self.train_sharded)
    # BaseAgent.train expects a dict-of-batches; wrap for lowering.
    data = {'ac': data, 'res': data}
    seed = self._next_seeds(self.train_sharded)
    carry = self.init_train(self.config.batch_size)
    allo = {k: v for k, v in self.params.items() if k in self.policy_keys}
    dona = {k: v for k, v in self.params.items() if k not in self.policy_keys}
    self._train = self._train.lower(allo, dona, data, carry, seed)

  def _lower_report(self):
    B = self.config.batch_size
    T = self.config.batch_length_eval
    data = self._dummy_batch(self.spaces, (B, T))
    data = jax.device_put(data, self.train_sharded)
    seed = self._next_seeds(self.train_sharded)
    carry = self.init_report(self.config.batch_size)
    self._report = self._report.lower(self.params, data, carry, seed)


def fetch_async(value):
  with jax._src.config.explicit_device_get_scope():
    [x.copy_to_host_async() for x in jax.tree_util.tree_leaves(value)]
  return value
