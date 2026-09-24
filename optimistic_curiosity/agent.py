import re
import jax
import jax.numpy as jnp
import numpy as np
import ruamel.yaml as yaml
from functools import partial as bind
from abc import ABC, abstractmethod

import embodied
from dreamerv3 import jaxutils
from dreamerv3 import ninjax as nj
from dreamerv3 import nets
from multimex.nets import IntrinsicRewardModel, Temperature
import optimistic_curiosity.nets as dr_nets
from optimistic_curiosity.nets import ShiftedDist
from .report import ReportMixin

f32 = jnp.float32
treemap = jax.tree_util.tree_map
sg = lambda x: treemap(jax.lax.stop_gradient, x)
cast = jaxutils.cast_to_compute
sample = lambda dist: {k: v.sample(seed=nj.seed()) for k, v in dist.items()}
sample_k = lambda dist, K: {k: jax.vmap(lambda key: v.sample(seed=key))(jax.random.split(nj.seed(), K)) for k, v in dist.items()}


def preprocess(obs, obs_space, act_space, aux_spaces):
    spaces = {**obs_space, **act_space, **aux_spaces}
    result = {}
    for key, value in obs.items():
        if key.startswith('log_') or key in ('reset', 'key', 'id'):
            continue
        space = spaces[key]
        if len(space.shape) >= 3 and space.dtype == jnp.uint8:
            value = cast(value) / 255.0
        result[key] = value
    result['cont'] = 1.0 - f32(result['is_terminal'])
    return result


def init_carry(dyn, act_space, batch_size):
    prevact = {k: jnp.zeros((batch_size, *v.shape), v.dtype) for k, v in act_space.items()}
    return (dyn.initial(batch_size), prevact)


class BaseAgent(nj.Module, ReportMixin):
    """DreamerV3 model-based reinforcement learning agent."""

    configs = yaml.YAML(typ='safe').load((embodied.Path(__file__).parent / 'configs.yaml').read())
    _dyn_heads = 1
    _value_heads = 1
    _actor_heads = 1
    _prior_special_keys = frozenset(('use_rff', 'length_scale', 'action_independent', 'corrector', 'epistemic', 'epistemic_dim', 'epistemic_std', 'epistemic_samples', 'epistemic_key', 'epistemic_shared_contexts'))

    # Initilization
    def build_spaces(self, obs_space, config):
        """Determine encoder and decoder observation subsets."""
        skip = ('is_first', 'is_last', 'is_terminal', 'reward')
        enc_space = {k: v for k, v in obs_space.items() if k not in skip and not k.startswith('log_') and re.match(config.enc.spaces, k)}
        dec_space = {k: v for k, v in obs_space.items() if k not in skip and not k.startswith('log_') and re.match(config.dec.spaces, k)}
        return enc_space, dec_space

    def _discrete_classes(self, space):
        classes = np.asarray(space.high - space.low)
        if classes.ndim == 0 or (classes.size and (classes == classes.flat[0]).all()):
            return int(classes.flat[0])
        raise NotImplementedError(
            f'Only equal class counts per discrete action dimension are supported: {classes}')

    def _action_shape(self, space):
        return (*space.shape, self._discrete_classes(space)) if space.discrete else space.shape

    def _discrete_entropy(self):
        total = 0.0
        for space in self.act_space.values():
            classes = np.asarray(space.high - space.low, np.float32)
            if classes.ndim == 0:
                classes = np.broadcast_to(classes, space.shape or ())
            total += float(np.log(classes).sum())
        return total

    def build_world_model(self, config, enc_space, dec_space):
        """Construct encoder, decoder, dynamics, reward, and continuation heads."""
        enc = {
            'simple': bind(nets.SimpleEncoder, **config.enc.simple),
            'drq': bind(dr_nets.DrQEncoder, **config.enc.drq),
        }[config.enc.typ](enc_space, name='enc')
        dec = {'simple': bind(nets.SimpleDecoder, **config.dec.simple),}[config.dec.typ](dec_space, name='dec')
        assert config.dyn.typ == 'rssm', config.dyn.typ
        dyn = bind(dr_nets.EnsembleRSSM, num_heads=self.dyn_heads, **config.dyn.rssm)(name='dyn')
        rew = dr_nets.EnsembleMLP((), num_heads=self.dyn_heads, **config.rewhead, name='rew')
        rew_prior = dr_nets.EnsembleMLP((), num_heads=self.dyn_heads, name='rew_prior',**self._prior_kwargs(config.rewhead, config.critic_prior))
        con = dr_nets.EnsembleMLP((), num_heads=self.dyn_heads, **config.conhead, name='con')
        return dict(enc=enc, dec=dec, dyn=dyn, rew=rew, rew_prior=rew_prior, con=con)

    def build_actor(self, config, act_space, heads, name='actor'):
        """Construct the policy network and normalization modules."""
        kwargs = {}
        kwargs['shape'] = {k: self._action_shape(s) for k, s in act_space.items()}
        # OFU's _z is sampled from a uniform-on-cube prior; parametrize the policy with a
        # truncated normal so its samples stay within the same support.
        def _dist(k, s):
            if k == '_z':
                return 'trunc_normal'
            return config.actor_dist_disc if s.discrete else config.actor_dist_cont
        kwargs['dist'] = {k: _dist(k, v) for k, v in act_space.items()}
        prefix = f'{name}_' if name != 'actor' else ''
        actor = dr_nets.EnsembleMLP(**kwargs, num_heads=heads, **config.actor, name=name)
        retnorm = jaxutils.Moments(**config.retnorm, name=f'{prefix}retnorm') # TODO: Per head moments
        valnorm = jaxutils.Moments(**config.valnorm, name=f'{prefix}valnorm')
        advnorm = jaxutils.Moments(**config.advnorm, name=f'{prefix}advnorm')
        return dict(actor=actor, retnorm=retnorm, valnorm=valnorm, advnorm=advnorm)

    def _prior_kwargs(self, base_config, prior_config):
        """Resolve prior network config from a base head config and prior overrides."""
        kw = dict(base_config)
        kw['dist'] = 'mse'
        for k, v in prior_config.items():
            if k in self._prior_special_keys:
                continue
            if v != 0:
                kw[k] = v
        return kw

    def _resolved_prior_config(self, base_config, prior_config): # TODO: Join with other methods
        kw = self._prior_kwargs(base_config, prior_config)
        prior_config = dict(prior_config.items())
        kw['use_rff'] = bool(prior_config.get('use_rff', False))
        kw['length_scale'] = float(prior_config.get('length_scale', 1.0))
        kw['action_independent'] = bool(prior_config.get('action_independent', False))
        kw['corrector'] = bool(prior_config.get('corrector', False))
        kw['epistemic'] = bool(prior_config.get('epistemic', False))
        kw['epistemic_dim'] = int(prior_config.get('epistemic_dim', 0))
        kw['epistemic_std'] = float(prior_config.get('epistemic_std', 1.0))
        kw['epistemic_samples'] = int(prior_config.get('epistemic_samples', 1))
        kw['epistemic_key'] = prior_config.get('epistemic_key', 'epistemic')
        kw['epistemic_shared_contexts'] = bool(prior_config.get('epistemic_shared_contexts', False))
        return kw

    def build_critic(self, config, name='critic'):
        """Construct critic, slow critic, prior, and the slow-update mechanism."""
        critic = dr_nets.EnsembleMLP((), num_heads=self.value_heads, name=name, **config.critic)
        slowcritic = dr_nets.EnsembleMLP((), num_heads=self.value_heads, name=f'slow{name}', **config.critic, dtype='float32')
        prior_kw = self._prior_kwargs(config.critic, config.critic_prior)
        v_prior = dr_nets.EnsembleMLP((), num_heads=self.value_heads, name=f'{name}_prior', **prior_kw)
        updater = jaxutils.SlowUpdater(critic, slowcritic, config.slow_critic_fraction, config.slow_critic_update, name=f'{name}_updater')
        return dict(critic=critic, slowcritic=slowcritic, updater=updater, v_prior=v_prior)

    def build_q_function(self, config, heads, use_multihead_q, names=['q', 'q_target', 'q_prior', 'q_updater'], pessimism=-1.0, prior_scale=None, residual_bootstrap=False):
        """Build q-functions wrapped with a shared frozen randomized prior."""
        head_kw = dict(config.critic)
        head_kw['inputs'] = self._q_input_keys()
        q_shape = (self._num_actions,) if use_multihead_q else ()
        N = heads
        if prior_scale is None:
            prior_scale = float(getattr(self, 'value_prior_scale', getattr(self, 'prior_scale', 0.0)))

        # Build the shared frozen prior. Lives outside self.modules and so is
        # never optimized; both online and target wrappers reference it.
        prior_kw = self._resolved_prior_config(config.critic, config.critic_prior)
        prior_use_rff = prior_kw.pop('use_rff')
        prior_length_scale = prior_kw.pop('length_scale')
        prior_action_indep = prior_kw.pop('action_independent')
        prior_corrector = prior_kw.pop('corrector') and prior_scale > 0
        prior_epistemic = prior_kw.pop('epistemic') and prior_scale > 0
        prior_epistemic_dim = prior_kw.pop('epistemic_dim')
        prior_epistemic_std = prior_kw.pop('epistemic_std')
        prior_epistemic_samples = prior_kw.pop('epistemic_samples')
        prior_epistemic_key = prior_kw.pop('epistemic_key')
        prior_epistemic_shared_contexts = prior_kw.pop('epistemic_shared_contexts')
        state_keys = ['visual_embed'] if config.get('visual_prior', 'none') != 'none' else list(config.critic.get('inputs', ['deter', 'stoch']))
        act_keys = [] if self._is_discrete else list(self.act_space.keys())
        if prior_action_indep:
            prior_kw['inputs'] = state_keys
            prior_shape = ()
        else:
            prior_kw['inputs'] = state_keys + act_keys
            prior_shape = q_shape
        corrector_kw = dict(prior_kw)
        corrector_shape = prior_shape
        if prior_corrector and prior_action_indep:
            corrector_kw['inputs'] = state_keys + act_keys
            corrector_shape = q_shape
        if prior_epistemic:
            prior_epistemic_dim = max(1, int(prior_epistemic_dim))
            prior_kw['inputs'] = list(prior_kw['inputs']) + [prior_epistemic_key]
            corrector_kw['inputs'] = list(corrector_kw['inputs']) + [prior_epistemic_key]
        else:
            prior_epistemic_dim = 0
        prior_projection = (
            getattr(self, 'visual_proj', None) if prior_scale > 0 else None)
        if prior_use_rff:
            prior_ctor = dr_nets.EnsembleRFFPrior
            prior_args = (prior_shape,)
            prior_kwargs = dict(
                num_heads=N, inputs=prior_kw['inputs'],
                units=prior_kw['units'], length_scale=prior_length_scale,
                dist=prior_kw['dist'],
                outact=prior_kw.get('outact', 'none'))
        else:
            prior_ctor = dr_nets.EnsembleMLP
            prior_args = (prior_shape,)
            prior_kwargs = dict(num_heads=N, **prior_kw)
        if prior_projection is not None:
            prior = dr_nets.EnsembleInputProjection(
                prior_ctor, prior_args, prior_kwargs, prior_projection,
                key='visual_embed', name=names[2])
        else:
            prior = prior_ctor(
                *prior_args, **prior_kwargs, name=names[2])

        wrap_kw = dict(
            prior=prior, prior_shape=prior_shape, prior_scale=prior_scale,
            prior_corrector=prior_corrector,
            prior_corrector_use_rff=prior_use_rff,
            prior_corrector_length_scale=prior_length_scale,
            prior_corrector_kw=corrector_kw,
            prior_corrector_shape=corrector_shape,
            prior_input_projection=prior_projection,
            prior_input_projection_key='visual_embed',
            epistemic_dim=prior_epistemic_dim,
            epistemic_std=prior_epistemic_std,
            epistemic_samples=prior_epistemic_samples,
            epistemic_key=prior_epistemic_key,
            epistemic_shared_contexts=prior_epistemic_shared_contexts,
            residual_bootstrap=bool(residual_bootstrap),
            residual_bootstrap_kw={**head_kw, 'dist': config.residual_bootstrap_dist},
            residual_bootstrap_shape=q_shape)
        q = dr_nets.PriorCritic(q_shape, num_heads=N, name=names[0], **wrap_kw, **head_kw)
        q_tgt = dr_nets.PriorCritic(q_shape, num_heads=N, name=names[1], **wrap_kw, **head_kw, dtype='float32')
        q_upd = jaxutils.SlowUpdater(q, q_tgt, config.slow_critic_fraction, config.slow_critic_update, name=names[3])
        online_heads, target_heads, updaters = [q], [q_tgt], [q_upd]
        if pessimism >= 0.0:
            q2 = dr_nets.PriorCritic(q_shape, num_heads=N, name=names[0] + '2', **wrap_kw, **head_kw)
            q2_tgt = dr_nets.PriorCritic(q_shape, num_heads=N, name=names[1] + '2', **wrap_kw, **head_kw, dtype='float32')
            q2_upd = jaxutils.SlowUpdater(q2, q2_tgt, config.slow_critic_fraction, config.slow_critic_update, name=names[3] + '2')
            online_heads.append(q2); target_heads.append(q2_tgt); updaters.append(q2_upd)
        return dict(
            q=dr_nets.DoubleQCritic(online_heads, pessimism),
            q_target=dr_nets.DoubleQCritic(target_heads, pessimism),
            q_updater=dr_nets.DoubleQUpdater(updaters),
            q_prior=prior,
        )

    def intrinsic_model_input_keys(self, model_inputs):
        if getattr(self, 'ac_inputs', 'wm') == 'obs':
            return tuple(self._state_obs_keys) + tuple(self.act_space.keys())
        return tuple(model_inputs) + tuple(self.act_space.keys())

    def _intrinsic_model_inputs(self, states, acts, data=None):
        if getattr(self, 'ac_inputs', 'wm') != 'obs':
            return states, acts
        if isinstance(states, dict) and all(
                k in states for k in self._state_obs_keys):
            obs_states = {k: states[k] for k in self._state_obs_keys}
        elif data is not None:
            obs_states = {k: data[k] for k in self._state_obs_keys}
        else:
            raise KeyError(
                'Need raw state observations for intrinsic reward model inputs.')
        return obs_states, acts

    def build_intrinsic_model_spec(self, config):
        """Build intrinsic model kwargs without mutating config objects."""
        model_kwargs = dict(config.int_rew_model.model.flat)
        dyn_cfg = getattr(config.dyn, config.dyn.typ)
        stoch_size = dyn_cfg.stoch
        num_classes = dyn_cfg.classes

        dist = {}
        normalization_impl = {}
        pooling_kwargs = {}
        shapes = {}

        if 'dist.obs' in model_kwargs:
            dist['obs'] = model_kwargs.pop('dist.obs')
            embedding_size = config.int_rew_model.pooling_output
            pooling_kwargs['output_size'] = {'obs': embedding_size}
            shapes['obs'] = (embedding_size,)
            normalization_impl['obs'] = model_kwargs.pop('normalization_impl.obs', 'mean_std')

        if 'dist.dyn' in model_kwargs:
            dist['dyn'] = model_kwargs.pop('dist.dyn')
            shapes['dyn'] = (stoch_size, num_classes)
            normalization_impl['dyn'] = model_kwargs.pop('normalization_impl.dyn', 'mean_std')

        if 'dist.rew' in model_kwargs:
            dist['rew'] = model_kwargs.pop('dist.rew')
            shapes['rew'] = ()
            normalization_impl['rew'] = model_kwargs.pop('normalization_impl.rew', 'mean_std')

        assert len(dist) > 0, "Need at least one target for intrinsic reward"

        model_kwargs['shape'] = shapes
        model_kwargs['inputs'] = self.intrinsic_model_input_keys(
            model_kwargs['inputs'])
        model_kwargs['dist'] = dist
        model_kwargs['normalization_impl'] = normalization_impl

        intr_rew_weights = config.int_rew_model.get(
            'intr_rew_weights',
            config.int_rew_model.get('expl_rew_weights', None))
        if intr_rew_weights:
            intr_rew_weights = intr_rew_weights.flat if hasattr(intr_rew_weights, 'flat') else dict(intr_rew_weights)

        return dict(
            model_kwargs=model_kwargs,
            pooling_kwargs=pooling_kwargs,
            ens_keys=tuple(dist.keys()),
            intr_rew_weights=intr_rew_weights,
        )

    def build_intrinsic_reward_model(self, config):
        spec = self.build_intrinsic_model_spec(config)
        self.ens_keys = spec['ens_keys']
        return IntrinsicRewardModel(
            num_heads=config.int_rew_model.num_heads,
            pooling=spec['pooling_kwargs'],
            model_kwargs=spec['model_kwargs'],
            disg_agg=config.int_rew_model.disg_agg,
            use_entropy=False,
            use_squared_disg=False,
            exploration_reward_weights=spec['intr_rew_weights'],
            name='int_rew_model',
        )

    def build_temperature(self, config):
        config_temp = config.temp.flat.copy()
        constraint_weight = config_temp.pop('constraint_weight')
        temp_wd = config_temp.pop('temp_wd')
        temp = Temperature(**config_temp, name='temp')
        temp.constraint_weight = constraint_weight
        temp.temp_wd = temp_wd
        return temp

    def build_slow_actor(self, config, source_actor, heads, name, updater_name):
        kwargs = {
            'shape': {
                k: self._action_shape(s)
                for k, s in self.act_space.items()},
            'dist': {
                k: config.actor_dist_disc if s.discrete else config.actor_dist_cont
                for k, s in self.act_space.items()},
        }
        slowactor = dr_nets.EnsembleMLP(
            **kwargs, num_heads=heads, **config.actor, name=name,
            dtype='float32')
        updater = jaxutils.SlowUpdater(
            source_actor, slowactor, config.slow_actor_fraction,
            config.slow_actor_update, name=updater_name)
        return slowactor, updater

    def build_intrinsic_components(self, config):
        """Build all intrinsic reward components."""
        int_rew_model = self.build_intrinsic_reward_model(config)
        
        # intrinsic Critic
        intr_critic = dr_nets.EnsembleMLP((), num_heads=self.value_heads, name='intr_critic', **config.critic)
        intr_slowcritic = dr_nets.EnsembleMLP((), num_heads=self.value_heads, name='intr_slowcritic', **config.critic, dtype='float32')
        intr_updater = jaxutils.SlowUpdater(intr_critic, intr_slowcritic, config.slow_critic_fraction, config.slow_critic_update, name='intr_updater')
        
        intr_retnorm = jaxutils.Moments(**config.retnorm, name='intr_retnorm')
        intr_valnorm = jaxutils.Moments(**config.valnorm, name='intr_valnorm')
        intr_advnorm = jaxutils.Moments(**config.advnorm, name='intr_advnorm')
        
        kwargs = {
            'shape': {k: self._action_shape(s) for k, s in self.act_space.items()},
            'dist': {k: config.actor_dist_disc if v.discrete else config.actor_dist_cont for k, v in self.act_space.items()}
        }
        slowactor = dr_nets.EnsembleMLP(**kwargs, num_heads=self.actor_heads, **config.actor, name='slowactor', dtype='float32')
        policy_updater = jaxutils.SlowUpdater(self.actor, slowactor, config.slow_actor_fraction, config.slow_actor_update, name='policy_updater')
        
        temp = self.build_temperature(config)
        
        return dict(
            int_rew_model=int_rew_model, intr_critic=intr_critic, intr_slowcritic=intr_slowcritic,
            intr_updater=intr_updater, intr_retnorm=intr_retnorm, intr_valnorm=intr_valnorm,
            intr_advnorm=intr_advnorm, slowactor=slowactor, policy_updater=policy_updater, temp=temp,
        )

    def build_optimizer(self, config, modules, dec):
        """Construct the optimizer and per-loss scaling dictionary."""
        kw = dict(config.opt)
        lr = kw.pop('lr')
        if config.separate_lrs:
            module_names = {m.name for m in modules}
            lr = {f'agent/{k}': v for k, v in config.lrs.items() if k in module_names}
        opt = jaxutils.Optimizer(lr, **kw, name='opt')

        scales = config.loss_scales.copy()
        cnn = scales.pop('dec_cnn')
        mlp = scales.pop('dec_mlp')
        scales.update({k: cnn for k in dec.imgkeys})
        scales.update({k: mlp for k in dec.veckeys})
        return opt, scales

    # ------------------------------------------------------------------
    # Spaces & initialization helpers
    # ------------------------------------------------------------------

    @property
    def aux_spaces(self):
        spaces = {}
        spaces['stepid'] = embodied.Space(np.uint8, 20)
        if self.config.replay_context:
            latdtype = jaxutils.COMPUTE_DTYPE
            latdtype = np.float32 if latdtype == jnp.bfloat16 else latdtype
            dyn_cfg = getattr(self.config.dyn, self.config.dyn.typ)
            deter_size = dyn_cfg.deter
            if deter_size > 0:
                spaces['deter'] = embodied.Space(latdtype, deter_size)
                spaces['stoch'] = embodied.Space(np.int32, dyn_cfg.stoch)
        return spaces

    @property
    def replay_context_keys(self):
        """Auxiliary replay fields used to restore recurrent state."""
        return tuple(k for k in self.aux_spaces if k != 'stepid')

    def init_policy(self, batch_size):
        lat, prevact = init_carry(self.dyn, self.act_space, batch_size)
        head_idx = jnp.zeros(batch_size, dtype=jnp.int32)
        return lat, prevact, head_idx

    def init_train(self, batch_size):
        return init_carry(self.dyn, self.act_space, batch_size)

    def init_report(self, batch_size):
        return self.init_train(batch_size)

    def preprocess(self, obs):
        return preprocess(obs, self.obs_space, self.act_space, self.aux_spaces)

    @property
    def dyn_heads(self):
        return self._dyn_heads

    @property
    def value_heads(self):
        return self._value_heads

    @property
    def actor_heads(self):
        return self._actor_heads

    # Bind on the Ninjax class so its metaclass preserves the report name scope.
    report = ReportMixin.report

    @abstractmethod
    def extra_report_metrics(self, data, carry):
        pass

    @abstractmethod
    def policy(self, obs, carry, mode='train'):
        pass

    @abstractmethod
    def train_updater(self):
        pass

    @abstractmethod
    def loss(self, data, carry, update=True):
        pass

    def _compute_exp_objective(self, *, particles=None, std_particles=None, mean_extr=None, std_extr=None, intr_mean=None, rb_mean=None, exp_obj=None):
        """Universal exploration objective.
        Two input modes (use one):
          - particles: [P, *batch, *q_shape] — function reduces to mean / safe_std internally.
            std_particles: optional [P', *batch, *q_shape] override for the std bonus.
          - mean_extr / std_extr: pre-computed scalar mean / std (already aggregated).
        intr_mean: intr Q mean for sombrl/ocb/intr.
        rb_mean: residual_bootstrap mean for rnd (acts as a learned uncertainty signal).
        OFU is encoded by passing particles already at fixed z, so `mean` IS the optimistic value."""
        obj = self.exp_obj if exp_obj is None else exp_obj
        mean = particles.mean(0) if particles is not None else mean_extr
        if obj in ('none', 'exploit', 'ofu'):
            return mean
        if obj == 'intr':
            return intr_mean
        beta = self._current_beta()
        std = (self.safe_std(std_particles if std_particles is not None else particles) if particles is not None else std_extr) if obj in ('ucb', 'ocb') else 0.0
        if obj == 'ucb':
            return mean + beta * std
        if obj == 'sombrl':
            alpha, _ = self.temp()
            return mean + alpha * intr_mean
        if obj == 'ocb':
            alpha, _ = self.temp()
            return mean + beta * std + alpha * intr_mean
        if obj == 'rnd':
            return mean + beta * rb_mean
        raise NotImplementedError(obj)

    def _lambda_returns(self, rewards_h, con_h, next_v, lam, discount, horizon=0, is_last_next=None):
        """TD(lambda) targets along time axis 2 of next_v ([N, B, T])."""
        cont = jnp.broadcast_to(discount * con_h, next_v.shape)
        one_step = jnp.broadcast_to(rewards_h, next_v.shape) + cont * next_v
        trace = jnp.full(next_v.shape, lam, f32)
        if is_last_next is not None:
            reset = jnp.asarray(is_last_next, dtype=bool)
            while reset.ndim < next_v.ndim:
                reset = reset[None]
            trace = jnp.where(jnp.broadcast_to(reset, next_v.shape), 0.0, trace)
        if 0 < horizon < next_v.shape[2]:
            cuts = (next_v.shape[2] - 1 - jnp.arange(next_v.shape[2])) % horizon == 0
            cut_shape = (1, 1, next_v.shape[2]) + (1,) * (next_v.ndim - 3)
            trace = jnp.where(cuts.reshape(cut_shape), 0.0, trace)
        def step(carry, inputs):
            bootstrap, value, discount_t, trace_t = inputs
            ret = bootstrap + trace_t * discount_t * (carry - value)
            return ret, ret
        values_t = jnp.moveaxis(next_v, 2, 0)
        scan_inputs = (jnp.moveaxis(one_step, 2, 0), values_t, jnp.moveaxis(cont, 2, 0), jnp.moveaxis(trace, 2, 0))
        _, returns = jax.lax.scan(step, values_t[-1], scan_inputs, reverse=True)
        return jnp.moveaxis(returns, 0, 2)

    def _temperature_actor(self):
        if getattr(self, 'use_exp_actor', False):
            return self.exp_actor, 1
        if hasattr(self, 'intr_actor'):
            return self.intr_actor, 1
        return self.actor, self.actor_heads

    def _temperature_actions(self, dyn_states, states, actor, actor_heads):
        if actor_heads < self.dyn_heads:
            actor_inp = self._align_ens_heads(dyn_states, actor_heads)
            bdims = 4 if self.dyn_heads > 1 else 3
            acts = sample(actor(actor_inp, bdims=bdims, has_ensemble=True))
            target_acts = sample(self.slowactor(actor_inp, bdims=bdims, has_ensemble=True))
            if self.dyn_heads > 1:
                return (treemap(lambda x: x[0, 0], acts),treemap(lambda x: x[0, 0], target_acts))
            return treemap(lambda x: x[0], acts), treemap(lambda x: x[0], target_acts)
        actor_inp = treemap(lambda x: self._broadcast_heads(x, actor_heads), states)
        take_actor_head = lambda tree: treemap(lambda x: self._index_head(x, 0, actor_heads), tree)
        acts = take_actor_head(sample(actor(actor_inp, bdims=3, has_ensemble=True)))
        target_acts = take_actor_head(sample(self.slowactor(actor_inp, bdims=3, has_ensemble=True)))
        return acts, target_acts

    def temp_loss(self, dyn_states, actor=None, actor_heads=None):
        states = self._take_dyn_head(dyn_states)
        if actor is None or actor_heads is None:
            default_actor, default_heads = self._temperature_actor()
            actor = default_actor if actor is None else actor
            actor_heads = default_heads if actor_heads is None else actor_heads
        acts, target_acts = self._temperature_actions(dyn_states, states, actor, actor_heads)
        model_states, model_acts = self._intrinsic_model_inputs(states, acts)
        target_states, target_model_acts = self._intrinsic_model_inputs(states, target_acts)
        sig, _ = self.int_rew_model(model_states, model_acts)
        target_sig, _ = self.int_rew_model(target_states, target_model_acts)
        constraint = sg(sig - target_sig)
        _, log_temp = self.temp()
        weight_decay = self.temp.temp_wd * jnp.square(jnp.exp(log_temp))
        return self.temp.constraint_weight * (log_temp * constraint) + weight_decay
    
    def _intrinsic_reward_model_input(self, replay_outs, prevacts, prevlat, data):
        if getattr(self, 'ac_inputs', 'wm') == 'obs':
            replay_outs = self._take_dyn_head(replay_outs)
            acts_prev = {k: val[:, 1:] for k, val in prevacts.items()}
            obs_prev = {k: f32(data[k][:, :-1]) for k in self._state_obs_keys}
            return obs_prev | acts_prev, replay_outs
        replay_outs = self._take_dyn_head(replay_outs)
        prevlat = self._take_dyn_head(prevlat)
        states_prev = {k: replay_outs[k][:, :-1] for k in replay_outs.keys() if k in prevlat.keys()}
        acts_prev = {k: val[:, 1:] for k, val in prevacts.items()}
        return states_prev | acts_prev, replay_outs

    def intrinsic_reward_loss(self, replay_outs, prevacts, prevlat, embed, data, learn_temp = None):
        if learn_temp is None:
            learn_temp = self.learn_temp
        replay_outs_headed = replay_outs
        model_input, replay_outs = self._intrinsic_reward_model_input(replay_outs, prevacts, prevlat, data)
        
        potential_labels = {'obs': embed[:, :-1], 'dyn': replay_outs['stoch'][:, 1:], 'rew': data['reward'][:, :-1],}
        labels = {k: sg(potential_labels[k]) for k in self.ens_keys}
        int_rew_loss, _ = self.int_rew_model.loss(labels=labels, model_input=sg(model_input))
        int_rew_loss = jnp.concatenate([int_rew_loss[:, 0:1] * 0, int_rew_loss], axis=-1)
        
        losses = {'int_rew_model': int_rew_loss}
        if learn_temp:
            losses['temp'] = self.temp_loss(replay_outs_headed)
        return losses

    def train(self, data, carry):
        self.config.jax.jit and embodied.print('Tracing train function', color='yellow')
        # data is always a dict-of-batches: {'ac': batch_ac, 'res': batch_res}.
        # `ac` drives the carry update and the replay-context / priority feedback; `res`
        # is used only by the corrector's (res+cor)^2 term inside ObserverAgent.loss.
        # When dual-batch training is off, jaxagent fills `res` with the same object as
        # `ac`; we detect that here and share the preprocessed batch.
        shared = data['ac'] is data['res']

        def _prep(batch, is_ac):
            nonlocal carry
            batch = self.preprocess(batch)
            sid = batch.pop('stepid')
            if self.config.replay_context:
                K = self.config.replay_context
                batch = batch.copy()
                context = {k: batch.pop(k)[:, :K] for k in self.replay_context_keys}
                if is_ac and 'stoch' in context:
                    dyn_cfg = getattr(self.config.dyn, self.config.dyn.typ)
                    context['stoch'] = f32(jax.nn.one_hot(context['stoch'], dyn_cfg.classes))
                    prevlat = self._dyn_outs_to_carry(context)
                    prevact = {k: batch[k][:, K - 1] for k in self.act_space}
                    carry = prevlat, prevact
                batch = {k: v[:, K:] for k, v in batch.items()}
                sid = sid[:, K:]
            if self.config.reset_context:
                keep = (jax.random.uniform(nj.seed(), batch['is_first'][:, :1].shape) > self.config.reset_context)
                batch['is_first'] = jnp.concatenate([batch['is_first'][:, :1] & keep, batch['is_first'][:, 1:]], 1)
            return batch, sid

        ac_batch, stepid_ac = _prep(data['ac'], is_ac=True)
        if shared:
            res_batch, stepid_res = ac_batch, stepid_ac
        else:
            res_batch, stepid_res = _prep(data['res'], is_ac=False)
        prepped = {'ac': ac_batch, 'res': res_batch}

        # Optimize
        mets, (out, carry, metrics) = self.opt(self.modules, self.loss, prepped, carry, has_aux=True)
        metrics.update(mets)
        self.train_updater()

        # Collect outputs for replay buffer. Format mirrors what Replay.update
        # expects: legacy flat (single sampler) when dual-batch is off, nested
        # per-sampler when on. Replay-context updates (e.g. 'stoch') always flow
        # through the 'ac' sampler's chunks (chunks are shared anyway).
        outs = {}
        ctx = {}
        if self.config.replay_context:
            ctx = {k: out['replay_outs'][k] for k in self.replay_context_keys}
            if 'stoch' in ctx:
                ctx['stoch'] = jnp.argmax(ctx['stoch'], -1).astype(jnp.int32)
        prio_ac = out.get('priority_ac', None)
        prio_res = out.get('priority_res', None)
        if self.config.dual_batch_training:
            ac_entry = {'stepid': stepid_ac, **ctx}
            if prio_ac is not None:
                ac_entry['priority'] = prio_ac
            res_entry = {'stepid': stepid_res}
            if prio_res is not None:
                res_entry['priority'] = prio_res
            outs['replay'] = {'ac': ac_entry, 'res': res_entry}
        elif self.config.replay_context or prio_ac is not None:
            outs['replay'] = {'stepid': stepid_ac, **ctx}
            if prio_ac is not None:
                outs['replay']['priority'] = prio_ac

        return outs, carry, metrics

    def _augment_images(self, data):
        """DrQ random-shift augmentation: edge-pad each image key by
        `config.image_shift_pad` and random-crop back to the original H×W, with an
        independent shift per image. Returns only the augmented image keys."""
        pad = int(self.config.image_shift_pad)
        if pad <= 0:
            return {}
        out = {}
        for k in self.enc.imgkeys:
            x = data[k]
            *lead, H, W, C = x.shape
            n = int(np.prod(lead)) if lead else 1
            flat = x.reshape((n, H, W, C))
            flat = jnp.pad(flat, [(0, 0), (pad, pad), (pad, pad), (0, 0)], mode='edge')
            dh = jax.random.randint(nj.seed(), (n,), 0, 2 * pad + 1)
            dw = jax.random.randint(nj.seed(), (n,), 0, 2 * pad + 1)
            crop = lambda img, i, j: jax.lax.dynamic_slice(img, (i, j, 0), (H, W, C))
            flat = jax.vmap(crop)(flat, dh, dw)
            out[k] = flat.reshape((*lead, H, W, C))
        return out

    def world_model_loss(self, data, carry, boot_prob=1.0, compute_losses=True, augment=False):
        """Run the replay rollout and (optionally) compute reconstruction / dynamics losses.

        When ``compute_losses=False`` the decoder / reward / cont heads and the
        dynamics-loss path are skipped: only the encoder + RSSM forward is run,
        which is all the corrector (res-batch) and obs-input observer need.
        """
        prevlat, prevact = carry
        prevacts = {k: jnp.concatenate([prevact[k][:, None], data[k][:, :-1]], 1) for k in self.act_space}
        prevacts = jaxutils.onehot_dict(prevacts, self.act_space)

        if augment:
            data = {**data, **self._augment_images(data)}
        embed = self.enc(data)
        target_embed = (
            self.target_enc(data)
            if getattr(self, 'target_enc', None) is not None else None)
        newlat, outs = self._dyn_observe(prevlat, prevacts, embed, data['is_first'], bdims=2)
        heads = self.dyn_heads
        replay_outs = (
            {**outs, 'target_embed': self._broadcast_heads(
                target_embed, heads)}
            if target_embed is not None else outs)
        newact = {k: data[k][:, -1] for k in self.act_space}

        if not compute_losses:
            return ({}, None, embed, replay_outs, prevacts, newlat, newact, {})

        outs_dec = treemap(lambda x: x.reshape((heads * x.shape[1],) + x.shape[2:])if self._has_head_axis(x, heads) and x.ndim >= 2 else x,outs)
        rew_feat = outs if self.config.reward_grad else sg(outs)
        dists = dict(**self.dec(outs_dec),reward=self._rew_dist(rew_feat, bdims=3, training=True),cont=self._con_dist(outs, bdims=3, training=True),)
        targets = {k: self._broadcast_heads(f32(data[k]), heads) for k in dists}
        if self.config.contdisc:
            targets['cont'] = self._broadcast_heads(f32(data['cont']) * (1 - 1 / self.config.horizon), heads)
        losses = {k: -self._dist_log_prob_headed(v, targets[k]) for k, v in dists.items()}
        dynlosses, wm_metrics = self.dyn.loss(outs, **self.config.rssm_loss)
        losses.update(dynlosses)

        boot_mask = self._sample_bootstrap_mask((heads, *data['reward'].shape), boot_prob=boot_prob)  # [heads, B, T] or None
        B, T = data['reward'].shape
        flat_mask = boot_mask.reshape(heads * B, T)  # for decoder losses
        for k in losses:
            if self._has_head_axis(losses[k], heads):
                losses[k] = losses[k] * boot_mask        # [heads, B, T]
            else:
                losses[k] = losses[k] * flat_mask         # [heads*B, T]
        return (losses, dists, embed, replay_outs, prevacts, newlat, newact, wm_metrics)

    

    def openloop_predict(self, data, carry, outs):
        B, T = data['is_first'].shape
        num_obs = min(self.config.report_openl_context, T // 2)

        img_start, rec_outs = self._dyn_observe(carry[0], {k: v[:, :num_obs] for k, v in outs['prevacts'].items()}, outs['embed'][:, :num_obs], data['is_first'][:, :num_obs], bdims=2)

        img_acts = {k: v[:, num_obs:] for k, v in outs['prevacts'].items()}
        img_outs = self._dyn_imagine(img_start, img_acts, bdims=2)[1]

        rec = dict(**self._decode_openloop(rec_outs), reward=self._rew_dist(rec_outs, bdims=3), cont=self._con_dist(rec_outs, bdims=3),)
        img = dict(**self._decode_openloop(img_outs), reward=self._rew_dist(img_outs, bdims=3),cont=self._con_dist(img_outs, bdims=3),)
        return rec, img, num_obs
    
    def compute_ret_adv(self, rew, con, tarval, a, b, retnorm=None, advnorm=None, update=True): # TODO: Refactor to make functionality much clearer
        """Compute MC lambda returns, where a and b determine the scale of the """
        rets = [tarval[..., -1]]
        interm = rew[..., 1:] + (1 - a) * b * tarval[..., 1:]
        for t in reversed(range(b.shape[-1])):
            rets.append(interm[..., t] + a[..., t] * b[..., t] * rets[-1])
        ret = jnp.stack(list(reversed(rets))[:-1], -1)

        if retnorm is not None: # retnorm
            roffset, rscale = retnorm(ret, update)
            adv = (ret - tarval[..., :-1]) / rscale
        else:
            adv = ret

        if advnorm is not None: # None by default
            aoffset, ascale = advnorm(adv, update)
            adv_normed = (adv - aoffset) / ascale
        else:
            adv_normed = adv

        metrics = {}
        metrics.update(jaxutils.tensorstats(ret, 'ret'))
        metrics.update(jaxutils.tensorstats(adv_normed, 'adv_normed'))

        return {'ret': ret, 'adv_normed': adv_normed, 'tarval': tarval}, metrics

    def safe_std(self, x, axis=0, eps=1e-8):
        variance = jnp.var(x, axis=axis)
        q_std = jnp.sqrt(jnp.maximum(variance, eps))
        q_std = jnp.where(jnp.isfinite(q_std), q_std, 0.0)
        return jnp.clip(q_std, 0, 100.0)

    def _rank_corr(self, x, y, eps=1e-8): # TODO: Move into report
        """Spearman rank correlation between flattened x and y."""
        x, y = x.reshape(-1), y.reshape(-1)
        rx = jnp.argsort(jnp.argsort(x)).astype(jnp.float32)
        ry = jnp.argsort(jnp.argsort(y)).astype(jnp.float32)
        rx = rx - rx.mean()
        ry = ry - ry.mean()
        return (rx * ry).sum() / (jnp.linalg.norm(rx) * jnp.linalg.norm(ry) + eps)

    def _sample_bootstrap_mask(self, ref_shape, boot_prob = None):
        """Bernoulli bootstrap mask ``(N, B, T)``.  None when disabled."""
        if boot_prob is None:
            boot_prob = self.config.boot_prob
        return jax.random.bernoulli(nj.seed(), boot_prob, shape=ref_shape).astype(f32)

    # ------------------------------------------------------------------
    # Head manipulation
    # ------------------------------------------------------------------

    def _align_ens_heads(self, inp, num_heads):
        """Align dyn-headed [E, B, ...] for a network with num_heads heads.
        If num_heads >= E: broadcast [E, ...] → [num_heads, ...].
        If num_heads == 1 < E: nest → [1, E, B, ...] (caller must use bdims+1)."""
        E = self._dyn_heads
        if E <= num_heads:
            return self._expand_to_heads(inp, num_heads)
        assert num_heads == 1, f"num_heads={num_heads} < dyn_heads={E}, only 1 supported"
        return treemap(lambda x: x[None], inp)

    def _expand_to_heads(self, tree, num_heads):
        """Broadcast headed [H, B, ...] → [num_heads, B, ...].  H must be 1 or num_heads."""
        def _bcast(x):
            assert x.ndim >= 1 and x.shape[0] in (1, num_heads), \
                f"_expand_to_heads: expected head dim 1 or {num_heads}, got shape {x.shape}"
            return jnp.broadcast_to(x, (num_heads, *x.shape[1:]))
        return treemap(_bcast, tree)
    
    def _broadcast_heads(self, x, heads):
        """Add head axis to unheaded tensor: [B, ...] → [heads, B, ...]."""
        assert x.ndim >= 1, f"_broadcast_heads: expected ≥1-D tensor, got {x.ndim}-D"
        return jnp.broadcast_to(x[None], (heads, *x.shape))

    def _has_head_axis(self, x, heads):
        return hasattr(x, 'ndim') and x.ndim > 0 and x.shape[0] == heads

    def _take_dyn_head(self, tree, index=0):
        """Select one dyn head: [E, B, ...] → [B, ...]. E=dyn_heads."""
        return treemap(lambda x: self._index_head(x, index, self.dyn_heads), tree)

    def _take_actor_head(self, tree, index=0):
        """Select one actor head: [A, B, ...] → [B, ...]. A=actor_heads."""
        return treemap(lambda x: self._index_head(x, index, self.actor_heads), tree)

    def _take_value_head(self, tree, index=0):
        """Select one value head: [V, B, ...] → [B, ...]. V=value_heads."""
        return treemap(lambda x: self._index_head(x, index, self.value_heads), tree)

    def _index_head(self, x, index, heads):
        """Select head(s) from a headed [H, B, ...] tensor. Supports scalar or per-batch index."""
        if not self._has_head_axis(x, heads):
            return x
        if heads == 1:
            return x[0]
        if isinstance(index, int) or (hasattr(index, 'ndim') and index.ndim == 0):
            return x[index]
        B = x.shape[1]
        return x[index, jnp.arange(B)]

    # ------------------------------------------------------------------
    # Dynamics wrappers
    # ------------------------------------------------------------------

    def _dyn_observe(self, carry, action, embed, reset, bdims=2):
        """Observe with ensemble dynamics.
        carry: [E, B, ...] headed.  action (dict), embed, reset: [B, ...] unheaded."""
        E = self.dyn_heads
        newlat, outs = self.dyn.observe(
            self._expand_to_heads(carry, E),
            treemap(lambda x: self._broadcast_heads(x, E), action),
            self._broadcast_heads(embed, E),
            self._broadcast_heads(reset, E),
            bdims=bdims,
        )
        # DrQ mode: expose the (gradient-carrying) image embedding as an AC input,
        # broadcast to the dyn-head axis to match the other state leaves.
        if getattr(self, 'ac_inputs', 'wm') == 'drq':
            outs = {**outs, 'embed': self._broadcast_heads(embed, E)}
        return newlat, outs

    def _dyn_imagine(self, carry, action, bdims=2):
        """Imagine with ensemble dynamics.
        carry: [E, B, ...] headed.  action (dict): [B, ...] unheaded."""
        E = self.dyn_heads
        return self.dyn.imagine(self._expand_to_heads(carry, E), treemap(lambda x: self._broadcast_heads(x, E), action), bdims=bdims,)

    def _dyn_outs_to_carry(self, outs):
        """Convert unheaded outs [B, ...] to carry by adding dyn head axis."""
        E = self.dyn_heads
        outs_h = treemap(lambda x: self._broadcast_heads(x, E), outs)
        return self.dyn.outs_to_carry(outs_h)

    # ------------------------------------------------------------------
    # Distribution helpers
    # ------------------------------------------------------------------

    def _rew_prior_mean(self, outs, bdims):
        """outs: [E, B, ...] headed."""
        reward_prior_scale = getattr(self, 'reward_prior_scale', getattr(self, 'prior_scale', 0.0))
        if not hasattr(self, 'rew_prior') or reward_prior_scale <= 0:
            return None
        prior = self.rew_prior(self._expand_to_heads(outs, self.dyn_heads), bdims=bdims, has_ensemble=True).mean()
        return reward_prior_scale * sg(prior)

    def _current_beta(self):
        # Read β for inference / non-dual loss sites. When learned, gradient is
        # stopped here so only the dual β-loss can update log_beta.
        if hasattr(self, 'log_beta'):
            return sg(self._beta_with_grad())
        beta = getattr(self, 'beta', 0.0)
        return beta.read() if hasattr(beta, 'read') else jnp.asarray(beta, f32)

    def _beta_with_grad(self):
        # β = exp(tanh-bounded log_beta), matching BRO's Adjustment (with offset=0).
        # `beta_pessimism` enters only in the dual loss, not here.
        if not hasattr(self, 'log_beta'):
            return self._current_beta()
        cfg = self.config
        log_min = float(cfg.beta_log_min)
        log_max = float(cfg.beta_log_max)
        raw = self.log_beta.read()
        bounded = log_min + (log_max - log_min) * 0.5 * (1.0 + jnp.tanh(raw))
        return jnp.exp(bounded)

    def _current_tau(self):
        # KL Lagrange multiplier (BRO's `regularizer()`). Stops gradient at read sites
        # so only the τ dual loss updates log_tau.
        if hasattr(self, 'log_tau'):
            return sg(self._tau_with_grad())
        return jnp.asarray(float(getattr(self.config, 'actor_kl_reg', 0.0)), f32)

    def _tau_with_grad(self):
        # τ = exp(tanh-bounded log_tau), same parameterization as β.
        if not hasattr(self, 'log_tau'):
            return self._current_tau()
        cfg = self.config
        log_min = float(cfg.tau_log_min)
        log_max = float(cfg.tau_log_max)
        raw = self.log_tau.read()
        bounded = log_min + (log_max - log_min) * 0.5 * (1.0 + jnp.tanh(raw))
        return jnp.exp(bounded)

    def _rew_dist(self, outs, bdims, training=False): # TODO: Remove reward prior.
        """outs: [E, B, ...] headed."""
        dist = self.rew(self._expand_to_heads(outs, self.dyn_heads), bdims=bdims, training=training, has_ensemble=True)
        prior = self._rew_prior_mean(outs, bdims)
        return ShiftedDist(dist, prior) if prior is not None else dist

    def _con_dist(self, outs, bdims, training=False):
        """outs: [E, B, ...] headed."""
        return self.con(self._expand_to_heads(outs, self.dyn_heads), bdims=bdims, training=training, has_ensemble=True)

    def _dist_log_prob_headed(self, dist, target):
        """Log prob aligning headed target [H, B, ...] with dist batch shape."""
        target = jnp.asarray(target)
        assert target.ndim >= 2, f'Expected target [heads, batch, ...], got shape {target.shape}'
        heads, batch = target.shape[:2]
        bshape = tuple(dist.batch_shape)
        if len(bshape) >= 2 and bshape[:2] == (heads, batch):
            return dist.log_prob(target)
        if len(bshape) >= 1 and bshape[0] == heads * batch:
            flat = target.reshape((heads * batch,) + target.shape[2:])
            logp = dist.log_prob(flat)
            return logp.reshape((heads, batch) + logp.shape[1:])
        if heads == 1 and len(bshape) >= 1 and bshape[0] == batch:
            return dist.log_prob(target[0])[None]
        raise ValueError(f'Cannot align target shape {target.shape} with distribution batch shape {bshape}.')

    def _flatten_imag_outs(self, outs):
        """Flatten replay/imagination states from [H, B, T, ...] to [H, B*(T-1), ...]."""
        return treemap(lambda x: jnp.swapaxes(x[:, :, :-1], 1, 2).reshape(x.shape[0], -1, *x.shape[3:]), outs,)

### Action Selection Helpers

    def _sample_action_candidates(self, states, K=4, actor_net=None, actor_heads=None, bdims=2, temperature=None):
        actor_net = actor_net or self.actor
        actor_heads = self.actor_heads if actor_heads is None else int(actor_heads)
        actor_inp = self._align_ens_heads(states, actor_heads)
        kwargs = dict(bdims=bdims, has_ensemble=True)
        if temperature is not None:
            kwargs['temperature'] = temperature
        actor_dist = actor_net(actor_inp, **kwargs)
        return sample_k(actor_dist, K)

    def _make_continuous_action_grid(self, points_per_axis):
        specs = []
        axes = []
        for key, space in self.act_space.items():
            assert not space.discrete, key
            shape = tuple(space.shape)
            size = int(np.prod(shape))
            low = np.broadcast_to(space.low, shape).reshape(-1)
            high = np.broadcast_to(space.high, shape).reshape(-1)
            low = np.where(np.isfinite(low), low, -1.0)
            high = np.where(np.isfinite(high), high, 1.0)
            specs.append((key, shape, size))
            axes.extend(np.linspace(low[i], high[i], points_per_axis, dtype=np.float32) for i in range(size))
        mesh = np.meshgrid(*axes, indexing='xy')
        flat = np.stack([x.reshape(-1) for x in mesh], axis=-1)
        grids = {}
        start = 0
        for key, shape, size in specs:
            grids[key] = flat[:, start:start + size].reshape((-1, *shape))
            start += size
        return grids

    def _grid_action_candidates(self, leading_shape):
        """Broadcast fixed continuous action candidates to [K, *leading, ...]."""
        leading_shape = tuple(leading_shape)
        grids = getattr(self, '_cand_action_grid', None)
        if grids is None:
            return None
        return {key: jnp.broadcast_to(jnp.asarray(grid, f32).reshape((grid.shape[0],) + (1,) * len(leading_shape) + grid.shape[1:]), (grid.shape[0], *leading_shape, *grid.shape[1:])) for key, grid in grids.items()}

    def _select_candidate_index(self, score, num_candidates, axis, mode='eval'):
        if mode == 'eval':
            return jnp.argmax(score, axis=axis)
        cand_policy = getattr(self, 'cand_policy', 'softmax')
        if cand_policy == 'epsilon_greedy':
            act_idx = jnp.argmax(score, axis=axis)
            rand = jax.random.randint(nj.seed(), act_idx.shape, 0, num_candidates)
            explore = jax.random.uniform(nj.seed(), act_idx.shape)
            return jnp.where(explore < getattr(self, 'greedy_eps', 0.1), rand, act_idx)
        if cand_policy == 'thompson_sampling':
            raise NotImplementedError('')
        temperature = getattr(self, 'sampling_temp', 1.0)
        return jax.random.categorical(nj.seed(), score / temperature, axis=axis)
