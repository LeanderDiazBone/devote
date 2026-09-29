import jax
import jax.numpy as jnp
import numpy as np

import embodied
from devote.agent import BaseAgent, init_carry
from dreamerv3 import jaxagent
from dreamerv3 import jaxutils
from dreamerv3 import ninjax as nj
from dreamerv3.nets import SquashedNormal

from . import nets as dr_nets
from .metrics import MetricsCollector
from .visual import build_visual_encoder

f32 = jnp.float32
treemap = jax.tree_util.tree_map
sg = lambda x: treemap(jax.lax.stop_gradient, x)
cast = jaxutils.cast_to_compute
sample = lambda dist: {k: v.sample(seed=nj.seed()) for k, v in dist.items()}


def _kl_loc_scale(d):
    # SquashedNormal.mean() returns tanh(loc) (the deterministic action), not the base loc.
    # The Tanh Jacobian cancels in KL between two SquashedNormals, so the Gaussian KL on the
    # base (loc, scale) is exact.
    if isinstance(d, SquashedNormal):
        return d._loc, d._scale
    return d.mean(), d.stddev()


class _DrQTargetQ:
    """Present the Polyak encoder feature to an otherwise unchanged target Q."""

    def __init__(self, q, target_embed):
        self.q = q
        self.target_embed = target_embed

    def particles(self, inputs, *args, **kwargs):
        embed = self.target_embed
        while embed.ndim < inputs['embed'].ndim:
            embed = embed[None]
        inputs = {
            **inputs,
            'embed': jnp.broadcast_to(embed, inputs['embed'].shape),
        }
        return self.q.particles(inputs, *args, **kwargs)


class ActorDisposition:
    """Routes per z component ('ensemble', 'epistemic'), each one of 'input'|'output'|'marginal'.
    'input': z is part of the actor's inputs (sampled per is_first, persisted in carry).
    'output': actor produces z as part of its action (OFU's _z, _z_head).
    'marginal': z is integrated out at this site (no routing)."""

    def __init__(self, ensemble='marginal', epistemic='marginal'):
        self.ensemble = ensemble
        self.epistemic = epistemic

    def __getitem__(self, key):
        return getattr(self, key)

    def components_with(self, value):
        return [c for c in ('ensemble', 'epistemic') if getattr(self, c) == value]


def _disp(actor):
    return getattr(actor, 'z_disposition', ActorDisposition())


class ObserverAgent(BaseAgent):
    """Off-policy agent based on world model features."""

    def _q_input_keys(self):
        return list(self.config.critic.inputs) if self._is_discrete else list(self.config.critic.inputs) + list(self.act_space.keys())

    extra_report_metrics = BaseAgent._report_observer

    def __init__(self, obs_space, act_space, config):
        self.obs_space = {k: v for k, v in obs_space.items() if not k.startswith('log_')}
        self.act_space = {k: v for k, v in act_space.items() if k != 'reset'}
        skip = {'is_first', 'is_last', 'is_terminal', 'reward', 'cont', 'stepid'}
        self._state_obs_keys = sorted(k for k, v in obs_space.items() if k not in skip and not k.startswith('log_') and len(v.shape) == 1 and np.issubdtype(v.dtype, np.floating))
        self.ac_inputs = config.ac_inputs
        if self.ac_inputs == 'obs':
            config = config.update({k: list(self._state_obs_keys) for k in ('actor.inputs', 'critic.inputs', 'critic_prior.inputs')})
        elif self.ac_inputs == 'drq':
            config = config.update({k: ['embed'] for k in ('actor.inputs', 'critic.inputs', 'critic_prior.inputs')})
        self.config = config

        # Cached config knobs (used in hot paths).
        self._dyn_heads = 1
        self._value_heads = config.ens_heads
        self._actor_heads = config.actor_ens_heads
        self.policy_mode = config.policy_mode
        self.exp_obj = config.exp_obj
        self._aug_pad = int(config.image_shift_pad)
        self._drq_num_views = int(getattr(config, 'drq_num_views', 1))
        if self._drq_num_views < 1:
            raise ValueError(
                f'drq_num_views must be positive, got {self._drq_num_views}.')
        if self._drq_num_views > 1 and (
                self.ac_inputs != 'drq' or self._aug_pad <= 0):
            raise ValueError(
                'drq_num_views > 1 requires ac_inputs=drq and '
                'image_shift_pad > 0.')
        self.normalize_exp_mean = bool(config.normalize_exp_mean)
        self.value_prior_scale = config.value_prior_scale
        self.reward_prior_scale = config.reward_prior_scale
        self.prior_scale = self.value_prior_scale
        self.visual_prior = config.visual_prior != 'none'
        self.use_exp_actor = self.policy_mode == 'exp_actor'
        self.use_intrinsic_reward = self.exp_obj in ('sombrl', 'ocb')
        self.use_temp = self.use_intrinsic_reward
        # Only train the world model when something actually consumes its
        # outputs: an intrinsic-reward model that depends on dynamics, or Q /
        # actor inputs that read from the latent state (ac_inputs == 'wm').
        # Otherwise we still need the WM forward for `replay_outs` plumbing,
        # but the decoder / reward / cont heads and dynamics losses are dead
        # weight. (ac_inputs='drq' reads the encoder embedding directly, so the
        # encoder is trained by the critic loss, not by reconstruction.)
        self.train_world_model = self.use_intrinsic_reward or self.ac_inputs == 'wm'
        self.learn_temp = self.use_temp and config.loss_scales.get('temp', 0.0) != 0
        self.discount = config.discount if config.discount >= 0 else 1 - 1 / config.horizon
        self.use_candidate_policy = self.policy_mode == 'exp_sampling'
        self._is_discrete = all(s.discrete for s in self.act_space.values())
        self.enumerate_actions = config.enumerate_actions and self._is_discrete
        if self._is_discrete:
            self._num_actions = self._discrete_classes(list(self.act_space.values())[0])
        self._cand_action_grid = None
        if int(config.cand_grid_points) > 0 and not self._is_discrete:
            self._cand_action_grid = self._make_continuous_action_grid(int(config.cand_grid_points))
        assert config.policy_mode in ('actor', 'exp_actor', 'exp_sampling', 'thompson_sampling'), config.policy_mode
        assert config.exp_obj in ('ucb', 'sombrl', 'ocb', 'ofu', 'rnd', 'none')
        assert config.pc_target_actor in ('actor', 'exp_actor'), config.pc_target_actor
        assert config.td_target_actor in ('actor', 'exp_actor'), config.td_target_actor
        assert config.td_target_mode in ('joint', 'separate', 'q_full', 'rnd'), config.td_target_mode
        assert self._actor_heads == 1 or self._actor_heads == self._value_heads
        if self.use_candidate_policy:
            assert self.enumerate_actions or self._cand_action_grid is not None, 'use_candidate_policy needs enumerate_actions=True or cand_grid_points>0'

        # Networks.
        self.learn_beta = bool(config.beta_learn) and self.use_exp_actor and not self.use_candidate_policy
        # τ (KL Lagrange) is tuned simultaneously with β against the same target.
        self.learn_tau = self.learn_beta
        if self.learn_beta:
            # BRO-style: store raw log_beta, map via tanh-bounded log to β = exp(·).
            # Initialize raw so that bounded β = config.beta_init.
            log_init = float(np.log(max(float(config.beta_init), 1e-12)))
            log_min, log_max = float(config.beta_log_min), float(config.beta_log_max)
            log_init = float(np.clip(log_init, log_min + 1e-6, log_max - 1e-6))
            # Invert bounded = log_min + (log_max - log_min) * 0.5 * (1 + tanh(raw)) → raw.
            u = (log_init - log_min) / (log_max - log_min)
            raw_init = float(np.arctanh(2.0 * u - 1.0))
            self.log_beta = nj.Variable(lambda: jnp.asarray(raw_init, f32), name='log_beta')
        else:
            self.beta = nj.Variable(lambda: jnp.asarray(config.beta, f32), name='beta')
        if self.learn_tau:
            # Same tanh-bounded log parameterization as β.
            log_init_t = float(np.log(max(float(config.tau_init), 1e-12)))
            log_min_t, log_max_t = float(config.tau_log_min), float(config.tau_log_max)
            log_init_t = float(np.clip(log_init_t, log_min_t + 1e-6, log_max_t - 1e-6))
            u_t = (log_init_t - log_min_t) / (log_max_t - log_min_t)
            raw_init_t = float(np.arctanh(2.0 * u_t - 1.0))
            self.log_tau = nj.Variable(lambda: jnp.asarray(raw_init_t, f32), name='log_tau')
        self._action_dim = float(sum(np.prod(s.shape) for s in self.act_space.values())) if not self._is_discrete else 1.0
        self.cur_step = nj.Variable(lambda: jnp.zeros((), jnp.int32), name='cur_step')
        enc_space, dec_space = self.build_spaces(obs_space, config)
        wm = self.build_world_model(config, enc_space, dec_space)
        self.enc, self.dec, self.dyn, self.rew, self.rew_prior, self.con = wm['enc'], wm['dec'], wm['dyn'], wm['rew'], wm['rew_prior'], wm['con']
        self.target_enc = None
        self.target_enc_updater = None
        if self.ac_inputs == 'drq':
            self.target_enc = dr_nets.DrQEncoder(
                enc_space, name='target_enc', **config.enc.drq)
            self.target_enc_updater = jaxutils.SlowUpdater(
                self.enc, self.target_enc, config.slow_critic_fraction,
                config.slow_critic_update, name='target_enc_updater')
        if self.visual_prior:
            # Frozen pretrained encoder + one fixed random projection per
            # Q-prior member. Neither joins self.modules, so both stay frozen.
            imgkeys = [k for k, v in self.obs_space.items() if len(v.shape) == 3]
            assert len(imgkeys) == 1, f'visual_prior needs exactly one image observation, got {imgkeys}'
            self._visual_key = imgkeys[0]
            (self.visual_enc, self._visual_output_key, self._visual_feature_axis,
             self._visual_embed_dim) = build_visual_encoder(config.visual_prior)
            d = int(config.visual_prior_dim)
            if d < 0:
                raise ValueError(
                    f'visual_prior_dim must be non-negative, got {d}.')
            self.visual_proj = nj.Variable(
                lambda: jax.random.normal(
                    nj.seed(), (self._value_heads, self._visual_embed_dim, d),
                    f32),
                name='visual_prior_proj') if d > 0 else None
        use_rb = config.td_target_mode in ('q_full', 'rnd') and bool(config.critic_prior.corrector) and self.value_prior_scale > 0
        if config.td_target_mode in ('q_full', 'rnd') and not use_rb:
            print(f"[observer] td_target_mode={config.td_target_mode!r} requires corrector + residual_bootstrap; falling back to 'joint'.")
            self._td_target_mode = 'joint'
        else:
            self._td_target_mode = config.td_target_mode
        critic = self.build_q_function(config, use_multihead_q=self._is_discrete, heads=self._value_heads, pessimism=float(config.pessimism), residual_bootstrap=use_rb)
        self.q, self.q_target, self.q_updater, self.q_prior = critic['q'], critic['q_target'], critic['q_updater'], critic['q_prior']
        init_alpha = float(np.log(config.sac.init_alpha))
        if not self.use_candidate_policy:
            actor = self.build_actor(config, self.act_space, heads=self._actor_heads, name='actor_sac')
            self.actor, self.retnorm = actor['actor'], actor['retnorm']
            self.log_alpha = nj.Variable(lambda: jnp.full(self._actor_heads, init_alpha, f32), name='log_alpha')
        if self.use_exp_actor:
            exp_act_space = dict(self.act_space)
            if self.exp_obj == 'ofu':
                z_space = self.q.z_space()
                if z_space is not None:
                    exp_act_space['_z'] = z_space
                exp_act_space['_z_head'] = embodied.Space(np.int32, (), low=0, high=self.q.num_heads)
            self._exp_act_space = exp_act_space
            exp_actor = self.build_actor(config, exp_act_space, heads=1, name='exp_actor_sac')
            self.exp_actor, self.exp_retnorm = exp_actor['actor'], exp_actor['retnorm']
            self.exp_log_alpha = nj.Variable(lambda: jnp.full(1, init_alpha, f32), name='exp_log_alpha')
        if self.use_intrinsic_reward:
            self.int_rew_model = self.build_intrinsic_reward_model(config)
            intr_q = self.build_q_function(config, use_multihead_q=self._is_discrete, heads=1, names=['intr_q', 'intr_q_target', 'intr_q_prior', 'intr_q_updater'], prior_scale=0.0)
            self.intr_q, self.intr_q_target, self.intr_q_updater = intr_q['q'], intr_q['q_target'], intr_q['q_updater']
            if not self.use_exp_actor:
                intr_actor = self.build_actor(config, self.act_space, heads=1, name='intr_actor_sac')
                self.intr_actor, self.intr_retnorm = intr_actor['actor'], intr_actor['retnorm']
                self.intr_log_alpha = nj.Variable(lambda: jnp.full(1, init_alpha, f32), name='intr_log_alpha')
            if self.use_temp:
                self.temp = self.build_temperature(config)
                if self.learn_temp:
                    self.slowactor, self.policy_updater = self.build_slow_actor(config, self.exp_actor if self.use_exp_actor else self.intr_actor, heads=1, name='slow_intr_actor', updater_name='intr_policy_updater')

        # Target entropy and disposition routing.
        scale = config.sac.target_entropy_scale_disc if self._is_discrete else config.sac.target_entropy_scale_cont
        self.target_entropy = scale * self._discrete_entropy() if self._is_discrete else -scale * float(sum(np.prod(s.shape) for s in self.act_space.values()))
        self._set_dispositions()

        # Modules + optimizer.
        self.modules = [self.enc, self.dyn, self.dec, self.rew, self.con] + list(self.q.heads)
        if not self.use_candidate_policy:
            self.modules += [self.actor, self.log_alpha]
        if self.use_exp_actor:
            self.modules += [self.exp_actor, self.exp_log_alpha]
            if self.learn_beta:
                self.modules.append(self.log_beta)
            if self.learn_tau:
                self.modules.append(self.log_tau)
        if self.use_intrinsic_reward:
            self.modules += [self.int_rew_model] + list(self.intr_q.heads)
            if not self.use_exp_actor:
                self.modules += [self.intr_actor, self.intr_log_alpha]
            if self.use_temp:
                self.modules.append(self.temp)
        
        if config.freeze_all:
            self.modules = []
        elif config.freeze_wm:
            self.modules = [m for m in self.modules if m not in (self.enc, self.dyn, self.dec, self.rew, self.con)]
        elif not self.train_world_model:
            # world_model_loss(compute_losses=False) skips dec/rew/con, so they
            # never lazily create params; the optimizer would then crash trying
            # to find any. enc/dyn are still called via _dyn_observe and stay.
            self.modules = [m for m in self.modules if m not in (self.dec, self.rew, self.con)]
        if config.run.script == 'policy_eval':
            # Zero loss does not prevent optimizer weight decay or parameter noise.
            # The policy and all non-Q modules must remain exactly unchanged.
            self.modules = list(self.q.heads)
        self.opt, self.scales = self.build_optimizer(config, self.modules, self.dec)
        if getattr(config, 'freeze_corrector', False):
            # Keep the parent Q learning-rate groups, but optimize only its raw
            # and bootstrap heads so decay and optimizer noise cannot move c.
            trainable = []
            for module in self.modules:
                if module in self.q.heads:
                    trainable.extend(m for m in (module.head, module.residual_bootstrap)
                                     if m is not None)
                else:
                    trainable.append(module)
            self.modules = trainable

        ls = config.loss_scales
        for i in range(len(self.q.heads)):
            self.scales['q' if i == 0 else f'q{i + 1}'] = ls.get('critic', 1.0)
            self.scales['q_bt' if i == 0 else f'q{i + 1}_bt'] = ls.get('critic_bt', ls.get('critic', 1.0))
            self.scales['q_pc' if i == 0 else f'q{i + 1}_pc'] = ls.get('critic_pc', ls.get('critic', 1.0))
        if not self.use_candidate_policy:
            self.scales['actor_sac'] = ls.get('actor', 1.0)
            self.scales['alpha'] = ls.get('alpha', 1.0)
        if self.use_exp_actor:
            self.scales['exp_actor_sac'] = ls.get('exp_actor', ls.get('actor', 1.0))
            self.scales['exp_alpha'] = ls.get('exp_alpha', 1.0)
            if self.learn_beta:
                self.scales['exp_beta'] = ls.get('beta', 1.0)
            if self.learn_tau:
                self.scales['exp_tau'] = ls.get('tau', ls.get('beta', 1.0))
        if self.use_intrinsic_reward:
            self.scales['int_rew_model'] = ls.get('int_rew_model', 1.0)
            self.scales['intr_q'] = ls.get('intr_q', 1.0)
            if not self.use_exp_actor:
                self.scales['intr_actor_sac'] = ls.get('actor', 1.0)
        if self.learn_temp:
            self.scales['temp'] = ls['temp']
        self._log_metrics_table = config.log_metrics_table

    def _set_dispositions(self):
        """Attach an ActorDisposition to each actor describing how z components flow."""
        if not self.use_candidate_policy:
            self.actor.z_disposition = ActorDisposition(ensemble='input' if self._actor_heads > 1 else 'marginal')
        if self.use_exp_actor:
            self.exp_actor.z_disposition = (ActorDisposition(ensemble='output', epistemic='output' if self.q.epistemic_dim > 0 else 'marginal') if self.exp_obj == 'ofu' else ActorDisposition())
        if self.use_intrinsic_reward and not self.use_exp_actor:
            self.intr_actor.z_disposition = ActorDisposition()

    @property
    def aux_spaces(self):
        """Policy outputs persisted alongside observations in replay."""
        spaces = {**super().aux_spaces, **{f'z_explore_{c}': s for c, s in self.q.z_aux_spaces().items()}}
        if self.visual_prior:
            spaces['visual_embed'] = embodied.Space(np.float32, self._visual_embed_dim)
        return spaces

    @property
    def replay_context_keys(self):
        return tuple(k for k in super().replay_context_keys if k != 'visual_embed')

    def _action_probs(self, actor_dist):
        key = next(k for k in actor_dist.keys() if k != '_z')
        return actor_dist[key].probs_parameter()

    def _q_input(self, states, actions=None):
        inp = dict(states)
        if not self._is_discrete and actions is not None:
            inp.update({k: actions[k] for k in self.act_space})
        return inp

    def preprocess(self, obs):
        obs = super().preprocess(obs)
        # Init batches contain a placeholder aux value; run the encoder once while
        # creating so its Flax state exists before cached replay features are used.
        if self.visual_prior and (nj.creating() or 'visual_embed' not in obs):
            x = f32(obs[self._visual_key])
            lead, hwc = x.shape[:-3], x.shape[-3:]
            feat = self.visual_enc(x.reshape((-1,) + hwc), train=False)
            if self._visual_output_key is not None:
                feat = feat[self._visual_output_key]
            axis = self._visual_feature_axis % feat.ndim
            pool_axes = tuple(i for i in range(1, feat.ndim) if i != axis)
            feat = feat.mean(pool_axes) if pool_axes else feat
            assert feat.shape[-1] == self._visual_embed_dim, (feat.shape, self._visual_embed_dim)
            feat = feat / jnp.maximum(jnp.linalg.norm(feat, axis=-1, keepdims=True), 1e-6)
            obs['visual_embed'] = sg(feat).reshape(lead + feat.shape[-1:])
        elif self.visual_prior:
            assert obs['visual_embed'].shape[-1] == self._visual_embed_dim, (
                obs['visual_embed'].shape, self._visual_embed_dim)
        return obs

    def _augment_ac_inputs(self, states, raw, headed=True):
        """Add raw obs keys to a state dict when ac_inputs='obs'.
        headed=True: raw[k] is broadcast with a leading dyn-head axis to match headed states."""
        if self.visual_prior:
            # Replay and agent state keep the full encoder feature. The frozen
            # Q-prior applies its member-specific projection at the call site.
            feat = sg(raw['visual_embed'])
            states = {**states, 'visual_embed': feat[None] if headed else feat}
        if self.ac_inputs != 'obs':
            return states
        out = dict(states)
        for k in self._state_obs_keys:
            out[k] = raw[k][None] if headed else raw[k]
        return out

    def _sg_ac_states(self, states):
        """Stop-gradient critic state inputs, but keep the DrQ image `embed`
        differentiable so the critic loss trains the shared encoder. The actor path
        sg's its states at the call site, so the encoder stays detached there (DrQ)."""
        if self.ac_inputs != 'drq' or 'embed' not in states:
            return sg(states)
        return {**sg(states), 'embed': states['embed']}

    def _select_action_values(self, q_values, actions):
        if not self._is_discrete:
            return q_values
        onehot = actions[next(iter(self.act_space.keys()))]
        if q_values.ndim == onehot.ndim and q_values.shape[0] != onehot.shape[0]:
            repeats = q_values.shape[0] // onehot.shape[0]
            onehot = jnp.repeat(onehot[None], repeats, axis=0)
            onehot = onehot.reshape((q_values.shape[0],) + onehot.shape[2:])
        return (q_values * onehot).sum(-1)

    # ----- z-context actor helpers ------------------------------------------
    # `z_input` is a dict shared between policy() and losses; per-component shape:
    #   z['ensemble']  : [*batch, num_heads]  one-hot
    #   z['epistemic'] : [*batch, epistemic_dim]
    # Keys present iff actor's disposition for that component is 'input'.

    def _refresh_z_input(self, actor, z_input, is_first):
        """Resample input-disposition z components on is_first; persist otherwise."""
        components = _disp(actor).components_with('input')
        if not components:
            return z_input
        fresh = self.q.sample_z(is_first.shape, components=components)
        out = {}
        for c in components:
            if c not in fresh: continue
            cur = z_input.get(c, fresh[c])
            mask = is_first.reshape(is_first.shape + (1,) * (cur.ndim - is_first.ndim))
            out[c] = jnp.where(mask, fresh[c], cur)
        return out
    
    def _critic_z_for_actor_eval(self, actor, z_output, *, batch_shape):
        """Build the z dict to thread into the critic when scoring an actor's actions.
        ensemble:input → diagonal one-hot (each actor head h ↔ critic head h).
        ensemble:output → use the actor's sampled z_head. epistemic:output → forward z."""
        disp = _disp(actor)
        z = {}
        if disp.ensemble == 'input':
            V = self.q.num_heads
            assert batch_shape[0] == V
            eye = jnp.eye(V, dtype=f32).reshape((V,) + (1,) * (len(batch_shape) - 1) + (V,))
            z['ensemble'] = jnp.broadcast_to(eye, batch_shape + (V,))
        elif disp.ensemble == 'output':
            z['ensemble'] = z_output['ensemble']
        if disp.epistemic == 'output':
            z['epistemic'] = z_output['epistemic']
        return z

    def _actor_step(self, actor, inputs, *, z_input=None, bdims=2, select_head=False):
        """Sample from `actor`, route z, optionally collapse H axis.
        select_head=False (training): acts has leading [H, *batch, ...].
        select_head=True (policy): acts is [*batch, ...]; head picked per ensemble disposition."""
        z_input = z_input or {}
        disp = _disp(actor)
        if disp.epistemic == 'input':
            inputs = {**inputs, 'epistemic': z_input['epistemic']}
        actor_dist = actor(inputs, bdims=bdims, has_ensemble=False)
        raw = treemap(cast, sample(actor_dist))                                # [H, *batch, ...]
        if select_head:
            if disp.ensemble == 'input':
                idx = jnp.argmax(z_input['ensemble'], axis=-1)
                def pick(x):
                    expand = (None,) + (slice(None),) * idx.ndim + (None,) * (x.ndim - idx.ndim - 1)
                    return jnp.take_along_axis(x, idx[expand], axis=0).squeeze(0)
                raw = treemap(pick, raw)
            else:
                raw = treemap(lambda x: x[0], raw)
        z_output = {}
        if disp.ensemble == 'output' and '_z_head' in raw:
            z_output['ensemble'] = raw.pop('_z_head')
        if disp.epistemic == 'output' and '_z' in raw:
            z_output['epistemic'] = raw.pop('_z')
        return raw, z_output, actor_dist

    def _frozen_call(self, module, fn):
        """Call `fn(module)` with sg'd params (gradients into module are dropped)."""
        params = module.find()
        module.put(sg(params))
        try:
            return fn(module)
        finally:
            module.put(params)

    def _actor_logpi(self, actor_dist, acts, z_out):
        """Sum log_probs of `acts` plus the actor's output-disposition components (`_z`, `_z_head`)."""
        logpi = sum(d.log_prob(acts[k]) for k, d in actor_dist.items() if k in acts)
        for zk, zv in (('_z', z_out.get('epistemic')), ('_z_head', z_out.get('ensemble'))):
            if zv is not None and zk in actor_dist:
                logpi = logpi + actor_dist[zk].log_prob(zv)
        return logpi

    def _exp_norm_rscale(self):
        """Percentile span used to normalize the extrinsic mean_term in exp_score.
        Returns None when normalize_exp_mean is off or no retnorm has been built."""
        if not self.normalize_exp_mean:
            return None
        if hasattr(self, 'exp_retnorm'):
            _, rscale = self.exp_retnorm.stats()
            return rscale
        if hasattr(self, 'retnorm'):
            _, rscale = self.retnorm.stats()
            return rscale
        return None

    def exp_score(self, particles, *, exp_obj=None, rscale=None):
        """Sum of mean + bonus (see exp_score_parts)."""
        m, b = self.exp_score_parts(particles, exp_obj=exp_obj, rscale=rscale)
        return m + b

    def exp_score_parts(self, particles, *, exp_obj=None, rscale=None):
        """(mean_term, bonus_term) for the configured exploration objective.
        particles: dict with leading P-axis; keys among {raw, prior, corrector, residual_bootstrap, intr}.
        Both terms reduce P; remaining shape matches input minus the leading axis.
        rscale: if provided and self.normalize_exp_mean is on, mean_term /= rscale so β is scale-free."""
        obj = self.exp_obj if exp_obj is None else exp_obj
        beta = self._current_beta()
        if obj == 'intr':
            alpha, _ = self.temp()
            mean_term = jnp.zeros_like(particles['intr'].mean(0))
            return mean_term, alpha * particles['intr'].mean(0)

        full_q = particles['raw'] + particles['prior'] + particles['corrector']
        mean_term = (full_q if obj == 'ofu' else particles['raw']).mean(0)
        if self.normalize_exp_mean and rscale is not None:
            mean_term = mean_term / rscale
        bonus_term = jnp.zeros_like(mean_term)
        if obj in ('ucb', 'ocb', 'rnd'):
            rb = particles.get('residual_bootstrap', full_q)
            if self.config.residual_transform > 0 and 'residual_bootstrap' in particles:
                alpha_rb = (1.0 - self.discount) * jnp.log(self.config.residual_transform_reduction)
                rb = jnp.exp(alpha_rb * rb)
            if obj in ('ucb', 'ocb'):
                bonus_term = bonus_term + beta * self.safe_std(rb)
            if obj == 'rnd':
                bonus_term = bonus_term + beta * rb.mean(0)
        if obj in ('sombrl', 'ocb'):
            alpha, _ = self.temp()
            bonus_term = bonus_term + alpha * particles['intr'].mean(0)
        return mean_term, bonus_term

    def _aggregate_targets(self, target_qs): # TODO: Remove we always use individual
        if self.config.sac.target_mode == 'individual':
            return target_qs
        elif self.config.sac.target_mode == 'min':
            return target_qs.min(axis=0)[None]
        elif self.config.sac.target_mode == 'random_min':
            num_heads = target_qs.shape[0]
            k = min(self.config.sac.random_target_k, num_heads)
            keys = jax.random.split(nj.seed(), num_heads)
            def rand_min(key):
                idx = jax.random.choice(key, num_heads, shape=(k,), replace=False)
                return target_qs[idx].min(axis=0)
            return jax.vmap(rand_min)(keys)
        elif self.config.sac.target_mode == 'mean':
            return target_qs.mean(axis=0)[None]

    # ----- Candidate-action evaluation (shared by policy(), PC channel, candidate TD) -----
    # All paths follow the same template:
    #   1) `_candidate_qs`: get Q-particles at every candidate action (enumerate or grid).
    #   2) Selection: argmax (greedy) or `_candidate_score`+`_select_candidate_index` (policy).
    #   3) `_gather_act`: index the grid by the selected idx (no-op for enumerate one-hot).
    # Output shapes:
    #   enumerate: qs [V[*S], *batch, A]; acts=None (action is the trailing axis).
    #   grid:      qs [V[*S], K, *batch];  acts={k: [K, *batch, d]}.

    def _candidate_qs(self, states_unh, q, *, samples=None, z=None, **eval_kw):
        """Q at all candidate actions (mixed internally if `q` is a double-Q wrapper)."""
        bdims = next(iter(states_unh.values())).ndim - 1
        def at(act):
            inp = self._q_input(states_unh, act) if act is not None else self._q_input(states_unh)
            return q.particles(inp, bdims=bdims, samples=samples, frozen=False, z=z, **eval_kw)
        if self.enumerate_actions:
            return at(None), None
        leading = next(iter(states_unh.values())).shape[:bdims]
        acts = self._grid_action_candidates(leading)
        qs = jnp.swapaxes(jax.vmap(at)(acts), 0, 1)
        return qs, acts

    def _candidate_score(self, states_unh, *, samples=None, z=None):
        """Exp-objective score at each candidate, plus the candidate acts (for grid).
        z: optional {'epistemic': [...], 'ensemble': [...]} dict threaded to component_particles."""
        bdims = next(iter(states_unh.values())).ndim - 1
        def parts_at(act):
            inp = self._q_input(states_unh, act) if act is not None else self._q_input(states_unh)
            p = self.q.component_particles(inp, bdims=bdims, samples=samples, frozen=False, z=z)
            if self.use_intrinsic_reward:
                p['intr'] = self.intr_q.particles(inp, bdims=bdims, samples=samples, frozen=False)
            return p

        if self.enumerate_actions:
            parts = parts_at(None)
            acts = None
        else:
            leading = next(iter(states_unh.values())).shape[:bdims]
            acts = self._grid_action_candidates(leading)
            parts = jax.vmap(parts_at)(acts)
            parts = {k: jnp.swapaxes(v, 0, 1) for k, v in parts.items()}
        return self.exp_score(parts, rscale=self._exp_norm_rscale()), acts

    def _gather_act(self, acts, idx):
        """Index grid candidates at per-batch idx. acts={k: [K, *batch, d]}, idx [*batch] → {k: [*batch, d]}."""
        grids = jnp.meshgrid(*[jnp.arange(s) for s in idx.shape], indexing='ij')
        return {k: a[(idx,) + tuple(grids)] for k, a in acts.items()}

    def _candidate_pick(self, states_unh, *, scoring='greedy', q=None, samples=1, z=None, **eval_kw):
        """Pick a candidate action from candidates.
          scoring='greedy': aggregated max over `q.particles` (mixed internally for double-Q). Returns (next_v [N, *batch], next_acts).
          scoring='exp':    argmax over `_candidate_score`'s exploration objective. Returns (None, next_acts)."""
        if scoring == 'exp':
            scored, acts = self._candidate_score(states_unh, samples=samples)
            scored = scored[None]                                        # [1, ...] for uniform downstream handling
            return_v = False
        else:
            qs, acts = self._candidate_qs(states_unh, q or self.q, samples=samples, z=z, **eval_kw)
            scored = self._aggregate_targets(qs)                         # [N, *batch, A] or [N, K, *batch]
            return_v = True
        if self.enumerate_actions:
            idx = scored[0].argmax(-1)
            key = next(iter(self.act_space.keys()))
            next_acts = {key: jax.nn.one_hot(idx, scored.shape[-1]).astype(scored.dtype)}
            next_v = scored.max(-1) if return_v else None
        else:
            idx = scored[0].argmax(0)
            next_acts = self._gather_act(acts, idx)
            next_v = scored.max(1) if return_v else None
        return next_v, next_acts

    def _select_candidate_action(self, states, mode='eval', z=None):
        """Pick action via candidate evaluation.
          mode='explore': exp-objective score (mean + bonus), sampled via cand_policy.
          mode='train':   raw Q (matching the TD target), sampled via cand_policy.
          mode='eval':    raw Q, argmax.
        z: optional dict threaded to scoring so Q is evaluated at the same z that will be persisted."""
        states_unh = self._take_dyn_head(states)
        if mode == 'explore':
            score, acts = self._candidate_score(states_unh, z=z)
        else:
            qs, acts = self._candidate_qs(states_unh, self.q, z=z, include_prior=False, include_corrector=False, include_residual_bootstrap=False)
            # Reduce V (value heads) with the configured aggregation, then take row 0
            # so `score` matches `_candidate_score`'s shape ([*batch, A] or [K, *batch]).
            score = self._aggregate_targets(qs)[0]
        if self.enumerate_actions:
            idx = self._select_candidate_index(score, self._num_actions, -1, mode)
            key = next(iter(self.act_space.keys()))
            return {key: cast(jax.nn.one_hot(idx, self._num_actions))}
        idx = self._select_candidate_index(score, score.shape[0], 0, mode)
        return self._gather_act(acts, idx)

    def _pc_next_components(self, q_module, s_unh, pc_acts, pc_actor, kw):
        """component_means at (s', a') with disposition-aware (V, H) → V reduction.
        s_unh: unheaded state dict, leaves [B, T, F].
        pc_acts: dict, leaves [H, B, T, ...] (H=actor's parallel heads, or H=1 for candidate).
        pc_actor: actor module providing disposition (or None for candidate path).
        kw: forwarded to component_means; epistemic broadcast to H if needed."""
        H = next(iter(pc_acts.values())).shape[0]
        s_h = treemap(lambda x: jnp.broadcast_to(x[None], (H,) + x.shape), s_unh)
        inputs = self._q_input(s_h, pc_acts) if not self._is_discrete else self._q_input(s_h)
        kw = {k: (jnp.broadcast_to(v[None], (H,) + v.shape) if k in ('epistemic', 'ensemble') and v is not None else v) for k, v in kw.items()}
        comps = q_module.component_means(inputs, bdims=3, has_ensemble=False, **kw)
        # comps[k]: [V, H, B, T, *q_shape]; reduce H per disposition.
        if pc_actor is not None and _disp(pc_actor).ensemble == 'input':
            V = next(iter(comps.values())).shape[0]
            idx = jnp.arange(V)
            return {k: v[idx, idx] for k, v in comps.items()}
        return {k: v[:, 0] for k, v in comps.items()}

    def _ofu_z_pair(self, td_target_mode, z_next, buffer_z, states_unh):
        """Per-component (z_cur, z_next) for OFU. Caller has already verified OFU is active.
        Three branches, all producing the same shape contract:
          bellman_z_from_target: RHS z' from π_exp(s'); z_cur from buffer (q_full) or shared with z_next.
          q_full / rnd:           sample one z, share between z_cur and z_next (symmetric LHS/RHS index).
          separate / joint:       z_cur from π_exp(s); z_next mirrors it."""
        if not self.q.z_aux_spaces():
            return {}, z_next
        is_q_full = td_target_mode in ('q_full', 'rnd')
        if self.config.bellman_z_from_target:
            assert z_next, 'ofu+bellman_z_from_target requires pc_target_actor=exp_actor'
            z_cur = ({c: sg(buffer_z[c]) for c in self.q.z_aux_spaces()} if is_q_full else dict(z_next))
        elif is_q_full:
            bshape = next(iter(states_unh.values())).shape[:2]
            z_cur = {c: sg(v) for c, v in self.q.sample_z(bshape, components=self.q.z_aux_spaces()).items()}
            z_next = dict(z_cur)
        else:
            _, exp_z_s, _ = self._actor_step(self.exp_actor, sg(states_unh), bdims=2)
            z_cur = {c: sg(v)[0] for c, v in exp_z_s.items()}
            z_next = dict(z_cur)
        return z_cur, z_next

    def _compute_q_targets(self, q_target, next_states_unh, con, rewards, alpha, discount, actor_net, actor_heads, value_heads, is_last_next=None, q=None, q_states_unh=None, q_actions_unh=None, include_prior=True, include_corrector=True, qctx_unh=None, return_aux=False):
        metrics = {}
        # When the caller asks for raw-only (include_prior=False), drop the RB shift too —
        # otherwise the raw head's TD target depends on the bootstrap output.
        kw = dict(include_prior=include_prior, include_corrector=include_corrector, include_residual_bootstrap=bool(include_prior), samples=1, frozen=False)

        if actor_net is None:
            # Candidate-action path: greedy TD target, no actor / entropy backup.
            z_target = {'epistemic': qctx_unh} if qctx_unh is not None else None
            next_v, next_acts = self._candidate_pick(next_states_unh, q=q_target, z=z_target, **{k: v for k, v in kw.items() if k.startswith('include_')})
            next_actor = None
        else:
            H = actor_heads
            acts, z_out, next_actor = self._actor_step(actor_net, next_states_unh, bdims=2)
            inputs_h = treemap(lambda x: jnp.broadcast_to(x[None], (H,) + x.shape), next_states_unh)
            inputs = self._q_input(inputs_h, acts) if not self._is_discrete else self._q_input(inputs_h)
            z_t = {'epistemic': jnp.broadcast_to(qctx_unh[None], (H,) + qctx_unh.shape)} if qctx_unh is not None else {}
            p_q = q_target.particles(inputs, z=z_t, bdims=3, **kw)
            # Disposition reduction: diagonal for ensemble:input (Thompson); broadcast (acts[0]) otherwise.
            V = p_q.shape[0]
            target_qs = self._aggregate_targets(p_q[jnp.arange(V), jnp.arange(V)] if _disp(actor_net).ensemble == 'input' else p_q[:, 0])
            # SAC entropy backup (use_entropy_backup_sac toggles the regularizer).
            alpha_v = self._expand_to_heads(alpha, value_heads)
            if self._is_discrete:
                probs_v = self._expand_to_heads(self._action_probs(next_actor), value_heads)
                next_v = (probs_v * (target_qs - self.config.use_entropy_backup_sac * alpha_v[..., None] * jnp.log(probs_v + 1e-8))).sum(-1)
            else:
                logpi = self._actor_logpi(next_actor, acts, z_out)
                next_v = target_qs - self.config.use_entropy_backup_sac * alpha_v * self._expand_to_heads(logpi, value_heads)
            next_acts = acts                                                 # [H, B, T, ...]

        rh, ch = rewards[None], con[None]
        targets = sg(self._lambda_returns(rh, ch, next_v, self.config.td_lambda, discount, self.config.td_lambda_horizon, is_last_next) if self.config.td_lambda > 0 else rh + discount * ch * next_v)
        metrics['target_mean'] = targets.mean()
        metrics.update(self.report_prior_target_metrics(q, q_states_unh, q_actions_unh, next_states_unh, next_acts, discount, qctx_unh))
        if return_aux: # TODO: Always return aux
            return targets, metrics, dict(next_actor=next_actor, next_acts=next_acts)
        return targets, metrics

    def _intr_q_loss(self, q, q_target, states, actions, rewards, next_states, con, weight, log_alpha, actor_net, is_last_next=None):
        """SAC-style TD loss for the single-head intrinsic Q-critic.
        intr_q is built with prior_scale=0 → no prior/corrector/RB and no epistemic z; the primary-critic
        machinery (PC channel, RB shared target, OFU z routing, per-head iteration) all reduces to no-ops,
        so we skip it entirely here."""
        losses, metrics = {}, {}
        alpha = jnp.exp(sg(log_alpha.read()))[:, None, None]
        states_unh = self._take_dyn_head(sg(states))
        actions_unh = self._take_dyn_head(sg(actions))
        next_states_unh = self._take_dyn_head(sg(next_states))
        q_inp_unh = self._q_input(states_unh, actions_unh) if not self._is_discrete else self._q_input(states_unh)

        targets, target_metrics, _ = self._compute_q_targets(q_target, next_states_unh, con, rewards, alpha, self.discount, actor_net, actor_heads=1, value_heads=1, is_last_next=is_last_next, q=q, q_states_unh=states_unh, q_actions_unh=actions_unh, include_prior=False, include_corrector=False, qctx_unh=None, return_aux=True)

        boot_mask = self._sample_bootstrap_mask((1,) + states_unh['deter'].shape[:2])
        w = sg(weight)[None] * boot_mask

        q_all = q.heads[0](q_inp_unh, bdims=2, has_ensemble=False, include_prior=False, include_corrector=False, include_residual_bootstrap=False)
        base = self._base_td_loss(q_all, targets, actions_unh)

        losses['intr_q'] = (w * base).mean(axis=0)
        metrics['expl/q-td-loss'] = (w * base).mean()
        metrics.update({f'expl/{k}': v for k, v in target_metrics.items()})
        metrics['expl/cont_mean'] = con.mean()
        return losses, metrics

    def _compute_q_losses(self, q, q_target, states, actions, rewards, next_states, con, weight, log_alpha, actor_net, actor_heads, value_heads, loss_name, metric_prefix, is_last_next=None, buffer_z=None, pc_states=None, pc_actions=None, pc_weight=None, target_result=None):
        """SAC-style TD loss for the primary critic (intr_q has its own _intr_q_loss)."""
        losses, metrics = {}, {}
        alpha = (jnp.exp(sg(log_alpha.read()))[:, None, None] if log_alpha is not None else None)
        # Unheaded views for all new-API critic calls.
        states_unh = self._take_dyn_head(self._sg_ac_states(states))
        actions_unh = self._take_dyn_head(sg(actions))
        next_states_unh = self._take_dyn_head(sg(next_states))
        q_inp_unh = self._q_input(states_unh, actions_unh) if not self._is_discrete else self._q_input(states_unh)
        qctx_unh = q.sample_context(q_inp_unh, bdims=2) if hasattr(q, 'sample_context') else None
        td_target_mode = self._td_target_mode

        if target_result is None:
            target_result = self._compute_q_targets(q_target, next_states_unh, con, rewards, alpha, self.discount, actor_net, actor_heads=actor_heads, value_heads=value_heads, is_last_next=is_last_next, q=q, q_states_unh=states_unh, q_actions_unh=actions_unh,include_prior=False, include_corrector=False, qctx_unh=qctx_unh, return_aux=True)
        targets, target_metrics, aux = target_result

        # ----- PC channel target action ----------------------------------------
        # pc_target_actor='actor' reuses the exploit a' from _compute_q_targets;
        # 'exp_actor' uses the exploration policy (exp_actor sample, or — under
        # use_candidate_policy — the argmax of the exp objective over candidates).
        use_exp_for_pc = (self.config.pc_target_actor == 'exp_actor' and (self.use_exp_actor or self.use_candidate_policy))
        ofu = self.exp_obj == 'ofu' and self.use_exp_actor
        z_next = {}
        if not use_exp_for_pc:
            pc_next_actor, pc_acts = aux['next_actor'], aux['next_acts']
        elif self.use_candidate_policy:
            pc_next_actor = None
            pc_acts = treemap(lambda x: x[None], self._candidate_pick(next_states_unh, scoring='exp')[1])
        else:
            pc_acts, pc_z_out, pc_next_actor = self._actor_step(self.exp_actor, sg(next_states_unh), bdims=2)
            pc_acts = sg(pc_acts)
            z_next = {c: sg(v)[0] for c, v in pc_z_out.items()}

        # OFU z routing: the corrector / RB losses index Q in z symmetrically per component.
        z_cur, z_next = self._ofu_z_pair(td_target_mode, z_next, buffer_z, states_unh) if ofu else ({}, z_next)

        # PC kw: epistemic falls back to qctx_unh; ensemble has no fallback.
        eps_fb = {'epistemic': qctx_unh} if qctx_unh is not None else {}
        pc_kw_cur = {**eps_fb, **z_cur}
        pc_kw_next = {**eps_fb, **z_next}

        boot_mask = self._sample_bootstrap_mask((value_heads,) + states_unh['deter'].shape[:2])
        w = sg(weight)[None] * boot_mask

        # pc inputs (res batch). When dual-batch is off, pc_states is None and we
        # fall back to the ac-batch inputs/weights so behaviour is unchanged.
        if pc_states is not None:
            pc_states_unh = self._take_dyn_head(sg(pc_states))
            pc_actions_unh = self._take_dyn_head(sg(pc_actions))
            q_inp_unh_pc = self._q_input(pc_states_unh, pc_actions_unh) if not self._is_discrete else self._q_input(pc_states_unh)
            qctx_unh_pc = q.sample_context(q_inp_unh_pc, bdims=2) if hasattr(q, 'sample_context') else None
            eps_fb_pc = {'epistemic': qctx_unh_pc} if qctx_unh_pc is not None else {}
            # z routing: keep symmetric with pc_kw_cur but drop the ac-batch z's, which are batch-aligned.
            pc_kw_cur_pc = {**eps_fb_pc}
            pc_boot_mask = self._sample_bootstrap_mask((value_heads,) + pc_states_unh['deter'].shape[:2])
            w_pc = sg(pc_weight)[None] * pc_boot_mask
        else:
            q_inp_unh_pc = None
            pc_actions_unh = None
            pc_kw_cur_pc = None
            w_pc = w

        def expected_next(v):
            """Reduce per-action [V, B, T, A] to [V, B, T] under the pc_next policy (no-op for continuous)."""
            if not self._is_discrete:
                return v
            weights = (pc_acts[next(iter(self.act_space.keys()))][0] if pc_next_actor is None else self._expand_to_heads(self._action_probs(pc_next_actor), value_heads))
            return (weights * v).sum(-1)

        # Mode-specific shared bootstrap target (q_full / rnd only).
        bt_target_shared, rb_metrics = self._rb_shared_target(q_target, td_target_mode, pc_acts, pc_next_actor, pc_kw_next, next_states_unh, con, expected_next, is_last_next=is_last_next) if (td_target_mode in ('q_full', 'rnd') and q.residual_bootstrap is not None) else (None, {})

        for i, (q_mod, q_tgt) in enumerate(zip(q.heads, q_target.heads)):
            suf = '' if i == 0 else str(i + 1)
            pre = '' if i == 0 else f'q{i + 1}-'
            if td_target_mode in ('q_full', 'rnd'):
                bl, pl, btl = self._head_loss_full(q_mod, q_inp_unh, actions_unh, qctx_unh, targets, pc_kw_cur, bt_target_shared, td_target_mode, rb_metrics, pre, q_inp_unh_pc=q_inp_unh_pc, actions_unh_pc=pc_actions_unh, pc_kw_cur_pc=pc_kw_cur_pc)
            else:
                bl, pl = self._head_loss_residual(q_mod, q_tgt, q_inp_unh, actions_unh, qctx_unh, targets, td_target_mode, pc_kw_cur, pc_kw_next, pc_acts, pc_next_actor, next_states_unh, con, expected_next)
                btl = jnp.zeros_like(bl)
            # base TD, residual bootstrap, and pc are exposed as separate loss keys
            # so the priority signal can target any of them (`td` vs `rb_td` vs `pc`).
            # All three train on the ac batch except pc, which trains on res.
            losses[loss_name + suf] = (w * bl).mean(axis=0)
            losses[loss_name + suf + '_bt'] = (w * btl).mean(axis=0)
            losses[loss_name + suf + '_pc'] = (w_pc * pl).mean(axis=0)
            metrics[f'{metric_prefix}/{pre}q-td-loss'] = (w * bl).mean()
            metrics[f'{metric_prefix}/{pre}uncertainty-td-loss'] = (w_pc * pl).mean()
            if q_mod.residual_bootstrap is not None:
                metrics[f'{metric_prefix}/{pre}residual-bootstrap-td-loss'] = (w * btl).mean()
        for k, v in rb_metrics.items():
            metrics[f'{metric_prefix}/{k}'] = v

        metrics.update({f'{metric_prefix}/{k}': v for k, v in target_metrics.items()})
        metrics[f'{metric_prefix}/cont_mean'] = con.mean()
        return losses, metrics

    def _compute_drq_targets(
            self, batches, log_alpha, actor_net, actor_heads, value_heads):
        """Average targets from DrQ views while reusing the standard target."""
        alpha = (
            jnp.exp(sg(log_alpha.read()))[:, None, None]
            if log_alpha is not None else None)
        results = []
        for batch in batches:
            states_unh = self._take_dyn_head(sg(batch['states']))
            actions_unh = self._take_dyn_head(sg(batch['actions']))
            next_states_unh = self._take_dyn_head(sg(batch['next_states']))
            target_embed = self._take_dyn_head(
                sg(batch['target_next_embed']))
            q_inp_unh = (
                self._q_input(states_unh, actions_unh)
                if not self._is_discrete else self._q_input(states_unh))
            qctx_unh = (
                self.q.sample_context(q_inp_unh, bdims=2)
                if hasattr(self.q, 'sample_context') else None)
            results.append(self._compute_q_targets(
                _DrQTargetQ(self.q_target, target_embed), next_states_unh,
                batch['con'], batch['rewards'], alpha, self.discount,
                actor_net, actor_heads=actor_heads,
                value_heads=value_heads,
                is_last_next=batch['is_last_next'], q=self.q,
                q_states_unh=states_unh, q_actions_unh=actions_unh,
                include_prior=False, include_corrector=False,
                qctx_unh=qctx_unh, return_aux=True))

        targets, metrics, aux = zip(*results)
        mean = lambda *xs: jnp.stack(xs).mean(0)
        return sg(mean(*targets)), treemap(mean, *metrics), aux

    def _apply_drq_views(
            self, lossfn, data, carry, first_batch, augment,
            metric_prefix='sac'):
        """Give the critic all independent DrQ views in one call.

        The first view is produced by the main world-model pass. Additional
        views rerun only the encoder and dynamics rollout with fresh shifts.
        Passing them together lets the critic average their target values once
        and reuse that shared target for every current-observation view, as in
        DrQ.
        """
        batches = [first_batch]
        if (augment and self.ac_inputs == 'drq'
                and self._drq_num_views > 1):
            for _ in range(1, self._drq_num_views):
                _, _, _, replay_outs, _, _, _, _ = self.world_model_loss(
                    data, carry, compute_losses=False, augment=True)
                batches.append(self._prepare_train_batch(data, replay_outs))

        losses, metrics = lossfn(batches)
        metrics[f'{metric_prefix}/drq_num_views'] = jnp.asarray(
            len(batches), f32)
        return losses, metrics

    def _base_td_loss(self, q_dist, targets, actions_unh):
        """NLL of `targets` under `q_dist`; selects the taken action for discrete actions."""
        if self._is_discrete:
            onehot = sg(actions_unh[next(iter(self.act_space.keys()))])
            ts = jnp.repeat(targets[..., None], onehot.shape[-1], axis=-1)
            return -(q_dist.log_prob(ts, sum_last=False) * onehot).sum(-1)
        return -q_dist.log_prob(targets)

    def _rb_shared_target(self, q_target, td_target_mode, pc_acts, pc_next_actor, pc_kw_next, next_states_unh, con, expected_next, is_last_next=None):
        """Shared (across q1/q2) RB TD target for q_full / rnd modes.
        q_full: γ·c·(rb_next + cp − Δ).   rnd: γ·c·(rb_next + |cp| − Δ).   cp = corrector + prior.
        Δ = residual_transform downshift; fixed point (c+p→0): rb⋆ = −γΔ/(1−γ), so OOD heads carry
        a non-zero prior on the bootstrap residual. Per-head bt_next is mixed via q_target.mix.
        With td_lambda > 0 the 1-step backup is replaced by a λ-return where the per-step
        pseudo-reward is 0 and the bootstrap value at s_{t+1} is `rb_next + cp − Δ`."""
        rb_metrics = {}
        kw_next_full = dict(pc_kw_next, prior_scale=1.0, corrector_scale=1.0)
        def _bt_next(q_tgt, name):
            cn = self._pc_next_components(q_tgt, next_states_unh, pc_acts, pc_next_actor, kw_next_full)
            cp_signed = cn['prior'] + cn['corrector']
            cp = jnp.abs(cp_signed) if td_target_mode == 'rnd' else cp_signed
            rb_metrics[f'{name}rb_next/target_mean'] = cn['residual_bootstrap'].mean()
            rb_metrics[f'{name}rb_next/target_abs_mean'] = jnp.abs(cn['residual_bootstrap']).mean()
            rb_metrics[f'{name}rb_next/target_head_std'] = self.safe_std(cn['residual_bootstrap'], axis=0).mean()
            rb_metrics[f'{name}rb_next/cp_signed_mean'] = cp_signed.mean()
            rb_metrics[f'{name}rb_next/cp_abs_mean'] = jnp.abs(cp_signed).mean()
            rb_metrics[f'{name}rb_next/source_mean'] = cp.mean()
            return expected_next(sg(cn['residual_bootstrap'] + cp - self.config.residual_transform))
        prefixes = [''] if len(q_target.heads) == 1 else ['', 'q2_']
        bts = [_bt_next(h, p) for h, p in zip(q_target.heads, prefixes)]
        bt_next = bts[0]
        if len(bts) > 1:
            rb_metrics['rb_next/q12_disagree_mean'] = jnp.abs(bts[0] - bts[1]).mean()
            rb_metrics['rb_next/q12_disagree_max'] = jnp.abs(bts[0] - bts[1]).max()
            bt_next = q_target.mix(bts[0], bts[1])
        if self.config.td_lambda > 0:
            rh = jnp.zeros((1,) + bt_next.shape[-2:], bt_next.dtype)
            ch = con[None]
            bt_target_shared = sg(self._lambda_returns(rh, ch, bt_next, self.config.td_lambda, self.discount, self.config.td_lambda_horizon, is_last_next))
        else:
            bt_target_shared = sg(self.discount * con[None] * bt_next)
        rb_metrics['rb_target/mean'] = bt_target_shared.mean()
        rb_metrics['rb_target/abs_mean'] = jnp.abs(bt_target_shared).mean()
        rb_metrics['rb_target/head_std'] = self.safe_std(bt_target_shared, axis=0).mean()
        return bt_target_shared, rb_metrics

    def _head_loss_full(self, q_module, q_inp_unh, actions_unh, qctx_unh, targets, pc_kw_cur, bt_target_shared, td_target_mode, rb_metrics, pre, q_inp_unh_pc=None, actions_unh_pc=None, pc_kw_cur_pc=None):
        """q_full / rnd per-head loss: corrector regresses toward −prior; RB tracks bt_target_shared.
        Returns (base, pc, bt). The pc term may be computed on a separate batch
        (`q_inp_unh_pc`/`actions_unh_pc`/`pc_kw_cur_pc`) when dual-batch training
        routes (res+cor)^2 through the corrector (`res`) batch. Defaults reuse the
        ac-batch inputs. Updates rb_metrics in place."""
        q_all = q_module(q_inp_unh, bdims=2, has_ensemble=False, include_prior=False, include_corrector=False, include_residual_bootstrap=False, epistemic=qctx_unh)
        base = self._base_td_loss(q_all, targets, actions_unh)
        kw_cur = dict(pc_kw_cur, prior_scale=1.0, corrector_scale=1.0)
        comps_cur = q_module.component_means(q_inp_unh, bdims=2, has_ensemble=False, **kw_cur)
        # pc may live on a different batch than base/bt.
        if q_inp_unh_pc is not None:
            kw_cur_pc = dict(pc_kw_cur_pc, prior_scale=1.0, corrector_scale=1.0)
            comps_pc = q_module.component_means(q_inp_unh_pc, bdims=2, has_ensemble=False, **kw_cur_pc)
            pc = self._select_action_values(comps_pc['corrector'] + sg(comps_pc['prior']), actions_unh_pc) ** 2
        else:
            pc = self._select_action_values(comps_cur['corrector'] + sg(comps_cur['prior']), actions_unh) ** 2
        bt = jnp.zeros_like(base)
        if bt_target_shared is not None:
            rb_at_a = self._select_action_values(comps_cur['residual_bootstrap'], actions_unh)
            bt = (rb_at_a - bt_target_shared) ** 2
            rb_metrics[f'{pre}rb_cur/mean'] = rb_at_a.mean()
            rb_metrics[f'{pre}rb_cur/abs_mean'] = jnp.abs(rb_at_a).mean()
            rb_metrics[f'{pre}rb_cur/head_std'] = self.safe_std(rb_at_a, axis=0).mean()
            rb_metrics[f'{pre}rb_residual/signed_mean'] = (rb_at_a - bt_target_shared).mean()
            rb_metrics[f'{pre}rb_residual/abs_mean'] = jnp.abs(rb_at_a - bt_target_shared).mean()
            rb_metrics[f'{pre}rb_residual/rel_mag'] = jnp.abs(rb_at_a - bt_target_shared).mean() / (jnp.abs(bt_target_shared).mean() + 1e-8)
        return base, pc, bt

    def _head_loss_residual(self, q_module, q_target_module, q_inp_unh, actions_unh, qctx_unh, targets, td_target_mode, pc_kw_cur, pc_kw_next, pc_acts, pc_next_actor, next_states_unh, con, expected_next):
        """separate / joint per-head loss: PC tracks the standard TD residual.
        joint folds the base × pc cross term into the base loss so that
        base + pc reconstructs ((q_all - target) + pc_residual)**2. Returns (base, pc)."""
        q_all = q_module(q_inp_unh, bdims=2, has_ensemble=False, include_prior=False, include_corrector=False, include_residual_bootstrap=False, epistemic=qctx_unh)
        base = self._base_td_loss(q_all, targets, actions_unh)
        pc_cur = self._select_action_values(q_module.component_means(q_inp_unh, bdims=2, has_ensemble=False, **pc_kw_cur)['prior_corrector'], actions_unh)
        pc_next = expected_next(self._pc_next_components(q_target_module, next_states_unh, pc_acts, pc_next_actor, pc_kw_next)['prior_corrector'])
        pc_residual = pc_cur - self.discount * con[None] * sg(pc_next)
        pc = pc_residual ** 2
        if td_target_mode == 'joint':
            base = base + 2 * (self._select_action_values(q_all.mean(), actions_unh) - targets) * pc_residual
        return base, pc

    def _compute_actor_alpha_losses(self, actor_net, states, weight, retnorm, log_alpha, update=True, actor_heads=1, objective='exploit', logging_prefix='exploit', reg_actor=None, alpha_scale=1.0):
        metrics = {}
        alpha = alpha_scale * jnp.exp(sg(log_alpha.read()))[:, None, None]
        H = actor_heads

        # Sample actor at all heads (no head selection at training time).
        s = self._take_dyn_head(states)
        acts, z_out, actor = self._actor_step(actor_net, s, bdims=2)
        inputs_h = treemap(lambda x: jnp.broadcast_to(x[None], (H,) + x.shape), s)
        inputs = self._q_input(inputs_h, acts) if not self._is_discrete else self._q_input(inputs_h)
        bt = next(iter(s.values())).shape[:2]
        z_critic = self._critic_z_for_actor_eval(actor_net, z_out, batch_shape=(H,) + bt)
        parts = self.q.component_particles(inputs, z=z_critic, bdims=3, samples=self.q.epistemic_samples, stop_prior_grad=False)
        if self.use_intrinsic_reward:
            # intr_q is single-headed and broadcast across actor heads; collapse H here.
            parts['intr'] = self.intr_q.particles(inputs, bdims=3)[:, 0]
        q_mean = parts['raw'].mean(0).mean(0)                                  # [B, T(, A)] — for metric

        # Update retnorm to track the percentile span of base-network Q values so
        # rscale can be used to normalize mean_term in exp_score (β scale-free).
        # Skip the intr objective (doesn't use mean_term) to avoid polluting intr_retnorm.
        rscale = None
        if self.normalize_exp_mean and retnorm is not None and objective != 'intr':
            retnorm(sg(parts['raw'].mean(0)), update=update)
            _, rscale = retnorm.stats()

        # Score reduces the leading P axis; H axis is preserved (exp_actor / intr_actor use H=1).
        obj_h = self.exp_score(parts, exp_obj=objective, rscale=rscale)

        # Entropy / actor loss
        if self._is_discrete:
            probs = self._action_probs(actor)
            log_probs = jnp.log(probs + 1e-8)
            entropy = -(probs * log_probs).sum(-1)
            actor_loss = -(probs * sg(obj_h)).sum(-1)
        else:
            entropy = -self._actor_logpi(actor, acts, z_out)
            actor_loss = -obj_h

        # Regularization toward `reg_actor` (nearest-head KL) — the OFU exp_actor is allowed
        # to track the closest plausible conservative head; min-over-R has a reachable zero.
        # KL is computed once with gradient through (μ_p, σ_p) and shared between:
        #   (a) the actor loss, weighted by τ (sg'd at this site so τ doesn't move via actor)
        #   (b) the β / τ dual losses, where the KL is sg'd and gradient flows only through
        #       _beta_with_grad / _tau_with_grad.
        # τ and β are tuned simultaneously against the same `beta_target_kl`.
        learn_duals = getattr(self, 'learn_beta', False) and reg_actor is not None
        use_kl_reg = reg_actor is not None and (self.learn_tau or self.config.actor_kl_reg > 0)
        if use_kl_reg or learn_duals:
            reg_dist = self._frozen_call(reg_actor, lambda r: r(s, bdims=2, has_ensemble=False))
            if self._is_discrete:
                p_e = probs[None]                                    # [1, H, B, T, A_dim]
                q_e = sg(self._action_probs(reg_dist))[:, None]      # [R, 1, B, T, A_dim]
                kl_per_r = (p_e * (jnp.log(p_e + 1e-8) - jnp.log(q_e + 1e-8))).sum(-1)  # [R, H, B, T]
            else:
                kl_per_r = 0
                for k, v in actor.items():
                    if k not in reg_dist:
                        continue
                    lp, sp = _kl_loc_scale(v)
                    lq, sq = _kl_loc_scale(reg_dist[k])
                    m_p, s_p = lp[None], sp[None]
                    m_q, s_q = sg(lq)[:, None], sg(sq)[:, None]
                    # Sum over the action axis; accumulate components into the joint KL before the min.
                    kl_per_r = kl_per_r + (jnp.log(s_q + 1e-8) - jnp.log(s_p + 1e-8) + (s_p ** 2 + (m_p - m_q) ** 2) / (2 * s_q ** 2 + 1e-8) - 0.5).sum(-1)
            kl_term = kl_per_r.min(0)                                # nearest-head KL, [H, B, T]

        if use_kl_reg:
            tau_w = self._current_tau() if self.learn_tau else jnp.asarray(self.config.actor_kl_reg, f32)
            actor_loss += tau_w * kl_term
            metrics[f'{logging_prefix}/kl_reg'] = kl_term.mean()
            metrics[f'{logging_prefix}/kl_reg_mean_heads'] = kl_per_r.mean(0).mean()
            metrics[f'{logging_prefix}/tau'] = tau_w
        else:
            actor_loss += -alpha * entropy

        alpha_loss = jnp.exp(log_alpha.read())[:, None, None] * sg(entropy - self.target_entropy)
        weight_h = jnp.broadcast_to(weight[None], (H,) + weight.shape)
        losses = {'actor_sac': sg(weight_h) * actor_loss, 'alpha': sg(weight_h) * alpha_loss,}

        # BRO-style dual updates: β (Q-upper-bound magnitude) and τ (KL Lagrange).
        #   loss_β = (β − pess) · (KL/|A| − KL*)    →  β grows when KL is below target
        #   loss_τ = −τ · (KL/|A| − KL*)            →  τ grows when KL is above target
        # The two move in opposite directions on the same constraint and together close it.
        if learn_duals:
            empirical_kl = sg(kl_term.mean() / max(self._action_dim, 1.0))
            target_kl = float(self.config.beta_target_kl)
            beta_for_grad = self._beta_with_grad()
            beta_loss = (beta_for_grad - float(self.config.beta_pessimism)) * (empirical_kl - target_kl)
            losses['beta'] = sg(weight_h.mean()) * beta_loss
            metrics[f'{logging_prefix}/beta'] = sg(beta_for_grad)
            metrics[f'{logging_prefix}/beta_kl'] = kl_term.mean()
            metrics[f'{logging_prefix}/beta_kl_mean_heads'] = kl_per_r.mean(0).mean()
            metrics[f'{logging_prefix}/beta_target_kl'] = jnp.asarray(target_kl, f32)
            if self.learn_tau:
                tau_for_grad = self._tau_with_grad()
                tau_loss = -tau_for_grad * (empirical_kl - target_kl)
                losses['tau'] = sg(weight_h.mean()) * tau_loss
                metrics[f'{logging_prefix}/tau_dual'] = sg(tau_for_grad)

        # Diagnostic q_std mirrors the actor's bonus source: full Q ensemble.
        full = parts['raw'] + parts['prior'] + parts['corrector']
        q_std = self.safe_std(full.reshape((-1,) + full.shape[2:]))

        # Metrics
        metrics.update({
            f'{logging_prefix}/alpha_mean': alpha.mean(),
            f'{logging_prefix}/entropy': entropy.mean(),
            f'{logging_prefix}/target_entropy': jnp.asarray(self.target_entropy, f32),
            f'{logging_prefix}/q_actor_mean': q_mean.mean(),
            f'{logging_prefix}/q_std_mean': q_std.mean(),
            f'{logging_prefix}/q_std_max': q_std.max(),
            f'{logging_prefix}/q_std_std': q_std.std(),
            f'{logging_prefix}/ucb_bonus_ratio': (self._current_beta() * q_std.mean()) / (jnp.abs(q_mean.mean()) + 1e-8),
        })
        return losses, metrics

    def _intrinsic_losses(self, batch, prevacts, prevlat, embed, data, update, update_mask=1.0):
        losses = self.intrinsic_reward_loss(batch['replay_outs'], prevacts, prevlat, embed, data, learn_temp=False)
        metrics = {}
        states = batch['states']
        actions = batch['actions']
        intr_states, intr_actions = self._intrinsic_model_inputs(self._take_dyn_head(states), self._take_dyn_head(actions))
        intr_rewards, int_rew_metrics = self.int_rew_model(intr_states, intr_actions, update=update)
        intr_actor, intr_log_alpha = (self.exp_actor, self.exp_log_alpha) if self.use_exp_actor else (self.intr_actor, self.intr_log_alpha)
        intr_q_losses, intr_q_metrics = self._intr_q_loss(self.intr_q, self.intr_q_target, states, actions, intr_rewards, batch['next_states'], batch['con'], batch['weight'], log_alpha=intr_log_alpha, actor_net=intr_actor, is_last_next=batch['is_last_next'])
        losses.update(intr_q_losses)
        if not self.use_exp_actor:
            reg_actor = None if self.use_candidate_policy else self.actor
            actor_losses, actor_metrics = self._compute_actor_alpha_losses(actor_net=self.intr_actor, states=sg(states), weight=batch['weight'], update=update, retnorm=self.intr_retnorm, log_alpha=self.intr_log_alpha, actor_heads=1, objective='intr', logging_prefix='intr', reg_actor=reg_actor)
            actor_losses = {k: v * update_mask for k, v in actor_losses.items()}
            losses.update(actor_losses)
            metrics.update(actor_metrics)
        if self.use_temp:
            alpha, log_alpha = self.temp()
            metrics.update({'temp/temp': jnp.mean(alpha), 'temp/log_temp': jnp.mean(sg(log_alpha)),})
        if self.learn_temp:
            losses['temp'] = self.temp_loss(states, actor=intr_actor, actor_heads=1)
        
        metrics.update({f'int_rew_model/{k}': v for k, v in int_rew_metrics.items()})
        metrics.update(intr_q_metrics)
        return losses, metrics

    def _prepare_train_batch(self, data, replay_outs):
        cur_acts = jaxutils.onehot_dict({k: data[k] for k in self.act_space}, self.act_space)
        replay_outs = self._augment_ac_inputs(replay_outs, data)
        target_embed = replay_outs.get('target_embed')
        if target_embed is not None:
            replay_outs = {
                k: v for k, v in replay_outs.items()
                if k != 'target_embed'}
        next_states = treemap(lambda x: x[:, :, 1:], replay_outs)
        target_next_embed = (
            target_embed[:, :, 1:]
            if target_embed is not None else next_states.get('embed'))
        is_terminal = data['is_terminal'][:, 1:]
        out = dict(
            replay_outs=replay_outs,
            states=treemap(lambda x: x[:, :, :-1], replay_outs),
            next_states=next_states,
            actions=treemap(lambda x: x[None][:, :, :-1], cur_acts),
            rewards=data['reward'][:, 1:],
            con=sg(f32(~is_terminal)),
            weight=sg(f32(~data['is_last'][:, :-1])),
            is_last_next=sg(f32(data['is_last'][:, 1:])),
            # Replay-stored z (one per timestep) corresponds to the action at that step;
            # slice [:, :-1] to align with current-state actions.
            buffer_z={c: sg(data[f'z_explore_{c}'][:, :-1]) for c in self.q.z_aux_spaces() if f'z_explore_{c}' in data},
        )
        if target_next_embed is not None:
            out['target_next_embed'] = target_next_embed
        return out

    def loss(self, data, carry, update=True):
        self._sync_frozen_correctors()
        mc = MetricsCollector()
        prevlat, _ = carry
        # Force beta state-entry creation during init so pure-report reads later don't crash.
        # Under use_candidate_policy the actor-alpha block (where this used to be touched) is skipped.
        _ = self._current_beta()
        # Dual-batch routing: `data['ac']` drives wm + all losses except (res+cor)^2;
        # `data['res']` carries the corrector regression batch. When dual-batch is off
        # (or in report/gradnorms paths), the two refer to the same batch.
        data_ac, data_res = data['ac'], data['res']
        augment = self._aug_pad > 0 and update
        wm_losses, dists, embed, replay_outs, prevacts, newlat, newact, wm_metrics = self.world_model_loss(
            data_ac, carry, compute_losses=self.train_world_model, augment=augment)
        self._newlat = newlat

        batch = self._prepare_train_batch(data_ac, replay_outs)
        states = batch['states']
        actions = batch['actions']

        # Encode the res batch through the world model so the corrector can read its
        # latent features. The wm losses for this batch are always discarded, so
        # skip the decoder / reward / cont / dyn-loss path entirely.
        if data_res is data_ac:
            batch_res = batch
        else:
            _, _, _, replay_outs_res, _, _, _, _ = self.world_model_loss(
                data_res, carry, compute_losses=False, augment=augment)
            batch_res = self._prepare_train_batch(data_res, replay_outs_res)

        # td_target_actor selects which policy bootstraps the base TD target on self.q.
        # In policy_eval the checkpoint is loaded into exp_actor while self.actor stays
        # at fresh random init — setting td_target_actor='exp_actor' makes raw learn
        # V^exp_actor instead of bootstrapping under a random policy. exp_actor is
        # built with heads=1, so the actor_heads passed downstream must match.
        use_exp_for_td = (self.config.td_target_actor == 'exp_actor' and self.use_exp_actor)
        actor_arg = None if self.use_candidate_policy else (self.exp_actor if use_exp_for_td else self.actor)
        log_alpha_arg = None if self.use_candidate_policy else (self.exp_log_alpha if use_exp_for_td else self.log_alpha)
        actor_heads_arg = 1 if use_exp_for_td else self.actor_heads
        pc_states_arg = batch_res['states'] if data_res is not data_ac else None
        pc_actions_arg = batch_res['actions'] if data_res is not data_ac else None
        pc_weight_arg = batch_res['weight'] if data_res is not data_ac else None
        def critic_loss(views):
            plain_drq = not (
                self.value_prior_scale
                or self.q.residual_bootstrap is not None
                or any(head.prior_corrector is not None
                       for head in self.q.heads))
            target_results = None
            if self.ac_inputs == 'drq' and plain_drq:
                target, target_metrics, target_aux = self._compute_drq_targets(
                    views, log_alpha_arg, actor_arg, actor_heads_arg,
                    self.value_heads)
                target_results = [
                    (target, target_metrics, aux) for aux in target_aux]

            outputs = []
            for index, view in enumerate(views):
                outputs.append(self._compute_q_losses(
                    self.q, self.q_target, view['states'], view['actions'],
                    view['rewards'], view['next_states'], view['con'],
                    view['weight'], log_alpha_arg, actor_arg,
                    actor_heads=actor_heads_arg,
                    value_heads=self.value_heads, loss_name='q',
                    metric_prefix='sac',
                    is_last_next=view['is_last_next'],
                    buffer_z=view.get('buffer_z'), pc_states=pc_states_arg,
                    pc_actions=pc_actions_arg, pc_weight=pc_weight_arg,
                    target_result=(
                        target_results[index]
                        if target_results is not None else None)))
            if len(outputs) == 1:
                return outputs[0]
            losses, metrics = zip(*outputs)
            reduce_losses = (
                (lambda *xs: jnp.stack(xs).sum(0))
                if target_results is not None
                else (lambda *xs: jnp.stack(xs).mean(0)))
            losses = treemap(reduce_losses, *losses)
            metrics = treemap(lambda *xs: jnp.stack(xs).mean(0), *metrics)
            return losses, metrics

        critic_losses, critic_metrics = self._apply_drq_views(
            critic_loss, data_ac, carry, batch, augment)
        wm_losses.update(critic_losses)

        cur = self.cur_step.read()
        actor_update_mask = f32((cur >= self.config.actor_update_warmup) & (cur % self.config.actor_update_delay == 0))
        # Frequency masks for the dual-batch split. mask_ac gates every loss that
        # consumes the ac batch (wm, base TD, residual bootstrap, actor, alpha,
        # intrinsic); mask_res gates the corrector (res+cor)^2 regression.
        mask_ac = f32(cur % self.config.ac_update_every == 0)
        mask_res = f32(cur % self.config.corrector_update_every == 0)

        if self.use_intrinsic_reward:
            intrinsic_losses, intrinsic_metrics = self._intrinsic_losses(batch, prevacts, prevlat, embed, data_ac, update, update_mask=actor_update_mask)
            wm_losses.update(intrinsic_losses)
            mc.update(intrinsic_metrics)

        if not self.use_candidate_policy:
            actor_losses, actor_metrics = self._compute_actor_alpha_losses(actor_net=self.actor, states=sg(states), weight=batch['weight'], update=update, retnorm=self.retnorm, log_alpha=self.log_alpha, actor_heads=self.actor_heads, objective='exploit', logging_prefix='exploit')
            actor_losses = {k: v * actor_update_mask for k, v in actor_losses.items()}
            wm_losses.update(actor_losses)
            mc.update(actor_metrics)

        if self.use_exp_actor:
            exp_actor_losses, exp_actor_metrics = self._compute_actor_alpha_losses(actor_net=self.exp_actor, states=sg(states), weight=batch['weight'], update=update, retnorm=self.exp_retnorm, log_alpha=self.exp_log_alpha, actor_heads=1, objective=self.exp_obj, logging_prefix='explore', reg_actor=self.actor, alpha_scale=self.config.exp_alpha_scale)
            exp_actor_losses = {k: v * actor_update_mask for k, v in exp_actor_losses.items()}
            wm_losses.update({f'exp_{k}': v for k, v in exp_actor_losses.items()})
            mc.update({f'exp/{k}': v for k, v in exp_actor_metrics.items()})

        mc.update(wm_metrics)
        mc.update(critic_metrics)
        mc.add('train/actor_update_mask', actor_update_mask)
        mc.add('train/mask_ac', mask_ac)
        mc.add('train/mask_res', mask_res)
        mc.add_loss_stats(wm_losses)
        if dists is not None:
            mc.add_distribution_stats(dists, data_ac)
        mc.add('activation/embed', jnp.abs(embed).mean())

        # Apply the dual-batch frequency masks: every loss key gets mask_ac except
        # `*_pc` keys (the corrector regression), which get mask_res.
        masked = {k: (mask_res if k.endswith('_pc') else mask_ac) * v for k, v in wm_losses.items()}
        losses = {k: v * self.scales[k] for k, v in masked.items()}
        loss = jnp.stack([v.mean() for v in losses.values()]).sum()
        replay_outs = treemap(lambda x: x[0], batch['replay_outs'])
        out = {'replay_outs': replay_outs, 'prevacts': prevacts, 'embed': embed}
        out.update({f'{k}_loss': v for k, v in losses.items()})

        # Priority signals for prioritized replay. Each per-sampler signal is a
        # [B, T] tensor padded with one zero at the trailing step (the q losses
        # live on [B, T-1] because they consume next_state).
        def _pad_priority(per_bt):
            B = per_bt.shape[0]
            return jnp.concatenate([per_bt, jnp.zeros((B, 1), per_bt.dtype)], axis=1)
        def _aggregate(keys):
            parts = [wm_losses[k] for k in keys if k in wm_losses]
            if not parts:
                return None
            return _pad_priority(sum(p.astype(f32) for p in parts))
        # AC sampler signal: select which Q-loss term feeds the priority.
        #   td     → base TD residual only        (q*  keys, no _bt / _pc suffix)
        #   rb_td  → residual-bootstrap TD only   (q*_bt keys)
        #   critic → all critic losses summed     (q* + q*_bt)
        sig = self.config.replay.priosignal_ac
        if sig != 'none':
            def _q_keys(suffix):
                return [k for k in wm_losses
                        if k.startswith('q') and not k.startswith('intr_q')
                        and not k.endswith('_pc')
                        and ((suffix and k.endswith(suffix)) or (not suffix and not k.endswith('_bt')))]
            if sig == 'td':
                ac_keys = _q_keys('')
            elif sig == 'rb_td':
                ac_keys = _q_keys('_bt')
            elif sig == 'critic':
                ac_keys = _q_keys('') + _q_keys('_bt')
            else:
                raise ValueError(f'Unknown priosignal_ac: {sig}')
            prio_ac = _aggregate(ac_keys)
            if prio_ac is not None:
                out['priority_ac'] = sg(prio_ac)
        # Res sampler signal: corrector regression magnitude.
        res_keys = [k for k in wm_losses if k.endswith('_pc')]
        if self.config.replay.priosignal_res == 'pc' and res_keys:
            prio_res = _aggregate(res_keys)
            if prio_res is not None:
                out['priority_res'] = sg(prio_res)

        self.cur_step.write(cur + 1)
        new_carry = (newlat, newact)
        return loss, (out, new_carry, mc.result())

    # ----- Policy / training updater / extra report metrics ---------------

    def init_policy(self, batch_size):
        lat, prevact = init_carry(self.dyn, self.act_space, batch_size)
        z = {}
        if not self.use_candidate_policy:
            disp = _disp(self.actor)
            if disp.ensemble == 'input':
                z['ensemble'] = jax.nn.one_hot(jnp.zeros(batch_size, jnp.int32), self.actor.num_heads, dtype=f32)
            if disp.epistemic == 'input':
                z['epistemic'] = jnp.zeros((batch_size, self.q.epistemic_dim), f32)
        return lat, prevact, z

    @property
    def policy_keys(self):
        return '/(enc|dyn|actor_sac|exp_actor_sac|q|q2|q_prior|intr_q|temp|beta|log_beta|log_tau|visual_prior_enc|visual_prior_proj)/'

    def policy(self, obs, carry, mode='train'):
        self.config.jax.jit and embodied.print('Tracing policy function', color='yellow')
        prevlat, prevact, z_input = carry
        obs = self.preprocess(obs)
        embed = self.enc(obs, bdims=1)
        prevact = jaxutils.onehot_dict(prevact, self.act_space)
        lat_h, out_h = self._dyn_observe(prevlat, prevact, embed, obs['is_first'], bdims=1)
        ac_inp = self._augment_ac_inputs(out_h, obs)
        cmps = self.q.z_aux_spaces()
        z_used = self.q.sample_z(obs['is_first'].shape, components=cmps.keys()) if cmps else {}

        actor = z_out = None
        if self.use_candidate_policy:
            act = self._select_candidate_action(ac_inp, mode=mode, z=(z_used or None))
        else:
            actor = self.exp_actor if (self.use_exp_actor and mode == 'explore') else self.actor
            z_input = self._refresh_z_input(actor, z_input, obs['is_first'])
            act, z_out, _ = self._actor_step(actor, self._take_dyn_head(ac_inp), z_input=z_input, bdims=1, select_head=True)

        out = treemap(lambda x: x[0], out_h)
        outs = {}
        if self.config.replay_context:
            outs.update({k: out[k] for k in self.replay_context_keys})
            if 'stoch' in outs:
                outs['stoch'] = jnp.argmax(outs['stoch'], -1).astype(jnp.int32)
        if self.visual_prior:
            outs['visual_embed'] = obs['visual_embed']

        if cmps:
            if self.use_candidate_policy:
                for c in cmps:
                    outs[f'z_explore_{c}'] = z_used[c]
            else:
                disp = _disp(actor)
                for c in cmps:
                    src = (z_out if disp[c] == 'output' else z_input if disp[c] == 'input' else None)
                    outs[f'z_explore_{c}'] = (src or {}).get(c, z_used[c])

        outs['finite'] = {'/'.join(x.key for x in k): (jnp.isfinite(v).all(range(1, v.ndim)),v.min(range(1, v.ndim)),v.max(range(1, v.ndim)),)
            for k, v in jax.tree_util.tree_leaves_with_path(dict(obs=obs, prevlat=prevlat, prevact=prevact, embed=embed, act=act, out=out, lat=lat_h))}
        assert all(k in outs for k in self.aux_spaces if k not in ('stepid', 'finite', 'is_online')), (list(outs.keys()), self.aux_spaces)

        act = {k: jnp.nanargmax(act[k], -1).astype(jnp.int32) if s.discrete else act[k] for k, s in self.act_space.items()}
        return act, outs, (lat_h, act, z_input)

    def q_values(self, obs, actions):
        """Q(s, a) at single-step (obs, action) pairs, used for off-loop evaluation.

        Treats every example as a fresh episode (is_first=True): runs the encoder
        and a single RSSM observe step from a zero carry, then evaluates the Q
        ensemble at (state, action). Useful for plotting V^pi predictions on a
        fixed anchor grid without going through the training driver.

        Args:
          obs:     dict[str, [B, ...]]. Standard obs dict; 'is_first' is forced True.
          actions: dict[str, [B, action_dim]] for each non-reset key in act_space.
                   For discrete tasks, pass integer action indices of shape [B].

        Returns:
          dict[str, [B, P]] with keys {'q', 'raw', 'prior', 'corrector'} and
          'residual_bootstrap' when the RB net is built. P = num_q_heads *
          (epistemic_samples if epistemic else 1). 'q' is the full mixed value
          (raw + prior + corrector [+ rb]); the others are the per-component
          contributions in matching units. Also returns [B] arrays
          'base_value', 'optimism_bonus', and 'optimistic_value', using the
          configured exploration objective and its normalization.
        """
        self.config.jax.jit and embodied.print('Tracing q_values function', color='yellow')
        obs = dict(obs)
        B = next(iter(obs.values())).shape[0]
        obs['is_first'] = jnp.ones((B,), bool)
        obs['is_last'] = jnp.zeros((B,), bool)
        obs['is_terminal'] = jnp.zeros((B,), bool)
        obs.setdefault('reward', jnp.zeros((B,), f32))
        obs = self.preprocess(obs)

        embed = self.enc(obs, bdims=1)
        prevlat, prevact, _ = self.init_policy(B)
        prevact = jaxutils.onehot_dict(prevact, self.act_space)
        _, out_h = self._dyn_observe(prevlat, prevact, embed, obs['is_first'], bdims=1)
        states_unh = self._take_dyn_head(out_h)
        # When ac_inputs='obs' the Q head reads raw proprio keys; merge them in
        # before _q_input. No-op otherwise. headed=False since states_unh has
        # already been un-headed by _take_dyn_head.
        states_unh = self._augment_ac_inputs(states_unh, obs, headed=False)

        if self._is_discrete:
            inputs = self._q_input(states_unh)
        else:
            inputs = self._q_input(states_unh, actions)

        samples = self.q.epistemic_samples if self.q.epistemic_dim > 0 else None
        p_q = self.q.particles(inputs, bdims=1, samples=samples, frozen=True)
        comps = self.q.component_particles(inputs, bdims=1, samples=samples, frozen=True)
        out = {'q': p_q, **comps}
        if self.use_intrinsic_reward:
            out['intr'] = self.intr_q.particles(inputs, bdims=1, frozen=True)

        if self._is_discrete:
            # [P, B, A] -> gather along A using the (scalar) action per batch.
            act_key = next(iter(self.act_space.keys()))
            a = actions[act_key]
            if a.ndim == 2 and a.shape[-1] > 1:        # one-hot
                a = jnp.argmax(a, axis=-1)
            a = a.astype(jnp.int32)
            out = {k: (v[:, jnp.arange(B), a] if v.ndim >= 3 else v) for k, v in out.items()}

        base, bonus = self.exp_score_parts(out, rscale=self._exp_norm_rscale())
        return {
            **{k: jnp.swapaxes(v, 0, 1) for k, v in out.items()},  # [B, P]
            'base_value': base,
            'optimism_bonus': bonus,
            'optimistic_value': base + bonus,
        }

    def _sync_frozen_correctors(self):
        if not getattr(self.config, 'freeze_corrector', False) or nj.creating():
            return
        # Lazy initialization creates source and target separately. Align them
        # before the first real loss, and copy exactly after each target EMA.
        for source, target in zip(self.q.heads, self.q_target.heads):
            src, dst = source.prior_corrector, target.prior_corrector
            if src is not None:
                dst.put({dst.path + k[len(src.path):]: sg(v)
                         for k, v in src.find().items()})

    def train_updater(self):
        self.q_updater()
        self._sync_frozen_correctors()
        if self.target_enc_updater is not None:
            self.target_enc_updater()
        if self.use_intrinsic_reward:
            self.intr_q_updater()
            if self.use_temp:
                if self.learn_temp:
                    self.policy_updater()
                else:
                    self.temp.update()

@jaxagent.Wrapper
class Agent(ObserverAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
