import jax
import jax.numpy as jnp
import numpy as np

import embodied
from dreamerv3 import jaxagent
from dreamerv3 import jaxutils
from dreamerv3 import ninjax as nj
from optimistic_curiosity.agent import BaseAgent
from multimex.nets import DataNormalizer

from .metrics import MetricsCollector

f32 = jnp.float32
treemap = jax.tree_util.tree_map
sg = lambda x: treemap(jax.lax.stop_gradient, x)
cast = jaxutils.cast_to_compute
sample = lambda dist: {k: v.sample(seed=nj.seed()) for k, v in dist.items()}


class DreamerAgent(BaseAgent):
    """DreamerV3 model-based reinforcement learning agent."""

    extra_report_metrics = BaseAgent._report_dreamer

    def __init__(self, obs_space, act_space, config):
        self.obs_space = {k: v for k, v in obs_space.items() if not k.startswith('log_')}
        self.act_space = {k: v for k, v in act_space.items() if k != 'reset'}
        self.config = config
        self._dyn_heads = config.ens_heads
        self._value_heads = config.ens_heads
        self._actor_heads = config.actor_ens_heads
        self.policy_mode = config.policy_mode
        self.exp_obj = config.exp_obj
        self.beta = nj.Variable(lambda: jnp.asarray(config.beta, f32), name='beta')
        assert config.get('visual_prior', 'none') == 'none', 'visual_prior is only supported by ObserverAgent'
        self.value_prior_scale = getattr(config, 'value_prior_scale', getattr(config, 'prior_scale', 0.0))
        self.reward_prior_scale = getattr(config, 'reward_prior_scale', self.value_prior_scale)
        self.num_cand_samples = config.num_samples
        self.sampling_temp = config.sampling_temp
        self._is_discrete = all(s.discrete for s in self.act_space.values())
        self.enumerate_actions = getattr(config, 'enumerate_actions', False) and self._is_discrete
        if self._is_discrete:
            self._num_actions = self._discrete_classes(list(self.act_space.values())[0])
        self.cand_grid_points = int(getattr(config, 'cand_grid_points', 0) or 0)
        self._cand_action_grid = None
        if self.cand_grid_points > 0 and not self._is_discrete:
            self._cand_action_grid = self._make_continuous_action_grid(
                self.cand_grid_points)
        self.use_exp_actor = self.policy_mode == "exp_actor"
        self.use_intrinsic_reward = self.exp_obj in ('sombrl', 'ocb')
        self.use_exp_critic = getattr(config, 'use_exp_critic', True)
        self.intrinsic_mode = config.intrinsic_mode
        enc_space, dec_space = self.build_spaces(obs_space, config)

        assert self.intrinsic_mode == "model"
        assert self.policy_mode in ('actor', 'exp_actor', 'exp_sampling', 'thompson_sampling'), self.policy_mode
        assert self.exp_obj in ('ucb', 'sombrl', 'ocb', 'none'), self.exp_obj
        assert self.actor_heads == 1 or self.actor_heads == self._value_heads
        assert self.config.imag_start == 'all', "No support for other imag_start values."

        # Modules: World model, Actor & Critic
        wm = self.build_world_model(config, enc_space, dec_space)
        self.enc, self.dec, self.dyn, self.rew, self.rew_prior, self.con = wm['enc'], wm['dec'], wm['dyn'], wm['rew'], wm['rew_prior'], wm['con']
        ac = self.build_actor(config, self.act_space, heads=self._actor_heads)
        self.actor, self.retnorm, self.valnorm, self.advnorm = ac['actor'], ac['retnorm'], ac['valnorm'], ac['advnorm']
        cr = self.build_critic(config)
        self.critic, self.slowcritic, self.updater, self.v_prior = cr['critic'], cr['slowcritic'], cr['updater'], cr['v_prior']
        self.modules = [self.enc, self.dyn, self.dec, self.rew, self.con, self.actor, self.critic]

        # Optimizer
        self.opt, self.scales = self.build_optimizer(config, self.modules, self.dec)

        # Intrinsic Reward Module and Critic
        if self.use_intrinsic_reward:
            intrinsic = self.build_intrinsic_components(config)
            self.int_rew_model, self.intr_critic, self.intr_slowcritic, self.intr_updater, self.intr_retnorm = intrinsic['int_rew_model'], intrinsic['intr_critic'], intrinsic['intr_slowcritic'], intrinsic['intr_updater'], intrinsic['intr_retnorm']
            self.intr_valnorm, self.intr_advnorm, self.slowactor, self.policy_updater, self.temp = intrinsic['intr_valnorm'], intrinsic['intr_advnorm'], intrinsic['slowactor'], intrinsic['policy_updater'], intrinsic['temp']
            if self.intrinsic_mode == 'model':
                self.modules.extend([self.int_rew_model, self.intr_critic])
                self.scales['int_rew_model'] = config.loss_scales.get('int_rew_model', 1.0)
            self.scales['intr_critic'] = config.loss_scales.get('intr_critic', 1.0)
            self.learn_temp = config.loss_scales.get('temp', 0.0) != 0
            if self.learn_temp:
                self.modules.append(self.temp)
                self.scales['temp'] = config.loss_scales['temp']

        # Exploration Actor (Optionally with critic)
        if self.use_exp_actor:
            exp_ac = self.build_actor(config, self.act_space, heads=1, name="exp_actor")
            self.exp_actor, self.exp_retnorm, self.exp_valnorm, self.exp_advnorm = exp_ac['actor'], exp_ac['retnorm'], exp_ac['valnorm'], exp_ac['advnorm']
            self.modules.extend([self.exp_actor])
            self.scales['exp_actor'] = config.loss_scales.get('exp_actor', config.loss_scales.get('actor', 1.0))
            if self.use_exp_critic:
                exp_cr = self.build_critic(config, name='exp_critic')
                self.exp_critic, self.exp_slowcritic, self.exp_updater = exp_cr['critic'], exp_cr['slowcritic'], exp_cr['updater']
                self.scales['exp_critic'] = config.loss_scales.get('exp_critic', config.loss_scales.get('critic', 1.0))
                self.modules.extend([self.exp_critic])
            if self.use_intrinsic_reward and self.learn_temp:
                slow_exp_ac = self.build_actor(config, self.act_space, heads=1, name='slow_exp_actor')
                self.slowactor = slow_exp_ac['actor']
                self.policy_updater = jaxutils.SlowUpdater(self.exp_actor, self.slowactor,config.slow_actor_fraction, config.slow_actor_update,name='exp_policy_updater')

        if getattr(config, 'freeze_all', False):
            self.scales = {k: 0.0 for k in self.scales}

        # Logging
        skip = {'is_first', 'is_last', 'is_terminal', 'reward', 'cont', 'stepid'}
        self._state_obs_keys = sorted([k for k, v in obs_space.items() if (k not in skip and not k.startswith('log_') and len(v.shape) == 1 and np.issubdtype(v.dtype, np.floating))]) #  # 1-D vectors only
        self._log_metrics_table = config.log_metrics_table

    def _assert_headed_batch(self, tree, heads, batch, name):
        for key, value in tree.items():
            assert value.ndim >= 2, f'{name}.{key} expected at least 2 dims, got {value.shape}'
            assert value.shape[:2] == (heads, batch), (f'{name}.{key} expected head/batch {(heads, batch)}, got {value.shape[:2]}')

    def _compute_q_values(self, out_h, actor, K, actor_heads=1):
        """Estimate candidate action values from imagined next-state critics."""
        if self.enumerate_actions:
            H = out_h[next(iter(out_h))].shape[0]
            B = out_h[next(iter(out_h))].shape[1]
            K = self._num_actions
            key = next(iter(self.act_space.keys()))
            one_hots = jnp.eye(K)
            acts = {key: jnp.broadcast_to(one_hots[:, None, None, :], (K, H, B, K))}    
        elif self._cand_action_grid is not None:
            H = out_h[next(iter(out_h))].shape[0]
            B = out_h[next(iter(out_h))].shape[1]
            acts = self._grid_action_candidates((H, B))
            K = acts[next(iter(acts))].shape[0]
        else:
            H = out_h[next(iter(out_h))].shape[0]
            B = out_h[next(iter(out_h))].shape[1]
            bdims = 3 if actor_heads < self.dyn_heads else 2
            acts = self._sample_action_candidates(
                out_h, K, actor_net=actor, actor_heads=actor_heads,
                bdims=bdims, temperature=self.sampling_temp)
            if actor_heads < self.dyn_heads:
                acts = treemap(lambda x: x[:, 0], acts)   # [K, H, B, ...]

        states = treemap(lambda x: jnp.broadcast_to(x[None], (K, *x.shape)), out_h)  # [K, H, B, ...]
        flat_states = treemap(lambda x: x.reshape(K * H * B, *x.shape[3:]), states)
        flat_acts = treemap(lambda x: x.reshape(K * H * B, *x.shape[3:]), acts)
        _, next_states = self.dyn.imagine(flat_states, flat_acts, bdims=1)
        inp_v = self._expand_to_heads(next_states, self.value_heads)  # [V, ...]
        critic = self.exp_critic if self.use_exp_critic else self.critic
        vals_extr = critic(inp_v, bdims=2, has_ensemble=True).mean()  # [V, ...]
        if self.value_prior_scale > 0:
            vals_extr = vals_extr + self.value_prior_scale * sg(self.v_prior(inp_v, bdims=2, has_ensemble=True).mean())
        vals_extr = vals_extr.reshape(self.value_heads, K, H, B)
        vals_intr = None
        if self.use_intrinsic_reward:
            vals_intr = self.intr_critic(inp_v, bdims=2, has_ensemble=True).mean()  # [V, ...]
            vals_intr = vals_intr.reshape(self.value_heads, K, H, B)
        return acts, vals_extr, vals_intr

    def _select_candidate_action(self, out_h, mode='eval'):
        acts, vals_extr, vals_intr = self._compute_q_values(out_h, actor=self.actor, actor_heads=self.actor_heads, K=self.num_cand_samples)
        H = acts[next(iter(acts))].shape[1]
        B = acts[next(iter(acts))].shape[2]
        _, extr_scale = self.retnorm.stats()
        scaled_extr = vals_extr / extr_scale
        intr_mean = None
        if vals_intr is not None:
            _, intr_scale = self.intr_retnorm.stats()
            intr_mean = (vals_intr / intr_scale).mean(0)
        score = self._compute_exp_objective(particles=scaled_extr, intr_mean=intr_mean)
        score = score.reshape(-1, B)
        best_flat = self._select_candidate_index(score, score.shape[0], axis=0, mode=mode)
        best_k, best_h = jnp.divmod(best_flat, H)
        batch_idx = jnp.arange(B)
        return treemap(lambda x: x[best_k, best_h, batch_idx], acts)

    @property
    def policy_keys(self):
        return '/(enc|dyn|actor|exp_actor|critic|critic_prior|intr_critic|temp|beta|exp_critic|retnorm|intr_retnorm)/'

    def policy(self, obs, carry, mode='train'):
        self.config.jax.jit and embodied.print('Tracing policy function', color='yellow')
        prevlat, prevact, head_idx = carry
        obs = self.preprocess(obs)
        embed = self.enc(obs, bdims=1)
        prevact = jaxutils.onehot_dict(prevact, self.act_space)
        lat_h, out_h = self._dyn_observe(prevlat, prevact, embed, obs['is_first'], bdims=1)

        if self.policy_mode == "exp_sampling" and mode == 'explore':
            act = self._select_candidate_action(out_h)
        elif self.policy_mode == "exp_actor" and mode == 'explore':
            head_idx = jnp.where(obs['is_first'], jax.random.randint(nj.seed(), head_idx.shape, 0, self.dyn_heads), head_idx)
            actor_inp = self._align_ens_heads(out_h, 1)  # [1, E, B, ...]
            actor_dist = self.exp_actor(actor_inp, bdims=3 if 1 < self.dyn_heads else 2, has_ensemble=True)
            act = cast(sample(actor_dist))  # [1, E, B, ...]
            act = treemap(lambda x: x[0], act) if 1 < self.dyn_heads else act  # → [E, B, ...] or [A, B, ...] # [E, B, ...]
            act = treemap(lambda x: x[head_idx, jnp.arange(x.shape[1])], act)  # [B, ...]
        else:
            head_idx = jnp.where(obs['is_first'], jax.random.randint(nj.seed(), head_idx.shape, 0, self.dyn_heads), head_idx)
            A = self.actor_heads
            actor_inp = self._align_ens_heads(out_h, A)  # [A, B, ...] if A>=E, else [1, E, B, ...]
            actor_dist = self.actor(actor_inp, bdims=3 if A < self.dyn_heads else 2, has_ensemble=True)
            act = cast(sample(actor_dist))
            act = treemap(lambda x: x[0], act) if A < self.dyn_heads else act  # → [E, B, ...] or [A, B, ...]
            act = treemap(lambda x: x[head_idx, jnp.arange(x.shape[1])], act)  # [B, ...]

        out = treemap(lambda x: x[0], out_h)
        outs = {}
        if self.config.replay_context:
            outs.update({k: out[k] for k in self.aux_spaces if k != 'stepid'})
            outs['stoch'] = jnp.argmax(outs['stoch'], -1).astype(jnp.int32)
        outs['finite'] = {'/'.join(x.key for x in k): (jnp.isfinite(v).all(range(1, v.ndim)), v.min(range(1, v.ndim)), v.max(range(1, v.ndim)),)
            for k, v in jax.tree_util.tree_leaves_with_path(dict( obs=obs, prevlat=prevlat, prevact=prevact, embed=embed,act=act, out=out, lat=lat_h,))}
        assert all(k in outs for k in self.aux_spaces if k not in ('stepid', 'finite', 'is_online')), (list(outs.keys()), self.aux_spaces)
        batch = obs['is_first'].shape[0]
        strip_singleton_head = (lambda x: x[0] if hasattr(x, 'ndim') and x.ndim >= 2 and x.shape[0] == 1 and x.shape[1] == batch else x)
        act = {k: strip_singleton_head(v) for k, v in act.items()}
        outs = {k: strip_singleton_head(v) for k, v in outs.items()}
        act = {k: jnp.nanargmax(act[k], -1).astype(jnp.int32) if s.discrete else act[k]for k, s in self.act_space.items()}
        return act, outs, (lat_h, act, head_idx)

    def train_updater(self):
        if getattr(self.config, 'freeze_all', False):
            return
        self.updater()
        if self.use_exp_critic:
            self.exp_updater()
        if self.use_intrinsic_reward and self.intrinsic_mode == 'model':
            self.intr_updater()
            if self.learn_temp:
                self.policy_updater()
            else:
                self.temp.update()

    def imagination_rollout(self, replay_outs, data, actor, actor_heads):
        """Roll out a policy in imagination from replay states."""
        # replay_outs: [E, B, T, ...] where E=dyn_heads (already headed from world_model_loss)
        rew = data['reward']
        con = 1 - f32(data['is_terminal'])

        B, T = data['is_first'].shape
        startlat = self.dyn.outs_to_carry(treemap(lambda x: x.reshape((self.dyn_heads, B * T, 1, *x.shape[3:])), replay_outs))
        startout = treemap(lambda x: x.reshape((self.dyn_heads, B * T, *x.shape[3:])), replay_outs)
        startout['prior_feat'] = jnp.zeros(startout['deter'].shape[:-1] + (self.dyn.hidden,),dtype=startout['deter'].dtype)
        startrew = self._broadcast_heads(rew.reshape((B * T,)), self.dyn_heads)
        startcon = self._broadcast_heads(con.reshape((B * T,)), self.dyn_heads)

        if self.config.imag_repeat > 1:
            N = self.config.imag_repeat
            startlat = treemap(lambda x: x.repeat(N, 1), startlat)
            startout = treemap(lambda x: x.repeat(N, 1), startout)
            startrew, startcon = startrew.repeat(N, 1), startcon.repeat(N, 1)
        start_batch = B * T * self.config.imag_repeat

        def sample_action(inp):
            actor_inp = self._align_ens_heads(inp, actor_heads)
            acts = cast(sample(actor(actor_inp, bdims=3 if actor_heads < self.dyn_heads else 2, has_ensemble=True)))
            acts = treemap(lambda x: x[0], acts) if actor_heads < self.dyn_heads else acts
            return acts

        def imgstep(carry, _):
            lat, act = carry
            lat, out = self.dyn.imagine(lat, act, bdims=1)
            out['stoch'] = sg(out['stoch'])
            act = sample_action(out)
            return (lat, act), (out, act)

        startact = sample_action(startout)
        self._assert_headed_batch(startlat, self.dyn_heads, start_batch, 'startlat')
        self._assert_headed_batch(startact, self.dyn_heads, start_batch, 'startact')
        _, (outs, acts) = jaxutils.scan(imgstep, sg((startlat, startact)), jnp.arange(self.config.imag_length), self.config.imag_unroll)
        outs, acts = treemap(lambda x: jnp.moveaxis(x, 0, 2), (outs, acts))
        outs, acts = treemap(lambda first, seq: jnp.concatenate([first[:, :, None], seq], 2),  (startout, startact), (outs, acts))

        # outs: [E, B, T, ...] where E=dyn_heads
        pred_rew = self._rew_dist(outs, bdims=3).mean()  # [E, B, T]
        pred_con = self.con(outs, bdims=3, has_ensemble=True).mean()  # [E, B, T]
        rew = jnp.concatenate([startrew[..., None], pred_rew[..., 1:]], -1)
        con = jnp.concatenate([startcon[..., None], pred_con[..., 1:]], -1)
        discount = 1 if self.config.contdisc else 1 - 1 / self.config.horizon
        weight = jnp.cumprod(discount * con, -1) / discount
        return outs, sg(acts), rew, con, weight
    
    def _compute_returns(self, inp, rew, con, critic, slowcritic, valnorm, retnorm, advnorm, discount, update, v_prior=None, prior_scale=0.0):
        """Compute critic values, lambda-returns, and normalized advantages. inp: [V, B, T, ...], rew: [V, B, T], con: [V, B, T] where V=value_heads."""
        V = self.value_heads
        assert inp[next(iter(inp))].shape[0] == V, f"inp head dim {inp[next(iter(inp))].shape[0]} != value_heads {V}"
        voffset, vscale = valnorm.stats()
        val = critic(inp, bdims=3, has_ensemble=True).mean() * vscale + voffset      # [V, B, T]
        slowval = slowcritic(inp, bdims=3, has_ensemble=True).mean() * vscale + voffset  # [V, B, T]
        if v_prior is not None and prior_scale > 0:
            prior_val = sg(v_prior(inp, bdims=3, has_ensemble=True).mean())  # [V, B, T]
            val = val + prior_scale * prior_val
            slowval += prior_scale * prior_val
        tarval = slowval if self.config.slowtar else val
        a = self.config.return_lambda + (con[..., 1:] * 0)
        b = con[..., 1:] * discount
        data, metrics = self.compute_ret_adv(rew, con, tarval, retnorm=retnorm, advnorm=advnorm, a=a, b=b, update=update)
        data['val'] = val
        if v_prior is not None and prior_scale > 0:
            data['prior_val'] = prior_val[..., :-1]
        return data, metrics

    def _intrinsic_critic_loss(self, inp, acts, outs, con_v, weight_v, discount, update): # TODO: Combine with _critic_loss - move rest outside
        """Train intrinsic critic on given rollouts. Returns (losses, metrics, intr_data) inp/outs/acts: [E, B, T, ...], con_v/weight_v: [V, B, T]."""
        V = self.value_heads
        inp_v = self._expand_to_heads(inp, V)  # [V, B, T, ...]
        losses, metrics = {}, {}
        if self.intrinsic_mode == 'model':
            sig, sig_metrics = self.int_rew_model(self._take_dyn_head(outs), self._take_value_head(acts), update=update)            
        # sig: [B, T]. Broadcast to value_heads.
        sig_v = self._broadcast_heads(sig, V)  # [V, B, T]
        intr_data, intr_metrics = self._compute_returns(inp_v, sig_v, con_v, self.intr_critic, self.intr_slowcritic, self.intr_valnorm, self.intr_retnorm, self.intr_advnorm, discount, update)
        losses['intr_critic'] = self._critic_loss(self.intr_critic, self.intr_slowcritic, inp, intr_data['ret'],weight_v, self.intr_valnorm, update)
        if self.exp_obj == 'sombrl':
            alpha, log_alpha = self.temp()
            metrics.update({'temp/temp': jnp.mean(alpha), 'temp/log_temp': jnp.mean(sg(log_alpha))})
        metrics.update({f'int_rew_model/{k}': v for k, v in sig_metrics.items()})
        metrics.update({f'int_{k}': v for k, v in intr_metrics.items()})
        return losses, metrics, intr_data

    def _critic_loss(self, critic, slowcritic, inp, ret, weight, valnorm, update):
        """Critic log-prob loss with slow regularization.
        inp: [E, B, T, ...] (dyn_heads), ret: [V, B, T], weight: [V, B, T] where V=value_heads."""
        V = self.value_heads
        inp_v = self._expand_to_heads(inp, V)                                         # [V, B, T, ...]
        critic_dist = critic(inp_v, bdims=3, has_ensemble=True)                       # batch_shape [V, B, T]
        slowcritic_dist = slowcritic(inp_v, bdims=3, has_ensemble=True)               # batch_shape [V, B, T]
        voffset, vscale = valnorm(ret, update)
        ret_normed = (ret - voffset) / vscale                                         # [V, B, T]
        ret_padded = jnp.concatenate([ret_normed, 0 * ret_normed[..., -1:]], -1)      # [V, B, T+1]
        critic_logp = critic_dist.log_prob(sg(ret_padded))[..., :-1]                  # [V, B, T]
        slow_logp = critic_dist.log_prob(sg(slowcritic_dist.mean()))[..., :-1]        # [V, B, T]
        return sg(weight)[..., :-1] * -(critic_logp + self.config.slowreg * slow_logp)
    
    def _replay_critic_loss(self, replay_outs, data, basic_data, update):
        replay_inp = replay_outs if self.config.replay_critic_grad else sg(replay_outs)
        heads = self.value_heads
        replay_weight = sg(self._broadcast_heads(f32(~data['is_last']), heads))
        # replay_inp: [E, B, T, ...]. Expand to value_heads for critic.
        replay_inp_v = self._expand_to_heads(replay_inp, heads)  # [V, B, T, ...]
        replay_boot = self.critic(replay_inp_v, bdims=3, has_ensemble=True).mean()  # [V, B, T]
        metrics = {}
        replay_prior = None
        if self.value_prior_scale > 0:
            replay_prior = sg(self.v_prior(replay_inp_v, bdims=3, has_ensemble=True).mean())  # [V, B, T]
            replay_boot = replay_boot + self.value_prior_scale * replay_prior
        imag_boot = basic_data['ret'][..., 0]
        B, T = data['reward'].shape
        if imag_boot.shape[1] == B:
            imag_boot = jnp.broadcast_to(imag_boot[..., None], (heads, B, T))
        else:
            repeats = imag_boot.shape[1] // (B * T)
            imag_boot = imag_boot.reshape((heads, repeats, B, T))[:, 0]
        tarval = dict(imag=imag_boot, critic=replay_boot)[self.config.replay_critic_bootstrap]
        replay_reward = self._broadcast_heads(data['reward'], heads)
        replay_con = self._broadcast_heads(1.0 - f32(data['is_terminal']), heads)
        a = self._broadcast_heads(f32(~data['is_last'])[:, 1:] * self.config.return_lambda_replay, heads)
        b = self._broadcast_heads(f32(~data['is_terminal'])[:, 1:] * (1 - 1 / self.config.horizon), heads)
        replay_data, replay_metrics = self.compute_ret_adv(replay_reward, replay_con, tarval, a=a, b=b, update=False)
        replay_ret = replay_data['ret']
        if self.value_prior_scale > 0:
            replay_ret = replay_ret - self.value_prior_scale * replay_prior[..., :-1]
        critic_loss = self._critic_loss(self.critic, self.slowcritic, replay_inp, replay_ret, replay_weight,self.valnorm, update)
        return {'replay_critic': critic_loss}, {f'replay_{k}': v for k, v in replay_metrics.items()}

    def _actor_loss(self, actor, inp, acts, advantage, weight, actor_heads=None):
        """Policy gradient actor loss. Returns (loss, stats).
        inp: [E, B, T, ...] (dyn_heads), acts: [E, B, T, ...], advantage: [V, B, T], weight: [V, B, T]."""
        A = self.actor_heads if actor_heads is None else int(actor_heads)
        E = self.dyn_heads
        V = self.value_heads

        # Align input to actor heads: [A, B, T, ...] if A>=E, else [1, E, B, T, ...]
        actor_inp = self._align_ens_heads(inp, A)
        bdims = 4 if A < E else 3
        actor_dist = actor(actor_inp, bdims=bdims, has_ensemble=True)

        if A < E: # Shared actors keep dyn heads as ordinary batch dims, so the distribution batch shape is [1, E, B, T], not [1, E * B, T].
            actor_acts = treemap(lambda x: x[None], sg(acts))   # [1, E, B, T, ...]
            actor_adv = advantage[None]                         # [1, V, B, T]
            actor_weight = weight[None]                         # [1, V, B, T]
            stats_acts = self._take_dyn_head(sg(acts))          # [B, T, ...]
        else: # Actor has A>=E heads. actor_dist batch_shape: [A, B, T].
            actor_acts = treemap(lambda x: self._expand_to_heads(x, A), sg(acts))    # [A, B, T, ...]
            actor_adv = self._expand_to_heads(advantage, A)                          # [A, B, T]
            actor_weight = self._expand_to_heads(weight, A)                          # [A, B, T]
            stats_acts = self._take_actor_head(sg(acts))  # [B, T, ...]

        logpi = sum(v.log_prob(actor_acts[k]) for k, v in actor_dist.items())  # [A, ?, T]
        ents = {k: v.entropy() for k, v in actor_dist.items()}                  # [A, ?, T]

        T = min(logpi.shape[-1], actor_adv.shape[-1], actor_weight.shape[-1] - 1, *(x.shape[-1] for x in ents.values()))
        logpi = logpi[..., :T]
        actor_adv = actor_adv[..., :T]
        actor_weight = actor_weight[..., :T]
        ents = {k: v[..., :T] for k, v in ents.items()}
        loss = sg(actor_weight) * -(logpi * sg(actor_adv) + self.config.actent * sum(ents.values()))

        if A < E:
            stats_ents = {k: v[0].mean(0) for k, v in ents.items()}
        else:
            stats_ents = {k: self._take_actor_head(v) for k, v in ents.items()}
        return loss, {'acts': stats_acts, 'ents': stats_ents, 'actor': actor_dist}

    def loss(self, data, carry, update=True):
        mc = MetricsCollector()
        prevlat, prevact = carry
        beta = self._current_beta()
        # DreamerAgent has no corrector; the `res` batch is unused. All losses
        # train on the `ac` batch (no frequency masking here — Dreamer doesn't
        # split corrector vs actor-critic updates).
        data = data['ac']
        (wm_losses, dists, embed, replay_outs, prevacts, newlat, newact, wm_metrics) = self.world_model_loss(data, carry, boot_prob=self.config.boot_prob)
        self._newlat = newlat
        self._assert_headed_batch(replay_outs, self.dyn_heads, data['reward'].shape[0], 'replay_outs')
        outs, acts, rew, con, weight = self.imagination_rollout(replay_outs, data, actor=self.actor, actor_heads=self.actor_heads)        

        # rew, con, weight: [E, B, T] where E=dyn_heads. Expand to value_heads.
        V = self.value_heads
        rew_v = self._expand_to_heads(rew, V)       # [V, B, T]
        con_v = self._expand_to_heads(con, V)       # [V, B, T]
        weight_v = self._expand_to_heads(weight, V) # [V, B, T]
        prepare_actor_input = lambda o: treemap({'none': lambda x: sg(x), 'first': lambda x: jnp.concatenate([x[:, :1], sg(x[:, 1:])], 1), 'all': lambda x: x,}[self.config.ac_grads], o)
        inp = prepare_actor_input(outs)

        # Actor Critic Losses
        inp_v = self._expand_to_heads(inp, V)       # [V, B, T, ...]
        discount = 1 if self.config.contdisc else 1 - 1 / self.config.horizon
        basic_data, basic_metrics = self._compute_returns(inp_v, rew_v, con_v, self.critic, self.slowcritic, self.valnorm, self.retnorm, self.advnorm, discount, update, v_prior=self.v_prior, prior_scale=self.value_prior_scale)
        value_metrics = {}
        if V > 1:
            val = basic_data['val']
            value_metrics.update({'dreamer/value_ensemble/mean': val.mean(axis=0).mean(), 'dreamer/value_ensemble/std': val.std(axis=0).mean(),})
        if self.value_prior_scale > 0:
            prior_val = basic_data['prior_val']
            value_metrics.update(jaxutils.tensorstats(prior_val, 'dreamer/prior/raw'))
            value_metrics.update(jaxutils.tensorstats(self.value_prior_scale * prior_val, 'dreamer/prior/scaled'))
            value_metrics['dreamer/prior/ensemble_mean'] = prior_val.mean(axis=0).mean()
            value_metrics['dreamer/prior/ensemble_std'] = self.safe_std(prior_val, axis=0).mean()

        actor_loss, ac_stats = self._actor_loss(self.actor, inp, acts, basic_data['adv_normed'], weight_v)
        critic_ret = basic_data['ret']
        if self.value_prior_scale > 0:
            critic_ret = critic_ret - self.value_prior_scale * basic_data['prior_val'] # TODO: Check if we need a lambda here?
        critic_loss = self._critic_loss(self.critic, self.slowcritic, inp, critic_ret, weight_v, self.valnorm, update)
        wm_losses['actor'] = actor_loss
        wm_losses['critic'] = critic_loss

        if self.config.replay_critic_loss:
            replay_critic_losses, rc_stats = self._replay_critic_loss(replay_outs, data, basic_data, update)
            wm_losses.update(replay_critic_losses)
            mc.update(rc_stats)

        if self.use_intrinsic_reward and self.intrinsic_mode == 'model':
            int_rew_losses = self.intrinsic_reward_loss(replay_outs, prevacts, prevlat, embed, data)
            wm_losses.update(int_rew_losses)

        if self.use_exp_actor:
            if self.intrinsic_mode == 'model':
                outs, acts, rew, con, weight = self.imagination_rollout(replay_outs, data, actor=self.exp_actor, actor_heads=1)
            inp = prepare_actor_input(outs)
            con_v = self._expand_to_heads(con, V)           # [V, B, T]
            weight_v = self._expand_to_heads(weight, V)     # [V, B, T]
        
        intr_data = None
        if self.use_intrinsic_reward: # Computed on the exp_actor rollout if exists else actor
            intr_losses, intr_metrics, intr_data = self._intrinsic_critic_loss(inp, acts, outs, con_v, weight_v, discount, update)
            wm_losses.update(intr_losses)
            mc.update(intr_metrics)

        if self.use_exp_actor:
            V = self.value_heads
            exp_inp_v = self._expand_to_heads(inp, V)       # [V, B, T, ...]
            exp_rew_v = self._expand_to_heads(self._rew_dist(outs, bdims=3).mean(), V)   # [V, B, T]
            critic, slow_critic, valnorm = (self.exp_critic, self.exp_slowcritic, self.exp_valnorm) if self.use_exp_critic else (self.critic, self.slowcritic, self.valnorm)
            exp_data, exp_metrics = self._compute_returns(exp_inp_v, exp_rew_v, con_v, critic, slow_critic, valnorm, self.exp_retnorm, advnorm=self.exp_advnorm, discount=discount, update=update, v_prior=self.v_prior, prior_scale=self.value_prior_scale)
            mean_intr_adv = None
            if intr_data is not None:
                mean_intr_adv = intr_data['adv_normed'].mean(0, keepdims=True)
            mean_extr_adv = exp_data['adv_normed'].mean(0, keepdims=True)
            std_extr_ret = self.safe_std(exp_data['ret'], axis=0)[None]
            std_extr_val = self.safe_std(exp_data['tarval'][..., :-1], axis=0)[None] 
            _, rscale = self.exp_retnorm.stats()
            std_extr_adv = (std_extr_ret - std_extr_val) / rscale # UCB adv: R_\mu + beta R_\sigma - (V_\mu + \beta V_\sigma)
            exp_advantage = self._compute_exp_objective(mean_extr=mean_extr_adv, std_extr=std_extr_adv, intr_mean=mean_intr_adv)
            exp_advantage = self._expand_to_heads(exp_advantage, V)  # [V, B, T]

            exp_actor_loss, exp_stats = self._actor_loss(self.exp_actor, inp, acts, exp_advantage, weight_v, actor_heads=1)
            if self.use_exp_critic:
                critic_ret = exp_data['ret']
                if self.value_prior_scale > 0:
                    critic_ret = critic_ret - self.value_prior_scale * exp_data['prior_val']
                exp_critic_loss = self._critic_loss(self.exp_critic, self.exp_slowcritic, inp, critic_ret, weight_v, self.exp_valnorm, update)
                wm_losses['exp_critic'] = exp_critic_loss

            wm_losses['exp_actor'] = exp_actor_loss
            mc.update({f'exp_{k}': v for k, v in exp_metrics.items()})

        # Metrics
        mc.update(wm_metrics)
        mc.update(basic_metrics)
        mc.update(value_metrics)
        mc.add_loss_stats(wm_losses)
        mc.add_action_stats(ac_stats['acts'], ac_stats['ents'], ac_stats['actor'], self.act_space)
        mc.add_reward_stats(data['reward'], self._take_dyn_head(rew))
        mc.add_distribution_stats(dists, data)
        mc.add('activation/embed', jnp.abs(embed).mean())
        mc.add('dreamer/beta', beta)

        # Aggregation
        losses = {k: v * self.scales[k] for k, v in wm_losses.items()}
        loss = jnp.stack([v.mean() for v in losses.values()]).sum()
        replay_outs = treemap(lambda x: x[0], replay_outs)
        out = {'replay_outs': replay_outs, 'prevacts': prevacts, 'embed': embed}
        out.update({f'{k}_loss': v for k, v in losses.items()})
        new_carry = (newlat, newact)
        return loss, (out, new_carry, mc.result())


@jaxagent.Wrapper
class Agent(DreamerAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
