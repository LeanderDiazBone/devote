"""Policy divergence, exploration objectives, reward mixing, and network plasticity."""

import jax.numpy as jnp
import jax
from dreamerv3 import jaxutils
from dreamerv3 import ninjax as nj

f32 = jnp.float32
treemap = jax.tree_util.tree_map
sg = lambda x: treemap(jax.lax.stop_gradient, x)
cast = jaxutils.cast_to_compute
sample = lambda dist: {k: v.sample(seed=nj.seed()) for k, v in dist.items()}


class PolicyReports:
    """Policy divergence, exploration objectives, reward mixing, and network plasticity."""

    def report_policy_kl(self, data, carry, inputs=None):
        """KL divergence between exploitation (actor) and exploration policy,
        and average pairwise KL between actor heads (when actor_heads > 1)."""
        metrics = {}
        if getattr(self, 'use_candidate_policy', False):
            return metrics
        policy_mode = getattr(self, 'policy_mode', 'actor')
        is_discrete = all(s.discrete for s in self.act_space.values())

        _, outs_h = self._observe_report(data, carry, inputs, actor_inputs=True)

        key = list(self.act_space.keys())[0]

        # --- Actor head divergence (when actor_heads > 1) ---
        if self.actor_heads > 1:
            actor_dist = self._report_actor_distribution(outs_h, inputs)
            head_dist = actor_dist[key]  # distribution with batch shape [H, B, T, ...]
            if is_discrete:
                probs = head_dist.probs_parameter()
                kl_sum = jnp.zeros(probs.shape[1:3])
                kl = lambda i, j: _categorical_kl(probs[i], probs[j])
            else:
                means, stds = head_dist.mean(), head_dist.stddev()
                kl_sum = jnp.zeros(means[0].shape[:-1])
                def kl(i, j):
                    dm = means[i] - means[j]
                    return (jnp.log(stds[j] + 1e-8) - jnp.log(stds[i] + 1e-8)
                            + (stds[i] ** 2 + dm ** 2) / (2 * stds[j] ** 2 + 1e-8) - 0.5).sum(-1)
            count = 0
            for i in range(self.actor_heads):
                for j in range(i + 1, self.actor_heads):
                    kl_sum = kl_sum + (kl(i, j) + kl(j, i)) / 2
                    count += 1
            metrics['policy_kl/actor_heads_mean_pairwise'] = (kl_sum / count).mean()

        # --- Exploit vs explore KL (for exp_actor / disc_ucb modes) ---
        if policy_mode == 'actor' or policy_mode == 'thompson_sampling':
            return metrics

        exploit_dist = self._report_actor_distribution(outs_h, inputs)

        if policy_mode == 'exp_actor':
            if hasattr(self.exp_actor, 'num_heads') and self.exp_actor.num_heads == 1:
                actor_inp = treemap(lambda x: x.mean(0, keepdims=True) if x.shape[0] == self.dyn_heads else x[None] if x.ndim >= 1 else x, outs_h)  # [1, B, T, ...]
                explore_dist = self.exp_actor(actor_inp, bdims=3, has_ensemble=True)
            else:
                explore_dist = self.exp_actor(sg(self._take_dyn_head(outs_h)))
        elif policy_mode == 'disc_ucb':
            explore_dist = self.exp_actor(sg(self._take_dyn_head(outs_h)), bdims=2)
        else:
            return metrics

        p = self._take_actor_head(exploit_dist)[key]
        q = explore_dist[key]
        if hasattr(self, '_take_actor_head'):
            q = self._take_actor_head({key: q})[key] if isinstance(q, type(p)) else q

        if is_discrete:
            p_probs = p.probs_parameter()
            q_probs = q.probs_parameter()
            kl_pq = _categorical_kl(p_probs, q_probs)
            kl_qp = _categorical_kl(q_probs, p_probs)
        else:
            kl_pq = p.kl_divergence(q)
            kl_qp = q.kl_divergence(p)

        metrics['policy_kl/exploit_to_explore'] = kl_pq.mean()
        metrics['policy_kl/explore_to_exploit'] = kl_qp.mean()
        metrics['policy_kl/symmetric'] = ((kl_pq + kl_qp) / 2).mean()
        return metrics

    def report_mixing(self, data, carry, inputs=None):
        metrics = {}

        # Reconstruct latent sequence from replay data
        embed, outs = self._observe_report(data, carry, inputs)
        val = self._report_value(outs, inputs)
        val_mean = val.mean(0)
        val_std = val.std(0)
        metrics['dreamer/mixing/value_ensemble_mean'] = val_mean.mean()
        metrics['dreamer/mixing/value_ensemble_std'] = val_std.mean()
        value_prior_scale = getattr(
            self, 'value_prior_scale', getattr(self, 'prior_scale', 0.0))
        if hasattr(self, 'v_prior') and value_prior_scale > 0:
            prior = self._report_value(outs, inputs, 'prior')
            prior_mean = prior.mean(0)
            prior_std = prior.std(0)
            metrics['dreamer/mixing/prior_mean'] = prior_mean.mean()
            metrics['dreamer/mixing/prior_std'] = prior_std.mean()
            metrics['dreamer/mixing/scaled_prior_mean'] = value_prior_scale * prior_mean.mean()
            metrics['dreamer/mixing/scaled_prior_std'] = abs(value_prior_scale) * prior_std.mean()

        beta = self._current_beta()
        metrics['dreamer/mixing/beta'] = beta
        metrics['dreamer/mixing/ucb_bonus'] = (beta * val_std).mean()
        metrics['dreamer/mixing/ucb_bonus_ratio'] = (beta * val_std.mean()) / (jnp.abs(val_mean).mean() + 1e-8)

        # Intrinsic / extrinsic reward mixing
        if getattr(self, 'use_intrinsic_reward', False):
            alpha, log_alpha = self.temp()
            metrics['dreamer/mixing/temp'] = jnp.mean(alpha)
            metrics['dreamer/mixing/log_temp'] = jnp.mean(sg(log_alpha))

            actor_dist = self._report_actor_distribution(outs, inputs)
            acts_single = self._take_actor_head(cast(sample(actor_dist)))
            sig, _ = self._report_intrinsic_reward(outs, data, acts_single)
            int_scale = jnp.abs(sig).mean()
            ext_scale = jnp.abs(data['reward']).mean()
            metrics['dreamer/mixing/intrinsic_reward_mean'] = sig.mean()
            metrics['dreamer/mixing/intrinsic_reward_abs'] = int_scale
            metrics['dreamer/mixing/extrinsic_reward_abs'] = ext_scale
            metrics['dreamer/mixing/weighted_int_ext_ratio'] = (jnp.mean(alpha) * int_scale) / (ext_scale + 1e-8)

        return metrics

    def report_exp_objective(self, data, carry, inputs=None):
        if hasattr(self, 'q'):
            return self._report_observer_exp_objective(data, carry, inputs)
        if self.policy_mode == 'exp_actor' and self.use_exp_actor:
            return self._report_exp_actor_objective(data, carry, inputs)
        if self.policy_mode == 'exp_sampling':
            return self._report_exp_sampling_objective(data, carry, inputs)
        return {}

    def _report_exp_actor_objective(self, data, carry, inputs=None):
        _, replay_outs = self._observe_report(data, carry, inputs, actor_inputs=True)
        probe_outs = None
        if getattr(self, 'intrinsic_mode', 'model') == 'dynamics':
            outs, probe_outs, acts, _, con, weight = self.exploration_imagination_rollout(
                replay_outs, data, actor=self.exp_actor, actor_heads=1)
        else:
            outs, acts, _, con, weight = self.imagination_rollout(
                replay_outs, data, actor=self.exp_actor, actor_heads=1)
        prepare_actor_input = lambda o: treemap({
            'none': lambda x: sg(x),
            'first': lambda x: jnp.concatenate([x[:, :1], sg(x[:, 1:])], 1),
            'all': lambda x: x,
        }[self.config.ac_grads], o)
        inp = prepare_actor_input(outs)

        discount = 1 if self.config.contdisc else 1 - 1 / self.config.horizon
        V = self.value_heads
        exp_inp_v = self._expand_to_heads(inp, V)
        exp_rew = self._rew_dist(outs, bdims=3).mean()
        exp_rew_v = self._expand_to_heads(exp_rew, V)
        con_v = self._expand_to_heads(con, V)
        weight_v = self._expand_to_heads(weight, V)
        critic, slow_critic, valnorm = (self.exp_critic, self.exp_slowcritic, self.exp_valnorm) if self.use_exp_critic else (self.critic, self.slowcritic, self.valnorm)
        exp_data, _ = self._compute_returns(exp_inp_v, exp_rew_v, con_v, critic, slow_critic, valnorm, self.exp_retnorm, self.exp_advnorm, discount, update=False)

        #mean_extr = exp_data['adv_normed'].mean(0, keepdims=True)
        #std_extr = self._normalized_return_std(exp_data, self.exp_retnorm)
        mean_extr = exp_data['adv_normed'].mean(0, keepdims=True)
        std_extr_ret = self.safe_std(exp_data['ret'], axis=0)[None]
        std_extr_val = self.safe_std(exp_data['tarval'][..., :-1], axis=0)[None]
        _, rscale = self.exp_retnorm.stats()
        std_extr = (std_extr_ret - std_extr_val) / rscale # UCB adv: R_\mu + beta R_\sigma - (V_\mu + \beta V_\sigma)

        mean_intr = None
        if self.use_intrinsic_reward:
            _, _, intr_data = self._intrinsic_critic_loss(inp, acts, outs, con_v, weight_v, discount, update=False)
            mean_intr = intr_data['adv_normed'].mean(0, keepdims=True)


        return self._report_exp_objective_metrics(
            mean_extr, std_extr, mean_intr=mean_intr)

    def _report_exp_sampling_objective(self, data, carry, inputs=None):
        _, replay_outs = self._observe_report(data, carry, inputs, actor_inputs=True)
        flat_outs = self._flatten_imag_outs(replay_outs)
        _, vals_extr, vals_intr = self._compute_q_values(flat_outs, actor=self.actor, actor_heads=self.actor_heads, K=self.num_cand_samples)

        mean_extr = vals_extr.mean(0)
        std_extr = self.safe_std(vals_extr)
        mean_intr = None
        if vals_intr is not None:
            mean_intr = vals_intr.mean(0)

        return self._report_exp_objective_metrics(
            mean_extr, std_extr, mean_intr=mean_intr)

    def _report_observer_exp_objective(self, data, carry, inputs=None):
        if getattr(self, 'exp_obj', 'none') == 'none':
            return {}
        _, replay_outs = self._observe_report(data, carry, inputs, actor_inputs=True)
        states = treemap(lambda x: x[:, :, :-1], replay_outs)
        states_unh = self._take_dyn_head(states)
        if getattr(self, 'use_candidate_policy', False) and not self.use_exp_actor:
            if self.enumerate_actions:
                q_for_actions = self.q(self._q_input(states_unh), bdims=2, has_ensemble=False).mean().mean(0)
                key = next(iter(self.act_space.keys()))
                idx = jnp.argmax(q_for_actions, axis=-1)
                acts = {key: cast(jax.nn.one_hot(idx, q_for_actions.shape[-1]))}
            else:
                qs, grid = self._candidate_qs(states_unh, self.q, samples=1)
                acts = self._gather_act(grid, qs.mean(0).argmax(0))
        else:
            actor_net = self.exp_actor if self.use_exp_actor else self.actor
            acts, _, _ = self._actor_step(actor_net, sg(states_unh), bdims=2)
            acts = treemap(lambda x: x[0], acts)                       # take first head
        q_inp = self._q_input(states_unh, acts) if not self._is_discrete else self._q_input(states_unh)
        q_extr = self.q.particles(q_inp, bdims=2, frozen=True)
        if self._is_discrete:
            q_extr = self._select_action_values(q_extr, acts)
        mean_extr = q_extr.mean(axis=0)
        std_extr = self.safe_std(q_extr)
        mean_intr = None
        if self.use_intrinsic_reward:
            q_intr = self.intr_q.particles(q_inp, bdims=2, frozen=True)
            if self._is_discrete:
                q_intr = self._select_action_values(q_intr, acts)
            mean_intr = q_intr.mean(axis=0)
        rb_mean = None
        if self.exp_obj == 'rnd' and getattr(self.q, 'residual_bootstrap', None) is not None:
            rb_p = self.q.component_particles(q_inp, bdims=2)['residual_bootstrap']
            if self._is_discrete:
                rb_p = self._select_action_values(rb_p, acts)
            rb_mean = rb_p.mean(axis=0)
        return self._report_exp_objective_metrics(
            mean_extr, std_extr, mean_intr=mean_intr, rb_mean=rb_mean)

    def _report_exp_objective_metrics(self, mean_extr, std_extr, mean_intr=None, rb_mean=None):
        metrics = {}
        beta = self._current_beta()
        metrics['exp_obj/beta'] = beta
        metrics.update(jaxutils.tensorstats(mean_extr, 'exp_obj/mean_extr'))
        metrics.update(jaxutils.tensorstats(std_extr, 'exp_obj/std_extr'))
        if mean_intr is not None:
            metrics.update(jaxutils.tensorstats(mean_intr, 'exp_obj/mean_intr'))
        if rb_mean is not None:
            metrics.update(jaxutils.tensorstats(rb_mean, 'exp_obj/rb_mean'))
        if self.exp_obj in ('ucb', 'ocb'):
            ucb = mean_extr + beta * std_extr
            metrics.update(jaxutils.tensorstats(ucb, 'exp_obj/ucb'))
        if self.exp_obj in ('sombrl', 'ocb') and mean_intr is not None:
            alpha, log_alpha = self.temp()
            metrics['exp_obj/temp'] = jnp.mean(alpha)
            metrics['exp_obj/log_temp'] = jnp.mean(sg(log_alpha))
            metrics.update(jaxutils.tensorstats(alpha * mean_intr, 'exp_obj/weighted_intr'))
        if mean_intr is not None:
            metrics['exp_obj/rank_corr_intr_extr_std'] = self._rank_corr(mean_intr, std_extr)
        obj = self._compute_exp_objective(mean_extr=mean_extr, std_extr=std_extr, intr_mean=mean_intr, rb_mean=rb_mean)
        metrics.update(jaxutils.tensorstats(obj, 'exp_obj/value'))
        return metrics

    def report_plasticity(self, data, carry, dormant_tau=0.025, rank_delta=0.01, inputs=None):
        """Plasticity diagnostics for the exp_actor and each critic ensemble member.

        Computed on post-activation features of the final hidden layer for an
        eval batch:
          * dormant_frac (Sokar et al. 2023): fraction of units whose mean
            absolute activation, normalized by the layer mean, is <= dormant_tau.
          * effective_rank (Kumar et al. srank_delta): smallest k such that the
            top-k singular values of the [N, U] feature matrix account for at
            least (1 - rank_delta) of the total singular-value mass.

        Logged separately for self.exp_actor and for each ensemble head of each
        PriorCritic in self.q.heads.
        """
        if not getattr(self.config, 'report_plasticity', True):
            return {}
        metrics = {}

        embed, outs_h = self._observe_report(data, carry, inputs)
        ac_states = self._take_dyn_head(outs_h)
        if getattr(self, 'ac_inputs', 'wm') == 'obs' or getattr(self, 'visual_prior', False):
            ac_states = self._augment_ac_inputs(ac_states, data, headed=False)

        if getattr(self, 'use_exp_actor', False) and hasattr(self, 'exp_actor'):
            actor_feat = self.exp_actor.features(
                ac_states, bdims=2, has_ensemble=False)            # [E=1, B, T, U]
            metrics.update(self._plasticity_metrics(
                actor_feat, prefix='plasticity/exp_actor',
                dormant_tau=dormant_tau, rank_delta=rank_delta))

        if hasattr(self, 'q') and hasattr(self.q, 'heads'):
            data_acts = jaxutils.onehot_dict(
                {k: data[k] for k in self.act_space}, self.act_space)
            data_acts = treemap(
                lambda x: x[0] if hasattr(x, 'ndim') and x.ndim > 0 and x.shape[0] == 1 else x,
                data_acts,
            )
            q_inp = (self._q_input(ac_states)
                     if self._is_discrete else self._q_input(ac_states, data_acts))
            for hi, critic in enumerate(self.q.heads):
                feat = critic.head.features(
                    q_inp, bdims=2, has_ensemble=False)            # [E, B, T, U]
                tag = (f'plasticity/q{hi}'
                       if len(self.q.heads) > 1 else 'plasticity/q')
                metrics.update(self._plasticity_metrics(
                    feat, prefix=tag,
                    dormant_tau=dormant_tau, rank_delta=rank_delta))
        return metrics

    @staticmethod
    def _plasticity_metrics(feat, prefix, dormant_tau, rank_delta):
        """Per-ensemble-head dormant fraction and srank_delta from [E, *batch, U]."""
        feat = f32(feat)
        E = feat.shape[0]
        flat = feat.reshape(E, -1, feat.shape[-1])                 # [E, N, U]
        metrics = {}
        for e in range(E):
            x = flat[e]
            act_mag = jnp.abs(x).mean(0)                           # [U]
            score = act_mag / (act_mag.mean() + 1e-8)
            dormant_frac = (score <= dormant_tau).mean()
            # Singular values via Gram-matrix eigvalsh: faster + more stable
            # than svd on GPU. s_i = sqrt(eigval_i(X^T X)).
            gram = x.T @ x                                         # [U, U]
            s2 = jnp.clip(jnp.linalg.eigvalsh(gram), 0.0, None)
            s = jnp.sqrt(s2)
            # srank_delta (Kumar et al.): smallest k such that top-k singular
            # values capture >= (1 - rank_delta) of the total mass. eigvalsh
            # is ascending, so cumulative-from-the-top is reverse cumsum.
            cum_from_top = jnp.cumsum(s[::-1]) / (s.sum() + 1e-8)
            eff_rank = (cum_from_top < (1.0 - rank_delta)).sum().astype(f32) + 1.0
            metrics[f'{prefix}_e{e}/dormant_frac'] = dormant_frac
            metrics[f'{prefix}_e{e}/effective_rank'] = eff_rank
        return metrics

    def _report_actor_distribution(self, outs, inputs=None):
        def evaluate():
            if getattr(self, 'use_candidate_policy', False):
                return None
            heads = max(int(getattr(self, 'actor_heads', 1)), 1)
            inp = (self._expand_to_heads(outs, heads) if heads >= self.dyn_heads else
                   treemap(lambda x: x.mean(0, keepdims=True), outs))
            return self.actor(inp, bdims=3, has_ensemble=True)
        return self._report_cached(inputs, 'actor', evaluate)

    def _report_exp_actor_distribution(self, outs_h):
        """Evaluate the exploration actor with its original head/batch layout."""
        if not hasattr(self, 'exp_actor'):
            return None
        if getattr(self, 'dyn_heads', 1) > 1:
            actor_inp = self._align_ens_heads(outs_h, 1)
            actor_dist = self.exp_actor(actor_inp, bdims=4, has_ensemble=True)
            return actor_dist, self.dyn_heads, True
        actor_dist = self.exp_actor(outs_h, bdims=3, has_ensemble=True)
        return actor_dist, 1, False

    def _report_intrinsic_reward(self, outs_h, data, actions=None):
        """Evaluate intrinsic rewards for either replay or sampled policy actions."""
        if not getattr(self, 'use_intrinsic_reward', False):
            return None

        if actions is None:
            cur_acts = jaxutils.onehot_dict(
                {k: data[k] for k in self.act_space}, self.act_space)
        else:
            cur_acts = actions

        intrinsic_mode = getattr(self, 'intrinsic_mode', 'model')
        disg_metrics = {}
        if intrinsic_mode == 'dynamics':
            outs_single = self._take_dyn_head(outs_h)
            batch_size, time_size = data['is_first'].shape
            probe_start = treemap(
                lambda x: x.reshape((batch_size * time_size, 1) + x.shape[2:]),
                outs_single)
            probe_carry = self._dyn_outs_to_carry(probe_start)
            probe_acts = treemap(
                lambda x: x.reshape((batch_size * time_size,) + x.shape[2:]),
                cur_acts)
            _, probe_outs = self._dyn_imagine(probe_carry, probe_acts, bdims=1)
            probe_outs = treemap(
                lambda x: x.reshape((x.shape[0], batch_size, time_size) + x.shape[2:]),
                probe_outs)
            intr_reward = self._dynamics_disagreement(probe_outs)
        else:
            if not hasattr(self, 'int_rew_model'):
                return None
            intr_states, intr_acts = self._take_dyn_head(outs_h), cur_acts
            if hasattr(self, '_intrinsic_model_inputs'):
                intr_states, intr_acts = self._intrinsic_model_inputs(
                    intr_states, intr_acts, data=data)
            intr_reward, disg_metrics = self.int_rew_model(intr_states, intr_acts)

        return intr_reward, disg_metrics


def _categorical_kl(p, q):
    return (p * (jnp.log(p + 1e-8) - jnp.log(q + 1e-8))).sum(-1)
