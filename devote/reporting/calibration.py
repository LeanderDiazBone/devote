"""Q/value calibration and the optional predictions used by calibration tables."""

import jax.numpy as jnp
import jax
from dreamerv3 import jaxutils

f32 = jnp.float32
treemap = jax.tree_util.tree_map
sg = lambda x: treemap(jax.lax.stop_gradient, x)
cast = jaxutils.cast_to_compute


class CalibrationReports:
    """Q/value calibration and the optional predictions used by calibration tables."""

    def report_q_calibration(self, data, carry, gamma, inputs=None):
        inputs = self._report_inputs(data, carry) if inputs is None else inputs
        metrics = {}
        state_table_mode = jnp.all(data['stepid'][..., 0] == 255)
        action_table_mode = jnp.all(data['stepid'][..., 0] == 254)
        table_mode = state_table_mode | action_table_mode

        # 1. Get Latents (Reconstruct sequence)
        embed, outs_h = self._observe_report(data, carry, inputs)
        _, ac_states_h = self._observe_report(data, carry, inputs, actor_inputs=True)
        ac_states = self._take_dyn_head(outs_h)
        if getattr(self, 'ac_inputs', 'wm') == 'obs' or getattr(self, 'visual_prior', False):
            ac_states = self._augment_ac_inputs(ac_states, data, headed=False)

        # 2. Get Q-values for actions actually taken
        data_acts = jaxutils.onehot_dict({k: data[k] for k in self.act_space}, self.act_space)
        data_acts = treemap(
            lambda x: x[0] if hasattr(x, 'ndim') and x.ndim > 0 and x.shape[0] == 1 else x,
            data_acts,
        )

        if getattr(self, 'use_candidate_policy', False):
            if self.enumerate_actions:
                q_for_actions = self.q(self._q_input(ac_states), bdims=2, has_ensemble=False).mean().mean(0)  # [B, T, A]
                key = next(iter(self.act_space.keys()))
                greedy = jnp.argmax(q_for_actions, axis=-1)
                policy_acts = {key: cast(jax.nn.one_hot(greedy, q_for_actions.shape[-1]))}
            else:
                qs, acts = self._candidate_qs(ac_states, self.q, samples=1)
                policy_acts = treemap(cast, self._gather_act(acts, qs.mean(0).argmax(0)))
        else:
            if self._actor_heads >= self.dyn_heads:
                actor_dist = self._report_actor_distribution(ac_states_h, inputs)
            else:
                actor_inp = self._align_ens_heads(ac_states_h, self._actor_heads)
                actor_dist = self.actor(actor_inp, bdims=3, has_ensemble=True)
            policy_acts = {}
            for key, space in self.act_space.items():
                if space.discrete:
                    probs = self._take_actor_head(
                        {key: actor_dist[key].probs_parameter()})[key]
                    policy_acts[key] = cast(jax.nn.one_hot(
                        jnp.argmax(probs, axis=-1), probs.shape[-1]))
                else:
                    policy_acts[key] = cast(self._take_actor_head(
                        {key: actor_dist[key].mean()})[key])

        cur_acts = treemap(
            lambda pol, dat: jnp.where(state_table_mode, pol, dat),
            policy_acts,
            data_acts,
        )

        q_inp = self._q_input(ac_states, cur_acts)
        q_particles = self.q.particles(q_inp, bdims=2, frozen=False)  # [V*S, B, T, *q_shape]
        q_mixed_ensemble = self._select_action_values(q_particles, cur_acts)
        q_mean = q_mixed_ensemble.mean(0)
        q_std = self.safe_std(q_mixed_ensemble)

        # Actor objective decomposition: mean + bonus that the policy actually optimizes.
        # Mirrors exp_score_parts for the configured exp_obj, so e.g. ucb logs Std-based
        # bonus, rnd logs β·mean(rb), sombrl logs α·mean(intr), etc.
        if hasattr(self, 'exp_score_parts') and hasattr(self.q, 'component_particles'):
            comp_parts = self.q.component_particles(q_inp, bdims=2)
            if self._is_discrete:
                key = next(iter(self.act_space.keys()))
                onehot = cur_acts[key]
                comp_parts = {k: (v * onehot).sum(-1) if v.shape[-1] == onehot.shape[-1] else v
                              for k, v in comp_parts.items()}
            if self.use_intrinsic_reward:
                intr_p = self.intr_q.particles(q_inp, bdims=2, frozen=True)
                if self._is_discrete:
                    intr_p = self._select_action_values(intr_p, cur_acts)
                comp_parts['intr'] = intr_p
            actor_mean_term, actor_bonus_term = self.exp_score_parts(comp_parts)
            metrics['sac/actor_obj_mean'] = actor_mean_term.mean()
            metrics['sac/actor_obj_bonus'] = actor_bonus_term.mean()
            metrics['sac/actor_obj_score'] = (actor_mean_term + actor_bonus_term).mean()
        stats, ret, error = self._calibration(data, q_mean, q_std, gamma)
        stats.pop('mean')
        corr = stats.pop('cal_err_std_cor')
        metrics.update({f'sac/q_{key}': value if key == 'std' else
                        jnp.where(table_mode, jnp.array(jnp.nan, f32), value)
                        for key, value in stats.items()})
        metrics['sac/cal_err_std_cor'] = jnp.where(table_mode, jnp.array(jnp.nan, f32), corr)

        if self._state_obs_keys and self._log_metrics_table:
            blocks = self._report_calibration_extras(
                data, outs_h, ac_states_h, inputs, states=ac_states, actions=cur_acts)
            table, names = self._calibration_table(
                data, q_mean, q_std, ret, error, 'q', blocks, table_mode)
            metrics[f"sac/q_calibration_table__{'__'.join(names)}"] = table
        return metrics

    def report_value_calibration(self, data, carry, gamma, inputs=None):
        inputs = self._report_inputs(data, carry) if inputs is None else inputs
        metrics = {}

        # Reconstruct latent sequence from replay data
        embed, outs_h = self._observe_report(data, carry, inputs)
        val = self._report_value(outs_h, inputs)

        val_mean = val.mean(0)   # [B, T]
        val_std = val.std(0)     # [B, T]

        stats, ret, error = self._calibration(
            data, val_mean, val_std, gamma, uncertainty=self.value_heads > 1)
        metrics.update({f'dreamer/val_{key}': value for key, value in stats.items()})

        if getattr(self, '_state_obs_keys', []) and self._log_metrics_table:
            heads = self._flatten_headed_table_features(val, self.value_heads, trim_last=True)
            blocks = [(heads, [f'v_head_{i}' for i in range(heads.shape[-1])])]
            blocks += self._report_calibration_extras(data, outs_h, outs_h, inputs)
            table, names = self._calibration_table(data, val_mean, val_std, ret, error, 'v', blocks)
            metrics[f"dreamer/val_calibration_table__{'__'.join(names)}"] = table
        return metrics

    def _calibration(self, data, mean, std, gamma, uncertainty=True):
        reward = data['reward']
        discount = 1.0 - f32(data['is_terminal'])
        a = jnp.ones_like(reward[:, 1:])
        b = jnp.full_like(reward[:, 1:], gamma) * (1 - f32(data['is_terminal'][:, 1:]))
        result, _ = self.compute_ret_adv(reward, discount, mean, a, b)
        ret = result['ret']
        error = jnp.abs(ret - mean[:, :-1])
        metrics = {'cal_rmse': jnp.sqrt((error ** 2).mean()),
                   'mean': mean.mean(), 'std': std.mean()}
        if uncertainty:
            for beta in (1, 2, 5):
                metrics[f'cal_cov_beta_{beta}'] = (error < (beta * std[:, :-1] + 1e-6)).mean()
            corr = jnp.corrcoef(error.reshape(-1), std[:, :-1].reshape(-1))[0, 1]
            metrics['cal_err_std_cor'] = jnp.where(jnp.isnan(corr), 0.0, corr)
        return metrics, ret, error

    def _report_value(self, outs, inputs=None, kind='value'):
        def evaluate():
            if kind == 'prior':
                scale = getattr(self, 'value_prior_scale', getattr(self, 'prior_scale', 0.0))
                if not hasattr(self, 'v_prior') or scale <= 0:
                    return None
                net, norm = self.v_prior, None
            elif kind == 'intrinsic':
                if not getattr(self, 'use_intrinsic_reward', False) or not hasattr(self, 'intr_critic'):
                    return None
                net, norm = self.intr_critic, self.intr_valnorm
            else:
                net, norm = self.critic, self.valnorm
            inp = self._expand_to_heads(outs, self.value_heads)
            value = net(inp, bdims=3, has_ensemble=True).mean()
            assert value.shape[0] == self.value_heads, value.shape
            if norm is not None:
                offset, scale = norm.stats()
                value = value * scale + offset
            return value
        return self._report_cached(inputs, kind, evaluate)

    def _report_calibration_extras(self, data, outs, ac_outs, inputs, states=None, actions=None):
        """Evaluate optional predictions and append their columns in logging order."""
        reward = self._rew_dist(outs, bdims=3).mean() if getattr(self, 'train_world_model', True) else None
        blocks = [self._reward_prediction_table_columns(reward, True)]
        if states is not None:
            components = self._report_q_components(states, actions)
            blocks.append(self._q_component_table_columns(components, True))
            intrinsic = self._report_intrinsic_q(states, actions)
            blocks.append(self._ensemble_table_columns(intrinsic, 1, 'intr_q', True))
        else:
            for kind, prefix in [('prior', 'prior'), ('intrinsic', 'intr_v')]:
                values = self._report_value(outs, inputs, kind)
                blocks.append(self._ensemble_table_columns(values, self.value_heads, prefix, True))
        intrinsic = self._report_intrinsic_reward(outs, data, actions)
        alpha = jnp.mean(self.temp()[0]) if intrinsic is not None and hasattr(self, 'temp') else None
        blocks.append(self._intrinsic_reward_table_columns(intrinsic, alpha, True))
        actor = self._report_actor_distribution(ac_outs, inputs)
        if actor is not None:
            blocks.append(self._actor_distribution_columns_from_dist(
                actor, max(int(self.actor_heads), 1), 'actor', True))
        exploration = self._report_exp_actor_distribution(ac_outs)
        if exploration is not None:
            dist, heads, nested = exploration
            blocks.append(self._actor_distribution_columns_from_dist(
                dist, heads, 'exp_actor', True, lambda x: x[0] if nested else x))
        return blocks

    def _report_intrinsic_q(self, states, actions):
        """Evaluate intrinsic Q on replay state-action pairs."""
        if not getattr(self, 'use_intrinsic_reward', False) or not hasattr(self, 'intr_q'):
            return None

        intr_q_all = self.intr_q(self._q_input(states, actions), bdims=2, has_ensemble=False).mean()
        if self._is_discrete:
            action_key = next(iter(self.act_space.keys()))
            intr_q = (intr_q_all * actions[action_key][None]).sum(-1)
        else:
            intr_q = intr_q_all

        return intr_q

    def _report_q_components(self, states, actions):
        """Evaluate table components with separate sample and ensemble axes."""
        if not hasattr(self, 'q') or not hasattr(self.q, 'component_samples'):
            return None
        states, actions = treemap(
            lambda x: self._broadcast_heads(x, self.value_heads), (states, actions))
        params = self.q.find()
        self.q.put(sg(params))
        try:
            comps = self.q.component_samples(
                self._q_input(states, actions), bdims=3, has_ensemble=True)
        finally:
            self.q.put(params)
        if self._is_discrete:
            onehot = actions[next(iter(self.act_space))]
            comps = {k: (v * onehot).sum(-1) if v.shape[-1] == onehot.shape[-1] else v
                     for k, v in comps.items()}
        return comps
