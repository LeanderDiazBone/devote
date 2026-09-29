"""Prior statistics, residual-bootstrap fit, and bonus propagation over trajectories."""

import numpy as np
import jax.numpy as jnp
import jax
from dreamerv3 import jaxutils
from .. import nets as dr_nets

f32 = jnp.float32
treemap = jax.tree_util.tree_map
sg = lambda x: treemap(jax.lax.stop_gradient, x)
cast = jaxutils.cast_to_compute


class ResidualReports:
    """Prior statistics, residual-bootstrap fit, and bonus propagation over trajectories."""

    def report_prior_stats(self, data, carry, inputs=None):
        metrics = {}
        gamma = getattr(
            self, 'discount',
            1 if self.config.contdisc else 1 - 1 / self.config.horizon)
        _, replay_outs = self._observe_report(data, carry, inputs, actor_inputs=True)
        comps_cur = None

        if hasattr(self, 'critic') and hasattr(self, 'v_prior'):
            value_prior_scale = getattr(
                self, 'value_prior_scale', getattr(self, 'prior_scale', 0.0))
            if value_prior_scale <= 0:
                return metrics

            q_all = self._report_value(replay_outs, inputs)
            prior_all = self._report_value(replay_outs, inputs, 'prior')
            q_cur = q_all[..., :-1]
            prior_cur = prior_all[..., :-1]
            prior_next = prior_all[..., 1:]

        elif hasattr(self, 'q') and hasattr(self, 'q_prior'):
            prior_scale = getattr(self, 'prior_scale', getattr(self, 'value_prior_scale', 0.0))
            if prior_scale <= 0:
                return metrics

            states = treemap(lambda x: x[:, :, :-1], replay_outs)
            next_states = treemap(lambda x: x[:, :, 1:], replay_outs)
            actions = jaxutils.onehot_dict(
                {k: data[k][:, :-1] for k in self.act_space}, self.act_space)
            states_unh = self._take_dyn_head(states)
            actions_unh = self._take_dyn_head(actions) if not self._is_discrete else actions
            next_states_unh = self._take_dyn_head(next_states)
            q_inp = self._q_input(states_unh, actions_unh)
            q_particles = self.q.particles(q_inp, bdims=2, frozen=True, include_prior=False)
            q_cur = self._select_action_values(q_particles, actions_unh)
            if getattr(self, 'use_candidate_policy', False):
                if self.enumerate_actions:
                    q_at_next = self.q(self._q_input(next_states_unh), bdims=2, has_ensemble=False).mean().mean(0)
                    key = next(iter(self.act_space.keys()))
                    idx = jnp.argmax(q_at_next, axis=-1)
                    next_acts = {key: cast(jax.nn.one_hot(idx, q_at_next.shape[-1]))}
                else:
                    qs, grid = self._candidate_qs(next_states_unh, self.q, samples=1)
                    next_acts = self._gather_act(grid, qs.mean(0).argmax(0))
            else:
                next_acts, _, _ = self._actor_step(self.actor, sg(next_states_unh), bdims=2)
                next_acts = treemap(lambda x: x[0], next_acts)            # take first actor head
            qctx = self.q.sample_context(q_inp, bdims=2) if hasattr(self.q, 'sample_context') else None
            qkw = {} if qctx is None else {'epistemic': qctx}
            inp_next = self._q_input(next_states_unh, next_acts) if not self._is_discrete else self._q_input(next_states_unh)
            comps_cur = comps_next = None
            if hasattr(self.q, 'component_means'):
                comps_cur = self.q.component_means(q_inp, bdims=2, has_ensemble=False, prior_scale=1.0, corrector_scale=1.0, **qkw)
                comps_next = self.q.component_means(inp_next, bdims=2, has_ensemble=False, prior_scale=1.0, corrector_scale=1.0, **qkw)
                prior_cur, prior_next = comps_cur['prior'], comps_next['prior']
            else:
                prior_cur = self.q_prior(q_inp, bdims=2, has_ensemble=False).mean()
                prior_next = self.q_prior(inp_next, bdims=2, has_ensemble=False).mean()
            if self._is_discrete:
                key = next(iter(self.act_space.keys()))
                a_cur, a_next = actions_unh[key], next_acts[key]
                def reduce_action(x, oh):
                    return (x * oh).sum(-1) if x.shape[-1] == oh.shape[-1] else x
                prior_cur = reduce_action(prior_cur, a_cur)
                prior_next = reduce_action(prior_next, a_next)
                if comps_cur is not None:
                    comps_cur = {k: reduce_action(v, a_cur) for k, v in comps_cur.items()}
                    comps_next = {k: reduce_action(v, a_next) for k, v in comps_next.items()}
            # Bootstrap diagnostics need the action at t+H as well as t, so
            # retain the complete sequence rather than the TD-pair slice.
            full_actions = jaxutils.onehot_dict(
                {k: data[k] for k in self.act_space}, self.act_space)
            full_inputs = self._q_input(self._take_dyn_head(replay_outs), full_actions)
            metrics.update(self.report_residual_diagnostics(
                data, full_inputs, full_actions, gamma))
        else:
            return metrics

        con = (1.0 - f32(data['is_terminal'][:, 1:]))[None]
        prior_delta = gamma * con * prior_next - prior_cur
        metrics['prior_stats/mean_q'] = q_cur.mean()
        metrics['prior_stats/mean_q_std'] = self.safe_std(q_cur, axis=0).mean()
        metrics['prior_stats/mean_prior_q'] = prior_cur.mean()
        metrics['prior_stats/mean_prior_std'] = self.safe_std(prior_cur, axis=0).mean()
        metrics['prior_stats/mean_prior_delta'] = prior_delta.mean()
        # Per-component magnitudes (mean and across-head std at the (s, a) selected above).
        if comps_cur is not None:
            for k in ('raw', 'corrector', 'prior_corrector', 'residual_bootstrap'):
                if k not in comps_cur:
                    continue
                v = comps_cur[k]
                metrics[f'prior_stats/{k}_abs_mean'] = jnp.abs(v).mean()
                metrics[f'prior_stats/{k}_head_std'] = self.safe_std(v, axis=0).mean()
        return metrics

    def report_residual_diagnostics(self, data, inputs, actions, gamma):
        """Paired c+g cumulants and bootstrap fit on recorded trajectories."""
        # Diagnose additive critics separately; the double-Q mix is nonlinear.
        heads = self.q.heads if hasattr(self.q, 'heads') else [self.q]
        horizons = [h for h in (1, 2, 4, 8, 16, 32, 64, 128, 200)
                    if h < data['is_first'].shape[1]]
        if getattr(heads[0], 'prior_corrector', None) is None or not horizons:
            return {}
        parts = self._report_residual_particles(heads[0], data, inputs, actions)
        metrics = {}
        mode = getattr(self, '_td_target_mode', self.config.td_target_mode)
        coverage = getattr(getattr(self.config, 'run', None), 'log_coverage', False)
        if mode in ('q_full', 'rnd') and 'bootstrap' in parts:
            for i, q in enumerate(heads):
                values = parts if i == 0 else self._report_residual_particles(
                    q, data, inputs, actions)
                suffix = '' if i == 0 else f'/q{i + 1}'
                metrics.update(bootstrap_fit_metrics(
                    values['residual'], values['bootstrap'], data, gamma,
                    mode, horizons, shift=self.config.residual_transform,
                    prefix=f'rb_fit{suffix}',
                    sample_prefix=f'_rb{suffix}' if coverage else None))
            if coverage:
                metrics['_rb/discount'] = f32(gamma)

        # Preserve the existing propagation metrics' current-state origin and
        # configured scales. The new RB diagnostics above start at t+1.
        delta = parts['bonus_residual'][:, :, :horizons[-1]]
        # Exclude reset crossings and terminal rows. Synthetic state/action
        # tables mark every observation as is_first and have no valid prefixes.
        live = (~data['is_last'][:, :-1] & ~data['is_terminal'][:, :-1]
                & ~data['is_first'][:, 1:])
        finite = jnp.isfinite(delta).all(0)
        # Invalid steps are excluded below; sanitize them without erasing the
        # usable prefix of a sequence that fails only at a later horizon.
        delta = jnp.where(finite[None], delta, 0.0)
        weights = f32(gamma) ** jnp.arange(delta.shape[2], dtype=f32)
        signed = jnp.cumsum(delta * weights, axis=2).std(0)
        absolute = jnp.cumsum(jnp.abs(delta).mean(0) * weights, axis=1)
        step_std = jnp.cumsum(delta.std(0) * weights, axis=1)


        metrics['bonus_propagation/particles'] = f32(delta.shape[0])
        for h in horizons:
            # Each horizon uses its own uninterrupted, finite prefixes.
            # Counts can decrease with H; shorter horizons need not be empty
            # just because no sequence reaches the longest horizon.
            live_h = live[:, :h].all(1)
            finite_h = finite[:, :h].all(1)
            valid = live_h & finite_h
            s, a, ref = signed[:, h - 1], absolute[:, h - 1], step_std[:, h - 1]
            for name, value in (('signed_std', s), ('absolute_mean', a), ('sum_step_std', ref)):
                metrics[f'bonus_propagation/{name}_h{h}'] = _masked_mean(value, valid)
            metrics[f'bonus_propagation/valid_sequences_h{h}'] = f32(valid.sum())
            metrics[f'bonus_propagation/boundary_sequences_h{h}'] = f32((~live_h).sum())
            metrics[f'bonus_propagation/nonfinite_sequences_h{h}'] = f32((~finite_h).sum())
            # Relative resolution floor; do not turn zero denominators into
            # artificial finite ratios or use safe_std's variance floor/clip.
            tol = jnp.finfo(f32).eps * jnp.maximum(a, ref)
            for name, num, den in (('signed_retention', s, ref), ('kappa', a, s)):
                ratio_valid = valid & (den > tol)
                ratio = num / jnp.where(ratio_valid, den, 1.0)
                metrics[f'bonus_propagation/{name}_h{h}'] = _masked_mean(ratio, ratio_valid)
                metrics[f'bonus_propagation/valid_{name}_h{h}'] = f32(ratio_valid.sum())
        return metrics

    def _report_residual_particles(self, q, data, inputs, actions):
        """Pair raw residual/RB particles; keep z fixed across each trajectory."""
        def components(context):
            if context is not None:
                context = jnp.broadcast_to(
                    context, data['is_first'].shape + (q.epistemic_dim,))
            parts = q.component_means(
                inputs, bdims=2, epistemic=context,
                prior_scale=1.0, corrector_scale=1.0)
            prior, corrector = f32(parts['prior']), f32(parts['corrector'])
            pscale, cscale = q._effective_scales()
            values = {
                'residual': prior + corrector,
                'bonus_residual': pscale * prior + cscale * corrector,
            }
            if 'residual_bootstrap' in parts:
                values['bootstrap'] = f32(parts['residual_bootstrap'])
            return {k: self._select_action_values(v, actions)
                    for k, v in values.items()}  # [E, B, T]

        if q.epistemic_dim:
            # Local constant seed: same bank across states/time/reports, no
            # consumption of the training RNG. Map to bound intermediate memory.
            contexts = jax.random.uniform(
                jax.random.PRNGKey(0), (64, q.epistemic_dim), dtype=f32,
                minval=-q.epistemic_std, maxval=q.epistemic_std)
            parts = jax.lax.map(components, contexts)
            return {k: v.reshape((-1,) + v.shape[2:]) for k, v in parts.items()}
        return components(None)  # Enumerate the existing ensemble exactly.

    def report_prior_target_metrics(self, q, q_states_unh, q_actions_unh, next_states_unh, next_acts, discount, qctx_unh):
        """Diagnostic stats on the frozen prior. No effect on training.
        Inputs are all unheaded; next_acts may be [H, B, T, ...] (from actor) or [B, T, ...] (candidate)."""
        if hasattr(q, 'heads'):  # DoubleQCritic wrapper → use the primary head's prior
            q = q.heads[0]
        if (not isinstance(q, dr_nets.PriorCritic) or q.prior_scale <= 0 or
                q._prior is None or q_states_unh is None or q_actions_unh is None):
            return {}
        scale = q.prior_scale
        qkw = {} if qctx_unh is None else {'epistemic': qctx_unh}
        # next_acts may have leading H axis from _actor_step_train; take first head for the report.
        next_acts_unh = treemap(lambda x: x[0] if x.ndim >= 4 else x, next_acts) if next_acts else None
        inp_cur = self._q_input(q_states_unh, q_actions_unh) if not self._is_discrete else self._q_input(q_states_unh)
        inp_next = self._q_input(next_states_unh, next_acts_unh) if (not self._is_discrete and next_acts_unh is not None) else self._q_input(next_states_unh)
        prior_cur = sg(q.component_means(inp_cur, bdims=2, has_ensemble=False, **qkw)['prior'] / scale)
        prior_next = sg(q.component_means(inp_next, bdims=2, has_ensemble=False, **qkw)['prior'] / scale)
        if self._is_discrete and next_acts_unh is not None:
            key = next(iter(self.act_space.keys()))
            if prior_cur.shape[-1] == q_actions_unh[key].shape[-1]:
                prior_cur = (prior_cur * q_actions_unh[key]).sum(-1)
            if prior_next.shape[-1] == next_acts_unh[key].shape[-1]:
                prior_next = (prior_next * next_acts_unh[key]).sum(-1)
        prior_delta = scale * (prior_next - prior_cur / discount)
        return {
            'q_prior_mean': prior_cur.mean(),
            'q_prior_delta_mean': prior_delta.mean(),
            'q_prior_delta_std': prior_delta.std(),
        }


def _masked_mean(value, valid):
    count = valid.sum()
    mean = jnp.where(valid, value, 0).sum() / jnp.maximum(count, 1)
    return jnp.where(count > 0, mean, jnp.nan)


def bootstrap_fit_metrics(
        residual, bootstrap, data, discount, mode, horizons, shift=0.0,
        prefix='rb_fit', sample_prefix=None):
    """Compare moments over paired [P, B, T] particles at every valid start.

    The H-step return starts at t+1 with weight gamma. Its companion adds
    gamma**H * rb[t+H] and subtracts the configured downshift at each live
    step, measuring consistency with the continuing training objective.
    Terminals close returns; timeouts, resets, and missing futures do not.
    """
    if mode not in ('q_full', 'rnd'):
        raise ValueError(mode)
    residual, bootstrap = map(jnp.float32, (residual, bootstrap))
    assert residual.shape == bootstrap.shape
    assert residual.ndim == 3
    assert residual.shape[1:] == data['is_first'].shape
    if not horizons:
        return {}
    assert min(horizons) > 0 and max(horizons) < residual.shape[-1]

    def next_step(x):
        return jnp.concatenate([x[..., 1:], jnp.zeros_like(x[..., :1])], -1)


    moment_name = 'var' if mode == 'q_full' else 'mean'
    moment = lambda x: x.var(0) if mode == 'q_full' else x.mean(0)
    source = jnp.abs(residual) if mode == 'rnd' else residual
    finite_source = jnp.isfinite(source).all(0)
    finite_bootstrap = jnp.isfinite(bootstrap).all(0)
    # Sanitize before the scan; validity tracks excluded values separately so
    # a NaN late in a sequence cannot contaminate its usable short windows.
    source = jnp.where(jnp.isfinite(source), source, 0)
    bootstrap = jnp.where(jnp.isfinite(bootstrap), bootstrap, 0)
    prediction = moment(bootstrap)
    prediction_valid = finite_bootstrap & jnp.isfinite(prediction)

    first = jnp.asarray(data['is_first'], bool)
    last = jnp.asarray(data['is_last'], bool)
    terminal = jnp.asarray(data['is_terminal'], bool)
    next_terminal = next_step(terminal)
    known_step = ~last & ~terminal & next_step(~first)
    usable_source = next_step(finite_source & ~last)
    source_next = next_step(source)

    def advance(_, carry):
        returns, completed, valid, valid_completed = carry
        returns = discount * jnp.where(
            next_terminal[None], 0, source_next + next_step(returns))
        completed = discount * jnp.where(
            next_terminal[None], 0,
            source_next - shift + next_step(completed))
        valid = known_step & (
            next_terminal | (usable_source & next_step(valid)))
        valid_completed = known_step & (
            next_terminal | (usable_source & next_step(valid_completed)))
        return returns, completed, valid, valid_completed

    def summarize(carry, steps):
        carry = jax.lax.fori_loop(0, steps, advance, carry)
        returns, completed, valid, valid_completed = carry
        target, tail_target = moment(returns), moment(completed)
        error = jnp.abs(prediction - target)
        tail_error = jnp.abs(prediction - tail_target)
        keep = valid & prediction_valid & jnp.isfinite(target) & jnp.isfinite(error)
        keep_tail = (valid_completed & prediction_valid
                     & jnp.isfinite(tail_target) & jnp.isfinite(tail_error))
        stats = {
            f'{moment_name}_abs_error': _masked_mean(error, keep),
            f'predicted_{moment_name}': _masked_mean(prediction, keep),
            f'return_{moment_name}': _masked_mean(target, keep),
            f'{moment_name}_consistency_abs_error': _masked_mean(tail_error, keep_tail),
            f'completed_return_{moment_name}': _masked_mean(tail_target, keep_tail),
            'valid_starts': jnp.float32(keep.sum()),
            'valid_consistency_starts': jnp.float32(keep_tail.sum()),
        }
        if sample_prefix is not None:
            stats['valid'] = keep
            stats['return_moment'] = target
        return (returns, completed, valid, valid_completed), stats

    initial = (jnp.zeros_like(source), bootstrap,
               jnp.ones_like(first), finite_bootstrap)
    # Advance every recurrence step, but only reduce/store requested horizons.
    ordered = sorted(set(horizons))
    indices = {h: i for i, h in enumerate(ordered)}
    gaps = jnp.asarray(np.diff([0] + ordered), jnp.int32)
    _, stats = jax.lax.scan(summarize, initial, gaps)
    metrics = {
        f'{prefix}/{key}_h{h}': values[indices[h]]
        for key, values in stats.items() if key not in ('valid', 'return_moment')
        for h in horizons}
    metrics[f'{prefix}/particles'] = jnp.float32(residual.shape[0])
    if sample_prefix is not None:
        metrics[f'{sample_prefix}/{moment_name}'] = jnp.where(
            prediction_valid, prediction, jnp.nan)
        metrics.update({f'{sample_prefix}/valid_h{h}': stats['valid'][indices[h]]
                        for h in horizons})
        metrics.update({f'{sample_prefix}/return_moment_h{h}': stats['return_moment'][indices[h]]
                        for h in horizons})
    return metrics
