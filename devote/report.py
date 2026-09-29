"""Reporting entry points and report-local replay context.

Implementation lives in reporting/: runner, world_model, calibration, policy,
residual, novelty, and tables. See reporting/README.md for metric ownership.
"""

import embodied
import jax.numpy as jnp
from dreamerv3 import jaxutils

from .reporting.runner import Reporter
from .reporting.world_model import WorldModelReports, ReshapedBatchDist
from .reporting.calibration import CalibrationReports
from .reporting.policy import PolicyReports, _categorical_kl
from .reporting.residual import ResidualReports, _masked_mean, bootstrap_fit_metrics
from .reporting.novelty import (
    NoveltyReports, _nearest_neighbor_distances, _coverage_residual_metrics,
    _coverage_bootstrap_metrics, _coverage_inputs, _report_batch_array,
    _pairwise_auc, _visitation_rank_auc, _average_ranks, _spearman_rank_corr,
)
from .reporting.tables import ReportTables, _table_columns


__all__ = ['Reporter', 'ReportMixin', 'ReshapedBatchDist', 'bootstrap_fit_metrics']


class ReportMixin(WorldModelReports, CalibrationReports, PolicyReports,
                  ResidualReports, NoveltyReports, ReportTables):
    """Compose report families without changing the agent's Ninjax scope.

    Diagnostics share a report-local replay sample and compatible evaluations.
    Training/loss and imagination keep their own samples. No cached arrays are
    stored on the agent or survive the call.
    """

    def report(self, data, carry):
        self.config.jax.jit and embodied.print('Tracing report function', color='yellow')
        if not self.config.report:
            return {}, carry
        metrics = {}
        data = self.preprocess(data)
        # loss expects a dict-of-batches; in report we reuse the same batch for both.
        _, (outs, carry_out, mets) = self.loss({'ac': data, 'res': data}, carry, update=False)
        metrics.update(mets)
        # Open-loop prediction / videos / per-step prediction losses all need the
        # decoder and the dyn `imagine` path. When the world model isn't trained
        # (e.g. Observer with ac_inputs='obs' and a non-WM exp_obj) those modules
        # have no params, so skip the whole block.
        if getattr(self, 'train_world_model', True):
            rec, img, num_obs = self.openloop_predict(data, carry, outs)
            metrics.update(self.report_prediction_losses(img, data, num_obs))
            metrics.update(self.report_videos(data, rec, img, num_obs))
        if self.config.report_gradnorms:
            metrics.update(self.report_gradnorms(data, carry))
        metrics.update(self.extra_report_metrics(data, carry))
        return metrics, carry_out

    def _report_observer(self, data, carry):
        inputs = self._report_inputs(data, carry)
        metrics = self.report_q_calibration(data, carry, self.discount, inputs=inputs)
        metrics.update(self.report_prior_stats(data, carry, inputs=inputs))
        metrics.update(self.report_policy_kl(data, carry, inputs=inputs))
        metrics.update(self.report_state_diagnostics(data, carry, inputs=inputs))
        metrics.update(self.report_exp_objective(data, carry, inputs=inputs))
        metrics.update(self.report_plasticity(data, carry, inputs=inputs))
        return metrics

    def _report_dreamer(self, data, carry):
        inputs = self._report_inputs(data, carry)
        metrics = self.report_value_calibration(data, carry, gamma=1 if self.config.contdisc else 1 - 1 / self.config.horizon, inputs=inputs)
        metrics.update(self.report_mixing(data, carry, inputs=inputs))
        metrics.update(self.report_policy_kl(data, carry, inputs=inputs))
        metrics.update(self.report_prior_stats(data, carry, inputs=inputs))
        metrics.update(self.report_state_diagnostics(data, carry, inputs=inputs))
        metrics.update(self.report_exp_objective(data, carry, inputs=inputs))
        return metrics

    def _report_inputs(self, data, carry):
        prevacts = {k: jnp.concatenate([carry[1][k][:, None], data[k][:, :-1]], 1)
                    for k in self.act_space}
        prevacts = jaxutils.onehot_dict(prevacts, self.act_space)
        embed = self.enc(data)
        _, outs = self._dyn_observe(carry[0], prevacts, embed, data['is_first'])
        ac_outs = (self._augment_ac_inputs(outs, data)
                   if getattr(self, 'ac_inputs', 'wm') == 'obs' or getattr(self, 'visual_prior', False)
                   else outs)
        return dict(embed=embed, outs=outs, ac_outs=ac_outs, evaluated={})

    def _observe_report(self, data, carry, inputs=None, *, actor_inputs=False):
        inputs = self._report_inputs(data, carry) if inputs is None else inputs
        return inputs['embed'], inputs['ac_outs' if actor_inputs else 'outs']

    @staticmethod
    def _report_cached(inputs, key, evaluate):
        if inputs is None:
            return evaluate()
        cache = inputs['evaluated']
        if key not in cache:
            cache[key] = evaluate()
        return cache[key]


# Compatibility for callers that imported this helper directly.
_report_cached = ReportMixin._report_cached
