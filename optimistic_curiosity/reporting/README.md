# Reporting: where to find things

Start at [`../report.py`](../report.py). It contains the agent report entry point,
Observer/Dreamer diagnostic order, and the context shared within one report.
`ReportMixin` composes the metric groups below. Existing imports of `Reporter`,
`ReportMixin`, and `bootstrap_fit_metrics` from `optimistic_curiosity.report`
continue to work.

| File | Owns | Metrics / main entry points |
| --- | --- | --- |
| [`runner.py`](runner.py) | Host state, replay datasets, episode aggregation, coverage callbacks, trajectory recording, big-report cadence, logger prefixes | `Reporter`; `train_episode/*`, `*_epstats/*`, `coverage/*`, `policy_eval/*` |
| [`world_model.py`](world_model.py) | Prediction losses, videos, gradient norms, distribution/head reshaping | `openl_*`, `openloop/*`, `gradnorm/*` |
| [`calibration.py`](calibration.py) | Q/value calibration, shared value predictions, optional prediction evaluation for calibration tables | `sac/q_*`, `sac/cal_err_std_cor`, `sac/actor_obj_*`, `dreamer/val_*` |
| [`policy.py`](policy.py) | Actor distributions, policy KL, exploration objectives, intrinsic-reward evaluation, reward mixing, actor/critic plasticity | `policy_kl/*`, `exp_obj/*`, `dreamer/mixing/*`, `plasticity/*` |
| [`residual.py`](residual.py) | Prior statistics, paired residual/RB particles, bootstrap-fit recurrence, bonus propagation | `prior_stats/*`, `rb_fit/*`, `bonus_propagation/*`; `bootstrap_fit_metrics()`, `report_prior_target_metrics()` |
| [`novelty.py`](novelty.py) | State density, private residual samples, host joins with coverage counts, novelty returns and rank comparisons | `state_density/*`, `coverage_novelty/*`, `rb_novelty/*`; `_coverage_residual_metrics()` |
| [`residual_mc.py`](residual_mc.py) | Matched MC returns and conditional moments; three-link metrics | `policy_eval/mc/*`; host lifecycle in `embodied/run/policy_eval_diagnostics.py` |
| [`tables.py`](tables.py) | Pure column formatting, names/order, flattening and synthetic-table masks; no model evaluations | Calibration tables and actor/component columns |

## Pipeline

1. **Collect:** run loops schedule work and send transitions to `Reporter`.
   It owns episode/report state and the coverage tracker. Coverage callbacks
   attach stable `coverage_bin` IDs before replay insertion. The generic tracker
   implementation remains in `embodied/core/coverage.py`.
2. **Sample:** `Reporter.report_replay()` obtains a replay batch and calls the
   agent's existing report API.
3. **Evaluate:** `ReportMixin.report()` collects loss diagnostics, optional
   world-model outputs and gradient norms. The Observer/Dreamer profile then
   shares one replay reconstruction across its diagnostic groups.
4. **Summarize:** each metric family computes its own scalars. Calibration
   evaluates optional table predictions and passes their outputs to the pure
   formatters in `tables.py`.
5. **Join novelty:** on the host, `novelty.py` combines `_rb/*` and
   `_coverage/*` diagnostic samples with replay bin IDs and one current coverage
   snapshot. These private arrays are removed before logging.
6. **Publish:** `Reporter` applies the existing report prefixes, table cadence,
   and episode aggregation, then writes through the logger.

## Boundaries to preserve

- Metric definitions belong to their family module. Logger routing and report
  lifetime belong to `runner.py`; selection/order belong to `report.py`.
- The report context is local to a call. It holds raw and augmented replay
  states plus compatible cached predictions. Dreamer value/prior evaluations
  use raw states; Observer actor evaluations use augmented states.
- Training/loss evaluation, imagined rollouts, different action choices and
  different Q mixing/scaling semantics retain their own evaluations. Shared
  replay samples do not justify merging these distinct quantities.
- The groups are stateless mixins, not separately instantiated agents or Ninjax
  modules. Keep the existing `BaseAgent.report` and agent-profile bindings so
  module scope and checkpoint paths remain stable.
- Implementation modules must not import `report.py`. Calls between diagnostic
  methods use the composed agent; host orchestration imports novelty helpers
  directly. There is no plugin registry or dynamic dispatch framework.
- Tests and standalone checks stay under the repository's `tests/` directory.
  See [`tests/README.md`](../../tests/README.md), or run the commands below.

## Validation

Run runtime checks in the dependency-equipped JAX environment:

```sh
python -m pytest tests/optimistic_curiosity/test_report*.py -q
```

Static checks do not import or execute project functionality:

```sh
python tests/checks/check_layout.py
python tests/checks/check_reporting.py
```

Policy-evaluation treatments, metric interpretation and saved artifacts are documented
in [`policy_eval/README.md`](../../policy_eval/README.md).
