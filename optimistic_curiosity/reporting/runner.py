"""Host reporting: replay batches, episodes, coverage tracking, logging, and cadence."""

import re
from collections import defaultdict
from functools import partial as bind
import embodied
import numpy as np

from .novelty import _coverage_residual_metrics


class Reporter:
    """Host-side report data, aggregation and output for a single run.

    Keep log_step and coverage_step separate: existing drivers observe them in
    different orders relative to replay insertion. Returned agent report carry
    is intentionally discarded, matching independent replay-window reports.
    Coverage remains process-local and is not added to agent checkpoints.
    """

    def __init__(self, agent, logger, args, *, policy_eval=False):
        self.agent, self.logger, self.args = agent, logger, args
        self.step = logger.step
        self.logdir = embodied.Path(args.logdir)
        self.policy_eval = policy_eval
        self.coverage = None
        self.project = None
        self.agg = embodied.Agg()
        self.episodes = {mode: defaultdict(embodied.Agg)
                         for mode in ('train', 'eval', 'explore')}
        self.epstats = {mode: embodied.Agg() for mode in self.episodes}
        self.cum_counts = {mode: [0, 0] for mode in self.episodes}
        self.big_report_every = (0 if policy_eval else
                                 int(getattr(args, 'big_report_every', 0)))
        self.big_report_count = 0
        self.big_report_tables = None

    def init_replays(self, **replays):
        self.replays = replays
        self.datasets = {
            mode: self.agent.dataset(bind(
                replay.dataset, self.args.batch_size,
                self.args.batch_length_eval, 'ac'))
            for mode, replay in replays.items() if replay is not None}

    def reset_carry(self):
        self.carry = self.agent.init_report(self.args.batch_size)

    def report_replay(self, mode):
        replay = self.replays[mode]
        if replay is None or not len(replay):
            return
        if not self.policy_eval:
            self.agent.set_global_step(self.step)
        batch = next(self.datasets[mode])
        metrics, _ = self.agent.report(batch, self.carry)
        metrics = _coverage_residual_metrics(metrics, batch, self.coverage)
        if self.big_report_every > 0:
            metrics = {k: v for k, v in metrics.items() if '_table__' not in k}
        prefix = 'report' if mode == 'train' else mode
        self.logger.add(metrics, prefix=prefix)
        if mode == 'train':
            self._maybe_big_report()

    def add_train_metrics(self, metrics):
        if self.policy_eval:
            self.logger.add(metrics, prefix='train')
        else:
            self.agg.add(metrics, prefix='train')

    def flush_train(self):
        self.logger.add(self.agg.result())

    def flush_episodes(self, mode):
        self.logger.add(self.epstats[mode].result(), prefix=f'{mode}_epstats')

    def log_coverage(self):
        if self.coverage is not None:
            self.logger.add(self.coverage.stats(), prefix='coverage')
            if getattr(self.args, 'log_coverage_heatmap', False):
                self.logger.add({'heatmap': self.coverage.heatmap()}, prefix='coverage')

    def init_coverage(self, make_env, env=None):
        if not getattr(self.args, 'log_coverage', False):
            return
        def probe(env):
            if not hasattr(env, 'get_coverage_geometry'):
                return None
            try:
                return env.get_coverage_geometry(bins_per_cell=int(
                    getattr(self.args, 'coverage_bins_per_cell', 1) or 1))
            except TypeError:
                return env.get_coverage_geometry()

        geometry = None
        if env is not None:
            geometry = probe(env)
        elif self.policy_eval:
            # Policy evaluation historically propagates probe failures.
            probe_env = None
            try:
                probe_env = make_env(0)
                geometry = probe(probe_env)
            finally:
                if probe_env is not None:
                    probe_env.close()
        else:
            try:
                probe_env = make_env(0)
                geometry = probe(probe_env)
                if hasattr(probe_env, 'close'):
                    probe_env.close()
            except Exception as e:
                print(f'Coverage tracking: failed to probe env ({e}); disabled.')
        if geometry is not None:
            self.coverage = embodied.make_coverage(geometry)
            self.project = geometry['project']
            if self.policy_eval:
                print(f'[policy_eval] coverage enabled over {self.coverage.axis_names}: '
                      f'{int(self.coverage.valid_mask.sum())} valid bins')
            else:
                print(f'Coverage tracking enabled over {self.coverage.axis_names}: '
                      f'{int(self.coverage.valid_mask.sum())} valid bins.')
        elif self.policy_eval:
            print('[policy_eval] env exposes no coverage geometry; novelty reports disabled')
        else:
            print('Coverage tracking: env exposes no geometry; disabled.')

    def attach_trajectories(self, driver, mode):
        if not getattr(self.args, f'{mode}_trajectory_dir', False):
            return
        from embodied.core.trajectory_recorder import TrajectoryRecorder, state_obs_keys
        recorder = TrajectoryRecorder(
            self.logdir / f'{mode}_trajectories', self.args.num_envs_eval,
            state_obs_keys(self.agent.obs_space),
            [key for key in self.agent.act_space if key != 'reset'],
            step_counter=self.step)
        driver.on_step(recorder)

    def attach_coverage(self, driver, *, count=False):
        if self.coverage is not None:
            driver.on_step(bind(self.coverage_step, count=count))

    def coverage_step(self, tran, worker, *, count=False):
        point = self.project(tran)
        ids = self.coverage.add(point) if count else self.coverage.bin_ids(point)
        ids = np.asarray(ids, np.int32).reshape(-1)
        tran['coverage_bin'] = ids[0] if len(ids) == 1 else np.int32(-1)

    @embodied.timer.section('log_step')
    def log_step(self, tran, worker, mode):
        episodes = self.episodes[mode]
        epstats = self.epstats[mode]

        episode = episodes[worker]
        episode.add('score', tran['reward'], agg='sum')
        episode.add('length', 1, agg='sum')
        episode.add('rewards', tran['reward'], agg='stack')

        if tran['is_first']:
            episode.reset()

        if worker < self.args.log_video_streams:
            for key in self.args.log_keys_video:
                if key in tran:
                    episode.add(f'policy_{key}', tran[key], agg='stack')
        for key, value in tran.items():
            if re.match(self.args.log_keys_sum, key):
                episode.add(key, value, agg='sum')
            if re.match(self.args.log_keys_avg, key):
                episode.add(key, value, agg='avg')
            if re.match(self.args.log_keys_max, key):
                episode.add(key, value, agg='max')
            if re.match(self.args.log_keys_min, key):
                episode.add(key, value, agg='min')

        if tran['is_last']:
            result = episode.result()
            score = result.pop('score')
            length = result.pop('length')
            if mode == 'train':
                self.logger.add({'score': score, 'length': length}, prefix='train_episode')
            else:
                # Eval: aggregate per-episode score/length into epstats (mean + std over
                # eval_eps episodes) so they get logged once per eval phase instead of
                # overwriting at the same step.
                epstats.add({'score': score, 'length': length}, agg=('avg', 'std'))
            rew = result.pop('rewards')
            if 'log_success' in result:
                success = bool(np.asarray(result['log_success']).max())
            elif len(rew) > 0:
                success = bool((np.asarray(rew) > 0).any())
            else:
                success = bool(score > 0)
            result['success'] = float(success)
            if len(rew) > 0:
                result['success_rate'] = float(success)
                self.cum_counts[mode][0] += 1
                self.cum_counts[mode][1] += int(success)
                self.logger.add({'cum_success_rate': self.cum_counts[mode][1] / self.cum_counts[mode][0]},
                                      prefix=f'{mode}_episode')
            if len(rew) > 1:
                result['reward_rate'] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
            epstats.add(result)

    def _maybe_big_report(self):
        if self.big_report_every <= 0:
            return
        self.big_report_count += 1
        if self.big_report_count % self.big_report_every:
            return
        from results_analysis import analyze_trajectories as analysis
        if self.big_report_tables is None:
            self.big_report_tables = {
                kind: getattr(analysis, f'build_{kind}_table')(self.agent.config)
                for kind in ('state', 'action')}
            for kind, table in self.big_report_tables.items():
                if table is None:
                    print(f'Big report: no environment-backed {kind} table available.')
                else:
                    rows = len(next(iter(table['obs'].values())))
                    print(f'Big report: built {kind} table with {rows} rows.')
        if not any(table is not None for table in self.big_report_tables.values()):
            return
        out_path = self.logdir / 'big_reports'
        out_path.mkdir()
        for kind, table in self.big_report_tables.items():
            if table is not None:
                getattr(analysis, f'analyze_{kind}_table')(
                    self.agent, table, self.agent.config, out_path,
                    str(int(self.step)), float(int(self.step)))

    def init_policy_episodes(self):
        # Policy-eval includes reset rewards; standard log_step excludes them.
        # Discount per agent step matches the MC reference convention.
        self.pe_discount = float(self.args.policy_eval_discount)
        # Reach through to agent.discount when available (matches TD).
        inner_agent = getattr(self.agent, 'agent', self.agent)
        self.pe_discount = float(getattr(inner_agent, 'discount', self.pe_discount))
        self.ep_undisc = np.zeros(self.args.num_envs, dtype=np.float64)
        self.ep_disc = np.zeros(self.args.num_envs, dtype=np.float64)
        self.ep_gamma = np.ones(self.args.num_envs, dtype=np.float64)
        self.ep_len = np.zeros(self.args.num_envs, dtype=np.int64)
        self.ep_stats = {'undisc_sum': 0.0, 'disc_sum': 0.0, 'len_sum': 0.0, 'count': 0}

    def policy_episode(self, tran, worker):
        r = float(tran['reward'])
        if bool(tran['is_first']):
            self.ep_undisc[worker] = 0.0
            self.ep_disc[worker] = 0.0
            self.ep_gamma[worker] = 1.0
            self.ep_len[worker] = 0
        self.ep_undisc[worker] += r
        self.ep_disc[worker] += self.ep_gamma[worker] * r
        self.ep_gamma[worker] *= self.pe_discount
        self.ep_len[worker] += 1
        if bool(tran['is_last']):
            self.ep_stats['undisc_sum'] += float(self.ep_undisc[worker])
            self.ep_stats['disc_sum'] += float(self.ep_disc[worker])
            self.ep_stats['len_sum'] += float(self.ep_len[worker])
            self.ep_stats['count'] += 1

    def flush_policy_episodes(self):
        if self.ep_stats['count'] > 0:
            n = self.ep_stats['count']
            self.logger.add({
                'policy_eval/train_return_undisc': self.ep_stats['undisc_sum'] / n,
                'policy_eval/train_return_disc': self.ep_stats['disc_sum'] / n,
                'policy_eval/train_episode_len': self.ep_stats['len_sum'] / n,
                'policy_eval/train_episodes': float(n),
            })
            self.ep_stats.update(undisc_sum=0.0, disc_sum=0.0, len_sum=0.0, count=0)

    def policy_values(self, td, mc_mean, anchor_meta):
        # Aggregate metrics: mean across heads, MSE vs MC at anchors.
        q_mean = np.nanmean(td['q'], axis=-1)
        err = q_mean - mc_mean
        self.logger.add({
            'policy_eval/q_mean_anchor': float(np.nanmean(q_mean[anchor_meta['kind'] == 'anchor'])),
            'policy_eval/mse_vs_mc': float(np.nanmean(err ** 2)),
            'policy_eval/corr_vs_mc': float(np.corrcoef(q_mean, mc_mean)[0, 1]),
        })
        self.logger.add({
            f'policy_eval/{key}_anchor': float(np.nanmean(td[key][anchor_meta['kind'] == 'anchor']))
            for key in ('base_value', 'optimism_bonus', 'optimistic_value')
            if key in td
        })
