import jax.numpy as jnp

from dreamerv3 import jaxutils

f32 = jnp.float32


class MetricsCollector:
    """Gathers training metrics from losses, distributions, and activations.

    This keeps metrics bookkeeping out of the core loss functions so that
    algorithmic logic is easier to read and edit.
    """

    def __init__(self):
        self._metrics = {}

    def update(self, d):
        """Merge a dictionary of metrics."""
        self._metrics.update(d)

    def add(self, key, value):
        """Add a single metric."""
        self._metrics[key] = value

    def result(self):
        """Return the collected metrics dictionary."""
        return self._metrics

    # ------------------------------------------------------------------
    # Convenience collectors
    # ------------------------------------------------------------------

    def add_loss_stats(self, losses):
        """Record mean and std for every loss term."""
        for k, v in losses.items():
            self._metrics[f'{k}_loss'] = v.mean()
            self._metrics[f'{k}_loss_std'] = v.std()

    def add_tensorstats(self, tensor, prefix):
        """Record standard summary statistics for a tensor."""
        self._metrics.update(jaxutils.tensorstats(tensor, prefix))

    def add_reward_stats(self, data_rew, pred_rew):
        """Record data vs. predicted reward summaries."""
        self._metrics['data_rew/max'] = jnp.abs(data_rew).max()
        self._metrics['pred_rew/max'] = jnp.abs(pred_rew).max()
        self._metrics['data_rew/mean'] = data_rew.mean()
        self._metrics['pred_rew/mean'] = pred_rew.mean()
        self._metrics['data_rew/std'] = data_rew.std()
        self._metrics['pred_rew/std'] = pred_rew.std()

    def add_distribution_stats(self, dists, data):
        """Record balance statistics for reward and continuation heads."""
        if 'reward' in dists:
            stats = jaxutils.balance_stats(dists['reward'], data['reward'], 0.1)
            self._metrics.update({f'rewstats/{k}': v for k, v in stats.items()})
        if 'cont' in dists:
            stats = jaxutils.balance_stats(dists['cont'], data['cont'], 0.5)
            self._metrics.update({f'constats/{k}': v for k, v in stats.items()})

    def add_action_stats(self, acts, ents, actor, act_space):
        """Record per-action statistics including entropy and randomness."""
        for k, space in act_space.items():
            act = f32(jnp.argmax(acts[k], -1) if space.discrete else acts[k])
            self._metrics.update(jaxutils.tensorstats(f32(act), f'act/{k}'))
            if hasattr(actor[k], 'minent'):
                lo, hi = actor[k].minent, actor[k].maxent
                rand = ((ents[k] - lo) / (hi - lo)).mean(range(2, len(ents[k].shape)))
                self._metrics.update(jaxutils.tensorstats(rand, f'rand/{k}'))
            self._metrics.update(jaxutils.tensorstats(ents[k], f'ent/{k}'))

    def add_imagination_stats(self, adv, rew, weight, val, ret, roffset, rscale):
        """Record core imagination-rollout diagnostics."""
        self.add_tensorstats(adv, 'adv')
        self.add_tensorstats(rew, 'rew')
        self.add_tensorstats(weight, 'weight')
        self.add_tensorstats(val, 'val')
        self.add_tensorstats(ret, 'ret')
        self.add_tensorstats((ret - roffset) / rscale, 'ret_normed')
        self._metrics['td_error'] = jnp.abs(ret - val[:, :-1]).mean()
        self._metrics['ret_rate'] = (jnp.abs(ret) > 1.0).mean()
