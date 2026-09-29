"""Pure table formatting: column names, ordering, flattening, and synthetic-row masks."""

import jax.numpy as jnp

f32 = jnp.float32


class ReportTables:
    """Pure table formatting: column names, ordering, flattening, and synthetic-row masks."""

    def _calibration_table(self, data, mean, std, ret, error, prefix, blocks, table_mode=False):
        state_keys = self._state_obs_keys
        state = jnp.concatenate([f32(data[k])[:, :-1] for k in state_keys], axis=-1)
        rows = state.shape[0] * state.shape[1]
        names = [name for k in state_keys for name in
                 ([k] if data[k].shape[-1] == 1 else [f'{k}_{i}' for i in range(data[k].shape[-1])])]
        columns = [(state.reshape(rows, -1), names),
                   (mean[:, :-1].reshape(rows, 1), [f'{prefix}_mean']),
                   (std[:, :-1].reshape(rows, 1), [f'{prefix}_std'])]
        for name, value in [('return', ret), ('error', error), ('reward', data['reward'][:, :-1])]:
            value = value.reshape(rows, 1)
            if table_mode is not False:  # Q synthetic tables supply a traced predicate.
                value = jnp.where(table_mode, jnp.full((rows, 1), jnp.nan, f32), value)
            columns.append((value, [name]))
        return _table_columns(columns + blocks)

    def _ensemble_table_columns(self, values, heads, prefix, trim_last=False, *, mean=None, std=None):
        if values is None:
            return None, []
        mean = values.mean(0) if mean is None else mean
        std = values.std(0) if std is None else std
        headed = self._flatten_headed_table_features(values, heads, trim_last=trim_last)
        table = jnp.concatenate([
            self._flatten_table_feature(mean, trim_last=trim_last),
            self._flatten_table_feature(std, trim_last=trim_last), headed], axis=-1)
        names = [f'{prefix}_mean', f'{prefix}_std']
        names += [f'{prefix}_head_{i}' for i in range(headed.shape[-1])]
        return table, names

    def _flatten_table_feature(self, tensor, trim_last=False):
        """Flatten ``[B, T, ...]`` features into ``[B * T, F]`` rows."""
        if trim_last:
            tensor = tensor[:, :-1]
        return f32(tensor).reshape(tensor.shape[0] * tensor.shape[1], -1)

    def _flatten_headed_table_features(self, tensor, heads, trim_last=False):
        """Flatten ``[heads, B, T, ...]`` features into ``[B * T, F]`` rows."""
        if not self._has_head_axis(tensor, heads):
            tensor = tensor[None]
        if trim_last:
            tensor = tensor[:, :, :-1]
        tensor = jnp.moveaxis(tensor, 0, 2)  # [B, T, heads, ...]
        return tensor.reshape(tensor.shape[0] * tensor.shape[1], -1)

    def _broadcast_table_metric(self, value, rows):
        value = f32(value)
        if value.ndim == 0:
            value = value[None]
        value = value.reshape(1, -1)
        return jnp.broadcast_to(value, (rows, value.shape[-1]))

    def _actor_distribution_columns_from_dist(
            self, actor_dist, heads, prefix, trim_last=False,
            tensor_transform=lambda x: x):
        """Format actor distributions; only the distribution readout differs."""
        blocks = []
        for key, space in self.act_space.items():
            dist = actor_dist[key]
            stats = (('prob', dist.probs_parameter()),) if space.discrete else (
                ('mean', dist.mean()), ('std', dist.stddev()))
            for stat, tensor in stats:
                flat = self._flatten_headed_table_features(
                    tensor_transform(tensor), heads, trim_last=trim_last)
                per_head = flat.shape[-1] // heads
                if per_head * heads != flat.shape[-1]:
                    raise ValueError(f'Cannot split actor {stat} for {key}: shape={flat.shape}, heads={heads}.')
                names = [f'{prefix}_{key}_head{head}_{stat}_{dim}'
                         for head in range(heads) for dim in range(per_head)]
                blocks.append((flat, names))
        return _table_columns(blocks)

    def _reward_prediction_table_columns(self, rew_pred, trim_last=False):
        if rew_pred is None:
            return None, []
        if self._has_head_axis(rew_pred, self.dyn_heads):
            rew_mean = rew_pred.mean(0)
            rew_std = rew_pred.std(0)
        else:
            rew_mean = rew_pred
            rew_std = jnp.zeros_like(rew_pred)
        table = jnp.concatenate([self._flatten_table_feature(rew_mean, trim_last=trim_last),self._flatten_table_feature(rew_std, trim_last=trim_last),], axis=-1)
        return table, ['reward_pred_mean', 'reward_pred_std']

    def _intrinsic_reward_table_columns(self, prediction, alpha=None, trim_last=False):
        if prediction is None:
            return None, []
        intr_reward, disg_metrics = prediction
        blocks = [self._flatten_table_feature(intr_reward, trim_last=trim_last)]
        col_names = ['intr_reward']
        rows = blocks[0].shape[0]
        for key in sorted(disg_metrics):
            if key.startswith('disg/norm_eps_') and key.endswith('_mean'):
                flat = self._broadcast_table_metric(disg_metrics[key], rows)
                blocks.append(flat)
                col_names.append(key)

        if alpha is not None:
            weighted = alpha * intr_reward
            blocks.append(self._flatten_table_feature(weighted, trim_last=trim_last))
            col_names.append('weighted_intr_reward')
        return jnp.concatenate(blocks, axis=-1), col_names

    def _q_component_table_columns(self, comp_samples, trim_last=False):
        if comp_samples is None:
            return None, []
        heads = self.value_heads
        blocks = []
        names = [
            ('raw_q', 'raw'),
            ('prior_q', 'prior'),
            ('corrector_q', 'corrector'),
            ('prior_corrector_q', 'prior_corrector'),
            ('mixed_q', 'mixed'),
        ]
        if 'residual_bootstrap' in comp_samples:
            names.append(('residual_bootstrap_q', 'residual_bootstrap'))
        for prefix, key in names:
            stacked = comp_samples[key]  # [S, H, B, T]
            flat = stacked.reshape(
                (stacked.shape[0] * stacked.shape[1],) + stacked.shape[2:])
            mean = flat.mean(0)
            std = self.safe_std(flat, axis=0)
            per_head = stacked.mean(0)
            blocks.append(self._ensemble_table_columns(per_head, heads, prefix, trim_last, mean=mean, std=std))
        return _table_columns(blocks)


def _table_columns(blocks):
    blocks = [(values, names) for values, names in blocks if values is not None]
    if not blocks:
        return None, []
    return jnp.concatenate([values for values, _ in blocks], axis=-1), [
        name for _, names in blocks for name in names]
