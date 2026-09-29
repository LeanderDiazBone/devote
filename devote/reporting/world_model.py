"""Prediction losses, open-loop videos, gradient norms, and distribution reshaping."""

import jax.numpy as jnp
import optax
import jax
from dreamerv3 import jaxutils
from dreamerv3 import ninjax as nj

f32 = jnp.float32
treemap = jax.tree_util.tree_map


class WorldModelReports:
    """Prediction losses, open-loop videos, gradient norms, and distribution reshaping."""

    def report_prediction_losses(self, img, data, num_obs):
        """Compute open-loop losses after the num_obs context steps."""
        metrics = {}
        data_img = {k: v[:, num_obs:] for k, v in data.items()}
        losses = {
            k: self._reduce_openloop_heads(
                -v.log_prob(self._align_openloop_target(v, data_img[k])))
            for k, v in img.items()
        }
        metrics.update({f'openl_{k}_loss': v.mean() for k, v in losses.items()})

        for key, threshold in [('reward', 0.1), ('cont', 0.5)]:
            stats = self._balance_stats_from_loss(losses[key], img[key].mean(), data_img[key], threshold)
            metrics.update({f'openl_{key}_{name}': value for name, value in stats.items()})
        return metrics

    def report_videos(self, data, rec, img, num_obs):
        """Build video grids of observations, predictions, and their errors."""
        if not getattr(self.config, 'report_videos', True):
            return {}
        metrics = {}
        for key in self.dec.imgkeys:
            true = f32(data[key][:6])
            pred = jnp.concatenate(
                [
                    self._reduce_openloop_heads(rec[key].mode(), 'first')[:6],
                    self._reduce_openloop_heads(img[key].mode(), 'first')[:6],
                ], 1)
            error = (pred - true + 1) / 2
            video = jnp.concatenate([true, pred, error], 2)
            metrics[f'openloop/{key}'] = jaxutils.video_grid(video)
        return metrics

    def report_gradnorms(self, data, carry):
        """Compute global gradient norms for available loss terms."""
        metrics = {}
        for key in self.scales:
            try:
                lossfn = lambda data, carry: self.loss({'ac': data, 'res': data}, carry, update=False)[1][0][f'{key}_loss'].mean()
                grad = nj.grad(lossfn, self.modules)(data, carry)[-1]
                metrics[f'gradnorm/{key}'] = optax.global_norm(grad)
            except KeyError:
                print(f'Skipping gradnorm summary for missing loss: {key}')
        return metrics

    def _align_openloop_target(self, dist, target):
        """Match report targets to distribution batch shape, including head axes."""
        target = f32(target)
        batch_shape = tuple(dist.batch_shape)
        target_batch = target.shape[:len(batch_shape)]
        if batch_shape == target_batch:
            return target

        heads = int(getattr(self, 'dyn_heads', 1))
        target_h = self._broadcast_heads(target, heads)
        target_h_batch = target_h.shape[:len(batch_shape)]
        if batch_shape == target_h_batch:
            return target_h

        flat_target_h = target_h.reshape((heads * target.shape[0],) + target.shape[1:])
        flat_batch = flat_target_h.shape[:len(batch_shape)]
        if batch_shape == flat_batch:
            return flat_target_h

        raise ValueError(
            f'Cannot align open-loop target shape {target.shape} with '
            f'distribution batch shape {batch_shape}.')

    def _reduce_openloop_heads(self, tensor, reducer='mean'):
        """Collapse a leading dynamics-head axis when present."""
        heads = int(getattr(self, 'dyn_heads', 1))
        if self._has_head_axis(tensor, heads):
            if reducer == 'mean':
                return tensor.mean(0)
            if reducer == 'first':
                return tensor[0]
            raise NotImplementedError(reducer)
        return tensor

    def _flatten_openloop_decoder_input(self, lat):
        """Flatten headed [E, B, T, ...] latents for the non-ensemble decoder."""
        heads = int(getattr(self, 'dyn_heads', 1))
        first = next(iter(lat.values()))
        if not self._has_head_axis(first, heads):
            return lat, None
        batch_shape = first.shape[:3]
        flat = treemap(
            lambda x: x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:])
            if self._has_head_axis(x, heads) and x.ndim >= 2 else x,
            lat,
        )
        return flat, batch_shape

    def _decode_openloop(self, lat):
        """Decode open-loop latents while preserving a headed batch view."""
        dec_in, batch_shape = self._flatten_openloop_decoder_input(lat)
        dists = self.dec(dec_in, bdims=2)
        if batch_shape is None:
            return dists
        return {k: ReshapedBatchDist(v, batch_shape) for k, v in dists.items()}

    def _balance_stats_from_loss(self, loss, pred, target, thres):
        """Replica of ``balance_stats`` for precomputed loss/pred tensors."""
        target = f32(target)
        loss = f32(self._reduce_openloop_heads(loss, 'mean'))
        pred = f32(self._reduce_openloop_heads(pred, 'mean'))
        pos = (target > thres).astype(f32)
        neg = (target <= thres).astype(f32)
        pred_pos = (pred > thres).astype(f32)
        return dict(
            pos_loss=(loss * pos).sum() / pos.sum(),
            neg_loss=(loss * neg).sum() / neg.sum(),
            pos_acc=(pred_pos * pos).sum() / pos.sum(),
            neg_acc=((1 - pred_pos) * neg).sum() / neg.sum(),
            rate=pos.mean(),
            avg=target.mean(),
            pred=pred.mean(),
        )


class ReshapedBatchDist:
    """View a flat-batch distribution through a higher-rank batch shape."""

    def __init__(self, dist, batch_shape):
        self._dist = dist
        self.batch_shape = tuple(batch_shape)
        self.event_shape = dist.event_shape
        if hasattr(dist, 'minent'):
            self.minent = dist.minent
        if hasattr(dist, 'maxent'):
            self.maxent = dist.maxent

    def _reshape_output(self, value):
        flat_ndims = len(tuple(self._dist.batch_shape))
        return value.reshape(self.batch_shape + value.shape[flat_ndims:])

    def mode(self):
        return self._reshape_output(self._dist.mode())

    def mean(self):
        return self._reshape_output(self._dist.mean())

    def log_prob(self, value):
        flat_batch = tuple(self._dist.batch_shape)
        value = value.reshape(flat_batch + value.shape[len(self.batch_shape):])
        logp = self._dist.log_prob(value)
        return logp.reshape(self.batch_shape + logp.shape[len(flat_batch):])

    def __getattr__(self, name):
        return getattr(self._dist, name)
