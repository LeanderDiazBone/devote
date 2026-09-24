import functools

import numpy as np
import chex
import jax
from typing import Tuple, Dict, Optional, Type

from dreamerv3 import ninjax as nj
from dreamerv3.nets import MLP, SimpleEncoder, SimpleDecoder
import jax.numpy as jnp
import optax
from dreamerv3 import jaxutils
import flax
from tensorflow_probability.substrates import jax as tfp
import collections

tfd = tfp.distributions
f32 = jnp.float32
sg = lambda x: jax.tree_util.tree_map(jax.lax.stop_gradient, x)
cast = jaxutils.cast_to_compute


@functools.partial(jax.jit, static_argnames=['output_size', 'bdims'])
def max_pool(x, output_size: int, bdims=2):
    # do max pooling for the objects for which we have specified key parameters

    x_shape = x.shape
    assert x_shape[-1] % output_size == 0, f"{x_shape[-1]} % {output_size}"
    window_size = x_shape[-1] // output_size
    stride_size = x_shape[-1] // output_size
    # x = x.reshape([-1, x.shape[-1], 1])
    assert bdims >= 1 and bdims <= 2

    def pool_vec(v):
        v = flax.linen.max_pool(inputs=v.reshape(-1, 1), window_shape=(window_size,),
                                strides=(stride_size,)).reshape(-1)
        return v

    if bdims == 1:
        x = jax.vmap(pool_vec)(x)
    else:
        x = jax.vmap(jax.vmap(pool_vec))(x)
    return x


class MergedEncoder(nj.Module):
    def __init__(self,
                 obs_encoder: Type[SimpleEncoder],
                 obs_enc_space: Dict,
                 tactile_enc_space: Dict,
                 tact_encoder: Optional[Type[SimpleEncoder]] = None,
                 ):
        self.obs_encoder = obs_encoder(obs_enc_space, name='obs_enc')
        if tact_encoder is not None:
            assert len(tactile_enc_space) > 0
            self.tact_encoder = tact_encoder(tactile_enc_space, name='tact_enc')
        else:
            self.tact_encoder = tact_encoder

    def __call__(self, inputs, bdims=2, return_all_embeddings: bool = False, tactile_mask: Optional[chex.Array] = None):
        embed = self.obs_encoder(inputs, bdims=bdims)
        outs = {'obs': embed}
        if self.has_tactile_encoder:
            embed_tact = self.tact_encoder(inputs, bdims=bdims)
            outs['tactile'] = embed_tact
            if tactile_mask is not None:
                # mask out tactile measurements based on the specified mask
                embed_tact = tactile_mask[..., jnp.newaxis] * embed_tact
            embed = jnp.concatenate([embed, embed_tact], -1)
        if return_all_embeddings:
            return embed, outs
        else:
            return embed

    @property
    def imgkeys(self):
        return self.obs_encoder.imgkeys

    @property
    def veckeys(self):
        return self.obs_encoder.veckeys

    @property
    def tackeys(self):
        if self.has_tactile_encoder:
            return self.tact_encoder.tackeys
        else:
            return []

    @property
    def has_tactile_encoder(self):
        return self.tact_encoder is not None


class Temperature(nj.Module):
    init_temp: float = 1.0
    slope: float = -1.0
    min_temp: float = 1e-8
    max_temp: float = 1e-8

    def __init__(self):
        init_fn = lambda size, dtype: jnp.log(self.init_temp) * jnp.ones(size, dtype=dtype)
        self.log_temp = nj.Variable(init_fn, (), f32, name='log_temp')

    def __call__(self):
        log_temp = self.log_temp.read()
        clamped_log_temp = sg(jnp.clip(log_temp,
                                       min=jnp.log(self.min_temp),
                                       max=jnp.log(self.max_temp)))
        # clips gradient but maintains derivative
        log_temp = log_temp - sg(log_temp) + clamped_log_temp
        return jnp.exp(sg(log_temp)), log_temp

    def update(self):
        if self.slope > 0:
            log_temp = self.log_temp.read()
            temp = jnp.exp(log_temp)
            new_temp = jnp.clip(temp - self.slope, min=self.min_temp, max=self.max_temp)
            new_log_temp = jnp.log(new_temp)
            self.log_temp.write(new_log_temp)


class DataNormalizer(nj.Module):
    max_num_points: int = 1_000_000

    def __init__(self, label_shape: Tuple, impl: str = 'mean_std'):
        self._impl = impl
        assert self._impl in ['mean_std', 'off'], "We only have data normalization with the mean and std implemented " \
                                                  "for the ensemble"
        self.num_points = nj.Variable(jnp.zeros, (), f32, name='data_counter')
        self.mean = nj.Variable(jnp.zeros, label_shape, f32, name='data_mean')
        self.std = nj.Variable(jnp.ones, label_shape, f32, name='data_std')
        self.label_shape = label_shape

    def __call__(self, x, bdims: int = 2, update: bool = True):
        if update:
            flattend_x = x.reshape((-1,) + self.label_shape)
            self._update(sg(flattend_x))
        m, s, num_points = self._stats()
        scale = lambda z: (z - m) / s
        if bdims == 1:
            scale_x = jax.vmap(scale)(x)
        elif bdims == 2:
            scale_x = jax.vmap(jax.vmap(scale))(x)
        else:
            raise NotImplementedError
        return m, s, num_points, scale_x

    def _update(self, x: jax.Array):
        if self._impl == 'off':
            return True
        # assert len(x.shape) == 2 and x.shape[1:] == self.label_shape
        num_points = f32(x.shape[0])
        current_num_points = self.num_points.read()
        current_mean = self.mean.read()
        current_std = self.std.read()
        total_points = num_points + current_num_points
        data_sum = jnp.sum(x, axis=0)

        new_mean = (current_mean * current_num_points + data_sum) / total_points

        # mean = (normalizer_state.mean * normalizer_state.num_points
        #        + jnp.sum(x, axis=0)) / total_points

        var_std = jnp.square(current_std) * current_num_points
        diff_data = jnp.sum(jnp.square(x - new_mean), axis=0)
        diff_mean = jnp.square(current_mean - new_mean) * current_num_points

        new_var = var_std + diff_data + diff_mean
        new_var = new_var / total_points
        new_std = jnp.clip(jnp.sqrt(new_var), min=1e-3)

        # new_s_n = jnp.square(normalizer_state.std) * normalizer_state.num_points \
        #          + jnp.sum(jnp.square(x - mean), axis=0) + \
        #          normalizer_state.num_points * jnp.square(normalizer_state.mean - mean)

        # new_var = new_s_n / total_points
        # std = jnp.clip(jnp.sqrt(new_var), min=1e-3)
        self.mean.write(new_mean)
        self.std.write(new_std)
        self.num_points.write(jnp.clip(total_points, max=self.max_num_points))
        return True

    def _stats(self):
        return sg(self.mean.read()), sg(self.std.read()), sg(self.num_points.read())


class MergedDecoder(nj.Module):
    def __init__(self,
                 obs_decoder: Type[SimpleDecoder],
                 obs_dec_space: Dict,
                 tactile_dec_space: Dict,
                 tact_decoder: Optional[Type[SimpleDecoder]] = None,
                 ):
        self.obs_decoder = obs_decoder(obs_dec_space, name='obs_dec')
        if tact_decoder is not None:
            assert len(tactile_dec_space) > 0
            self.tact_decoder = tact_decoder(tactile_dec_space, name='tact_dec')
        else:
            self.tact_decoder = tact_decoder

    def __call__(self, inputs, bdims=2):
        outs = self.obs_decoder(inputs, bdims=bdims)
        if self.has_tactile_decoder:
            tactile = self.tact_decoder(inputs, bdims=bdims)
            outs = outs | tactile
        return outs

    @property
    def has_tactile_decoder(self):
        return self.tact_decoder is not None

    @property
    def imgkeys(self):
        return self.obs_decoder.imgkeys

    @property
    def veckeys(self):
        return self.obs_decoder.veckeys

    @property
    def tackeys(self):
        if self.has_tactile_decoder:
            return self.tact_decoder.tackeys
        else:
            return []


class EnsembleMLP(nj.Module):
    def __init__(self,
                 shape: int | Tuple,
                 num_heads: int = 5,
                 agg_disagreement: str = 'mean',
                 dist: str | Dict = 'mse',
                 normalization_impl: str | Dict = 'mean_std',
                 use_squared_disg: bool = True,
                 use_entropy: bool = True,
                 **kwargs):
        # assert isinstance(dist, str)
        if isinstance(dist, str):
            dist_dict = {'output_1': dist}
            if isinstance(shape, int):
                shape = {'output_1': (shape,)}
            else:
                shape = {'output_1': shape}
        else:
            dist_dict = dist
        if isinstance(normalization_impl, str):
            normalization_impl = {key: normalization_impl for key in dist_dict.keys()}
        assert normalization_impl.keys() == dist_dict.keys(), "Normalization and distribution dict can only have " \
                                                              "the same keys"
        self.base_dist = dist_dict
        self._normalization_impl = normalization_impl
        self.shape = shape
        self.kw = {'shape': shape, 'dist': dist_dict} | {k: v for k, v in kwargs.items()}
        self.num_heads = num_heads
        self._output_shape = shape
        # self.transform_predictions = transform_predictions
        self.agg_disagreement = agg_disagreement
        self.use_squared_disg = use_squared_disg
        self.use_entropy = use_entropy

    def __call__(self, inputs, bdims=2, training=False):
        outputs = collections.defaultdict(list)
        for i in range(self.num_heads):
            out = self.get(f'mlp_{i}', MLP, **self.kw)(inputs, bdims=bdims, training=training)
            assert isinstance(out, dict)
            for key, value in out.items():
                outputs[key].append(value)
        disg_rew, disg_metrics = self.get_disagreement(outputs)
        return outputs, (disg_rew, disg_metrics)

    def _unnormalize_output(self, key, out, bdims: int = 2):
        norm_kw = {'label_shape': self.shape[key],
                   'impl': self._normalization_impl[key]}
        dummy_inp = jnp.zeros((1,) + self.shape[key])
        m, s, _, _ = self.get(f'{key}_normalizer',
                              DataNormalizer,
                              **norm_kw)(dummy_inp,
                                         bdims=1,
                                         update=False)
        unnormalize = lambda x: (x * s) + m

        if bdims == 1:
            return jax.vmap(unnormalize)(out)
        elif bdims == 2:
            return jax.vmap(jax.vmap(unnormalize))(out)
        else:
            raise NotImplementedError

    def particle_moments(self, inputs, bdims=2, training=False):
        out_likelihood_dicts, _ = self(inputs, bdims=bdims, training=training)
        particles, mean, epistemic_std = collections.defaultdict(), collections.defaultdict(), collections.defaultdict()
        for key, val in self.base_dist.items():
            particle = self._get_ensemble_predictions(dist=out_likelihood_dicts[key], dist_name=val)
            # Model predicts normalized outputs
            particle = jax.vmap(lambda out: self._unnormalize_output(key, out, bdims),
                                in_axes=-1, out_axes=-1)(particle)
            particles[key] = particle
            mean[key] = particle.mean(axis=-1)
            epistemic_std[key] = particle.std(axis=-1)
            # particles = jnp.stack([dist.mean() for dist in out_likelihood_dicts.values()], axis=-1)
        return particles, mean, epistemic_std

    def _get_ensemble_predictions(self, dist: list, dist_name: str):
        assert isinstance(dist, list)
        if dist_name in ['symlog_mse', 'hyperbolic_mse']:
            ensemble_predictions = []
            for y in dist:
                assert isinstance(y, jaxutils.TransformedMseDist)
                predictions = y.mean()
                # TODO: See if need to get the true predictions for disagreement
                # if transform_predictions:
                #    predictions = y._fwd(predictions)
                ensemble_predictions.append(predictions)
        elif dist_name in ['mse', 'huber']:
            ensemble_predictions = []
            for y in dist:
                assert (isinstance(y, jaxutils.MSEDist) and dist_name == 'mse') | \
                       (isinstance(y, jaxutils.HuberDist) and dist_name == 'huber')
                ensemble_predictions.append(y.mean())
        elif dist_name in ['normal', 'trunc_normal']:
            ensemble_predictions = []
            for y in dist:
                assert (isinstance(y.distribution, tfd.Normal) and dist_name == 'normal') | \
                       (isinstance(y.distribution, tfd.TruncatedNormal) and dist_name == 'trunc_normal')
                ensemble_predictions.append(y.mean())
        elif dist_name in ['binary', 'softmax', 'onehot']:
            ensemble_predictions = []
            for y in dist:
                # TODO: See if logits or mean should be used
                if isinstance(y, tfd.Independent):
                    assert (isinstance(y.distribution, tfd.Bernoulli) and dist_name == 'binary') | \
                           (isinstance(y.distribution, tfd.Categorical) and dist_name == 'softmax') | \
                           (isinstance(y.distribution, tfd.OneHotCategorical) and dist_name == 'onehot')
                    ensemble_predictions.append(y.distribution.mean())
                else:
                    assert (isinstance(y, tfd.Bernoulli) and dist_name == 'binary') | \
                           (isinstance(y, tfd.Categorical) and dist_name == 'softmax') | \
                           (isinstance(y, tfd.OneHotCategorical) and dist_name == 'onehot')
                    ensemble_predictions.append(y.mean())

        elif dist_name in ['symlog_and_twohot', 'symexp_twohot', 'hyperbolic_twohot']:
            ensemble_predictions = []
            for y in dist:
                assert isinstance(y, jaxutils.TwoHotDist)
                # TODO: See if logits or mean should be used
                ensemble_predictions.append(y.mean())
        else:
            raise NotImplementedError(dist_name)
        ensemble_predictions = jnp.stack(ensemble_predictions, axis=-1)
        return ensemble_predictions

    def get_disagreement(self, dist: Dict):
        disagreement_reward = collections.defaultdict()
        disg_metrics = {}
        for key, val in self.base_dist.items():
            disagreement_reward[key], metrics = self.get_disagreement_reward_for_dist(
                dist=dist[key], dist_name=val,
                key=key,
                agg=self.agg_disagreement,
                shape=self.shape[key],
            )
            disg_metrics.update(metrics)
        return disagreement_reward, disg_metrics

    def normalize(self, outputs: Dict, bdims: int = 2, update: bool = True):
        normalized_outputs = outputs.copy()
        for key, val in outputs.items():
            norm_kw = {'label_shape': self.shape[key], 'impl': self._normalization_impl[key]}
            m, s, _, norm_val = self.get(f'{key}_normalizer', DataNormalizer, **norm_kw)(
                val, bdims=bdims, update=update)
            normalized_outputs[key] = norm_val
        return normalized_outputs

    def get_disagreement_reward_for_dist(self,
                                         dist: list,
                                         dist_name: str,
                                         shape: int | Tuple,
                                         key: str,
                                         agg: str = 'mean',
                                         ):
        ensemble_predictions = self._get_ensemble_predictions(dist, dist_name)
        metrics = {}

        norm_kw = {'label_shape': self.shape[key],
                   'impl': self._normalization_impl[key]}
        dummy_inp = jnp.zeros((1,) + self.shape[key])
        normalizer_mean, normalizer_std, normalizer_points, _ = self.get(f'{key}_normalizer', DataNormalizer,
                                                                         **norm_kw)(dummy_inp, bdims=1, update=False)
        ep_std = ensemble_predictions.std(axis=-1)

        metrics[f'disg/eps_{key}_max'] = jnp.max(ep_std)
        metrics[f'disg/eps_{key}_min'] = jnp.min(ep_std)
        metrics[f'disg/eps_{key}_mean'] = jnp.mean(ep_std)
        metrics[f'disg/eps_{key}_std'] = jnp.std(ep_std)


        metrics[f'disg/normalizer_{key}_mean'] = normalizer_mean.mean()
        metrics[f'disg/normalizer_{key}_std'] = normalizer_std.mean()
        metrics[f'disg/normalizer_{key}_num_points'] = normalizer_points

        # ep_std = ep_std / ep_std_scale

        metrics[f'disg/norm_eps_{key}_max'] = jnp.max(ep_std * normalizer_std)
        metrics[f'disg/norm_eps_{key}_min'] = jnp.min(ep_std * normalizer_std)
        metrics[f'disg/norm_eps_{key}_mean'] = jnp.mean(ep_std * normalizer_std)
        metrics[f'disg/norm_eps_{key}_std'] = jnp.std(ep_std * normalizer_std)
        if self.use_squared_disg:
            norm_ep_var = jnp.square(ep_std)
        else:
            norm_ep_var = ep_std
        if self.use_entropy:
            EPS = 1e-4
            norm_ep_var = jnp.log(1 + norm_ep_var / EPS)
        if agg == 'sum':
            norm_ep_var = norm_ep_var.sum([-(i + 1) for i in range(len(shape))])
        elif agg == 'mean':
            norm_ep_var = norm_ep_var.mean([-(i + 1) for i in range(len(shape))])
        else:
            raise NotImplementedError(agg)

        return norm_ep_var, metrics


class Pooling:
    def __init__(self, output_size: Dict | int = 128):
        if isinstance(output_size, int):
            output_size = {'output': output_size}
        self.output_size = output_size

    def __call__(self, x: Dict, bdims=2):
        y = {}
        for key, v in x.items():
            if key not in self.output_size:
                y[key] = v
            else:
                y[key] = max_pool(v, self.output_size[key], bdims)
        return y


class IntrinsicRewardModel(nj.Module):
    def __init__(self,
                 pooling: Dict,
                 model_kwargs: Dict,
                 disg_agg: str = 'sum',
                 num_heads: int = 5,
                 use_entropy: bool = False,
                 use_squared_disg: bool = False,
                 exploration_reward_weights: Optional[Dict] = None,
                 ):
        self.pool = Pooling(**pooling)
        self.ens = EnsembleMLP(num_heads=num_heads,use_squared_disg=use_squared_disg,use_entropy=use_entropy,**model_kwargs, name='intrinsic_reward_ens')
        self.shape = self.ens.shape

        self.disg_agg = disg_agg
        self.num_heads = num_heads
        if exploration_reward_weights is None:
            exploration_reward_weights = {key: 1.0 for key in self.shape.keys()}
        self.exploration_reward_weights = exploration_reward_weights

    def loss(self, labels: Dict, model_input, bdims: int = 2):
        """Given latent state, state action pairs, predict embeddings, rew, and next state. For all passed parameters,
        the gradient is stopped."""
        # Training ensemble models for disagreement reward
        # get loss for ensemble models
        # ensemble models to predict the embeddings from outputs --> f([z_t, h_t]) -> e_t
        # s_t -> e_t
        # note that gradients are already stopped before the labels are passed
        y = self.pool(labels, bdims=bdims)
        y = self.ens.normalize(y, update=True)
        outputs, _ = self.ens(model_input, bdims=bdims, training=True)
        losses = collections.defaultdict()
        total_loss = 0.0
        for key, val in outputs.items():
            y_key = y[key]
            curr_loss = jnp.stack([-dist.log_prob(f32(y_key)) for dist in val], axis=-1).mean(axis=-1)
            losses[key] = curr_loss.mean()
            total_loss = total_loss + curr_loss
        return total_loss, losses

    def __call__(self, lat, acts, bdims: int = 2, update: bool = True):
        lat_acts = lat | acts
        dist, (disg, disg_metrics) = self.ens(lat_acts, bdims=bdims)
        total_disg = jnp.stack([val * self.exploration_reward_weights[key] for key, val in disg.items()], axis=-1)
        if self.disg_agg == 'sum':
            total_disg = total_disg.sum(axis=-1)
        elif self.disg_agg == 'mean':
            total_disg = total_disg.mean(axis=-1)
        elif self.disg_agg == 'max':
            total_disg = total_disg.max(axis=-1)
        else:
            raise NotImplementedError
        return total_disg, disg_metrics


class Trainer(nj.Module):
    def __init__(self, model: EnsembleMLP, lr: float = 1e-3):
        self.model = model
        self.opt = nj.FromOptax(optax.adam)(lr, name='opt')

    def train(self, x, y):
        if isinstance(y, jax.Array):
            assert len(self.model.shape.keys()) == 1
            y = {list(self.model.shape.keys())[0]: y}
        y = self.model.normalize(y, bdims=1)

        # Take grads wrt. to submodules or state keys
        def loss(x, y):
            y_dist, _ = self.model(x, bdims=1, training=True)
            loss = 0.0
            for key, val in y_dist.items():
                y_key = y[key]
                loss += jnp.stack([-dist.log_prob(y_key) for dist in val], axis=-1).mean()
            return loss

        return self.opt(loss, [self.model], x, y)[0]


def main():
    import jax.numpy as jnp
    import jax.random as jr
    import matplotlib.pyplot as plt
    ensemble = EnsembleMLP(shape=1, layers=2, units=256, num_heads=5, act='silu', name='ens',
                           normalization_impl='mean_std')
    noise_level = 0.1
    d_l, d_u = -2, 10
    amplitude = 100
    xs = jnp.linspace(d_l, d_u, 64).reshape(-1, 1)
    ys = jnp.sin(xs) * amplitude
    ys = ys + noise_level * jr.normal(key=jr.PRNGKey(0), shape=ys.shape)
    trainer = Trainer(model=ensemble, name='trainer')
    state = {}
    state = nj.init(trainer.train)(state, xs, ys, seed=0)
    print(state.keys())
    predict = jax.jit(nj.pure(lambda x: ensemble.particle_moments(x, bdims=1)))
    train = jax.jit(nj.pure(trainer.train))
    for i in range(1_000):
        print('Norm state counter:', state['ens/output_1_normalizer/data_counter/value'])
        state, loss = train(state, xs, ys)
        print('Loss:', float(loss))
    num_test_points = 1000
    test_xs = jnp.linspace(-5, 15, num_test_points).reshape(-1, 1)
    test_ys = jnp.sin(test_xs) * amplitude
    state, outs = predict(state, test_xs)
    particles, mean_pred, epistemic_std = outs
    particles, mean_pred, epistemic_std = list(particles.values())[0], \
        list(mean_pred.values())[0], list(epistemic_std.values())[0]
    plt.scatter(xs.reshape(-1), ys, label='Data', color='red')
    for i in range(ensemble.num_heads):
        plt.plot(test_xs, particles[..., i], label='NN prediction', color='black', alpha=0.3)
    plt.plot(test_xs, mean_pred, label='Mean', color='blue')
    plt.fill_between(test_xs.reshape(-1),
                     (mean_pred - 2 * epistemic_std).reshape(-1),
                     (mean_pred + 2 * epistemic_std).reshape(-1),
                     label=r'$2\sigma$', alpha=0.3, color='blue')
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.plot(test_xs.reshape(-1), test_ys, label='True', color='green')
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()


if __name__ == '__main__':
    main()
    x = jax.random.normal(shape=(1, 32), key=jax.random.PRNGKey(0))
    import time

    pool = Pooling(output_size=8)
    for i in range(5):
        start_time = time.time()
        print(f'step_{i}', pool({'output': x}, bdims=1), f'time: {time.time() - start_time} s')
