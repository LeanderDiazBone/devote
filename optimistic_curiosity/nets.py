import einops
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

import embodied
from dreamerv3 import jaxutils
from dreamerv3 import ninjax as nj
from dreamerv3.nets import (
    Conv2D, Input, Initializer, Linear, Norm, RSSM, SquashedNormal,
    get_act,
)

f32 = jnp.float32
tfd = tfp.distributions
tfb = tfp.bijectors
sg = lambda x: jax.tree_util.tree_map(jax.lax.stop_gradient, x)
cast = jaxutils.cast_to_compute


class DrQEncoder(nj.Module):
    """Canonical DrQ visual encoder adapted to channels-last inputs."""

    feature_dim: int = 50

    def __init__(self, spaces):
        self.spaces = spaces
        self.veckeys = [k for k, s in spaces.items() if len(s.shape) <= 2]
        self.imgkeys = [k for k, s in spaces.items() if len(s.shape) == 3]
        if self.veckeys or len(self.imgkeys) != 1:
            raise ValueError(
                'DrQEncoder requires exactly one image observation and no '
                f'vector observations, got images={self.imgkeys}, '
                f'vectors={self.veckeys}.')
        shape = spaces[self.imgkeys[0]].shape
        if shape[0] != shape[1]:
            raise ValueError(
                f'DrQEncoder requires square images, got shape {shape}.')
        self.imginp = Input(self.imgkeys, featdims=3)

    def __call__(self, data, bdims=2):
        assert bdims in (1, 2), bdims
        shape = data['is_first'].shape[:bdims]
        data = {k: data[k] for k in self.spaces}
        x = self.imginp(data, bdims, jaxutils.COMPUTE_DTYPE)
        x = x.reshape((-1, *x.shape[bdims:]))

        for i, stride in enumerate((2, 1, 1, 1)):
            x = self.get(
                f'conv{i}', Conv2D, 32, 3, stride,
                pad='valid', norm='none', act='relu', winit='ortho',
                outscale=float(np.sqrt(2.0)))(x)

        x = x.reshape((x.shape[0], -1))
        x = self.get(
            'head', Linear, self.feature_dim,
            norm='none', act='none', winit='ortho')(x)
        x = self.get('head_norm', Norm, 'layer', eps=1e-5)(x)
        x = jnp.tanh(x)
        return x.reshape((*shape, self.feature_dim))


class ShiftedDist:
    """Wrap a distribution with an additive value-space shift."""

    def __init__(self, dist, shift):
        self._dist = dist
        self._shift = shift
        self.batch_shape = dist.batch_shape
        self.event_shape = dist.event_shape
        if hasattr(dist, 'minent'):
            self.minent = dist.minent
        if hasattr(dist, 'maxent'):
            self.maxent = dist.maxent

    def mode(self):
        return self._dist.mode() + self._shift

    def mean(self):
        return self._dist.mean() + self._shift

    def log_prob(self, value, **kw):
        return self._dist.log_prob(value - self._shift, **kw)

    def __getattr__(self, name):
        return getattr(self._dist, name)


class EnsembleDist(nj.Module):
    # same interface as Dist, but params have a leading [E] axis
    outscale: float = 0.1
    minstd: float = 1.0
    maxstd: float = 1.0
    unimix: float = 0.0
    bins: int = 255
    outact: str = 'none'

    def __init__(self, shape, dist='mse', num_heads: int = 1, **kw):
        assert all(isinstance(dim, (int, np.integer)) for dim in shape), shape
        forbidden = ('binit', 'norm', 'act')
        assert all(k not in kw for k in forbidden), (forbidden, kw)
        self.shape = shape
        self.dist = dist
        self.num_heads = int(num_heads)
        self.kw = dict(**kw, outscale=self.outscale)

    def __call__(self, inputs, temperature=1.0):
        # inputs: [E, ..., F]
        dist = self.inner(inputs, temperature=temperature)
        # Dist.batch_shape should be [E, ...] (all but last dim)
        assert tuple(dist.batch_shape) == tuple(inputs.shape[:-1]), (dist.batch_shape, dist.event_shape, inputs.shape)
        return dist

    def inner(self, inputs, temperature=1.0):
        shape = self.shape
        padding = 0

        if 'twohot' in self.dist or self.dist == 'softmax':
            padding = int(self.bins % 2)
            shape = (*self.shape, self.bins + padding)
        out_units = int(np.prod(shape))
        out = self.get('out', EnsembleLinear, out_units, self.num_heads, **self.kw)(inputs)
        out = out.reshape(inputs.shape[:-1] + shape).astype(f32)
        if self.outact == 'tanh':
            out = jnp.tanh(out)
        out = out[..., :-padding] if padding else out

        if 'normal' in self.dist:
            units = int(np.prod(self.shape))
            std = self.get('std', EnsembleLinear, units, self.num_heads, **self.kw)(inputs)
            std = std.reshape(inputs.shape[:-1] + self.shape).astype(f32)

        if self.dist == 'symlog_mse':
            fwd, bwd = jaxutils.symlog, jaxutils.symexp
            return jaxutils.TransformedMseDist(out, len(self.shape), fwd, bwd)

        if self.dist == 'mse':
            return jaxutils.MSEDist(out, len(self.shape), 'sum')

        if self.dist == 'huber':
            return jaxutils.HuberDist(out, len(self.shape), 'sum')

        if self.dist == 'normal':
            lo, hi = self.minstd, self.maxstd
            std = (hi - lo) * jax.nn.sigmoid(std + 2.0) + lo
            std = std * temperature
            dist = tfd.Normal(jnp.tanh(out), std)
            dist = tfd.Independent(dist, len(self.shape))
            dist.minent = np.prod(self.shape) * tfd.Normal(0.0, lo).entropy()
            dist.maxent = np.prod(self.shape) * tfd.Normal(0.0, hi).entropy()
            return dist

        if self.dist == 'trunc_normal':
            lo, hi = self.minstd, self.maxstd
            std = (hi - lo) * jax.nn.sigmoid(std + 2.0) + lo
            dist = tfd.TruncatedNormal(jnp.tanh(out), std, -1, 1)
            dist = tfd.Independent(dist, len(self.shape))
            dist.minent = np.prod(self.shape) * tfd.TruncatedNormal(1.0, lo, -1, 1).entropy()
            dist.maxent = np.prod(self.shape) * tfd.TruncatedNormal(0.0, hi, -1, 1).entropy()
            return dist

        if self.dist == 'squashed_normal':
            # SAC tanh-squashed Gaussian. Uses a hand-rolled SquashedNormal whose
            # log_prob is numerically stable (softplus Jacobian + clip-before-atanh),
            # avoiding the NaN-on-saturation failure of TransformedDistribution(Tanh).
            lo, hi = self.minstd, self.maxstd
            std = (hi - lo) * jax.nn.sigmoid(std + 2.0) + lo
            std = std * temperature
            dist = SquashedNormal(out, std, event_ndims=len(self.shape))
            dist.minent = np.prod(self.shape) * tfd.TruncatedNormal(0.0, lo, -1, 1).entropy()
            dist.maxent = np.prod(self.shape) * np.log(2.0)
            return dist

        if self.dist == 'binary':
            dist = tfd.Bernoulli(out)
            if self.shape:
                dist = tfd.Independent(dist, len(self.shape))
            return dist

        if self.dist == 'softmax':
            dist = tfd.Categorical(out / temperature)
            if len(self.shape) > 1:
                dist = tfd.Independent(dist, len(self.shape) - 1)
            return dist

        if self.dist == 'onehot':
            if self.unimix:
                probs = jax.nn.softmax(out, -1)
                uniform = jnp.ones_like(probs) / probs.shape[-1]
                probs = (1 - self.unimix) * probs + self.unimix * uniform
                out = jnp.log(probs)
            dist = jaxutils.OneHotDist(out / temperature)
            if len(self.shape) > 1:
                dist = tfd.Independent(dist, len(self.shape) - 1)
            dist.minent = 0.0
            dist.maxent = np.prod(self.shape[:-1]) * np.log(self.shape[-1])
            return dist
        if self.dist == 'symexp_twohot':
            if out.shape[-1] % 2 == 1:
                half = jnp.linspace(-20, 0, (out.shape[-1] - 1) // 2 + 1, dtype=f32)
                half = jaxutils.symexp(half)
                bins = jnp.concatenate([half, -half[:-1][::-1]], 0)
            else:
                half = jnp.linspace(-20, 0, out.shape[-1] // 2, dtype=f32)
                half = jaxutils.symexp(half)
                bins = jnp.concatenate([half, -half[::-1]], 0)
            return jaxutils.TwoHotDist(out / temperature, bins, len(self.shape))

        raise NotImplementedError(self.dist)


class EnsembleRFFPrior(nj.Module):
    """Frozen random Fourier feature prior for Q-functions."""

    units: int = 128
    num_heads: int = 1
    length_scale: float = 1.0
    outact: str = 'none'
    dtype: str = 'default'

    def __init__(self, shape, dist='mse', inputs=['tensor']):
        shape = (shape,) if isinstance(shape, (int, np.integer)) else shape
        assert isinstance(shape, tuple), shape
        self.shape = shape
        self.dist = dist
        self.inputs = Input(inputs, featdims=1)
        if self.units <= 0:
            raise ValueError(f'RFF prior width must be positive, got {self.units}.')
        if self.length_scale <= 0:
            raise ValueError(f'RFF prior length_scale must be positive, got {self.length_scale}.')

    def __call__(self, inputs, bdims=2, training=False, has_ensemble=False, temperature=1.0):
        del training, temperature
        feat = self.inputs(inputs, bdims, jaxutils.COMPUTE_DTYPE)
        E = int(self.num_heads)
        if has_ensemble:
            assert feat.shape[0] == E, (feat.shape, E)
            x = feat
        else:
            x = jnp.broadcast_to(feat[None, ...], (E,) + feat.shape)
        out = self._forward(x)
        return self._dist(out)

    def _forward(self, x):
        E = int(self.num_heads)
        in_units = x.shape[-1]
        out_units = int(np.prod(self.shape))

        omega = self.get('omega', self._omega_init, (E, in_units, self.units)).astype(x.dtype)
        phase = self.get('phase', self._phase_init, (E, self.units)).astype(x.dtype)
        weight = self.get('weight', self._weight_init, (E, self.units, out_units)).astype(x.dtype)

        proj = jnp.einsum('e...i,eim->e...m', x, omega)
        phase = phase.reshape((E,) + (1,) * (proj.ndim - 2) + (self.units,))
        scale = jnp.sqrt(jnp.asarray(2.0 / self.units, x.dtype))
        feat = scale * jnp.cos(proj + phase)
        out = jnp.einsum('e...m,emo->e...o', feat, weight)
        out = self._outact(out)
        return out.reshape(x.shape[:-1] + self.shape).astype(f32)

    def _outact(self, out):
        if self.outact == 'none':
            return out
        if self.outact == 'tanh':
            return jnp.tanh(out)
        return get_act(self.outact)(out)

    def _dist(self, out):
        if self.dist == 'mse':
            return jaxutils.MSEDist(out, len(self.shape), 'sum')
        if self.dist == 'huber':
            return jaxutils.HuberDist(out, len(self.shape), 'sum')
        raise NotImplementedError(f'RFF prior only supports direct regression dists, got {self.dist}.')

    def _param_dtype(self):
        dtype = jaxutils.PARAM_DTYPE if self.dtype == 'default' else self.dtype
        return getattr(jnp, dtype) if isinstance(dtype, str) else dtype

    def _omega_init(self, shape):
        dtype = self._param_dtype()
        return (jax.random.normal(nj.seed(), shape, dtype) / self.length_scale).astype(dtype)

    def _phase_init(self, shape):
        dtype = self._param_dtype()
        return jax.random.uniform(nj.seed(), shape, dtype, 0.0, 2 * np.pi)

    def _weight_init(self, shape):
        dtype = self._param_dtype()
        return jax.random.normal(nj.seed(), shape, dtype)


class EnsembleMLP(nj.Module):
    layers: int = None
    units: int = None
    block_fans: bool = False
    block_norm: bool = False
    num_heads: int = 1
    outact: str = 'none'
    residual: bool = False
    hidden_outscale: float = 1.0

    def __init__(self, shape, dist='mse', inputs=['tensor'], **kw):
        shape = (shape,) if isinstance(shape, (int, np.integer)) else shape
        assert isinstance(shape, (tuple, dict, type(None))), shape
        assert isinstance(dist, (str, dict)), dist
        assert isinstance(dist, dict) == isinstance(shape, dict), (dist, shape)

        self.shape = shape
        self.dist = dist
        self.inputs = Input(inputs, featdims=1)

        distonly = ('outscale', 'minstd', 'maxstd', 'unimix', 'bins')
        self.lkw = {k: v for k, v in kw.items() if k not in distonly}
        self.lkw['outscale'] = self.hidden_outscale
        forbidden = ('binit', 'norm', 'act')
        self.dkw = {k: v for k, v in kw.items() if k not in forbidden}
        self.lkw_noact = {**self.lkw, 'act': 'none'}

    def __call__(self, inputs, bdims=2, training=False, has_ensemble=False, temperature=1.0):
        x = self.features(inputs, bdims=bdims, has_ensemble=has_ensemble)
        if self.shape is None:
            return x
        if isinstance(self.shape, dict):
            return {k: self._out(k, v, self.dist[k], x, temperature=temperature) for k, v in self.shape.items()}
        else:
            return self._out('dist', self.shape, self.dist, x, temperature=temperature)

    def features(self, inputs, bdims=2, has_ensemble=False):
        """Post-activation features of the final hidden layer.

        Returns ``[E, *bdims, units]``; reuses the same submodules as ``__call__``.
        """
        feat = self.inputs(inputs, bdims, jaxutils.COMPUTE_DTYPE)   # [..., F]
        E = int(self.num_heads)
        if has_ensemble:
            assert feat.shape[0] == E, (feat.shape, E)
            x = feat.reshape((E, -1, feat.shape[-1]))               # [E, B*, F]
        else:
            x = feat.reshape([-1, feat.shape[-1]])                  # [B*, F]
            x = jnp.broadcast_to(x[None, ...], (E,) + x.shape)      # [E, B*, F]

        if self.residual:
            x = self.get('stem', EnsembleLinear, self.units, E, **self.lkw)(x)
            for i in range(self.layers):
                h = self.get(f'h{i}a', EnsembleLinear, self.units, E, **self.lkw)(x)
                h = self.get(f'h{i}b', EnsembleLinear, self.units, E, **self.lkw_noact)(h)
                x = x + h
        else:
            for i in range(self.layers):
                x = self.get(f'h{i}', EnsembleLinear, self.units, E, **self.lkw)(x)

        if has_ensemble:
            x = x.reshape((E, *feat.shape[1:bdims], -1))
        else:
            x = x.reshape((E, *feat.shape[:bdims], -1))
        return x

    def _out(self, name, shape, dist, x, temperature=1.0):
        name = name.replace('/', '_').replace('.', '_')
        return self.get(name, EnsembleDist, shape, dist, num_heads=int(self.num_heads), outact=self.outact, **self.dkw)(x, temperature=temperature)


class EnsembleInputProjection(nj.Module):
    """Apply one fixed input projection per ensemble member."""

    def __init__(self, module_ctor, module_args, module_kwargs, projection,
                 key='visual_embed'):
        self.module = module_ctor(
            *module_args, **module_kwargs, name='model')
        self.projection = projection
        self.key = key

    def __call__(self, inputs, bdims=2, training=False,
                 has_ensemble=False, temperature=1.0):
        if not isinstance(inputs, dict) or self.key not in inputs:
            raise KeyError(
                f"EnsembleInputProjection needs input key '{self.key}'.")
        matrix = self.projection.read()
        heads = int(self.module.num_heads)
        if matrix.ndim != 3 or matrix.shape[0] != heads:
            raise ValueError(
                f'Expected projection shape [E, F, D] with E={heads}, '
                f'got {matrix.shape}.')
        feature = inputs[self.key]
        if feature.shape[-1] != matrix.shape[-2]:
            raise ValueError(
                f"Projection input '{self.key}' has width {feature.shape[-1]}, "
                f'expected {matrix.shape[-2]}.')
        matrix = matrix.astype(feature.dtype)

        if has_ensemble:
            if feature.shape[0] != heads:
                raise ValueError(
                    f"Projection input '{self.key}' needs leading ensemble "
                    f'axis {heads}, got {feature.shape}.')
            projected = jnp.einsum('e...f,efd->e...d', feature, matrix)
            projected_inputs = {**inputs, self.key: projected}
            return self.module(
                projected_inputs, bdims=bdims, training=training,
                has_ensemble=True, temperature=temperature)

        projected = jnp.einsum('...f,efd->e...d', feature, matrix)
        projected_inputs = {
            key: (projected if key == self.key else
                  jnp.broadcast_to(value[None], (heads,) + value.shape))
            for key, value in inputs.items()}
        return self.module(
            projected_inputs, bdims=bdims + 1, training=training,
            has_ensemble=True, temperature=temperature)


class PriorCritic(nj.Module):
    """Critic with randomized-prior components baked into its forward pass.

    The frozen prior module is referenced rather than owned, so its parameters
    live outside this module's namespace and are shared between online and target
    wrappers. The optional corrector lives under this critic and is optimized
    together with the regular critic head.
    """

    def __init__(
            self, shape, num_heads, prior=None, prior_shape=(),
            prior_scale=0.0, prior_corrector=False,
            prior_corrector_use_rff=False, prior_corrector_length_scale=1.0,
            prior_corrector_kw=None, prior_corrector_shape=None,
            prior_input_projection=None,
            prior_input_projection_key='visual_embed',
            corrector_scale=None,
            epistemic_dim=0, epistemic_std=1.0,
            epistemic_samples=1, epistemic_key='epistemic',
            epistemic_shared_contexts=False,
            residual_bootstrap=False, residual_bootstrap_kw=None, residual_bootstrap_shape=None,
            **head_kw):
        shape = (shape,) if isinstance(shape, (int, np.integer)) else tuple(shape)
        prior_shape = (prior_shape,) if isinstance(prior_shape, (int, np.integer)) else tuple(prior_shape)
        prior_corrector_shape = prior_shape if prior_corrector_shape is None else prior_corrector_shape
        prior_corrector_shape = (prior_corrector_shape,) if isinstance(prior_corrector_shape, (int, np.integer)) else tuple(prior_corrector_shape)
        self.shape = shape
        self.head = EnsembleMLP(shape, num_heads=num_heads, name='head', **head_kw)
        self.num_heads = int(num_heads)
        self._prior = prior
        self._broadcast_dims = max(0, len(shape) - len(prior_shape))
        self._corrector_broadcast_dims = max(0, len(shape) - len(prior_corrector_shape))
        self.prior_scale = float(prior_scale)
        self.corrector_scale = self.prior_scale if corrector_scale is None else float(corrector_scale)
        self.epistemic_dim = int(epistemic_dim or 0)
        self.epistemic_std = float(epistemic_std)
        self.epistemic_samples = max(1, int(epistemic_samples))
        self.epistemic_key = epistemic_key
        self.epistemic_shared_contexts = bool(epistemic_shared_contexts)
        self.prior_corrector = None
        if prior_corrector:
            prior_corrector_kw = dict(prior_corrector_kw or {})
            if 'dtype' in head_kw and 'dtype' not in prior_corrector_kw:
                prior_corrector_kw['dtype'] = head_kw['dtype']
            if prior_corrector_use_rff:
                corrector_ctor = EnsembleRFFPrior
                corrector_args = (prior_corrector_shape,)
                corrector_kwargs = dict(
                    num_heads=num_heads,
                    inputs=prior_corrector_kw['inputs'],
                    units=prior_corrector_kw['units'],
                    length_scale=prior_corrector_length_scale,
                    dist=prior_corrector_kw['dist'],
                    outact=prior_corrector_kw.get('outact', 'none'),
                    dtype=prior_corrector_kw.get('dtype', 'default'))
            else:
                corrector_ctor = EnsembleMLP
                corrector_args = (prior_corrector_shape,)
                corrector_kwargs = dict(
                    num_heads=num_heads, **prior_corrector_kw)
            if prior_input_projection is not None:
                self.prior_corrector = EnsembleInputProjection(
                    corrector_ctor, corrector_args, corrector_kwargs,
                    prior_input_projection, key=prior_input_projection_key,
                    name='prior_corrector')
            else:
                self.prior_corrector = corrector_ctor(
                    *corrector_args, **corrector_kwargs,
                    name='prior_corrector')
        # Optional residual_bootstrap: a sibling ensemble (parallel to prior_corrector)
        # that the agent trains via TD with the frozen prior as immediate reward, so
        # std-across-heads gives a persistent bootstrapped uncertainty signal. Uses the
        # standard critic head architecture, but conditions on the epistemic z when present.
        self.residual_bootstrap = None
        if residual_bootstrap:
            tracker_shape = shape if residual_bootstrap_shape is None else residual_bootstrap_shape
            tracker_shape = (tracker_shape,) if isinstance(tracker_shape, (int, np.integer)) else tuple(tracker_shape)
            tracker_kw = dict(residual_bootstrap_kw) if residual_bootstrap_kw is not None else dict(head_kw)
            if 'dtype' in head_kw and 'dtype' not in tracker_kw:
                tracker_kw['dtype'] = head_kw['dtype']
            if self.epistemic_dim > 0:
                tracker_kw['inputs'] = list(tracker_kw['inputs']) + [self.epistemic_key]
            self.residual_bootstrap = EnsembleMLP(tracker_shape, num_heads=num_heads, name='residual_bootstrap', **tracker_kw)

    def z_space(self):
        """Embodied space describing this prior's epistemic index z. None means no z.
        The cube is [-epistemic_std, +epistemic_std]^epistemic_dim."""
        if self.epistemic_dim <= 0:
            return None
        return embodied.Space(np.float32, (self.epistemic_dim,), low=-self.epistemic_std, high=self.epistemic_std)

    def z_aux_spaces(self):
        """Replay-aux spaces for z components ('epistemic', 'ensemble') to persist (ensemble as one-hot f32)."""
        return {**({'epistemic': self.z_space()} if self.epistemic_dim > 0 else {}),
                **({'ensemble': embodied.Space(np.float32, (self.num_heads,), 0.0, 1.0)} if self.num_heads > 1 else {})}

    def sample_epistemic(self, inputs, bdims=2, shared_contexts=None):
        if self.epistemic_dim <= 0:
            return None
        if not isinstance(inputs, dict):
            inputs = {'tensor': inputs}
        first = next(iter(inputs.values()))
        leading = tuple(first.shape[:bdims])
        shared = self.epistemic_shared_contexts if shared_contexts is None else bool(shared_contexts)
        sample_shape = leading
        if shared and len(leading) > 1:
            sample_shape = leading[:1] + (1,) * (len(leading) - 1)
        z = jax.random.uniform(
            nj.seed(), sample_shape + (self.epistemic_dim,),
            jaxutils.COMPUTE_DTYPE,
            minval=-self.epistemic_std, maxval=self.epistemic_std)
        return jnp.broadcast_to(z, leading + (self.epistemic_dim,))

    def _with_epistemic(self, inputs, bdims, epistemic=None):
        if self.epistemic_dim <= 0:
            return inputs
        if not isinstance(inputs, dict):
            inputs = {'tensor': inputs}
        if epistemic is None:
            epistemic = self.sample_epistemic(inputs, bdims)
        return {**inputs, self.epistemic_key: epistemic}

    def sample_epistemics(self, inputs, bdims=2, samples=None, shared_contexts=None):
        samples = self.epistemic_samples if samples is None else int(samples)
        samples = samples if self.epistemic_dim > 0 else 1
        return [
            self.sample_epistemic(inputs, bdims, shared_contexts)
            for _ in range(max(1, samples))]

    def sample_context(self, inputs, bdims=2, shared_contexts=None):
        return self.sample_epistemic(inputs, bdims, shared_contexts)

    def sample_contexts(self, inputs, bdims=2, samples=None, shared_contexts=None):
        return self.sample_epistemics(inputs, bdims, samples, shared_contexts)

    def _broadcast_shift(self, shift, ref, broadcast_dims=None):
        broadcast_dims = self._broadcast_dims if broadcast_dims is None else broadcast_dims
        for _ in range(broadcast_dims):
            shift = shift[..., None]
        return jnp.zeros_like(ref) + shift

    def _effective_scales(self, prior_scale=None, corrector_scale=None):
        prior_override = prior_scale is not None
        prior_scale = self.prior_scale if prior_scale is None else prior_scale
        if corrector_scale is None:
            corrector_scale = prior_scale if prior_override else self.corrector_scale
        return prior_scale, corrector_scale

    def component_means(self, inputs, bdims=2, training=False, has_ensemble=False, temperature=1.0, epistemic=None, ensemble=None, prior_scale=None, corrector_scale=None):
        indexed_inputs = self._with_epistemic(inputs, bdims, epistemic)
        raw = self.head(inputs, bdims=bdims, training=training, has_ensemble=has_ensemble, temperature=temperature).mean()
        prior = jnp.zeros_like(raw)
        corrector = jnp.zeros_like(raw)
        prior_scale, corrector_scale = self._effective_scales(prior_scale, corrector_scale)
        if self._prior is not None and prior_scale != 0:
            prior = prior_scale * self._prior(indexed_inputs, bdims=bdims, has_ensemble=has_ensemble).mean()
            prior = self._broadcast_shift(sg(prior), raw)
        if self.prior_corrector is not None and corrector_scale != 0:
            corrector = corrector_scale * self.prior_corrector(indexed_inputs, bdims=bdims, training=training, has_ensemble=has_ensemble).mean()
            corrector = self._broadcast_shift(corrector, raw, self._corrector_broadcast_dims)
        prior_corrector = prior + corrector
        mixed = raw + prior_corrector
        out = dict(raw=raw, prior=prior, corrector=corrector, prior_corrector=prior_corrector)
        if self.residual_bootstrap is not None:
            residual_bootstrap = prior_scale * self.residual_bootstrap(indexed_inputs, bdims=bdims, training=training, has_ensemble=has_ensemble).mean()
            out['residual_bootstrap'] = residual_bootstrap
            mixed = mixed + residual_bootstrap
        out['mixed'] = mixed
        if ensemble is not None:
            # Anchor each component to the head picked by `ensemble` (one-hot) and rebroadcast across V.
            out = {k: jnp.broadcast_to(self._select_ensemble(v, ensemble)[None], v.shape) for k, v in out.items()}
        return out

    def mean_samples(self, inputs, bdims=2, training=False, has_ensemble=False, temperature=1.0, samples=None, contexts=None, shared_contexts=None, stop_prior_grad=True, **kw):
        values = []
        contexts = (
            self.sample_epistemics(inputs, bdims, samples, shared_contexts)
            if contexts is None else contexts)
        for context in contexts:
            values.append(self(inputs, bdims=bdims, training=training, has_ensemble=has_ensemble, temperature=temperature, epistemic=context, stop_prior_grad=stop_prior_grad, **kw).mean())
        values = jnp.stack(values, 0)
        return values.reshape((values.shape[0] * values.shape[1],) + values.shape[2:])

    def component_samples(self, inputs, bdims=2, training=False, has_ensemble=False, temperature=1.0, samples=None, contexts=None, shared_contexts=None, prior_scale=None, corrector_scale=None):
        outs = {}
        contexts = (
            self.sample_epistemics(inputs, bdims, samples, shared_contexts)
            if contexts is None else contexts)
        for context in contexts:
            comps = self.component_means(inputs, bdims=bdims, training=training, has_ensemble=has_ensemble, temperature=temperature, epistemic=context, prior_scale=prior_scale, corrector_scale=corrector_scale)
            for key, value in comps.items():
                outs.setdefault(key, []).append(value)
        return {k: jnp.stack(v, 0) for k, v in outs.items()}

    def __call__(self, inputs, bdims=2, training=False, has_ensemble=False, temperature=1.0, include_prior=True, include_corrector=True, include_residual_bootstrap=True, epistemic=None, prior_scale=None, corrector_scale=None, stop_prior_grad=True):
        indexed_inputs = self._with_epistemic(inputs, bdims, epistemic)
        dist = self.head(inputs, bdims=bdims, training=training, has_ensemble=has_ensemble, temperature=temperature)
        shift = None
        prior_scale, corrector_scale = self._effective_scales(prior_scale, corrector_scale)
        if include_prior and self._prior is not None and prior_scale != 0:
            prior_shift = prior_scale * self._prior(indexed_inputs, bdims=bdims, has_ensemble=has_ensemble).mean()
            for _ in range(self._broadcast_dims):
                prior_shift = prior_shift[..., None]
            if stop_prior_grad:
                prior_shift = sg(prior_shift)
            shift = prior_shift if shift is None else shift + prior_shift
        if include_corrector and self.prior_corrector is not None and corrector_scale != 0:
            corrector_shift = corrector_scale * self.prior_corrector(indexed_inputs, bdims=bdims, training=training, has_ensemble=has_ensemble).mean()
            for _ in range(self._corrector_broadcast_dims):
                corrector_shift = corrector_shift[..., None]
            shift = corrector_shift if shift is None else shift + corrector_shift
        if include_residual_bootstrap and self.residual_bootstrap is not None:
            boot_shift = prior_scale * self.residual_bootstrap(indexed_inputs, bdims=bdims, training=training, has_ensemble=has_ensemble).mean()
            shift = boot_shift if shift is None else shift + boot_shift
        if shift is None:
            return dist
        return ShiftedDist(dist, shift)

    # ----- Unified z-context API -------------------------------------------
    # Public contract:
    #   inputs : dict[str, [*batch, F]]            (no leading particle axis)
    #   z      : optional dict, may contain:
    #              'ensemble' : [*batch, num_heads]   one-hot
    #              'epistemic': [*batch, epistemic_dim]
    #            Missing keys are marginalized; present keys select.
    #   bdims  : number of leading batch dims in `inputs`.
    # Particle axis is *only* exposed by `particles()` and never elsewhere.

    def sample_z(self, batch_shape, components=None):
        """Fresh z dict. Components with no degrees of freedom are omitted."""
        components = set(components) if components is not None else {'ensemble', 'epistemic'}
        z = {}
        if 'ensemble' in components and self.num_heads > 1:
            idx = jax.random.randint(nj.seed(), batch_shape, 0, self.num_heads)
            z['ensemble'] = jax.nn.one_hot(idx, self.num_heads, dtype=f32)
        if 'epistemic' in components and self.epistemic_dim > 0:
            z['epistemic'] = jax.random.uniform(
                nj.seed(), tuple(batch_shape) + (self.epistemic_dim,), f32,
                minval=-self.epistemic_std, maxval=self.epistemic_std)
        return z

    def particles(self, inputs, *, z=None, bdims=2, samples=None, prior_scale=None, corrector_scale=None, frozen=True, stop_prior_grad=True, include_prior=True, include_corrector=True, include_residual_bootstrap=True):
        """All un-fixed particles, shape [P, *batch, *q_shape].
        P = (epistemic samples if epistemic free else 1) * (num_heads if ensemble free else 1).
        Kwargs match the legacy `__call__` semantics (frozen=True wraps params in sg)."""
        z = z or {}
        eps = z.get('epistemic')
        if eps is not None:
            eps_contexts = [eps]
        elif self.epistemic_dim > 0:
            eps_contexts = self.sample_epistemics(inputs, bdims, samples=samples)
        else:
            eps_contexts = [None]

        def _eval():
            vals = []
            for ctx in eps_contexts:
                d = self(
                    inputs, bdims=bdims, has_ensemble=False, epistemic=ctx,
                    prior_scale=prior_scale, corrector_scale=corrector_scale,
                    stop_prior_grad=stop_prior_grad,
                    include_prior=include_prior,
                    include_corrector=include_corrector,
                    include_residual_bootstrap=include_residual_bootstrap)
                v = d.mean()                                # [E, *batch, *q_shape]
                if 'ensemble' in z:
                    v = self._select_ensemble(v, z['ensemble'])[None]
                vals.append(v)
            out = jnp.stack(vals, 0)                       # [S_or_1, E_or_1, *batch, *q_shape]
            return out.reshape((-1,) + out.shape[2:])      # [P, *batch, *q_shape]

        # During nj.init we can't freeze yet (state hasn't been created), so just
        # run the forward — that's what populates the state in the first place.
        if not frozen or nj.creating():
            return _eval()
        params = self.find()
        self.put(sg(params))
        try:
            return _eval()
        finally:
            self.put(params)

    def component_particles(self, inputs, *, z=None, bdims=2, samples=None, prior_scale=None, corrector_scale=None, frozen=True, stop_prior_grad=True):
        """Per-component particles as dict[str, [P, *batch, *q_shape]].
        Keys: 'raw', 'prior', 'corrector', and 'residual_bootstrap' (if present).
        Each component carries the same leading P axis so they can be combined consistently."""
        z = z or {}
        eps = z.get('epistemic')
        if eps is not None:
            eps_contexts = [eps]
        elif self.epistemic_dim > 0:
            eps_contexts = self.sample_epistemics(inputs, bdims, samples=samples)
        else:
            eps_contexts = [None]
        p_scale, c_scale = self._effective_scales(prior_scale, corrector_scale)
        has_rb = self.residual_bootstrap is not None

        def _eval():
            keys = ['raw', 'prior', 'corrector'] + (['residual_bootstrap'] if has_rb else [])
            comps = {k: [] for k in keys}
            for ctx in eps_contexts:
                indexed = self._with_epistemic(inputs, bdims, ctx)
                raw = self.head(inputs, bdims=bdims, has_ensemble=False).mean()
                prior = jnp.zeros_like(raw)
                corrector = jnp.zeros_like(raw)
                if self._prior is not None and p_scale != 0:
                    prior = p_scale * self._prior(indexed, bdims=bdims, has_ensemble=False).mean()
                    prior = self._broadcast_shift(sg(prior) if stop_prior_grad else prior, raw)
                if self.prior_corrector is not None and c_scale != 0:
                    corrector = c_scale * self.prior_corrector(indexed, bdims=bdims, has_ensemble=False).mean()
                    corrector = self._broadcast_shift(corrector, raw, self._corrector_broadcast_dims)
                local = {'raw': raw, 'prior': prior, 'corrector': corrector}
                if has_rb:
                    rb = p_scale * self.residual_bootstrap(indexed, bdims=bdims, has_ensemble=False).mean()
                    local['residual_bootstrap'] = rb
                if 'ensemble' in z:
                    local = {k: self._select_ensemble(v, z['ensemble'])[None] for k, v in local.items()}
                for k, v in local.items():
                    comps[k].append(v)
            out = {}
            for k, vs in comps.items():
                stacked = jnp.stack(vs, 0)
                out[k] = stacked.reshape((-1,) + stacked.shape[2:])
            return out

        if not frozen or nj.creating():
            return _eval()
        params = self.find()
        self.put(sg(params))
        try:
            return _eval()
        finally:
            self.put(params)

    def mean(self, inputs, *, z=None, bdims=2, **kw):
        """Marginal mean over un-fixed particles, shape [*batch, *q_shape]."""
        return self.particles(inputs, z=z, bdims=bdims, **kw).mean(0)

    def std(self, inputs, *, z=None, bdims=2, **kw):
        """Std over un-fixed particles. Returns zeros when nothing is free.
        Matches BaseAgent.safe_std (eps floor, NaN→0, clipped)."""
        p = self.particles(inputs, z=z, bdims=bdims, **kw)
        if p.shape[0] == 1:
            return jnp.zeros(p.shape[1:], dtype=p.dtype)
        s = jnp.sqrt(jnp.maximum(jnp.var(p, axis=0), 1e-8))
        s = jnp.where(jnp.isfinite(s), s, 0.0)
        return jnp.clip(s, 0, 100.0)

    @staticmethod
    def aggregate(particles, mode='mean'):
        """Reduce over the leading particle axis. mode ∈ {mean, min, individual}."""
        if mode == 'mean':       return particles.mean(0)
        if mode == 'min':        return particles.min(0)
        if mode == 'individual': return particles
        raise NotImplementedError(mode)

    def _select_ensemble(self, v, ens_oh):
        """Per-batch head selection.
        v: [E, *batch, *q_shape]; ens_oh: [*batch, E]; returns [*batch, *q_shape]."""
        E = self.num_heads
        n_batch = ens_oh.ndim - 1
        assert v.shape[0] == E and ens_oh.shape[-1] == E, (v.shape, ens_oh.shape, E)
        v = jnp.moveaxis(v, 0, n_batch)                    # [*batch, E, *q_shape]
        for _ in range(len(self.shape)):
            ens_oh = ens_oh[..., None]                     # broadcast over q_shape
        return (v * ens_oh).sum(n_batch)


class DoubleQCritic:
    """Wraps 1 or 2 PriorCritics and applies the double-Q pessimism mix.

    pessimism < 0: single critic; calls forward to heads[0] unchanged.
    pessimism >= 0: two critics; particles/component_particles/component_means
    return 0.5*(a+b) - pessimism*|a-b|.

    Other PriorCritic attributes (num_heads, epistemic_dim, residual_bootstrap,
    z_space, sample_z, sample_context, ...) are forwarded to heads[0]. Both
    inner critics are constructed with identical configs, so this is consistent.
    """

    def __init__(self, heads, pessimism):
        assert 1 <= len(heads) <= 2, len(heads)
        self.heads = list(heads)
        self.pessimism = float(pessimism)
        self.use_double = len(self.heads) == 2

    def mix(self, a, b):
        if isinstance(a, dict):
            return {k: self.mix(a[k], b[k]) for k in a}
        return 0.5 * (a + b) - self.pessimism * jnp.abs(a - b)

    def __call__(self, *args, **kw):
        return self.heads[0](*args, **kw)

    def particles(self, *args, **kw):
        p = self.heads[0].particles(*args, **kw)
        if self.use_double:
            p = self.mix(p, self.heads[1].particles(*args, **kw))
        return p

    def component_particles(self, *args, **kw):
        p = self.heads[0].component_particles(*args, **kw)
        if self.use_double:
            p = self.mix(p, self.heads[1].component_particles(*args, **kw))
        return p

    def component_means(self, *args, **kw):
        p = self.heads[0].component_means(*args, **kw)
        if self.use_double:
            p = self.mix(p, self.heads[1].component_means(*args, **kw))
        return p

    def __getattr__(self, name):
        if name == 'heads':
            raise AttributeError(name)
        return getattr(self.heads[0], name)


class DoubleQUpdater:
    """Calls a list of SlowUpdaters in sequence."""

    def __init__(self, updaters):
        self.updaters = list(updaters)

    def __call__(self):
        for u in self.updaters:
            u()


# ---------------------------------------------------------------------
# Per-ensemble normalization
# ---------------------------------------------------------------------

class EnsembleNorm(nj.Module):
    """
    Per-head version of your Norm module.

    Input:  x  with leading ensemble axis: [E, ..., H]
    Params: scale/offset per head: [E, H]
    """

    act: str = 'none'

    def __init__(self, impl, num_heads: int, eps=1e-4):
        if '1em' in impl:
            impl, exponent = impl.split('1em')
            eps = 10 ** -int(exponent)
        self._impl = impl
        self._eps = eps
        self.num_heads = int(num_heads)

    def __call__(self, x):
        x = self._norm(x)
        x = get_act(self.act)(x)
        return x

    def _norm(self, x):
        if self._impl == 'none':
            return x

        E = self.num_heads
        assert x.shape[0] == E, (x.shape, E)

        # We'll keep the same dtype behavior as your Norm.
        if self._impl == 'layer':
            # layer norm over last axis
            dtype_in = x.dtype
            x_ = x.astype(f32)
            mean = x_.mean(-1, keepdims=True)
            mean2 = jnp.square(x_).mean(-1, keepdims=True)
            var = jnp.maximum(0, mean2 - jnp.square(mean))

            H = x.shape[-1]
            scale = self.get('scale', jnp.ones, (E, H), f32).astype(x_.dtype)
            offset = self.get('offset', jnp.zeros, (E, H), f32).astype(x_.dtype)

            # broadcast scale/offset to [E, 1, 1, ..., H]
            bshape = (E,) + (1,) * (x.ndim - 2) + (H,)
            scale = scale.reshape(bshape)
            offset = offset.reshape(bshape)

            mult = scale * jax.lax.rsqrt(var + self._eps)
            y = (x_ - mean) * mult + offset
            return cast(y).astype(dtype_in)

        elif self._impl == 'rms':
            dtype_in = x.dtype
            x_ = f32(x) if x.dtype == jnp.float16 else x
            H = x.shape[-1]
            scale = self.get('scale', jnp.ones, (E, H), f32).astype(x_.dtype)
            bshape = (E,) + (1,) * (x.ndim - 2) + (H,)
            scale = scale.reshape(bshape)

            mult = jax.lax.rsqrt((x_ * x_).mean(-1, keepdims=True) + self._eps) * scale
            return (x_ * mult).astype(dtype_in)

        elif self._impl == 'rms_instance':
            # like your rms_instance: RMS over spatial dims (-3,-2), per-channel scale
            dtype_in = x.dtype
            x_ = x.astype(f32)
            H = x.shape[-1]
            scale = self.get('scale', jnp.ones, (E, H), f32).astype(x_.dtype)
            bshape = (E,) + (1,) * (x.ndim - 2) + (H,)
            scale = scale.reshape(bshape)

            mult = jax.lax.rsqrt((x_ * x_).mean((-3, -2), keepdims=True) + self._eps)
            return cast(x_ * (mult * scale)).astype(dtype_in)

        elif self._impl == 'grn':
            # global response norm variant; per-head scale/offset
            assert x.ndim >= 4, x.shape
            dtype_in = x.dtype
            x_ = x.astype(f32)
            H = x.shape[-1]

            norm = jnp.linalg.norm(x_, 2, (-3, -2), keepdims=True)
            norm = norm / (norm.mean(-1, keepdims=True) + self._eps)

            scale = self.get('scale', jnp.ones, (E, H), f32).astype(x_.dtype)
            offset = self.get('offset', jnp.zeros, (E, H), f32).astype(x_.dtype)
            bshape = (E,) + (1,) * (x.ndim - 2) + (H,)
            scale = scale.reshape(bshape)
            offset = offset.reshape(bshape)

            y = (norm * scale + 1) * x_ + offset
            return cast(y).astype(dtype_in)

        elif self._impl == 'instance':
            # instance norm over (-3,-2) with per-head affine
            dtype_in = x.dtype
            x_ = x.astype(f32)
            H = x.shape[-1]

            mean = x_.mean(axis=(-3, -2), keepdims=True)
            var = x_.var(axis=(-3, -2), keepdims=True)

            scale = self.get('scale', jnp.ones, (E, H), f32).astype(x_.dtype)
            offset = self.get('offset', jnp.zeros, (E, H), f32).astype(x_.dtype)
            bshape = (E,) + (1,) * (x.ndim - 2) + (H,)
            scale = scale.reshape(bshape)
            offset = offset.reshape(bshape)

            y = (scale * jax.lax.rsqrt(var + self._eps)) * (x_ - mean) + offset
            return cast(y).astype(dtype_in)

        else:
            raise NotImplementedError(self._impl)


# ---------------------------------------------------------------------
# EnsembleLinear with per-head norm
# ---------------------------------------------------------------------

class EnsembleLinear(nj.Module):
    act: str = 'none'
    norm: str = 'none'
    bias: bool = True
    outscale: float = 1.0
    winit: str = 'normal'
    binit: bool = False
    fan: str = 'in'
    dtype: str = 'default'
    fanin: int = 0
    norm_eps: float = 1e-4

    def __init__(self, units, num_heads: int):
        self.units = (units,) if isinstance(units, int) else tuple(units)
        self.num_heads = int(num_heads)
        self._winit = Initializer(self.winit, self.outscale, self.fan, self.dtype)
        self._binit = Initializer('zeros', 1.0, self.fan, self.dtype)
        self._norm = EnsembleNorm(self.norm, self.num_heads, eps=self.norm_eps, name='norm')

    def __call__(self, x):
        # x: [E, ..., Din]
        assert x.dtype == jaxutils.COMPUTE_DTYPE, (x.dtype, x.shape)
        x = self._layer(x)
        x = self._norm(x)
        x = get_act(self.act)(x)
        return x

    def _layer(self, x):
        E = self.num_heads
        assert x.shape[0] == E, (x.shape, E)
        din = x.shape[-1]
        dout = int(np.prod(self.units))

        # kernel: [E, Din, Dout]
        shape = (E, din, dout)
        fan_shape = (E, self.fanin, dout) if self.fanin else None
        init = self._ensemble_ortho_init if self.winit == 'ortho' else self._winit
        W = self.get('kernel', init, shape, fan_shape).astype(x.dtype)

        # [E, ..., Din] @ [E, Din, Dout] -> [E, ..., Dout]
        y = jnp.einsum('e...i,eio->e...o', x, W)

        if self.bias:
            if self.binit:
                b = self.get('bias', self._winit, (E, dout), shape).astype(x.dtype)
            else:
                b = self.get('bias', self._binit, (E, dout)).astype(x.dtype)

            b = b.reshape((E,) + (1,) * (y.ndim - 2) + (dout,))
            y = y + b

        if len(self.units) > 1:
            y = y.reshape(y.shape[:-1] + self.units)
        assert y.dtype == jaxutils.COMPUTE_DTYPE, (y.dtype, y.shape)
        return y

    def _ensemble_ortho_init(self, shape, fan_shape=None):
        del fan_shape
        E, din, dout = shape
        dtype = jaxutils.PARAM_DTYPE if self.dtype == 'default' else self.dtype
        dtype = getattr(jnp, dtype) if isinstance(dtype, str) else dtype
        keys = jax.random.split(nj.seed(), E)

        def init_one(key):
            nrows, ncols = dout, din
            matshape = (nrows, ncols) if nrows > ncols else (ncols, nrows)
            mat = jax.random.normal(key, matshape, dtype)
            qmat, rmat = jnp.linalg.qr(mat)
            qmat *= jnp.sign(jnp.diag(rmat))
            qmat = qmat.T if nrows < ncols else qmat
            return jnp.moveaxis(qmat.reshape(nrows, din), 0, -1)

        return self.outscale * jax.vmap(init_one)(keys)


# ---------------------------------------------------------------------
# EnsembleBlockLinear: per-head BlockLinear with groups
# ---------------------------------------------------------------------

class EnsembleBlockLinear(nj.Module):
    """
    Per-head version of BlockLinear.

    Input:  x: [E, ..., Din]
    Output: y: [E, ..., Dout] (or reshaped to units)

    Parameters:
      kernel: [E, G, Din/G, Dout/G]
      bias:   [E, Dout] (broadcast over middle dims)
    """
    act: str = 'none'
    norm: str = 'none'
    bias: bool = True
    outscale: float = 1.0
    winit: str = 'normal'
    binit: bool = False
    fan: str = 'in'
    dtype: str = 'default'
    block_fans: bool = False
    block_norm: bool = False
    norm_eps: float = 1e-4

    def __init__(self, units, groups: int, num_heads: int):
        self.units = (units,) if isinstance(units, int) else tuple(units)
        self.groups = int(groups)
        self.num_heads = int(num_heads)
        assert self.groups <= int(np.prod(self.units)), (self.groups, self.units)
        self._winit = Initializer(
            self.winit, self.outscale, self.fan, self.dtype,
            block_fans=self.block_fans
        )
        self._binit = Initializer('zeros', 1.0, self.fan, self.dtype)

        # Per-head norm (either per-block or full)
        if self.block_norm:
            self._norm = [EnsembleNorm(self.norm, self.num_heads, eps=self.norm_eps, name=f'norm{i}') for i in range(self.groups)]
        else:
            self._norm = EnsembleNorm(self.norm, self.num_heads, eps=self.norm_eps, name='norm')

    def __call__(self, x):
        assert x.dtype == jaxutils.COMPUTE_DTYPE, (x.dtype, x.shape)
        y = self._layer(x)
        if self.block_norm and self.norm != 'none':
            # split channels into groups, apply per-head norm per group, then concat
            ys = jnp.split(y, self.groups, axis=-1)
            y = jnp.concatenate([f(z) for f, z in zip(self._norm, ys)], axis=-1)
        else:
            y = self._norm(y)
        y = get_act(self.act)(y)
        return y

    def _layer(self, x):
        E = self.num_heads
        assert x.shape[0] == E, (x.shape, E)

        bdims = x.shape[1:-1]            # everything except E and feature
        indim = x.shape[-1]
        outdim = int(np.prod(self.units))
        G = self.groups

        # pad indim so indim % G == 0 (same behavior as BlockLinear)
        if indim % G != 0:
            pad = int(np.ceil(indim / G)) * G - indim
            x = jnp.concatenate([x, jnp.zeros((E, *bdims, pad), x.dtype)], axis=-1)
            indim = x.shape[-1]

        assert indim % G == 0, (indim, G)
        assert outdim % G == 0, (outdim, G)

        inpg = indim // G
        outpg = outdim // G

        # kernel: [E, G, inpg, outpg]
        shape = (E, G, inpg, outpg)
        kernel = self.get('kernel', self._winit, shape, shape).astype(x.dtype)

        # reshape x to [E, ..., G, inpg]
        xg = x.reshape((E, *bdims, G, inpg))

        # einsum over inpg:
        # [E, ..., G, inpg] x [E, G, inpg, outpg] -> [E, ..., G, outpg]
        yg = jnp.einsum('e...ki,ekio->e...ko', xg, kernel)

        # back to [E, ..., outdim]
        y = yg.reshape((E, *bdims, outdim))

        if self.bias:
            if self.binit:
                b = self.get('bias', self._winit, (E, outdim), shape).astype(x.dtype)
            else:
                b = self.get('bias', self._binit, (E, outdim)).astype(x.dtype)
            b = b.reshape((E,) + (1,) * (y.ndim - 2) + (outdim,))
            y = y + b

        if len(self.units) > 1:
            y = y.reshape(y.shape[:-1] + self.units)

        assert y.dtype == jaxutils.COMPUTE_DTYPE, (y.dtype, y.shape)
        return y

# ---------------------------------------------------------------------
# EnsembleRSSM supporting per-head norm AND blockgru
# ---------------------------------------------------------------------

class EnsembleRSSM(RSSM):
    """
    RSSM with an explicit ensemble axis E=num_heads leading all tensors.

    carry['deter']: [E,B,deter]
    carry['stoch']: [E,B,stoch,classes]
    """

    # Re-declare RSSM fields so Ninjax registers them on this subclass.
    deter: int = 4096
    hidden: int = 2048
    stoch: int = 32
    classes: int = 32
    norm: str = 'rms'
    act: str = 'gelu'
    unroll: bool = False
    unimix: float = 0.01
    outscale: float = 1.0
    imglayers: int = 2
    obslayers: int = 1
    dynlayers: int = 1
    absolute: bool = False
    cell: str = 'gru'
    blocks: int = 8
    block_fans: bool = False
    block_norm: bool = False
    num_heads: int = 1

    def __init__(self, **kw):
        super().__init__(**kw)

    # ---- init / carry helpers ----

    def initial(self, bsize):
        E = self.num_heads
        carry = dict(deter=jnp.zeros([E, bsize, self.deter], f32), stoch=jnp.zeros([E, bsize, self.stoch, self.classes], f32),)
        if self.cell == 'stack':
            carry['feat'] = jnp.zeros([E, bsize, self.hidden], f32)
        return cast(carry)

    def outs_to_carry(self, outs):
        keys = ('deter', 'stoch')
        if self.cell == 'stack':
            keys += ('feat',)
        carry = {}
        for k in keys:
            x = outs[k]
            min_ndim = 5 if k == 'stoch' else 4
            has_ensemble_axis = x.ndim >= min_ndim and x.shape[0] == self.num_heads
            if not has_ensemble_axis:
                x = jnp.broadcast_to(x[None, ...], (self.num_heads,) + x.shape)
            carry[k] = x[:, :, -1]
        return carry

    # ---- small utilities ----

    def _EL(self, name, units, **kw):
        return self.get(name, EnsembleLinear, units, self.num_heads, **kw)

    def _EBL(self, name, units, groups: int, **kw):
        return self.get(name, EnsembleBlockLinear, units, groups, self.num_heads, **kw)

    def _split_keys(self):
        return jax.random.split(nj.seed(), self.num_heads)

    def _sample_stoch(self, logit):
        keys = self._split_keys()
        def sample_one(logit_e, key_e):
            return self._dist(logit_e).sample(seed=key_e)
        return cast(jax.vmap(sample_one, in_axes=(0, 0))(logit, keys))

    def _flatten_EB(self, x):
        E, B = x.shape[:2]
        return x.reshape((E * B,) + x.shape[2:])

    def _unflatten_EB(self, x, E, B):
        return x.reshape((E, B) + x.shape[1:])

    def _ensure_ensemble_axis(self, x, bsize, min_ndim):
        x = jnp.asarray(x)
        has_ensemble_axis = x.ndim >= min_ndim and x.shape[0] == self.num_heads and x.shape[1] == bsize
        if has_ensemble_axis:
            return x
        return jnp.broadcast_to(x[None, ...], (self.num_heads,) + x.shape)

    def _ensure_carry(self, carry):
        first = next(iter(carry.values()))
        bsize = first.shape[1] if (first.ndim >= 3 and first.shape[0] == self.num_heads) else first.shape[0]
        return {k: self._ensure_ensemble_axis(v, bsize, min_ndim=3)for k, v in carry.items()}

    # ---- observe / imagine ----

    def observe(self, carry, action, embed, reset, bdims=2):
        kw = dict(**self.kw, norm=self.norm, act=self.act)
        assert bdims in (1, 2)

        if isinstance(action, dict):
            action = jaxutils.concat_dict(action)

        carry, action, embed = cast((carry, action, embed))
        carry = self._ensure_carry(carry)
        bsize = carry['deter'].shape[1]
        action = self._ensure_ensemble_axis(action, bsize, min_ndim=bdims + 2)
        embed = self._ensure_ensemble_axis(embed, bsize, min_ndim=bdims + 2)
        reset = self._ensure_ensemble_axis(reset, bsize, min_ndim=bdims + 1)

        if bdims == 2:
            def step(c, inp):
                a_t, e_t, r_t = inp
                return self.observe(c, a_t, e_t, r_t, bdims=1)

            # Scan over time while preserving per-step [E,B,...] layout.
            xs = (
                jnp.moveaxis(action, 2, 0),  # [T,E,B,A]
                jnp.moveaxis(embed, 2, 0),   # [T,E,B,D]
                jnp.moveaxis(reset, 2, 0),   # [T,E,B]
            )
            carry, outs = jaxutils.scan(step, carry, xs, self.unroll, axis=0)
            outs = jax.tree_util.tree_map(lambda x: jnp.moveaxis(x, 0, 2), outs)  # [E,B,T,...]
            return cast(carry), cast(outs)

        # bdims == 1
        deter, stoch, action = jaxutils.reset((carry['deter'], carry['stoch'], action), reset)
        deter, feat = self._gru(deter, stoch, action)
        x = embed if self.absolute else jnp.concatenate([feat, embed], -1)
        for i in range(self.obslayers):
            x = self._EL(f'obs{i}', self.hidden, **kw)(x)
        logit = self._elogit('obslogit', x)
        stoch = self._sample_stoch(logit)
        carry = dict(deter=deter, stoch=stoch)
        outs = dict(deter=deter, stoch=stoch, logit=logit)
        if self.cell == 'stack':
            carry['feat'] = feat
            outs['feat'] = feat
        return cast(carry), cast(outs)

    def imagine(self, carry, action, bdims=2):
        assert bdims in (1, 2)
        if isinstance(action, dict):
            action = jaxutils.concat_dict(action)
        carry, action = cast((carry, action))
        carry = self._ensure_carry(carry)
        bsize = carry['deter'].shape[1]
        action = self._ensure_ensemble_axis(action, bsize, min_ndim=bdims + 2)

        if bdims == 2: # Full T step rollout
            def step(c, a_t):
                return self.imagine(c, a_t, bdims=1)
            # Scan over time while preserving per-step [E,B,...] layout.
            xs = jnp.moveaxis(action, 2, 0)  # [T,E,B,A]
            carry, outs = jaxutils.scan(step, carry, xs, self.unroll, axis=0)
            outs = jax.tree_util.tree_map(lambda x: jnp.moveaxis(x, 0, 2), outs)  # [E,B,T,...]
            return cast(carry), cast(outs)

        # bdims == 1
        # TODO: assert carry and action in E, B, T, ...
        deter, feat = self._gru(carry['deter'], carry['stoch'], action)
        logit, prior_feat = self._prior(feat, return_feat=True)
        stoch = self._sample_stoch(logit)
        carry = dict(deter=deter, stoch=stoch)
        outs = dict(deter=deter, stoch=stoch, logit=logit, prior_feat=prior_feat)
        if self.cell == 'stack':
            carry['feat'] = feat
            outs['feat'] = feat
        return cast(carry), cast(outs)

    # ---- prior / logits ----

    def _prior(self, feat, return_feat=False):
        kw = dict(**self.kw, norm=self.norm, act=self.act)
        x = feat
        for i in range(self.imglayers):
            x = self._EL(f'img{i}', self.hidden, **kw)(x)
        logit = self._elogit('imglogit', x)
        return (logit, x) if return_feat else logit

    def _elogit(self, name, x):
        kw = dict(**self.kw, outscale=self.outscale)
        kw['binit'] = False
        x = self._EL(name, self.stoch * self.classes, **kw)(x)
        logit = x.reshape(x.shape[:-1] + (self.stoch, self.classes))
        if self.unimix:
            probs = jax.nn.softmax(logit, -1)
            uniform = jnp.ones_like(probs) / probs.shape[-1]
            probs = (1 - self.unimix) * probs + self.unimix * uniform
            logit = jnp.log(probs)
        return logit

    # ---- dynamics core (now supports blockgru with per-head norms + per-head block params) ----

    def _gru(self, deter, stoch, action):
        kw = dict(**self.kw, norm=self.norm, act=self.act)
        inkw = {**self.kw, 'norm': self.norm, 'binit': False}

        # stoch: [E,B,stoch,classes] -> [E,B,stoch*classes]
        stoch = stoch.reshape((stoch.shape[0], stoch.shape[1], -1))
        action /= sg(jnp.maximum(1, jnp.abs(action)))

        if self.cell == 'gru':
            x0 = self.get('dynnorm', EnsembleNorm, self.norm, self.num_heads)(deter)
            x1 = self._EL('dynin1', self.hidden, **inkw)(stoch)
            x2 = self._EL('dynin2', self.hidden, **inkw)(action)
            x = jnp.concatenate([x0, x1, x2], -1)
            for i in range(self.dynlayers):
                x = self._EL(f'dyn{i}', self.hidden, **kw)(x)
            x = self._EL('dyncore', 3 * self.deter, **self.kw)(x)
            reset, cand, update = jnp.split(x, 3, -1)
            reset = jax.nn.sigmoid(reset)
            cand = jnp.tanh(reset * cand)
            update = jax.nn.sigmoid(update - 1)
            deter = update * cand + (1 - update) * deter
            out = deter
            return deter, out

        if self.cell == 'mgu':
            x0 = self.get('dynnorm', EnsembleNorm, self.norm, self.num_heads)(deter)
            x1 = self._EL('dynin1', self.hidden, **inkw)(stoch)
            x2 = self._EL('dynin2', self.hidden, **inkw)(action)
            x = jnp.concatenate([x0, x1, x2], -1)
            for i in range(self.dynlayers):
                x = self._EL(f'dyn{i}', self.hidden, **kw)(x)
            x = self._EL('dyncore', 2 * self.deter, **self.kw)(x)
            cand, update = jnp.split(x, 2, -1)
            update = jax.nn.sigmoid(update - 1)
            cand = jnp.tanh((1 - update) * cand)
            deter = update * cand + (1 - update) * deter
            out = deter
            return deter, out

        if self.cell == 'blockgru':
            g = self.blocks
            flat2group = lambda x: einops.rearrange(x, 'e ... (g h) -> e ... g h', g=g)
            group2flat = lambda x: einops.rearrange(x, 'e ... g h -> e ... (g h)', g=g)

            # NOTE: use EnsembleLinear here too (with per-head norm inside it).
            x0 = self._EL('dynin0', self.hidden, **kw)(deter)
            x1 = self._EL('dynin1', self.hidden, **kw)(stoch)
            x2 = self._EL('dynin2', self.hidden, **kw)(action)

            # replicate hidden per group: [E,B,H] -> [E,B,g,H]
            x = jnp.concatenate([x0, x1, x2], -1)[..., None, :].repeat(g, -2)

            # concat with deter split into groups
            x = group2flat(jnp.concatenate([flat2group(deter), x], -1))

            for i in range(self.dynlayers):
                x = self._EBL(
                    f'dyn{i}', self.deter, g, **kw,
                    block_norm=self.block_norm, block_fans=self.block_fans
                )(x)

            x = self._EBL(
                'dyncore', 3 * self.deter, g, **self.kw,
                block_fans=self.block_fans, block_norm=self.block_norm
            )(x)

            gates = jnp.split(flat2group(x), 3, -1)
            reset, cand, update = [group2flat(z) for z in gates]
            reset = jax.nn.sigmoid(reset)
            cand = jnp.tanh(reset * cand)
            update = jax.nn.sigmoid(update - 1)
            deter = update * cand + (1 - update) * deter
            out = deter
            return deter, out

        if self.cell == 'stack':
            # This is also made per-head by using EnsembleLinear and EnsembleNorm.
            result = []
            deters = jnp.split(deter, self.dynlayers, -1)
            x = jnp.concatenate([stoch, action], -1)
            x = self._EL('in', self.hidden, **kw)(x)
            for i in range(self.dynlayers):
                skip = x
                x = get_act(self.act)(jnp.concatenate([
                    self.get(f'dyngru{i}norm1', EnsembleNorm, self.norm, self.num_heads)(deters[i]),
                    self.get(f'dyngru{i}norm2', EnsembleNorm, self.norm, self.num_heads)(x),
                ], -1))
                x = self._EL(f'dyngru{i}core', 3 * deters[i].shape[-1], **self.kw)(x)
                reset, cand, update = jnp.split(x, 3, -1)
                reset = jax.nn.sigmoid(reset)
                cand = jnp.tanh(reset * cand)
                update = jax.nn.sigmoid(update - 1)
                deter_i = update * cand + (1 - update) * deters[i]
                result.append(deter_i)

                x = self._EL(f'dyngru{i}proj', self.hidden, **self.kw)(x)
                x += skip

                skip = x
                x = self.get(f'dynmlp{i}norm', EnsembleNorm, self.norm, self.num_heads)(x)
                x = self._EL(f'dynmlp{i}up', deters[i].shape[-1], **self.kw)(x)
                x = get_act(self.act)(x)
                x = self._EL(f'dynmlp{i}down', self.hidden, **self.kw)(x)
                x += skip

            out = self.get('outnorm', EnsembleNorm, self.norm, self.num_heads)(x)
            deter = jnp.concatenate(result, -1)
            return deter, out

        raise NotImplementedError(self.cell)
