import os

import jax
import jax.numpy as jnp

from dreamerv3 import ninjax as nj


DEFAULT_DINOV2_MODEL = '~/.cache/devote/dinov2-small-flax'
DEFAULT_VC1_BASE_MODEL = '~/.cache/devote/vc1-vitb-flax'
DEFAULT_VC1_LARGE_MODEL = '~/.cache/devote/vc1-vitl-flax'


def _preprocess_imagenet_vit(image):
    # Bicubic resize, center crop, and ImageNet normalization. Inputs have
    # already been scaled to [0, 1].
    shape = (image.shape[0], 256, 256, image.shape[-1])
    image = jax.image.resize(image, shape, method='cubic', antialias=True)
    image = image[:, 16:240, 16:240]
    mean = jnp.asarray((0.485, 0.456, 0.406), jnp.float32)
    std = jnp.asarray((0.229, 0.224, 0.225), jnp.float32)
    return ((image - mean) / std).transpose((0, 3, 1, 2))


class Dinov2Encoder(nj.Module):

    def __init__(self, model_id):
        from transformers import FlaxDinov2Model
        with jax.transfer_guard('allow'):
            self.model = FlaxDinov2Model.from_pretrained(model_id)
        self.output_dim = self.model.config.hidden_size

    def __call__(self, image, train=False):
        image = _preprocess_imagenet_vit(image)
        params = self.get('flax', lambda: self.model.params)
        return self.model(image, params=params, train=train).pooler_output


class Vc1Encoder(nj.Module):

    def __init__(self, model_id):
        from transformers import FlaxViTModel
        with jax.transfer_guard('allow'):
            self.model = FlaxViTModel.from_pretrained(
                model_id, add_pooling_layer=False)
        self.output_dim = self.model.config.hidden_size

    def __call__(self, image, train=False):
        image = _preprocess_imagenet_vit(image)
        params = self.get('flax', lambda: self.model.params)
        output = self.model(image, params=params, train=train)
        return output.last_hidden_state[:, 0]


def build_visual_encoder(model):
    """Return (encoder, output key, feature axis, dim) for a visual model."""
    if model == 'resnet18':
        import flaxmodels
        encoder = nj.FromFlax(flaxmodels.ResNet18)(
            output='activations', pretrained='imagenet', normalize=True,
            ckpt_dir=os.path.expanduser('~/.cache/flaxmodels'), name='visual_prior_enc')
        return encoder, 'block4_1', -1, 512
    if model == 'dinov2' or model.startswith('dinov2:'):
        model_id = model.partition(':')[2] or DEFAULT_DINOV2_MODEL
        model_id = os.path.expanduser(model_id)
        if model == 'dinov2' and not os.path.exists(model_id):
            raise FileNotFoundError(
                f'DINOv2-small Flax checkpoint not found at {model_id!r}. '
                'Run `cd experiments && python convert_dinov2.py` once.')
        encoder = Dinov2Encoder(model_id, name='visual_prior_enc')
        return encoder, None, -1, encoder.output_dim
    if model in ('vc1', 'vc1-base', 'vc1-large') or model.startswith('vc1:'):
        defaults = {
            'vc1': DEFAULT_VC1_BASE_MODEL,
            'vc1-base': DEFAULT_VC1_BASE_MODEL,
            'vc1-large': DEFAULT_VC1_LARGE_MODEL,
        }
        model_id = model.partition(':')[2] or defaults[model]
        model_id = os.path.expanduser(model_id)
        if model in defaults and not os.path.exists(model_id):
            variant = 'large' if model == 'vc1-large' else 'base'
            raise FileNotFoundError(
                f'VC-1 {variant} Flax checkpoint not found at {model_id!r}. '
                f'Run `cd experiments && python convert_vc1.py --model {variant}` once.')
        encoder = Vc1Encoder(model_id, name='visual_prior_enc')
        return encoder, None, -1, encoder.output_dim
    raise ValueError(f'Unknown visual_prior: {model!r}')
