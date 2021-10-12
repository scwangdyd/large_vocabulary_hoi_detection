from choir.utils.registry import Registry

TRANSFORMER_REGISTRY = Registry("TRANSFORMER")  # noqa F401 isort:skip
TRANSFORMER_REGISTRY.__doc__ = """
Registry for HOI Transformers.
Transformer take features of detected person and perform per-region computation
for interacting objects.

The registered object will be called with `obj(cfg, input_shape)`.
The call is expected to return an :class:`Transformer`.
"""


def build_transformers(cfg, input_shape, query_shape):
    """
    Build Transformers defined by `cfg.MODEL.TRANSFORMER`.
    """
    name = cfg.MODEL.TRANSFORMER.BASE
    return TRANSFORMER_REGISTRY.get(name)(cfg, input_shape, query_shape)