from choir.layers import ShapeSpec
from .backbone import (
    BACKBONE_REGISTRY,
    FPN,
    Backbone,
    ResNet,
    ResNetBlockBase,
    build_backbone,
    build_resnet_backbone,
    make_stage,
)
from .meta_arch import (
    META_ARCH_REGISTRY,
    build_model,
)
from .proposal_generator import (
    PROPOSAL_GENERATOR_REGISTRY,
    ANCHOR_GENERATOR_REGISTRY,
    RPN_HEAD_REGISTRY,
    build_proposal_generator,
    build_anchor_generator,
    build_rpn_head,
)
from .roi_heads import (
    ROI_BOX_HEAD_REGISTRY,
    ROI_HEADS_REGISTRY,
    ROIHeads,
    StandardROIHeads,
    FastRCNNOutputLayers,
    build_box_head,
    build_roi_heads,
)
from .transformers import TRANSFORMER_REGISTRY, build_transformers
from .test_time_augmentation import DatasetMapperTTA, HOIRWithTTA

_EXCLUDE = {"ShapeSpec"}
__all__ = [k for k in globals().keys() if k not in _EXCLUDE and not k.startswith("_")]
