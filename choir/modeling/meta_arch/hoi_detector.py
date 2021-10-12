import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, Tuple

from choir.config import configurable
from choir.structures import ImageList, Boxes, Instances
import choir.utils.box_ops as box_ops

from ..backbone import Backbone, build_backbone
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads
from ..transformers import build_transformers
from .build import META_ARCH_REGISTRY

__all__ = ["HOIR"]

@META_ARCH_REGISTRY.register()
class HOIR(nn.Module):
    """
    Human-Object Interaction Detector with Transformers (HOIR).
    It contains the following components:
        1. Per-image feature extraction (aka backbone)
        2. Person detection
        3. Interaction detection with Transformers given the person queries.
    """
    
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        person_generator: nn.Module,
        person_heads: nn.Module,
        transformers: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        in_features: str,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface.
            person_generator: a Region Proposal Network only for person boxes.
            person_heads: a ROI head that performs per-region computation for person boxes.
            transformers: a stack of Transformers to aggregate the information for
                          the given person queries.
            setforward: separate feedforward heads to regress the bounding boxes, recognize
                        the object categories, and classify the interactions.
            pixel_mean, pixel_std: list or tuple with #channels element,
                representing the per-channel mean and std to be used to normalize
                the input image
            in_features: The input features to the Transformers.
        """
        super().__init__()
        self.backbone = backbone
        self.person_generator = person_generator
        self.person_heads = person_heads
        self.transformers = transformers
        self.in_features = in_features

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1))
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        output_shape = backbone.output_shape()
        person_heads = build_roi_heads(cfg, output_shape)
        transformers = build_transformers(cfg, output_shape, person_heads.output_shape)
        return {
            "backbone": backbone,
            "person_generator": build_proposal_generator(cfg, output_shape),
            "person_heads": person_heads,
            "transformers": transformers,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "in_features": cfg.MODEL.ROI_HEADS.IN_FEATURES[0],
        }

    @property
    def device(self):
        return self.pixel_mean.device
        
    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        """
        if not self.training:
            return self.inference(batched_inputs)
        
        images, masks = self.preprocess_image(batched_inputs)
        gt_instances = self.prepare_gt_instances(batched_inputs)

        features = self.backbone(images.tensor)
        masks = self.interpolate_mask(masks, features)
        
        proposals, rpn_losses = self.person_generator(images, features, gt_instances)
        person_instances, person_losses = self.person_heads(images, features, proposals, gt_instances)
        
        results, hoi_losses = self.transformers(features, masks, person_instances, images.image_sizes, gt_instances)

        losses = {}
        losses.update(rpn_losses)
        losses.update(person_losses)
        losses.update(hoi_losses)
        return losses

    def inference(self, batched_inputs, do_postprocess=True):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            do_postprocess (bool): whether to apply post-processing on the outputs.
        """
        assert not self.training
        
        images, masks = self.preprocess_image(batched_inputs)

        features = self.backbone(images.tensor)
        masks = self.interpolate_mask(masks, features)
        
        proposals, _ = self.person_generator(images, features, None)
        person_instances, _ = self.person_heads(images, features, proposals, None)

        results, _ = self.transformers(features, masks, person_instances, images.image_sizes, None)

        if do_postprocess:
            return HOIR._postprocess(results, batched_inputs, images.image_sizes)
        return results
        
    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) / 255. for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images, masks = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images, masks
    
    def interpolate_mask(self, masks, features):
        """
        Interpolate the masks based on the size of features from the backbone.
        """
        features_shape = features[self.in_features].shape
        masks = F.interpolate(
            masks[:, None, ...].float(), size=features_shape[-2:]
        ).to(torch.bool)[:, 0, ...]
        return masks
    
    def prepare_gt_instances(self, batched_inputs):
        """
        Prepare the ground truth instances. Convert the object gt boxes from (x1, y1, x2, y2) to
        (cx, cy, w, h) and normalize them to [0, 1].
        """
        if "instances" not in batched_inputs[0]:
            return None
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        if gt_instances[0].has("interactions"):
            for x in gt_instances:
                # normalize object boxes in gt_instances to [0, 1]
                scale_x, scale_y = 1.0 / x.image_size[1], 1.0 / x.image_size[0]
                for i in x.interactions:
                    gt_boxes = i.gt_boxes.tensor
                    gt_boxes = box_ops.box_xyxy_to_cxcywh(gt_boxes)
                    i.gt_boxes = Boxes(gt_boxes)
                    i.gt_boxes.scale(scale_x, scale_y)
                x.interactions = [i.to(self.device) for i in x.interactions]
                
        return gt_instances
    
    @staticmethod
    def _postprocess(results, batched_inputs, image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            results, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"results": r})
        return processed_results


def detector_postprocess(results, output_height, output_width):
    """
    Resize the output instances.
    The input images are often resized when entering an object detector.
    As a result, we often need the outputs of the detector in a different
    resolution from its inputs.

    This function will resize the raw outputs of an R-CNN detector
    to produce outputs according to the desired output resolution.

    Args:
        results (Instances): the raw outputs from the detector.
            `results.image_size` contains the input image resolution the detector sees.
            This object might be modified in-place.
        output_height, output_width: the desired output resolution.

    Returns:
        Instances: the resized output from the model, based on the output resolution
    """
    # Converts integer tensors to float temporaries
    #   to ensure true division is performed when
    #   computing scale_x and scale_y.
    if isinstance(output_width, torch.Tensor):
        output_width_tmp = output_width.float()
    else:
        output_width_tmp = output_width

    if isinstance(output_height, torch.Tensor):
        output_height_tmp = output_height.float()
    else:
        output_height_tmp = output_height

    scale_x = output_width_tmp / results.image_size[1]
    scale_y = output_height_tmp / results.image_size[0]
    results = Instances((output_height, output_width), **results.get_fields())

    for field in ["object_boxes", "person_boxes"]:
        if results.has(field):
            output_boxes = results.get(field)
            output_boxes.scale(scale_x, scale_y)
            output_boxes.clip(results.image_size)
            results.set(field, output_boxes)

    return results