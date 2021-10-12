# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import numpy as np
from itertools import count
import torch
from fvcore.transforms import HFlipTransform, NoOpTransform
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from choir.data.detection_utils import read_image
from choir.data.transforms import (
    RandomFlip,
    ResizeShortestEdge,
    ResizeTransform,
    apply_augmentations,
)
from choir.structures import Boxes, Instances
from choir.layers import batched_nms
from .meta_arch import HOIR, CHOIR

__all__ = ["DatasetMapperTTA", "HOIRWithTTA"]


class DatasetMapperTTA:
    """
    Implement test-time augmentation for detection data.
    It is a callable which takes a dataset dict from a detection dataset,
    and returns a list of dataset dicts where the images
    are augmented from the input image by the transformations defined in the config.
    This is used for test-time augmentation.
    """

    def __init__(self, cfg):
        self.min_sizes = cfg.TEST.AUG.MIN_SIZES
        self.max_size = cfg.TEST.AUG.MAX_SIZE
        self.flip = cfg.TEST.AUG.FLIP
        self.image_format = cfg.INPUT.FORMAT

    def __call__(self, dataset_dict):
        """
        Args:
            dict: a dict in standard model input format. See tutorials for details.

        Returns:
            list[dict]:
                a list of dicts, which contain augmented version of the input image.
                The total number of dicts is ``len(min_sizes) * (2 if flip else 1)``.
                Each dict has field "transforms" which is a TransformList,
                containing the transforms that are used to generate this image.
        """
        numpy_image = dataset_dict["image"].permute(1, 2, 0).numpy()
        shape = numpy_image.shape
        orig_shape = (dataset_dict["height"], dataset_dict["width"])
        if shape[:2] != orig_shape:
            # It transforms the "original" image in the dataset to the input image
            pre_tfm = ResizeTransform(orig_shape[0], orig_shape[1], shape[0], shape[1])
        else:
            pre_tfm = NoOpTransform()

        # Create all combinations of augmentations to use
        aug_candidates = []  # each element is a list[Augmentation]
        for min_size in self.min_sizes:
            resize = ResizeShortestEdge(min_size, self.max_size)
            aug_candidates.append([resize])  # resize only
            if self.flip:
                flip = RandomFlip(prob=1.0)
                aug_candidates.append([resize, flip])  # resize + flip

        # Apply all the augmentations
        ret = []
        for aug in aug_candidates:
            new_image, tfms = apply_augmentations(aug, np.copy(numpy_image))
            torch_image = torch.from_numpy(np.ascontiguousarray(new_image.transpose(2, 0, 1)))

            dic = copy.deepcopy(dataset_dict)
            dic["transforms"] = pre_tfm + tfms
            dic["image"] = torch_image
            ret.append(dic)
        return ret


class HOIRWithTTA(nn.Module):
    """
    A HOIST with test-time augmentation enabled.
    Its :meth:`__call__` method has the same interface as :meth:`HOIST.forward`.
    """
    def __init__(self, cfg, model, tta_mapper=None, batch_size=3):
        """
        Args:
            cfg (CfgNode):
            model (HOIST): a HOIST to apply TTA on.
            tta_mapper (callable): takes a dataset dict and returns a list of
                augmented versions of the dataset dict. Defaults to
                `DatasetMapperTTA(cfg)`.
            batch_size (int): batch the augmented images into this batch size for inference.
        """
        super().__init__()
        if isinstance(model, DistributedDataParallel):
            model = model.module
        assert isinstance(
            model, HOIR
        ), "TTA is only supported on HOIST. Got a model of type {}".format(type(model))
        self.cfg = cfg.clone()

        self.model = model

        if tta_mapper is None:
            tta_mapper = DatasetMapperTTA(cfg)
        self.tta_mapper = tta_mapper
        self.batch_size = batch_size

    def _batch_inference(self, batched_inputs):
        """
        Execute inference on a list of inputs,
        using batch size = self.batch_size, instead of the length of the list.

        Inputs & outputs have the same format as :meth:`GeneralizedRCNN.inference`
        """
        outputs = []
        inputs = []
        for idx, input in zip(count(), batched_inputs):
            inputs.append(input)
            if len(inputs) == self.batch_size or idx == len(batched_inputs) - 1:
                outputs.extend(self.model.inference(inputs, do_postprocess=False))
                inputs = []
        return outputs

    def __call__(self, batched_inputs):
        """
        Same input/output format as :meth:`GeneralizedRCNN.forward`
        """

        def _maybe_read_image(dataset_dict):
            ret = copy.copy(dataset_dict)
            if "image" not in ret:
                image = read_image(ret.pop("file_name"), self.tta_mapper.image_format)
                image = torch.from_numpy(image).permute(2, 0, 1)  # CHW
                ret["image"] = image
            if "height" not in ret and "width" not in ret:
                ret["height"] = image.shape[1]
                ret["width"] = image.shape[2]
            return ret

        return [self._inference_one_image(_maybe_read_image(x)) for x in batched_inputs]

    def _inference_one_image(self, input):
        """
        Args:
            input (dict): one dataset dict with "image" field being a CHW tensor

        Returns:
            dict: one output dict
        """
        orig_shape = (input["height"], input["width"])
        augmented_inputs, tfms = self._get_augmented_inputs(input)
        # Detect boxes from all augmented versions
        augmented_outputs = self._get_augmented_boxes(augmented_inputs, tfms, orig_shape)
        # merge all detected boxes to obtain final predictions for boxes
        merged_instances = self._merge_detections(augmented_outputs)
        return {"results": merged_instances}

    def _get_augmented_inputs(self, input):
        augmented_inputs = self.tta_mapper(input)
        tfms = [x.pop("transforms") for x in augmented_inputs]
        return augmented_inputs, tfms

    def _get_augmented_boxes(self, augmented_inputs, tfms, orig_shape):
        # 1: forward with all augmented images
        outputs = self._batch_inference(augmented_inputs)
        # 2: union the results
        for output, tfm in zip(outputs, tfms):
            output._image_size = orig_shape
            # Need to inverse the transforms on boxes, to obtain results on original image
            person_boxes = output.person_boxes.tensor
            object_boxes = output.object_boxes.tensor
            original_person_boxes = tfm.inverse().apply_box(person_boxes.cpu().numpy())
            original_object_boxes = tfm.inverse().apply_box(object_boxes.cpu().numpy())
            device = person_boxes.device
            output.person_boxes = Boxes(torch.from_numpy(original_person_boxes).to(device))
            output.object_boxes = Boxes(torch.from_numpy(original_object_boxes).to(device))

        outputs = Instances.cat(outputs)
        return outputs

    def _merge_detections(self, augmented_outputs, nms_thresh=0.5):
        # select from the union of all results
        action_classes = augmented_outputs.action_classes
        object_classes = augmented_outputs.object_classes
        hois = set([tuple([int(action_id), int(object_id)])
                    for action_id, object_id in zip(action_classes, object_classes)])
        hois_to_contiguous_id = {hoi: i for i, hoi in enumerate(hois)}

        device = action_classes.device
        hoi_scores = augmented_outputs.scores
        person_boxes = augmented_outputs.person_boxes.tensor
        object_boxes = augmented_outputs.object_boxes.tensor
        hoi_classes = torch.tensor([hois_to_contiguous_id[tuple([int(action_id), int(object_id)])]
                       for action_id, object_id in zip(action_classes, object_classes)]).to(device)
        # Apply NMS on person boxes and object boxes separately
        person_filter_ids = batched_nms(person_boxes, hoi_scores, hoi_classes, nms_thresh)
        object_filter_ids = batched_nms(object_boxes, hoi_scores, hoi_classes, nms_thresh)

        person_filter_mask = torch.zeros_like(hoi_scores, dtype=torch.bool)
        object_filter_mask = torch.zeros_like(hoi_scores, dtype=torch.bool)
        person_filter_mask[person_filter_ids] = True
        object_filter_mask[object_filter_ids] = True
        filter_mask = torch.logical_or(person_filter_mask, object_filter_mask)
        
        merged_instances = augmented_outputs[filter_mask]
        return merged_instances

    def _rescale_detected_boxes(self, augmented_inputs, merged_instances, tfms):
        augmented_instances = []
        for input, tfm in zip(augmented_inputs, tfms):
            # Transform the target box to the augmented image's coordinate space
            pred_boxes = merged_instances.pred_boxes.tensor.cpu().numpy()
            pred_boxes = torch.from_numpy(tfm.apply_box(pred_boxes))

            aug_instances = Instances(
                image_size=input["image"].shape[1:3],
                pred_boxes=Boxes(pred_boxes),
                pred_classes=merged_instances.pred_classes,
                scores=merged_instances.scores,
            )
            augmented_instances.append(aug_instances)
        return augmented_instances

    def _reduce_pred_masks(self, outputs, tfms):
        # Should apply inverse transforms on masks.
        # We assume only resize & flip are used. pred_masks is a scale-invariant
        # representation, so we handle flip specially
        for output, tfm in zip(outputs, tfms):
            if any(isinstance(t, HFlipTransform) for t in tfm.transforms):
                output.pred_masks = output.pred_masks.flip(dims=[3])
        all_pred_masks = torch.stack([o.pred_masks for o in outputs], dim=0)
        avg_pred_masks = torch.mean(all_pred_masks, dim=0)
        return avg_pred_masks
