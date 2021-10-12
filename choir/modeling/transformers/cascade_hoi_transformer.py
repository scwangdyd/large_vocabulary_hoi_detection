import logging
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import List, Dict, Tuple, Union

import fvcore.nn.weight_init as weight_init
import choir.utils.box_ops as box_ops
from choir.structures import Boxes
from choir.config import configurable
from choir.layers import ShapeSpec, Conv2d
from choir.structures.instances import Instances
from choir.modeling.backbone import Backbone
from choir.modeling.backbone.fpn import _assert_strides_are_log2_contiguous
from .position_encoding import build_box_encoding, build_pos_encoding
from .build import TRANSFORMER_REGISTRY
from .transformer import TransformerDecoderLayer, TransformerDecoder
from .hoi_transformer import MLP
from .set_criterion import SetCriterion

__all__ = ["CascadeHOITransformer"]

logger = logging.getLogger(__name__)


@TRANSFORMER_REGISTRY.register()
class CascadeHOITransformer(nn.Module):
    """
    Cascade HOI Transformer class.
    Copy-paste from torch.nn.Transformer with modifications:
        * positional encodings are passed in MHattention
        * extra LN at the end of encoder is removed
        * decoder returns a stack of activations from all decoding layers
    """
    
    @configurable
    def __init__(
        self,
        input_shape: List[ShapeSpec],
        query_shape: ShapeSpec,
        in_features: List[str],
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        num_duplicates: int = 20,
        return_intermediate: bool = True,
        normalize_before: bool = False,
        area_thresholds: Dict = None,
        num_classes: int = 80,
        num_actions: int = 117,
        test_score_thresh: float = 0.01,
        detections_per_image: int = 100,
        value_with_pos: bool = False,
        set_criterion: nn.Module = None,
    ):
        super().__init__()

        input_channels = [input_shape[f].channels for f in in_features]
        query_channels = query_shape.channels
        
        self.pos_encoding = build_pos_encoding(hidden_dim, position_embedding_type="sine")
        self.box_encoding = build_box_encoding(hidden_dim, position_embedding_type="sine")
        
        self.query_proj = nn.Linear(query_channels, hidden_dim)
        self.query_norm = nn.LayerNorm(hidden_dim)
        
        decoder_layer = TransformerDecoderLayer(
            hidden_dim, num_heads, dim_feedforward, dropout,
            activation, normalize_before, value_with_pos)
        decoder_norm = nn.LayerNorm(hidden_dim)

        self.decoder = nn.ModuleList()
        self.offset_embed = nn.ModuleList()
        for i, c in enumerate(input_channels):
            self.offset_embed.append(nn.Embedding(num_duplicates, hidden_dim))
            self.decoder.append(
                TransformerDecoder(
                    decoder_layer, num_decoder_layers, decoder_norm, return_intermediate
                )
            )

        self.fpn = FPN(in_features, input_shape, hidden_dim)
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_duplicates = num_duplicates
        self.num_decoder_layers = num_decoder_layers
        self.area_thresholds = {f: thresh for f, thresh in zip(in_features, area_thresholds)}

        self._reset_parameters()
        
        # Feedforward heads
        self.num_classes = num_classes # number of object categories
        self.num_actions = num_actions # number of action categories
        self.test_score_thresh = test_score_thresh
        self.detections_per_image = detections_per_image
        
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.action_embed = MLP(hidden_dim, hidden_dim, num_actions, 3)
        
        self.set_criterion = set_criterion
        

    @classmethod
    def from_config(cls, cfg, input_shape, query_shape):
        return {
            "input_shape": input_shape,
            "query_shape": query_shape,
            "in_features": cfg.MODEL.TRANSFORMER.IN_FEATURES,
            "hidden_dim": cfg.MODEL.TRANSFORMER.HIDDEN_DIM,
            "num_heads": cfg.MODEL.TRANSFORMER.NUM_HEADS,
            "num_decoder_layers": cfg.MODEL.TRANSFORMER.NUM_DECODER_LAYERS,
            "dim_feedforward": cfg.MODEL.TRANSFORMER.DIM_FEEDFORWARD,
            "dropout": cfg.MODEL.TRANSFORMER.DROPOUT,
            "num_duplicates": cfg.MODEL.TRANSFORMER.DUPLICATES_PER_QUERY,
            "activation": cfg.MODEL.TRANSFORMER.ACTIVATION,
            "num_classes": cfg.MODEL.TRANSFORMER.NUM_OBJECT_CLASSES,
            "num_actions": cfg.MODEL.TRANSFORMER.NUM_ACTION_CLASSES,
            "test_score_thresh": cfg.MODEL.TRANSFORMER.SCORE_THRESH_TEST,
            "detections_per_image": cfg.MODEL.TRANSFORMER.DETECTIONS_PER_IMAGE,
            "value_with_pos": cfg.MODEL.TRANSFORMER.VALUE_WITH_POS,
            "area_thresholds": cfg.MODEL.TRANSFORMER.AREA_THRESHOLDS,
            "set_criterion": SetCriterion(cfg),
        }

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        inputs: Dict[Union[str, int], Tensor],
        masks: Dict[Union[str, int], Tensor],
        queries: List[Instances],
        image_sizes: List[Tuple],
        gt_instances: List[Instances] = None,
    ):
        """
        Args:
            image_sizes (List[Tuple]): ([height, width], ...)
            inputs (Dict[Union[str, int], Tensor]): feature maps given by the backbone.
            masks (Dict[Union[str, int], Tensor]): All images in the mini-batch are padded
                to the same size. The tensor indicates the padded region for each feature map.
            queries (List[Instances]): Detected person instances, including the RoIAligned
                features, bounding boxes, and confidence scores.
            gt_instances (Instances): Optional. Annotations will be only provided for training.
        """
        if not self.training:
            return self.inference(inputs, masks, queries, image_sizes)
        
        inputs = self.fpn(inputs)

        # First round detection on the coarsest feature maps.
        queries = self.prepare_initial_queries(queries)
        memories = self.prepare_initial_memories(inputs, masks)
        hidden, attn = self.decoder[-1](
            tgt=queries.features,
            tgt_mask=queries.query_masks.permute(2, 0, 1),
            query_pos=queries.offset_embed,
            memory=memories.features,
            memory_key_padding_mask=memories.pad_masks.transpose(0, 1),
            pos=memories.pos_embed,
        )
        preds = self.predictor(hidden)
        losses = self.set_criterion(preds, queries, gt_instances)
        
        # Cascaded detection on the higher-resolution feature maps.
        for f_idx in range(-2, -1-len(self.in_features), -1):
            _, queries = self.prepare_cascade_queries(masks, queries, preds, hidden, f_idx)
            roi_masks = self.vote_rois(masks, queries, memories, attn, f_idx)
                
            memories = self.prepare_cascade_memories(inputs, masks, roi_masks, f_idx)
            
            hidden, attn = self.decoder[f_idx](
                tgt=queries.features,
                tgt_mask=queries.query_masks.permute(2, 0, 1),
                query_pos=queries.offset_embed,
                memory=memories.features,
                memory_key_padding_mask=memories.pad_masks.transpose(0, 1),
                pos=memories.pos_embed,
            )
            preds = self.predictor(hidden)
            losses_f = self.set_criterion(preds, queries, gt_instances)
            fmap = self.in_features[f_idx]
            losses.update({k + f"_f{fmap}": v for k, v in losses_f.items()})

        return {}, losses
    
    def inference(
        self,
        inputs: Dict[Union[str, int], Tensor],
        masks: Dict[Union[str, int], Tensor],
        queries: List[Instances],
        image_sizes: List[Tuple],
    ):
        """
        Inference process.
        Args:
            image_sizes (List[Tuple]): ([height, width], ...)
            inputs (Dict[Union[str, int], Tensor]): feature maps given by the backbone.
            masks (Dict[Union[str, int], Tensor]): All images in the mini-batch are padded
                to the same size. The tensor indicates the padded region for each feature map.
            queries (List[Instances]): Detected person instances, including the RoIAligned
                features, bounding boxes, and confidence scores.
        """
        inputs = self.fpn(inputs)
        cascade_predictions = []
        # First round detection on the coarsest feature maps.
        queries = self.prepare_initial_queries(queries)
        memories = self.prepare_initial_memories(inputs, masks)
        hidden, attn = self.decoder[-1](
            tgt=queries.features,
            tgt_mask=queries.query_masks.permute(2, 0, 1),
            query_pos=queries.offset_embed,
            memory=memories.features,
            memory_key_padding_mask=memories.pad_masks.transpose(0, 1),
            pos=memories.pos_embed,
        )
        preds = self.predictor(hidden)
        base_queries = queries
        base_preds = preds.copy()
        base_preds.pop("aux_outputs")
        
        # Cascaded detection on the higher-resolution feature maps.
        cascade_predictions = []
        bn, bs, _ = base_preds["pred_logits"].shape
        device = base_preds["pred_logits"].device
        count = torch.ones(bn, bs).to(device)
        for f_idx in range(-2, -1-len(self.in_features), -1):
            stop, queries = self.prepare_cascade_queries(masks, queries, preds, hidden, f_idx)
            if stop:
                break
            roi_masks = self.vote_rois(masks, queries, memories, attn, f_idx)
            
            memories = self.prepare_cascade_memories(inputs, masks, roi_masks, f_idx)
            
            hidden, attn = self.decoder[f_idx](
                tgt=queries.features,
                tgt_mask=queries.query_masks.permute(2, 0, 1),
                query_pos=queries.offset_embed,
                memory=memories.features,
                memory_key_padding_mask=memories.pad_masks.transpose(0, 1),
                pos=memories.pos_embed,
            )
            preds = self.predictor(hidden)
            cascade_predictions.append({"query_indices": queries.query_indices, "preds": preds})
            query_indices = queries.query_indices
            for b in range(bs):
                count[query_indices[:, b], b] += 1
            for k, v in base_preds.items():
                for b in range(v.shape[1]):
                    v[query_indices[:, b], b] = v[query_indices[:, b], b] + preds[k][:, b]
            
        for k, v in base_preds.items():
            v /= count[:, :, None]

        pred_instances = self.generate_pred_instances(image_sizes, base_preds, base_queries)
        
        return pred_instances, {}
    
    def predictor(self, hidden: Tensor):
        scores = self.class_embed(hidden)
        boxes = self.bbox_embed(hidden).sigmoid()
        actions = self.action_embed(hidden)
        predictions = {
            "pred_logits": scores[-1],
            "pred_boxes": boxes[-1],
            "pred_actions": actions[-1],
        }
        
        if self.set_criterion.aux_loss:
            predictions["aux_outputs"] = self.aux_preds(scores, boxes, actions)

        return predictions

    @torch.jit.unused
    def aux_preds(self, scores: Tensor, boxes: Tensor, actions: Tensor):
        # this is a workaround to make torchscript happy, as torchscript doesn't support dictionary
        # with non-homogeneous values, such as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, "pred_actions": c}
                for a, b, c in zip(scores[:-1], boxes[:-1], actions[:-1])]
    
    def prepare_initial_memories(self, inputs, masks):
        coarsest = self.in_features[-1]
        bs, _, h, w = inputs[coarsest].shape
        features = inputs[coarsest].flatten(2).permute(2, 0, 1)
        pos_embed = self.pos_encoding(masks[coarsest]).flatten(2).permute(2, 0, 1)
        pad_masks = masks[coarsest].flatten(1).transpose(0, 1)
        patch_indices = torch.tensor([[y, x] for y in range(h) for x in range(w)])[:, None].repeat(1, bs, 1)

        memory_instances = Instances(image_size=tuple(masks[coarsest].shape[-2:]))
        memory_instances.features = features
        memory_instances.pos_embed = pos_embed
        memory_instances.pad_masks = pad_masks
        memory_instances.patch_indices = patch_indices
        return memory_instances

    def prepare_cascade_memories(self, inputs, masks, roi_masks, f_idx):
        """
        Prepare the memories by masking the region of interests in the high resolution
        feature maps based on the predictions from the last level.
        Args:
            inputs (Dict[str: Tensor]): feature maps from the backbone.
            masks (Dict[str: Tensor]): padding masks for each feature level.
            roi_masks (List[Instances]): prepared roi regions.
            f_idx (int): The index of current feature map.
        """
        f_map = self.in_features[f_idx]
        bs, _, h, w = inputs[f_map].shape
        device = inputs[f_map].device
        
        features = inputs[f_map].flatten(2).permute(2, 0, 1)
        pos_embed = self.pos_encoding(masks[f_map])
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        pad_masks = masks[f_map].flatten(1).transpose(0, 1)
        patch_indices = torch.tensor([[y, x] for y in range(h) for x in range(w)])[:, None].repeat(1, bs, 1)

        roi_patches = int(max([(roi_masks[b] >= 1.).sum() for b in range(bs)]))
        if self.training:
            # Control the number of patch per image
            images_height = (~masks[f_map]).sum(dim=1).max(-1)[0]
            images_width = (~masks[f_map]).sum(dim=2).max(-1)[0]
            area_thresh = self.area_thresholds[f_map]
            patch_thresh = int(max(1000, max(images_height) * max(images_width) * area_thresh))

            k = min(patch_thresh, h * w)
            
            for b in range(bs):
                patches_per_img = (roi_masks[b] >= 1.).sum()
                if patches_per_img > k:
                    roi_masks[b][roi_masks[b] >= 1.] += torch.rand(patches_per_img).to(device)
        else:
            k = roi_patches

        rois_masks = roi_masks.flatten(1).transpose(0, 1)        
        topk_value, topk_indices = torch.topk(rois_masks, k, dim=0, largest=True)
        
        features = torch.stack([features[topk_indices[:, b], b, :] for b in range(bs)], dim=1)
        pos_embed = torch.stack([pos_embed[topk_indices[:, b], b, :] for b in range(bs)], dim=1)
        pad_masks = torch.stack([pad_masks[topk_indices[:, b], b] for b in range(bs)], dim=1)
        patch_indices = torch.stack([patch_indices[topk_indices[:, b], b] for b in range(bs)], dim=1)
        
        memory_instances = Instances(image_size=tuple(masks[f_map].shape[-2:]))
        memory_instances.features = features
        memory_instances.pos_embed = pos_embed
        memory_instances.pad_masks = pad_masks
        memory_instances.patch_indices = patch_indices
        return memory_instances

    @torch.no_grad()
    def vote_rois(self, masks, queries, memories, attn, f_idx, score_thresh=0.05):
        f_map = self.in_features[f_idx]
        area_thresh = self.area_thresholds[f_map]
        bs, h, w = masks[f_map].shape
        device = masks[f_map].device
        
        # Valid queries to refine in the next round
        pred_boxes = queries.pred_boxes
        cat_scores = queries.cat_scores
        hoi_scores = queries.hoi_scores
        box_areas = pred_boxes[..., 2] * pred_boxes[..., 3]
        
        images_height = (~masks[f_map]).sum(dim=1).max(-1)[0]
        images_width = (~masks[f_map]).sum(dim=2).max(-1)[0]
        images_area = images_height * images_width
        box_areas = box_areas * images_area[None, :]
        area_thresh = int(max(1000, max(images_height) * max(images_width) * area_thresh))
        
        kept_masks = torch.logical_and(box_areas < area_thresh, cat_scores > score_thresh)
        kept_masks = torch.logical_and(kept_masks, hoi_scores > score_thresh)
        kept_indices, batch_indices = torch.nonzero(kept_masks, as_tuple=True)
        
        prev_f_map = self.in_features[f_idx + 1]
        roi_masks_2x_small = torch.rand(masks[prev_f_map].shape).to(device) * (10**-6)
        # Vote rois based on the attention weights
        patch_indices = memories.patch_indices
        for q, b in zip(kept_indices, batch_indices):
            attn_per_q = attn[:, b, q].max(dim=0)[0]
            attn_patch_indices = patch_indices[:, b]
            attn_patch_indices = (attn_patch_indices[:, 0], attn_patch_indices[:, 1])
            roi_masks_2x_small[b][attn_patch_indices] = torch.max(
                roi_masks_2x_small[b][attn_patch_indices],
                attn_per_q
            )
        roi_masks = F.interpolate(roi_masks_2x_small[None, ...], size=(h, w))[0]
        k = min(area_thresh, h * w)
        # We observe that it is important to kept the high attention regions.
        topk_value, topk_indices = torch.topk(roi_masks.flatten(1).transpose(0, 1), k, dim=0, largest=True, sorted=True)
        for b, v in enumerate(topk_value[-1]):
            roi_masks[b][roi_masks[b] >= v] = 1.
        
        pad = 1 * abs(f_idx)
        for q, b in zip(kept_indices, batch_indices):
            x1 = torch.clamp(pred_boxes[q, b, 0] - pred_boxes[q, b, 2] / 2, min=0., max=1.) 
            x2 = torch.clamp(pred_boxes[q, b, 0] + pred_boxes[q, b, 2] / 2, min=0., max=1.)
            y1 = torch.clamp(pred_boxes[q, b, 1] - pred_boxes[q, b, 3] / 2, min=0., max=1.)
            y2 = torch.clamp(pred_boxes[q, b, 1] + pred_boxes[q, b, 3] / 2, min=0., max=1.)

            x1 = torch.clamp(torch.floor(x1 * (~masks[f_map][b]).sum(dim=1).max() - pad).long(), min=0, max=w)
            x2 = torch.clamp(torch.floor(x2 * (~masks[f_map][b]).sum(dim=1).max() + pad).long(), min=0, max=w)
            y1 = torch.clamp(torch.ceil(y1 * (~masks[f_map][b]).sum(dim=0).max() - pad).long(), min=0, max=h)
            y2 = torch.clamp(torch.ceil(y2 * (~masks[f_map][b]).sum(dim=0).max() + pad).long(), min=0, max=h)

            roi_masks[b, y1:y2, x1:x2] = 1.

        if self.training:
            target_classes = queries.target_classes
            target_boxes = queries.target_boxes
            box_areas = target_boxes[..., 2] * target_boxes[..., 3]
            box_areas = box_areas * images_area[None, :]
            # Ignore boxes exceed the area threshold and background boxes
            keep = torch.logical_and(box_areas < area_thresh, box_areas > 0)
            keep = torch.logical_and(keep, target_classes < self.num_classes)
            kept_indices, batch_indices = torch.nonzero(keep, as_tuple=True)
            # Assign value `num_duplicates` to positions within ground boxes.
            for q, b in zip(kept_indices, batch_indices):
                x1 = torch.clamp(target_boxes[q, b, 0] - target_boxes[q, b, 2] / 2, min=0., max=1.) 
                x2 = torch.clamp(target_boxes[q, b, 0] + target_boxes[q, b, 2] / 2, min=0., max=1.)
                y1 = torch.clamp(target_boxes[q, b, 1] - target_boxes[q, b, 3] / 2, min=0., max=1.)
                y2 = torch.clamp(target_boxes[q, b, 1] + target_boxes[q, b, 3] / 2, min=0., max=1.)

                x1 = torch.clamp(torch.floor(x1 * (~masks[f_map][b]).sum(dim=1).max() - pad).long(), min=0, max=w)
                x2 = torch.clamp(torch.floor(x2 * (~masks[f_map][b]).sum(dim=1).max() + pad).long(), min=0, max=w)
                y1 = torch.clamp(torch.ceil(y1 * (~masks[f_map][b]).sum(dim=0).max() - pad).long(), min=0, max=h)
                y2 = torch.clamp(torch.ceil(y2 * (~masks[f_map][b]).sum(dim=0).max() + pad).long(), min=0, max=h)

                roi_masks[b, y1:y2, x1:x2] = 1. + torch.rand((y2-y1, x2-x1)).to(device)
                
        return roi_masks
        
    def prepare_initial_queries(self, queries: List[Instances]):
        """
        Convert the person detection instances to format required by Transformers.
        """
        for x in queries:
            x.pred_boxes = x.pred_boxes.tensor.detach()
            x.scores = x.scores.detach()
            x.features = x.features.detach()

        bs = len(queries)
        vis_box = self.hidden_dim
        vis_dim = queries[0].features.shape[-1]
        num_pad = max([len(x) for x in queries]) * self.num_duplicates
        device = queries[0].features.device
        
        query_masks   = torch.ones((num_pad, num_pad, bs), dtype=torch.bool).to(device)
        query_boxes   = torch.zeros((num_pad, bs, 4), dtype=torch.float32).to(device)
        query_scores  = torch.zeros((num_pad, bs), dtype=torch.float32).to(device)
        query_classes = torch.full((num_pad, bs), -1, dtype=torch.int64).to(device)
        query_indices = torch.zeros((num_pad, bs), dtype=torch.int64).to(device)
        query_vis_emb = torch.zeros((num_pad, bs, vis_dim), dtype=torch.float32).to(device)
        query_box_emb = torch.zeros((num_pad, bs, vis_box), dtype=torch.float32).to(device)
        
        for b, x in enumerate(queries):
            size = len(x) * self.num_duplicates
            query_indices[:, b] = torch.arange(num_pad)

            query_boxes[:size, b, :] = x.pred_boxes.repeat(1, self.num_duplicates).view(-1, 4)
            query_scores[:size, b] = x.scores[:, None].repeat(1, self.num_duplicates).view(-1)

            classes = x.pred_classes[:, None].repeat(1, self.num_duplicates).view(-1)
            query_classes[:size, b] = classes
            
            box_emb = self.box_encoding(x.image_size, x.pred_boxes.clone())
            query_box_emb[:size, b, :] = box_emb.repeat(1, self.num_duplicates).view(-1, vis_box)
            
            vis_emb = x.features.repeat(1, self.num_duplicates).view(-1, vis_dim)
            query_vis_emb[:size, b, :] = vis_emb

        num_qrs = max([len(x) for x in queries])
        offset_embed = torch.cat([self.offset_embed[-1].weight] * num_qrs, dim=0)
        offset_embed = offset_embed.unsqueeze(1).repeat(1, bs, 1)
        offset_indices = torch.cat([torch.arange(self.num_duplicates)] * num_qrs, dim=0)
        offset_indices = offset_indices.unsqueeze(1).repeat(1, bs).to(device)
        batch_indices = torch.stack([torch.tensor([b] * num_pad) for b in range(bs)], dim=1).to(device)
        query_vis_emb = self.query_norm(self.query_proj(query_vis_emb))
        # This mask ensures that position is allowed to attend the unmasked positions.
        # If a BoolTensor is provided, positions with True is not allowed to attend
        # while False values will be unchanged.
        for q in range(num_qrs):
            stx = q * self.num_duplicates
            end = (q + 1) * self.num_duplicates
            query_masks[stx:end, stx:end, :] = False

        query_instances = Instances(image_size=(0, 0))
        query_instances.features = query_vis_emb + query_box_emb
        query_instances.query_boxes = query_boxes
        query_instances.query_masks = query_masks
        query_instances.query_scores = query_scores
        query_instances.query_classes = query_classes
        query_instances.query_indices = query_indices
        query_instances.query_box_emb = query_box_emb
        query_instances.offset_embed = offset_embed
        query_instances.offset_indices = offset_indices
        query_instances.batch_indices = batch_indices
        return query_instances

    def prepare_cascade_queries(
        self,
        masks: List[Tensor],
        queries: List[Instances],
        predictions: Dict[str, Tensor],
        hidden: List[Tensor],
        f_idx: int,
        score_thresh=0.05,
    ):
        """
        Prepare queries for the next higher-resolution feature map.
        Args:
            queries (List[Instances]): prepared query instances from the last level.
            predictions (Dict[str, Tensor]): prediction outputs from the model.
            hidden (List[Tensor]): hidden feature embeddings.
            f_idx (int): the index of current feature map.
        """
        for field, value in queries.get_fields().items():
            queries.set(field, value.detach())
        
        queries.features = hidden[-1].detach()
        
        offset_indices = queries.offset_indices
        queries.offset_embed = self.offset_embed[f_idx].weight[offset_indices]

        bs = len(queries.features[0])
        keep, num_kept_max = self.find_valid_queries(masks, queries, predictions, f_idx, score_thresh)

        filtered_queries = Instances(image_size=queries.image_size)
        for field, value in queries.get_fields().items():
            value_per_img = []
            if field == "query_masks":
                for b in range(bs):
                    value_per_img.append(value[keep[:, b], :, b][:, keep[:, b]])
                stacked_value = torch.stack(value_per_img, dim=2)
            else:
                for b in range(bs):
                    value_per_img.append(value[keep[:, b], b])
                stacked_value = torch.stack(value_per_img, dim=1)
            filtered_queries.set(field, stacked_value)

        # Replace the target labels
        if self.training:
            filtered_queries = self.replace_targets(masks, filtered_queries, f_idx)

        stop = num_kept_max == 0
        return stop, filtered_queries
    
    @torch.no_grad()
    def find_valid_queries(self, masks, queries, predictions, f_idx, score_thresh):
        num_qrs, bs = len(queries.features) // self.num_duplicates, len(queries.features[0])
        # Update predictions
        queries.pred_boxes = predictions["pred_boxes"]
        queries.pred_classes = predictions["pred_logits"].softmax(dim=-1)[..., :-1].argmax(dim=-1)
        queries.cat_scores = predictions["pred_logits"].softmax(dim=-1)[..., :-1].max(dim=-1)[0]
        queries.hoi_scores = predictions["pred_actions"].sigmoid().max(dim=-1)[0]
        
        f_map = self.in_features[f_idx]
        
        # Filtering out queries which will be be used in the next round.
        pred_box_areas = queries.pred_boxes[..., 2] * queries.pred_boxes[..., 3]
        images_height = (~masks[f_map]).sum(dim=1).max(-1)[0]
        images_width = (~masks[f_map]).sum(dim=2).max(-1)[0]
        images_area = images_height * images_width
        pred_box_areas = pred_box_areas * images_area[None, :]
        
        area_thresh = self.area_thresholds[f_map]
        area_thresh = max(1000, max(images_height) * max(images_width) * area_thresh)
        # area_thresh = max(images_height) * max(images_width) * area_thresh
        
        cat_scores = queries.cat_scores
        hoi_scores = queries.hoi_scores
        keep = torch.logical_and(pred_box_areas < area_thresh, cat_scores > score_thresh)
        keep = torch.logical_and(keep, hoi_scores > score_thresh)
        
        if self.training: # Ensure the queries assigned with foreground targets are kept
            gt_box_areas = queries.target_boxes[..., 2] * queries.target_boxes[..., 3]
            gt_box_areas = gt_box_areas * images_area[None, :]
            gt_keep = torch.logical_and(gt_box_areas > 0, gt_box_areas < area_thresh)
            keep = torch.logical_or(keep, gt_keep)
            
        num_kept_per_img = [sum(keep[:, b]) for b in range(bs)]
        num_kept_max = max(2, max(num_kept_per_img)) if self.training else max(num_kept_per_img)
        # Pad background queries such that the length is same.
        for b, num_kept in enumerate(num_kept_per_img):
            num_choice = num_kept_max - num_kept
            if num_choice > 0:
                bgs = torch.nonzero(~keep[:, b], as_tuple=True)[0]
                bgs = bgs[torch.randperm(len(bgs))[:num_choice]]
                keep[bgs, b] = True
        
        return keep, num_kept_max

    @torch.no_grad()
    def replace_targets(self, masks, queries, f_idx):
        """
        Replace assigned labels if the predicted ROIs do not include the true targets.
        Args:
            queries (Instances): prepared query instances from the last level
            area_thresh (float): area threshold to filter boxes
        """
        f_map = self.in_features[f_idx]
        area_thresh = self.area_thresholds[f_map]
        images_height = (~masks[f_map]).sum(dim=1).max(-1)[0]
        images_width = (~masks[f_map]).sum(dim=2).max(-1)[0]
        images_area = images_height * images_width
        area_thresh = max(1000, max(images_height) * max(images_width) * area_thresh)
        # area_thresh = max(images_height) * max(images_width) * area_thresh

        target_classes = queries.target_classes
        target_actions = queries.target_actions
        target_boxes = queries.target_boxes
        
        box_areas = target_boxes[..., 2] * target_boxes[..., 3]
        box_areas = box_areas * images_area
        indices = box_areas >= area_thresh

        # For targets whose area > threshold, we only predict its category
        # but not regress its bounding box.
        target_classes[indices] = self.num_classes
        target_actions[indices] = 0.
        target_boxes[indices] = 0.

        queries.target_classes = target_classes
        queries.target_actions = target_actions
        queries.target_boxes = target_boxes
        return queries

    def generate_pred_instances(
        self,
        image_sizes: List[Tuple],
        predictions: Dict[str, Tensor],
        queries: List[Instances],
    ):
        """
        image_sizes (List[Tuple]): each image size is a tuple (w, h).
        predictions (Dict): predictions made at various feature maps.
        query_instances (List[Instances]): query instances with query_boxes,
            query_scores, query_classes
        """
        scores = predictions["pred_logits"].softmax(-1)
        boxes = predictions["pred_boxes"]
        actions = predictions["pred_actions"].sigmoid()

        results = [self.inference_single_image(
                        boxes[:, b], scores[:, b], actions[:, b],
                        queries.query_boxes[:, b], queries.query_scores[:, b],
                        queries.query_classes[:, b], image_size)
                        for b, image_size in enumerate(image_sizes)]
        
        return results
    
    def inference_single_image(
        self,
        boxes: Tensor,
        object_scores: Tensor,
        action_scores: Tensor,
        query_boxes: Tensor,
        query_scores: Tensor,
        query_classes: Tensor,
        image_size: Tuple,
    ):
        """
        boxes (Tensor): Predicted boxes with shape (#queries, #lvls, 4)
        scores (Tensor): Predicted scores with shape (#queries, #lvls, #cls)
        queries (Instances): Predicted query instances with field query_boxes,
                             query_classes, query_scores
        image_size (Tuple): A tuple with the original size of image (h, w)
        """
        device = object_scores.device
        num_levels = len(boxes) // len(query_boxes)
        # Convert boxes from format (cx, cy, w, h) to format (x, y, w, h).
        # Rescale the boxes to image scale.
        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        boxes = box_ops.box_scale(boxes, scale_x=image_size[1], scale_y=image_size[0])

        query_boxes = torch.cat([query_boxes] * num_levels, dim=0)
        query_scores = torch.cat([query_scores] * num_levels, dim=0)
        query_classes = torch.cat([query_classes] * num_levels, dim=0)

        action_scores = action_scores.unsqueeze(-1).repeat(1, 1, self.num_classes)
        object_scores = object_scores[:, :-1].unsqueeze(1).repeat(1, self.num_actions, 1)
        person_scores = query_scores[:, None, None].repeat(1, self.num_actions, self.num_classes)
        
        combo_scores = object_scores * action_scores * person_scores
        
        filter_mask = combo_scores > self.test_score_thresh
        if filter_mask.sum() > self.detections_per_image:
            topk_thresh = torch.sort(combo_scores[filter_mask])[0][-self.detections_per_image]
            filter_mask = combo_scores >= topk_thresh
        filter_inds = filter_mask.nonzero(as_tuple=False)

        res = Instances(image_size)
        res.scores = combo_scores[filter_mask]
        res.object_scores = object_scores[filter_mask]
        res.action_scores = action_scores[filter_mask]
        res.person_scores = person_scores[filter_mask]
        res.object_boxes = Boxes(boxes[filter_inds[:, 0]])
        res.person_boxes = Boxes(query_boxes[filter_inds[:, 0]])
        res.object_classes = filter_inds[:, 2]
        res.action_classes = filter_inds[:, 1]
        res.person_classes = query_classes[filter_inds[:, 0]]
        res.offset_indices = filter_inds[:, 0]

        return res



        
class FPN(Backbone):
    def __init__(self, in_features, input_shapes, out_channels, fuse_type="sum"):
        super(FPN, self).__init__()
        strides = [input_shapes[f].stride for f in in_features]
        in_channels_per_feature = [input_shapes[f].channels for f in in_features]
        
        _assert_strides_are_log2_contiguous(strides)
        lateral_convs = []
        output_convs = []

        for idx, in_channels in enumerate(in_channels_per_feature):

            lateral_conv = Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
            output_conv = Conv2d(out_channels, out_channels, kernel_size=1, bias=True)
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)
            stage = int(math.log2(strides[idx]))
            self.add_module("fpn_lateral{}".format(stage), lateral_conv)
            self.add_module("fpn_output{}".format(stage), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]
        self.in_features = in_features
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        # self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in strides}
        self._out_feature_strides = {f: s for f, s in zip(in_features, strides)}

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = strides[-1]
        assert fuse_type in {"avg", "sum"}
        self._fuse_type = fuse_type
    
    @property
    def size_divisibility(self):
        return self._size_divisibility
    
    def forward(self, inputs):
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "res5") to
                feature map tensor for each feature level in high to low resolution order.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        # Reverse feature maps into top-down order (from low to high resolution)
        x = [inputs[f] for f in self.in_features[::-1]]
        results = []
        prev_features = self.lateral_convs[0](x[0])
        results.append(self.output_convs[0](prev_features))
        for features, lateral_conv, output_conv in zip(
            x[1:], self.lateral_convs[1:], self.output_convs[1:]
        ):
            top_down_features = F.interpolate(prev_features, scale_factor=2, mode="nearest")
            lateral_features = lateral_conv(features)
            prev_features = lateral_features + top_down_features
            if self._fuse_type == "avg":
                prev_features /= 2
            results.insert(0, output_conv(prev_features))

        assert len(self._out_features) == len(results)
        return dict(zip(self._out_features, results))