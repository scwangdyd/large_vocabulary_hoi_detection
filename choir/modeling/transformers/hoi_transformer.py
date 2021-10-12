import logging
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import List, Optional, Dict, Tuple

from choir.config import configurable
from choir.structures import Boxes
from choir.layers import ShapeSpec
from choir.structures.instances import Instances
import choir.utils.box_ops as box_ops
from .transformer import TransformerDecoderLayer, TransformerDecoder
from .position_encoding import build_box_encoding, build_pos_encoding
from .set_criterion import SetCriterion
from .build import TRANSFORMER_REGISTRY

__all__ = ["HOITransformer"]

logger = logging.getLogger(__name__)


@TRANSFORMER_REGISTRY.register()
class HOITransformer(nn.Module):
    """
    Transformer class.
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
        num_duplicates: int = 100,
        return_intermediate: bool = True,
        normalize_before: bool = False,
        num_classes: int = 80,
        num_actions: int = 117,
        test_score_thresh: float = 0.01,
        detections_per_image: int = 100,
        value_with_pos: bool = False,
        set_criterion = None,
    ):
        super().__init__()

        input_channels = input_shape[in_features[0]].channels
        query_channels = query_shape.channels
        
        self.pos_encoding = build_pos_encoding(hidden_dim, position_embedding_type="sine")
        self.box_encoding = build_box_encoding(hidden_dim, position_embedding_type="sine")
        self.offset_embed = nn.Embedding(num_duplicates, hidden_dim)

        self.input_proj = nn.Conv2d(input_channels, hidden_dim, kernel_size=1)
        self.query_proj = nn.Linear(query_channels, hidden_dim)
        self.query_norm = nn.LayerNorm(hidden_dim)

        decoder_layer = TransformerDecoderLayer(
            hidden_dim, num_heads, dim_feedforward, dropout,
            activation, normalize_before, value_with_pos)
        decoder_norm = nn.LayerNorm(hidden_dim)
        self.decoder = TransformerDecoder(
            decoder_layer, num_decoder_layers, decoder_norm, return_intermediate)

        self._reset_parameters()
        
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_duplicates = num_duplicates
        self.num_decoder_layers = num_decoder_layers

        # Feedforward heads
        self.num_classes = num_classes # number of object categories
        self.num_actions = num_actions # number of action categories
        self.test_score_thresh = test_score_thresh
        self.detections_per_image = detections_per_image
        
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.action_embed = nn.Linear(hidden_dim, num_actions)
        
        # Set Criterion
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
            "set_criterion": SetCriterion(cfg)
        }

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, inputs, masks, queries, image_sizes, gt_instances):
        # Image memory features
        f = self.in_features[0]
        bs, c, h, w = inputs[f].shape
        input_feat = self.input_proj(inputs[f]).flatten(2).permute(2, 0, 1)
        input_pos = self.pos_encoding(masks).flatten(2).permute(2, 0, 1)
        input_mask = masks.flatten(1)
        
        # Person query features
        query_instances = self.prepare_queries(queries, masks)
        query_feat = query_instances.query_feat
        query_masks = query_instances.query_masks.permute(2, 0, 1)

        num_queries, bs = max([len(x) for x in queries]), len(queries)
        offset_embed = torch.cat([self.offset_embed.weight] * num_queries, dim=0)
        offset_embed = offset_embed.unsqueeze(1).repeat(1, bs, 1)
        
        hidden, attn = self.decoder(
            tgt=query_feat,
            tgt_mask=query_masks,
            query_pos=offset_embed,
            memory=input_feat,
            memory_key_padding_mask=input_mask,
            pos=input_pos)
        
        predictions = self.predictor(hidden)
        
        if self.training:
            losses = self.set_criterion(predictions, query_instances, gt_instances)
            return predictions, losses

        pred_instances = self.inference(image_sizes, predictions, query_instances)
        return pred_instances, {}

    def prepare_queries(self, queries: List[Instances], mask):
        """
        Convert the person detection instances to format required by Transformer.
        """
        for x in queries:
            x.pred_boxes = Boxes(x.pred_boxes.tensor.detach())
            x.scores = x.scores.detach()
            x.features = x.features.detach()
        
        bs, vis_dim, box_dim = len(queries), queries[0].features.shape[-1], self.hidden_dim
        device = queries[0].features.device
        
        num_queries = max([len(x) for x in queries]) * self.num_duplicates

        query_masks = torch.ones((num_queries, num_queries, bs), dtype=torch.bool, device=device)
        query_vis_embs = torch.zeros((num_queries, bs, vis_dim), dtype=torch.float32, device=device)
        query_box_embs = torch.zeros((num_queries, bs, box_dim), dtype=torch.float32, device=device)
        query_boxes = torch.zeros((num_queries, bs, 4), dtype=torch.float32, device=device)
        query_classes = torch.full((num_queries, bs), -1, dtype=torch.int64, device=device)
        query_scores = torch.zeros((num_queries, bs), dtype=torch.float32, device=device)
        for i, x in enumerate(queries):
            size = len(x) * self.num_duplicates

            query_vis_embs[:size, i, :] = x.features.repeat(1, self.num_duplicates).view(-1, vis_dim)
            
            box = x.pred_boxes.tensor
            query_boxes[:size, i, :] = box.repeat(1, self.num_duplicates).view(-1, 4)
            
            box_embs_x = self.box_encoding(x.image_size, x.pred_boxes.tensor.clone())
            query_box_embs[:size, i, :] = box_embs_x.repeat(1, self.num_duplicates).view(-1, box_dim)

            classes_x = x.pred_classes.unsqueeze(-1).repeat(1, self.num_duplicates).view(-1)
            query_classes[:size, i] = classes_x

            scores_x = x.scores.unsqueeze(-1).repeat(1, self.num_duplicates).view(-1)
            query_scores[:size, i] = scores_x

        for q in range(max([len(x) for x in queries])):
            stx = q * self.num_duplicates
            end = (q + 1) * self.num_duplicates
            query_masks[stx:end, stx:end, :] = False

        query_vis_embs = self.query_norm(self.query_proj(query_vis_embs))
        query_instance = Instances(image_size=tuple(mask.shape[-2:]))
        query_instance.query_feat = query_vis_embs + query_box_embs
        query_instance.query_boxes = query_boxes
        query_instance.query_masks = query_masks
        query_instance.query_classes = query_classes
        query_instance.query_scores = query_scores

        return query_instance
    
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

    def inference(
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



class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x