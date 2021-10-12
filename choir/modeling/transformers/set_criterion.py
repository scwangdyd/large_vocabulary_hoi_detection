import logging
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import List, Dict, Tuple
from fvcore.nn import sigmoid_focal_loss_jit

from choir.config import configurable
from choir.structures import Instances
from choir.utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from ..matcher import build_matcher

__all__ = ["SetCriterion"]

logger = logging.getLogger(__name__)


class SetCriterion(nn.Module):
    """
    Copy-paste from DETR
    """
    @configurable
    def __init__(
        self,
        num_classes: int,
        num_actions: int,
        matcher: nn.Module,
        losses: List[str],
        aux_loss: bool = True,
        loss_weights: Dict = None,
        num_duplicates: int = 100,
        eos_coefficient: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_actions = num_actions

        self.matcher = matcher
        self.losses = losses
        self.aux_loss = aux_loss
        self.loss_weights = loss_weights
        self.num_duplicates = num_duplicates
        
        self.register_loss_weights(eos_coefficient)
        
    @classmethod
    def from_config(cls, cfg):
        loss_weights = {
            "cls": cfg.MODEL.TRANSFORMER.SET_COST_CLASS,
            "bbox": cfg.MODEL.TRANSFORMER.SET_COST_BBOX,
            "giou": cfg.MODEL.TRANSFORMER.SET_COST_GIOU,
            "action": cfg.MODEL.TRANSFORMER.SET_COST_ACTION
        }
        matcher = build_matcher(cfg)
        return {
            "num_classes": cfg.MODEL.TRANSFORMER.NUM_OBJECT_CLASSES,
            "num_actions": cfg.MODEL.TRANSFORMER.NUM_ACTION_CLASSES,
            "matcher": matcher,
            "losses": cfg.MODEL.TRANSFORMER.LOSSES,
            "aux_loss": cfg.MODEL.TRANSFORMER.AUX_LOSS,
            "loss_weights": loss_weights,
            "eos_coefficient": cfg.MODEL.TRANSFORMER.EOS_COEFFICIENT,
            "num_duplicates": cfg.MODEL.TRANSFORMER.DUPLICATES_PER_QUERY,
        }

    def register_loss_weights(self, eos_coefficient):
        """
        Register parameters for re-weighting object/action classes.
        """
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = eos_coefficient
        self.register_buffer("empty_weight", empty_weight)

        if "actions_bce" in self.losses:
            action_weights = torch.ones(self.num_actions) * 5.
            self.register_buffer("action_weights", action_weights)
    
    def forward(
        self,
        predictions: Dict[str, Tensor],
        query_instances: List[Instances],
        gt_instances: List[Instances],
    ):
        outputs_without_aux = {k: v for k, v in predictions.items() if k != "aux_outputs"}
        # Bipartite matching to find one-to-one matching
        indices, targets = self.matcher(
            outputs_without_aux, query_instances, gt_instances)
        # losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs_without_aux, targets, indices))

        if self.aux_loss:
            num_aux = len(predictions["aux_outputs"])
            for i in range(num_aux):
                aux_outputs = {k: v for k, v in predictions["aux_outputs"][i].items()}
                indices, targets = self.matcher(
                    aux_outputs, query_instances, gt_instances)
                for loss in self.losses:
                    aux_losses = self.get_loss(loss, aux_outputs, targets, indices)
                    aux_losses = {k + f"_aux{i}": v for k, v in aux_losses.items()}
                    losses.update(aux_losses)

        return losses
    
    def get_loss(
        self,
        loss: str,
        outputs: Dict[str, Tensor],
        targets: List[Instances],
        indices: List[Tensor],
        **kwargs
    ):
        loss_map = {
            "labels": self.loss_labels,
            "boxes": self.loss_boxes,
            "actions_bce": self.loss_actions_bce,
            "actions_focal": self.loss_actions_focal,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def _get_src_permutation_idx(self, indices: List[Tensor]):
        # permute predictions following indices
        src_idx = torch.cat([src for src in indices if src is not None])
        batch_idx = torch.cat([torch.full_like(src, i)
                              for i, src in enumerate(indices) if src is not None])
        return src_idx, batch_idx

    def _get_tgt_permutation_idx(self, indices: List[Tensor]):
        # permute targets following indices
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        return tgt_idx, batch_idx

    def loss_labels(
        self,
        outputs: Dict[str, Tensor],
        targets: List[Instances],
        indices: List[Tensor],
    ):
        """
        Classification loss (NLL).
        
        Args:
            outputs (Dict): outputs dicts must contain the key "pred_logits" and value is a
                Tensor with shape (num_queries, bs, num_classes).
            targets (Dict): targets dicts must contain the key "labels" containing a tensor
                of dim [num_target_boxes].
            indices (Dict): indices of one-to-one matching
            
        """
        assert 'pred_logits' in outputs
        _, bs, _ = outputs["pred_logits"].shape

        src_logits = outputs["pred_logits"]
        target_classes = targets["target_classes"]
        src_logits = torch.cat([src_logits[:, b, :] for b in range(bs)], dim=0)
        target_classes = torch.cat([target_classes[:, b] for b in range(bs)], dim=0)
        if "aux_classes" not in targets:
            loss_ce = F.cross_entropy(src_logits, target_classes, self.empty_weight)
        else:
            # Apply label smoothing, mainly for SWiG-HOI
            aux_classes = targets["aux_classes"]
            aux_classes = torch.cat([aux_classes[:, b] for b in range(bs)], dim=0)
            loss_ce = self.loss_label_smoothing(src_logits, target_classes, smoothing=0.01)
            loss_ce += self.loss_label_smoothing(
                src_logits[aux_classes >= 0], aux_classes[aux_classes >= 0], smoothing=0.01)

        losses = {'loss_tx_cls': loss_ce * self.loss_weights["cls"]}
        return losses

    def loss_boxes(
        self,
        outputs: Dict[str, Tensor],
        targets: List[Instances],
        indices: List[Tensor],
    ):
        """
        Compute the losses related to the bounding boxes, the L1 regression loss and
        the GIoU loss targets dicts must contain the key "boxes" containing a tensor
        of dim [num_target_boxes, 4]. The target boxes are expected in format
        (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs

        tgt_boxes = targets["target_boxes"][indices]
        src_boxes = outputs["pred_boxes"][indices]

        keep = (tgt_boxes[:, 2] * tgt_boxes[:, 3]) > 0
        if torch.sum(keep) == 0:
            src_boxes = outputs["pred_boxes"]
            trivial_tgts = torch.zeros_like(src_boxes)
            trivial_loss = F.l1_loss(src_boxes, trivial_tgts, reduction="mean")
            return {"loss_tx_bbox": trivial_loss * 0.0, "loss_tx_giou": trivial_loss * 0.0}

        loss_bbox = F.l1_loss(src_boxes[keep], tgt_boxes[keep], reduction="mean")

        giou = generalized_box_iou(box_cxcywh_to_xyxy(src_boxes[keep]),
                                   box_cxcywh_to_xyxy(tgt_boxes[keep]))
        loss_giou = (1 - torch.diag(giou)).mean()

        return {"loss_tx_bbox": loss_bbox * self.loss_weights["bbox"],
                "loss_tx_giou": loss_giou * self.loss_weights["giou"]}

    def loss_actions_bce(
        self,
        outputs: Dict[str, Tensor],
        targets: List[Instances],
        indices: List[Tensor],
    ):
        """
        Compute the losses of interaction/action recognition. Each action is treated as an
        independent binary classification problem. The loss is the sum of binary cross entropy 
        over all actions.
        """
        assert 'pred_actions' in outputs
        _, bs, _ = outputs["pred_actions"].shape
        action_logits = outputs["pred_actions"]

        idx = self._get_src_permutation_idx(indices)
        target_actions_o = torch.cat([t.gt_actions for t in targets if t])
        loss_action = F.binary_cross_entropy_with_logits(
            action_logits[idx], target_actions_o, reduction="mean", pos_weight=self.action_weights)
        return {"loss_tx_action": loss_action * self.loss_weights["action"]}
    
    def loss_actions_focal(
        self,
        outputs: Dict[str, Tensor],
        targets: List[Instances],
        indices: List[Tensor],
    ):
        """
        Compute the losses of interaction/action recognition using focal loss.
        """
        assert "pred_actions" in outputs
        action_logits = outputs["pred_actions"]
        target_actions_o = targets["target_actions"][indices]
        if len(indices[0]) == 0:
            trivial_tgts = torch.zeros_like(action_logits)
            trivial_loss = sigmoid_focal_loss_jit(action_logits, trivial_tgts, alpha=0.25, gamma=2.0, reduction="none")
            trivial_loss = trivial_loss.sum(dim=-1, keepdim=True).mean()
            return {"loss_tx_action": 0. * trivial_loss}
        loss_action = sigmoid_focal_loss_jit(
            action_logits[indices], target_actions_o, alpha=0.25, gamma=2.0, reduction="none")
        loss_action = loss_action.sum(dim=-1, keepdim=True).mean()
        return {"loss_tx_action": loss_action * self.loss_weights["action"]}
    
    def loss_label_smoothing(self, x, target, smoothing=0.1):
        """
        NLL loss with label smoothing.
        """
        device = x.device
        confidence = 1 - smoothing
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        empty_weight = torch.ones_like(nll_loss, device=device)
        empty_weight[target == self.num_classes] = 0.1
        nll_loss = nll_loss * empty_weight
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()
    
    