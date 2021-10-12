import math
import torch
import torch.nn as nn
from typing import List
from fvcore.nn import sigmoid_focal_loss_jit

from scipy.optimize import linear_sum_assignment
from choir.structures import Boxes, pairwise_iou, Instances
from choir.layers import nonzero_tuple
from choir.utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from choir.utils.memory import retry_if_cuda_oom


class Matcher(object):
    """
    This class assigns to each predicted "element" (e.g., a box) a ground-truth
    element. Each predicted element will have exactly zero or one matches; each
    ground-truth element may be matched to zero or more predicted elements.

    The matching is determined by the MxN match_quality_matrix, that characterizes
    how well each (ground-truth, prediction)-pair match each other. For example,
    if the elements are boxes, this matrix may contain box intersection-over-union
    overlap values.

    The matcher returns (a) a vector of length N containing the index of the
    ground-truth element m in [0, M) that matches to prediction n in [0, N).
    (b) a vector of length N containing the labels for each prediction.
    """

    def __init__(
        self, thresholds: List[float], labels: List[int], allow_low_quality_matches: bool = False
    ):
        """
        Args:
            thresholds (list): a list of thresholds used to stratify predictions
                into levels.
            labels (list): a list of values to label predictions belonging at
                each level. A label can be one of {-1, 0, 1} signifying
                {ignore, negative class, positive class}, respectively.
            allow_low_quality_matches (bool): if True, produce additional matches
                for predictions with maximum match quality lower than high_threshold.
                See set_low_quality_matches_ for more details.

            For example,
                thresholds = [0.3, 0.5]
                labels = [0, -1, 1]
                All predictions with iou < 0.3 will be marked with 0 and
                thus will be considered as false positives while training.
                All predictions with 0.3 <= iou < 0.5 will be marked with -1 and
                thus will be ignored.
                All predictions with 0.5 <= iou will be marked with 1 and
                thus will be considered as true positives.
        """
        # Add -inf and +inf to first and last position in thresholds
        thresholds = thresholds[:]
        assert thresholds[0] > 0
        thresholds.insert(0, -float("inf"))
        thresholds.append(float("inf"))
        # Currently torchscript does not support all + generator
        assert all([low <= high for (low, high) in zip(thresholds[:-1], thresholds[1:])])
        assert all([l in [-1, 0, 1] for l in labels])
        assert len(labels) == len(thresholds) - 1
        self.thresholds = thresholds
        self.labels = labels
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix):
        """
        Args:
            match_quality_matrix (Tensor[float]): an MxN tensor, containing the
                pairwise quality between M ground-truth elements and N predicted
                elements. All elements must be >= 0 (due to the us of `torch.nonzero`
                for selecting indices in :meth:`set_low_quality_matches_`).

        Returns:
            matches (Tensor[int64]): a vector of length N, where matches[i] is a matched
                ground-truth index in [0, M)
            match_labels (Tensor[int8]): a vector of length N, where pred_labels[i] indicates
                whether a prediction is a true or false positive or ignored
        """
        assert match_quality_matrix.dim() == 2
        if match_quality_matrix.numel() == 0:
            default_matches = match_quality_matrix.new_full(
                (match_quality_matrix.size(1),), 0, dtype=torch.int64
            )
            # When no gt boxes exist, we define IOU = 0 and therefore set labels
            # to `self.labels[0]`, which usually defaults to background class 0
            # To choose to ignore instead, can make labels=[-1,0,-1,1] + set appropriate thresholds
            default_match_labels = match_quality_matrix.new_full(
                (match_quality_matrix.size(1),), self.labels[0], dtype=torch.int8
            )
            return default_matches, default_match_labels

        assert torch.all(match_quality_matrix >= 0)

        # match_quality_matrix is M (gt) x N (predicted)
        # Max over gt elements (dim 0) to find best gt candidate for each prediction
        matched_vals, matches = match_quality_matrix.max(dim=0)

        match_labels = matches.new_full(matches.size(), 1, dtype=torch.int8)

        for (l, low, high) in zip(self.labels, self.thresholds[:-1], self.thresholds[1:]):
            low_high = (matched_vals >= low) & (matched_vals < high)
            match_labels[low_high] = l

        if self.allow_low_quality_matches:
            self.set_low_quality_matches_(match_labels, match_quality_matrix)

        return matches, match_labels

    def set_low_quality_matches_(self, match_labels, match_quality_matrix):
        """
        Produce additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth G find the set of predictions that have
        maximum overlap with it (including ties); for each prediction in that set, if
        it is unmatched, then match it to the ground-truth G.

        This function implements the RPN assignment case (i) in Sec. 3.1.2 of
        :paper:`Faster R-CNN`.
        """
        # For each gt, find the prediction with which it has highest quality
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        # Find the highest quality match available, even if it is low, including ties.
        # Note that the matches qualities must be positive due to the use of
        # `torch.nonzero`.
        _, pred_inds_with_highest_quality = nonzero_tuple(
            match_quality_matrix == highest_quality_foreach_gt[:, None]
        )
        # If an anchor was labeled positive only due to a low-quality match
        # with gt_A, but it has larger overlap with gt_B, it's matched index will still be gt_B.
        # This follows the implementation in Detectron, and is found to have no significant impact.
        match_labels[pred_inds_with_highest_quality] = 1


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best
    predictions, while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        losses: List[str],
        num_classes: int,
        num_actions: int,
        num_duplicates: int = 100,
        cost_class: float = 1.,
        cost_bbox: float = 1.,
        cost_giou: float = 1.,
        cost_action: float = 1.,
        match_thresh: float = 0.5,
    ):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates
            in the matching cost cost_giou: This is the relative weight of the giou loss of the
            bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_action = cost_action
        assert cost_class + cost_bbox + cost_giou + cost_action > 0, "all costs cant be 0"
        
        self.match_thresh = match_thresh
        self.num_duplicates = num_duplicates
        self.losses = losses
        self.num_classes = num_classes
        self.num_actions = num_actions

    @torch.no_grad()
    def forward(self, outputs, query_instances, gt_instances):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the
                                classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted
                                box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a
                     dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of
                           ground-truth objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        # match queries with gt instances to find gt target objects
        qr_indices, qr_gts = self.match_queries_and_gts(query_instances, gt_instances)
        # match outputs with gt target objects
        num_qrs, bs, _ = query_instances.query_boxes.shape
        num_dets, num_dps = num_qrs // self.num_duplicates, self.num_duplicates

        out_probs = outputs["pred_logits"].softmax(-1).view(num_dets, num_dps, bs, -1)
        out_boxes = outputs["pred_boxes"].view(num_dets, num_dps, bs, -1)

        out_actions = None
        if "actions_ce" in self.losses:
            out_actions = outputs["pred_actions"].softmax(-1).view(num_dets, num_dps, bs, -1)
        elif "actions_bce" in self.losses:
            out_actions = outputs["pred_actions"].sigmoid().view(num_dets, num_dps, bs, -1)
        elif "actions_focal" in self.losses:
            out_actions = outputs["pred_actions"].view(num_dets, num_dps, bs, -1)
        
        indices, targets = self.match_outputs_and_gts(
            out_probs, out_boxes, out_actions, qr_indices, qr_gts)
        return indices, targets

    def match_queries_and_gts(self, query_instances, gt_instances):
        """
        Match queries with gt instances to find targets of interacting objects.
        """
        query_boxes = query_instances.query_boxes
        num_qrs, bs, _ = query_boxes.shape
        assert num_qrs % self.num_duplicates == 0, "num_qrs cannot divided by num_duplicates"
        num_qrs, num_dps = num_qrs // self.num_duplicates, self.num_duplicates
        # Take unique query boxes. move batch axis to the first -> [bs, num_queries, 4]
        qr_boxes = Boxes(query_boxes[::num_dps].permute(1, 0, 2).contiguous().view(-1, 4))
        gt_boxes = Boxes.cat([x.gt_boxes for x in gt_instances])
        # Assert all person boxes are valid
        width = gt_boxes.tensor[:, 2] - gt_boxes.tensor[:, 0]
        height = gt_boxes.tensor[:, 3] - gt_boxes.tensor[:, 1]
        area = width * height
        assert all(area > 0), "find invalid gt person boxes due to random cropping"

        match_quality_matrix = retry_if_cuda_oom(pairwise_iou)(qr_boxes, gt_boxes)
        match_quality_matrix = match_quality_matrix.view(bs, num_qrs, -1).permute(1, 0, 2)

        qr_indices = []
        sizes = [len(v) for v in gt_instances]
        for b, C in enumerate(match_quality_matrix.split(sizes, -1)):
            if C.numel() == 0:
                qr_indices.append(([], []))
            else:
                max_ious, max_args = torch.max(C[:, b, :], dim=-1)
                kept_idxs = torch.nonzero(max_ious >= self.match_thresh, as_tuple=True)[0]
                qr_indices.append((kept_idxs, max_args[kept_idxs]))
        
        qr_gts = []
        for (_, gt_idxs), gt_x in zip(qr_indices, gt_instances):
            qr_gts.append([gt_x.interactions[i] for i in gt_idxs])
        
        return qr_indices, qr_gts
        

    def match_outputs_and_gts(self, out_probs, out_boxes, out_actions, qr_indices, qr_gts):
        """
        Match queries with gt instances to find targets of interacting objects.
        """
        device = out_probs.device
        num_qrs, num_dps, bs, _ = out_probs.shape
        
        indices = [[] for _ in range(bs)]
        targets = [[] for _ in range(bs)]
        for b, ((qr_idxs_b, _), qr_gts_b) in enumerate(zip(qr_indices, qr_gts)):
            for qr_ix, qr_gt in zip(qr_idxs_b, qr_gts_b):
                if len(qr_gt) == 0:
                    indices[b].append(torch.tensor([], dtype=torch.int64, device=device))
                    continue
                tgt_boxes = qr_gt.gt_boxes.tensor
                tgt_classes = qr_gt.gt_classes
                tgt_actions = qr_gt.gt_actions

                # Ignore boxes whose area is zero (due to the random cropping)
                kept = (tgt_boxes[:, 2] * tgt_boxes[:, 3]) > 0
                tgt_boxes = tgt_boxes[kept]
                tgt_classes = tgt_classes[kept]
                tgt_actions = tgt_actions[kept]
                if tgt_boxes.numel() == 0:
                    continue
                
                idxs = self.match_and_label_outputs_single_query(
                            out_probs[qr_ix, :, b, :], out_boxes[qr_ix, :, b, :],
                            out_actions[qr_ix, :, b, :], tgt_classes, tgt_boxes, tgt_actions)

                indices[b].append(idxs[0] + qr_ix * num_dps)
                targets[b].append(qr_gt[idxs[1]])
        
        for b in range(bs):
            if len(indices[b]) == 0 or len(targets[b]) == 0:
                indices[b], targets[b] = None, None
                continue
            indices[b] = torch.cat([x for x in indices[b]])
            targets[b] = Instances.cat([x for x in targets[b]])
        
        idx = self._get_src_permutation_idx(indices, device)
        nq = num_qrs * num_dps
        target_classes = torch.full((nq, bs), self.num_classes, dtype=torch.int64, device=device)
        target_boxes = torch.full((nq, bs, 4), 0, dtype=torch.float32, device=device)
        target_actions = torch.full((nq, bs, self.num_actions), 0, dtype=torch.float32, device=device)
        if len([x for x in targets if x is not None]) > 0:
            target_classes_o = torch.cat([t.gt_classes for t in targets if t])
            target_classes[idx] = target_classes_o

            target_boxes_o = torch.cat([t.gt_boxes.tensor for t in targets if t])
            target_boxes[idx] = target_boxes_o
            
            target_actions_o = torch.cat([t.gt_actions for t in targets if t])
            target_actions[idx] = target_actions_o

        output_targets = {"target_classes": target_classes,
                          "target_boxes": target_boxes, 
                          "target_actions": target_actions}
        
        if any([t.has("aux_classes") for t in targets if t]):
            aux_classes = torch.full((nq, bs), self.num_classes, dtype=torch.int64, device=device)
            aux_classes_o = torch.cat([t.aux_classes for t in targets if t])
            aux_classes[idx] = aux_classes_o
            output_targets["aux_classes"] = aux_classes
        
        return idx, output_targets

    def match_and_label_outputs_single_query(
        self,
        out_prob,
        out_bbox,
        out_action,
        tgt_class,
        tgt_bbox,
        tgt_action
    ):
        device = out_prob.device
        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_class]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost between boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Compute the action classification cost
        if "actions_ce" in self.losses:
            cost_action = -out_action[:, tgt_action]
        elif "actions_bce" in self.losses:
            num_positives = tgt_action.sum(dim=-1, keepdim=True).repeat(1, len(out_action))
            cost_action = -torch.matmul(out_action, tgt_action.T) / num_positives.T
        elif "actions_focal" in self.losses:
            cost_action = []
            for i in range(len(tgt_action)):
                tgt_action_i = tgt_action[i:i+1].repeat(len(out_action), 1)
                cost_action_i = sigmoid_focal_loss_jit(
                        out_action, tgt_action_i, alpha=0.25, gamma=2.0, reduction="none")
                cost_action_i = cost_action_i.sum(dim=-1, keepdim=True)
                cost_action.append(cost_action_i)
            cost_action = torch.cat(cost_action, dim=-1)
        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + \
            self.cost_giou * cost_giou + self.cost_action * cost_action

        indices = linear_sum_assignment(C.cpu())
        
        return (torch.tensor(indices[0], dtype=torch.int64, device=device),
                torch.tensor(indices[1], dtype=torch.int64, device=device))

    def _get_src_permutation_idx(self, indices, device):
        if len([src for src in indices if src is not None]) == 0:
            src_idx = torch.empty((0, ), dtype=torch.int64).to(device)
            batch_idx = torch.empty((0, ), dtype=torch.int64).to(device)
            return src_idx, batch_idx
        # permute predictions following indices
        src_idx = torch.cat([src for src in indices if src is not None])
        batch_idx = torch.cat([torch.full_like(src, i)
                              for i, src in enumerate(indices) if src is not None])
        return src_idx, batch_idx

class CascadeMatcher(HungarianMatcher):
    """
    This class computes an assignment between the targets and predictions for the
    cascade detection across feature maps with different resolution. At the coarsest feature map,
    it does the exactly same as :class:HungarianMatcher. At the higher-resolution feature map,
    the assignment is solely based on the previous matching results.
    """
    @torch.no_grad()
    def forward(self, outputs, query_instances, gt_instances, rematch=False):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                - pred_logits (Tensor): classification logits with shape
                                        [num_queries, batch_size, num_classes]            
                - pred_boxes (Tensor): predicted box coordinates with shape
                                        [num_queries, batch_size, 4] 
                - pred_actions (Tensor): classification logits with shape [num_queries, batch_size, num_actions] with 
            query_instances: The prepared query instances, containing:
                - "query_boxes" (Tensor): [num_queries, batch_size, 4] 
            gt_instances: This is a list of targets (len(targets) = batch_size), where each
                    target is a Instances containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of
                           ground-truth objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        if not query_instances.has("target_boxes") or rematch:
            # No matching has been conducted, which means it is at the coarsest feature level.
            # Match queries with gt instances to find gt target objects
            qr_indices, qr_gts = self.match_queries_and_gts(query_instances, gt_instances)
            # Match outputs with gt target objects
            num_qrs, bs, _ = query_instances.query_boxes.shape
            num_dets, num_dps = num_qrs // self.num_duplicates, self.num_duplicates

            out_probs = outputs["pred_logits"].softmax(-1).view(num_dets, num_dps, bs, -1)
            out_boxes = outputs["pred_boxes"].view(num_dets, num_dps, bs, -1)

            out_actions = None
            if "actions_ce" in self.losses:
                out_actions = outputs["pred_actions"].softmax(-1).view(num_dets, num_dps, bs, -1)
            elif "actions_bce" in self.losses:
                out_actions = outputs["pred_actions"].sigmoid().view(num_dets, num_dps, bs, -1)
            elif "actions_focal" in self.losses:
                out_actions = outputs["pred_actions"].view(num_dets, num_dps, bs, -1)
            
            indices, targets = self.match_outputs_and_gts(
                out_probs, out_boxes, out_actions, qr_indices, qr_gts)
            
            if not rematch:
                query_instances.target_classes = targets["target_classes"]
                query_instances.target_boxes = targets["target_boxes"]
                query_instances.target_actions = targets["target_actions"]
        else:
            valid = query_instances.target_classes < self.num_classes
            valid = torch.logical_and(valid, query_instances.target_boxes.sum(-1) > 0)
            indices = torch.nonzero(valid, as_tuple=True)
            sorted_idxs = torch.argsort(indices[1] + indices[0] * 10**-5)
            indices = (indices[0][sorted_idxs], indices[1][sorted_idxs])
            targets = {"target_classes": query_instances.target_classes,
                       "target_boxes": query_instances.target_boxes, 
                       "target_actions": query_instances.target_actions}
        return indices, targets
        

class MultiScaleMatcher(HungarianMatcher):
    def __init__(
        self,
        scales: List[float],
        losses: List[str],
        num_classes: int,
        num_actions: int,
        num_duplicates: int = 100,
        cost_class: float = 1.,
        cost_bbox: float = 1.,
        cost_giou: float = 1.,
        cost_action: float = 1.,
        match_thresh: float = 0.5,
        canonical_level: int = 3,
        canonical_box_size: int = 32.,
    ):
        super().__init__(
            losses, num_classes, num_actions, num_duplicates, cost_class,
            cost_bbox, cost_giou, cost_action, match_thresh
        )

        self.scales = scales
        min_level = -(math.log2(scales[0]))
        max_level = -(math.log2(scales[-1]))
        assert math.isclose(min_level, int(min_level)) and math.isclose(
            max_level, int(max_level)
        ), "Featuremap stride is not power of 2!"
        self.min_level = int(min_level)
        self.max_level = int(max_level)
        self.num_levels = int(max_level) - int(min_level) + 1
        assert (
            len(scales) == self.max_level - self.min_level + 1
        ), "[FPN Matcher] Sizes of input featuremaps do not form a pyramid!"
        assert 0 <= self.min_level and self.min_level <= self.max_level
        self.canonical_level = canonical_level
        assert canonical_box_size > 0
        self.canonical_box_size = canonical_box_size

    @torch.no_grad()
    def forward(self, outputs, query_instances, gt_instances):
        # match queries with gt instances to find gt target objects
        qr_indices, qr_gts = self.match_queries_and_gts(query_instances, gt_instances)
        # match outputs with gt target objects
        num_qrs, bs, _ = query_instances.query_boxes.shape
        num_dets, num_dps = num_qrs // self.num_duplicates, self.num_duplicates

        out_probs = outputs["pred_logits"].softmax(-1)
        out_boxes = outputs["pred_boxes"]
        out_probs_per_lvl = torch.split(out_probs, num_dets * num_dps, dim=0)
        out_boxes_per_lvl = torch.split(out_boxes, num_dets * num_dps, dim=0)
        out_probs_per_lvl = [x.view(num_dets, num_dps, bs, -1) for x in out_probs_per_lvl]
        out_boxes_per_lvl = [x.view(num_dets, num_dps, bs, -1) for x in out_boxes_per_lvl]
        out_probs = torch.cat(out_probs_per_lvl, dim=1)
        out_boxes = torch.cat(out_boxes_per_lvl, dim=1)

        out_actions = None
        if "actions_ce" in self.losses:
            out_actions = outputs["pred_actions"].softmax(-1)
        elif "actions_bce" in self.losses:
            out_actions = outputs["pred_actions"].sigmoid()
        elif "actions_focal" in self.losses:
            out_actions = outputs["pred_actions"]
        if out_actions is not None:
            out_actions_per_lvl = torch.split(out_actions, num_dets * num_dps, dim=0)
            out_actions_per_lvl = [x.view(num_dets, num_dps, bs, -1) for x in out_actions_per_lvl]
            out_actions = torch.cat(out_actions_per_lvl, dim=1)
        
        image_sizes = [x.image_size for x in gt_instances]
        indices, targets = self.match_outputs_and_gts(
            out_probs, out_boxes, out_actions, qr_indices, qr_gts, image_sizes)
        return indices, targets
    
    def match_outputs_and_gts(
        self,
        out_probs,
        out_boxes,
        out_actions,
        qr_indices,
        qr_gts,
        image_sizes
    ):
        """
        Match queries with gt instances to find targets of interacting objects.
        """
        device = out_probs.device
        num_qrs, num_dps_x_lvls, bs, _ = out_probs.shape
        num_dps = self.num_duplicates
        
        indices = [[] for _ in range(bs)]
        targets = [[] for _ in range(bs)]
        for b, ((qr_idxs_b, _), qr_gts_b) in enumerate(zip(qr_indices, qr_gts)):
            for qr_ix, qr_gt in zip(qr_idxs_b, qr_gts_b):
                # If the person box is not matched with any ground truth boxes,
                # we will ignore the subsequent matching for target objects.
                # If none of person detections hit ground truth, the indices
                # and targets will be empty (none).
                if len(qr_gt) == 0:
                    indices[b].append(torch.tensor([], dtype=torch.int64, device=device))
                    continue

                tgt_boxes = qr_gt.gt_boxes.tensor
                tgt_classes = qr_gt.gt_classes
                tgt_actions = qr_gt.gt_actions

                # Ignore boxes whose area is zero (due to the random cropping)
                kept = (tgt_boxes[:, 2] * tgt_boxes[:, 3]) > 0
                tgt_boxes = tgt_boxes[kept]
                tgt_classes = tgt_classes[kept]
                tgt_actions = tgt_actions[kept]
                if tgt_boxes.numel() == 0:
                    continue

                lvl_idxs = self.boxes_to_levels(tgt_boxes, image_sizes[b])
                
                idxs = self.match_and_label_outputs_single_query(
                    out_probs[qr_ix, :, b, :], out_boxes[qr_ix, :, b, :],
                    out_actions[qr_ix, :, b, :], tgt_classes,
                    tgt_boxes, tgt_actions, lvl_idxs
                )

                # convert to prediction idxs idxs[0] // num_dps
                indices[b].append(idxs[0]//num_dps*num_qrs*num_dps + qr_ix*num_dps + idxs[0]%num_dps)
                targets[b].append(qr_gt[idxs[1]])
        
        for b in range(bs):
            if len(indices[b]) == 0 or len(targets[b]) == 0:
                indices[b], targets[b] = None, None
                continue
            indices[b] = torch.cat([x for x in indices[b]])
            targets[b] = Instances.cat([x for x in targets[b]])
        
        idx = self._get_src_permutation_idx(indices, device)
        nq = num_qrs * num_dps_x_lvls
        target_classes = torch.full((nq, bs), self.num_classes, dtype=torch.int64, device=device)
        target_boxes = torch.full((nq, bs, 4), 0, dtype=torch.float32, device=device)
        target_actions = torch.full((nq, bs, self.num_actions), 0, dtype=torch.float32, device=device)
        if len([x for x in targets if x is not None]) > 0:
            target_classes_o = torch.cat([t.gt_classes for t in targets if t])
            target_classes[idx] = target_classes_o

            target_boxes_o = torch.cat([t.gt_boxes.tensor for t in targets if t])
            target_boxes[idx] = target_boxes_o
            
            target_actions_o = torch.cat([t.gt_actions for t in targets if t])
            target_actions[idx] = target_actions_o

        output_targets = {"target_classes": target_classes,
                          "target_boxes": target_boxes, 
                          "target_actions": target_actions}
        
        if any([t.has("aux_classes") for t in targets if t]):
            aux_classes = torch.full((nq, bs), self.num_classes, dtype=torch.int64, device=device)
            aux_classes_o = torch.cat([t.aux_classes for t in targets if t])
            aux_classes[idx] = aux_classes_o
            output_targets["aux_classes"] = aux_classes
        
        return idx, output_targets

    def match_and_label_outputs_single_query(
        self,
        out_prob,
        out_bbox,
        out_action,
        tgt_class,
        tgt_bbox,
        tgt_action,
        level_idxs,
    ):
        device = out_prob.device
        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_class]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost between boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Compute the action classification cost
        if "actions_ce" in self.losses:
            cost_action = -out_action[:, tgt_action]
        elif "actions_bce" in self.losses:
            num_positives = tgt_action.sum(dim=-1, keepdim=True).repeat(1, len(out_action))
            cost_action = -torch.matmul(out_action, tgt_action.T) / num_positives.T
        elif "actions_focal" in self.losses:
            cost_action = []
            for i in range(len(tgt_action)):
                tgt_action_i = tgt_action[i:i+1].repeat(len(out_action), 1)
                cost_action_i = sigmoid_focal_loss_jit(
                        out_action, tgt_action_i, alpha=0.25, gamma=2.0, reduction="none")
                cost_action_i = cost_action_i.sum(dim=-1, keepdim=True)
                cost_action.append(cost_action_i)
            cost_action = torch.cat(cost_action, dim=-1)
            
        # Compute the level cost
        cost_level = torch.full(cost_class.shape, 100.).to(device)
        num_dps_x_lvls = cost_class.shape[0]
        num_dps = num_dps_x_lvls // self.num_levels
        for i, lvl_id in enumerate(level_idxs):
            stx, end = lvl_id * num_dps, (lvl_id + 1) * num_dps
            cost_level[stx:end, i] = 0

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + \
            self.cost_giou * cost_giou + self.cost_action * cost_action + cost_level

        indices = linear_sum_assignment(C.cpu())
        
        return (torch.tensor(indices[0], dtype=torch.int64, device=device),
                torch.tensor(indices[1], dtype=torch.int64, device=device))

    def boxes_to_levels(self, boxes, image_size):
        w = boxes[:, 2] * image_size[1]
        h = boxes[:, 3] * image_size[0]
        box_sizes = torch.sqrt(w * h)
        # Eqn.(1) in FPN paper
        level_assignments = torch.floor(
            self.canonical_level + torch.log2(box_sizes / self.canonical_box_size + 1e-8)
        )
        # clamp level to (min, max), in case the box size is too large or too small
        # for the available feature maps
        level_assignments = torch.clamp(level_assignments, min=self.min_level, max=self.max_level)
        return level_assignments.to(torch.int64) - self.min_level

    def _get_src_permutation_idx(self, indices, device=None):
        if len([src for src in indices if src is not None]) == 0:
            src_idx = torch.empty((0, ), dtype=torch.int64).to(device)
            batch_idx = torch.empty((0, ), dtype=torch.int64).to(device)
            return src_idx, batch_idx
        # permute predictions following indices
        src_idx = torch.cat([src for src in indices if src is not None])
        batch_idx = torch.cat([torch.full_like(src, i)
                              for i, src in enumerate(indices) if src is not None])
        return src_idx, batch_idx


def build_matcher(cfg):
    if cfg.MODEL.META_ARCHITECTURE == "HOIR":
        return HungarianMatcher(
            cost_class=cfg.MODEL.TRANSFORMER.SET_COST_CLASS,
            cost_bbox=cfg.MODEL.TRANSFORMER.SET_COST_BBOX,
            cost_giou=cfg.MODEL.TRANSFORMER.SET_COST_GIOU,
            cost_action=cfg.MODEL.TRANSFORMER.SET_COST_ACTION,
            match_thresh=cfg.MODEL.TRANSFORMER.MATCH_THRESH,
            num_duplicates=cfg.MODEL.TRANSFORMER.DUPLICATES_PER_QUERY,
            losses=cfg.MODEL.TRANSFORMER.LOSSES,
            num_actions=cfg.MODEL.TRANSFORMER.NUM_ACTION_CLASSES,
            num_classes=cfg.MODEL.TRANSFORMER.NUM_OBJECT_CLASSES,
        )
    elif cfg.MODEL.META_ARCHITECTURE == "CHOIR":
        return CascadeMatcher(
            cost_class=cfg.MODEL.TRANSFORMER.SET_COST_CLASS,
            cost_bbox=cfg.MODEL.TRANSFORMER.SET_COST_BBOX,
            cost_giou=cfg.MODEL.TRANSFORMER.SET_COST_GIOU,
            cost_action=cfg.MODEL.TRANSFORMER.SET_COST_ACTION,
            match_thresh=cfg.MODEL.TRANSFORMER.MATCH_THRESH,
            num_duplicates=cfg.MODEL.TRANSFORMER.DUPLICATES_PER_QUERY,
            losses=cfg.MODEL.TRANSFORMER.LOSSES,
            num_actions=cfg.MODEL.TRANSFORMER.NUM_ACTION_CLASSES,
            num_classes=cfg.MODEL.TRANSFORMER.NUM_OBJECT_CLASSES,
        )