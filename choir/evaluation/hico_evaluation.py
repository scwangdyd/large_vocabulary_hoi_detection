# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
import pickle
import shutil
import datetime
import torch
from collections import OrderedDict, defaultdict
from fvcore.common.file_io import PathManager, file_lock
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate

import choir.utils.comm as comm
import choir.utils.box_ops as box_ops
from choir.data import MetadataCatalog, DatasetCatalog
from choir.evaluation.fast_eval_api import COCOeval_opt
from choir.structures import Boxes, BoxMode, pairwise_iou
from choir.utils.logger import create_small_table
from choir.layers import batched_nms
from .evaluator import DatasetEvaluator


class HICOEvaluator(DatasetEvaluator):
    """
    Evaluate object proposal, instance detection, using COCO's metrics and APIs.
    Evaluate human-object interaction detection using HICO-DET's metrics and APIs.
    """

    def __init__(self, dataset_name, cfg, distributed, output_dir=None, use_fast_impl=True):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:

                    "json_file": the path to the COCO format annotation;
                    "matlab_file": the original matlab annotation files,

                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            cfg (CfgNode): config instance
            distributed (True): if True, will collect results from all ranks and run evaluation
                in the main process. Otherwise, will evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:

                1. "instance_predictions.pth" a file in torch serialization
                    format that contains all the raw original predictions.
                2. "coco_instances_results.json" a json file in COCO's result format.
                3. "hico_interaction_results.mat" a matlab file
                    used for HICO-DET official evaluation.
            use_fast_impl (bool): use a fast but **unofficial** implementation to compute AP.
                Although the results should be very close to the official implementation in COCO
                API, it is still recommended to compute results with the official API for use in
                papers.
        """
        self._tasks = self._tasks_from_config(cfg)
        self._distributed = distributed
        self._output_dir = output_dir
        self._use_fast_impl = use_fast_impl

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._metadata = MetadataCatalog.get(dataset_name)
        cache_path = os.path.join(output_dir, f"{dataset_name}_coco_format.json")
        if not hasattr(self._metadata, "coco_file") and os.path.isfile(cache_path):
            self._metadata.coco_file = cache_path
        if not hasattr(self._metadata, "coco_file"):
            self._logger.info(
                f"'{dataset_name}' is not registered by `register_coco_instances`."
                " Therefore trying to convert it to COCO format ..."
            )
            self._metadata.coco_file = cache_path
            convert_to_coco_json(dataset_name, cache_path)
            
        coco_file = PathManager.get_local_path(self._metadata.coco_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(coco_file)

        # Test set json files do not contain annotations (evaluation must be
        # performed using the COCO evaluation server).
        self._do_evaluation = "annotations" in self._coco_api.dataset
        
        cache_path = os.path.join(output_dir, f"{dataset_name}_hico_format.json")
        if not hasattr(self._metadata, "hico_file") and os.path.isfile(cache_path):
            self._metadata.hico_file = cache_path
        if not hasattr(self._metadata, "hico_file"):
            prepare_hico_gts(dataset_name, cache_path)
            self._metadata.hico_file = cache_path

        hico_file = PathManager.get_local_path(self._metadata.hico_file)
        self._hico_gts = pickle.load(open(hico_file, "rb"))

    def reset(self):
        self._predictions = []

    def _tasks_from_config(self, cfg):
        """
        Returns:
            tuple[str]: tasks that can be evaluated under the given configuration.
        """
        tasks = cfg.TEST.EVAL_TASKS
        return tasks

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model (e.g., HOITX). It is a list of dict.
                Each dict corresponds to an image and contains keys
                like "height", "width", "file_name", "image_id".
            outputs: the outputs of a model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            img_id = input["image_id"]
            prediction = {"image_id": img_id}

            if "results" in output:
                results = output["results"].to(self._cpu_device)
                if "proposal" in self._tasks:
                    prediction["proposals"] = results
                if "bbox" in self._tasks:
                    prediction["box_instances"] = instances_to_coco_json(results, img_id)
                if "hoi" in self._tasks:
                    prediction["hoi_instances"] = instances_to_hico_with_nms(results, self._metadata)

            self._predictions.append(prediction)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[HICOEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            # file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            # with PathManager.open(file_path, "wb") as f:
            #     torch.save(predictions, f)

        self._results = OrderedDict()
        if "proposal" in self._tasks and "proposals" in predictions[0]:
            self._eval_box_proposals(copy.deepcopy(predictions))
        if "bbox" in self._tasks and "box_instances" in predictions[0]:
            self._eval_box_predictions(copy.deepcopy(predictions))
        if "hoi" in self._tasks and "hoi_instances" in predictions[0]:
            self._eval_hoi_predictions(copy.deepcopy(predictions))
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_hoi_predictions(self, predictions):
        """
        Evaluate predictions on the human-object interactions.
        Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for HICO-DET matlab format ...")
        images = [x["image_id"] for x in predictions]
        results = [x["hoi_instances"] for x in predictions]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "hico_interaction_results.pkl")
            file_path = os.path.abspath(file_path)
            self._logger.info("Saving results to {}".format(file_path))
            write_results_hico_format(images, results, file_path)
        else:
            file_path = "./output"

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        if not self._use_fast_impl:
            self._logger.info("Evaluating interaction using HICO-DET official MATLAB code ...")
            self._evaluate_hico_on_matlab(
                file_path, self._hico_official_anno_file, self._hico_official_bbox_file)
        else:
            hico_ap, hico_rec = np.zeros(600), np.zeros(600)
            scores = [[] for _ in range(600)]
            boxes = [[] for _ in range(600)]
            keys = [[] for _ in range(600)]
            
            for img_id, dets in zip(images, results):
                for det in dets:
                    hoi_id, box, score = int(det[0]), det[1:-2], float(det[-2])
                    scores[hoi_id].append(score)
                    boxes[hoi_id].append(box)
                    keys[hoi_id].append(img_id)
            
            for hoi_id in range(600):
                gts = self._hico_gts[hoi_id]
                ap, rec = calc_ap(scores[hoi_id], boxes[hoi_id], keys[hoi_id], gts)
                hico_ap[hoi_id], hico_rec[hoi_id] = ap, rec

            self._derive_hico_results(hico_ap, hico_rec, self._metadata, print_full_table=True)
            with open(os.path.join(self._output_dir, "hico_ap_rec_res.pkl"), "wb") as f:
                pickle.dump({"hico_ap": hico_ap, "hico_rec": hico_rec}, f)
            
        
    def _eval_box_predictions(self, predictions, iou_type="bbox"):
        """
        Evaluate box predictions.
        Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for COCO format ...")
        coco_results = list(itertools.chain(*[x["box_instances"] for x in predictions]))

        # unmap the category ids for objects
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
            for result in coco_results:
                category_id = result["category_id"]
                assert (
                    category_id in reverse_id_mapping
                ), "A prediction has category_id={}, which is not available in the dataset.".format(
                    category_id
                )
                result["category_id"] = reverse_id_mapping[category_id]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info(
            "Evaluating predictions with {} COCO API...".format(
                "unofficial" if self._use_fast_impl else "official"
            )
        )

        coco_eval = (
            _evaluate_predictions_on_coco(
                self._coco_api, coco_results, iou_type, use_fast_impl=self._use_fast_impl
            )
            if len(coco_results) > 0
            else None  # cocoapi does not handle empty results very well
        )

        res = self._derive_coco_results(
            coco_eval, iou_type, class_names=self._metadata.get("thing_classes"),
        )
        self._results[iou_type] = res

    def _eval_box_proposals(self, predictions):
        """
        Evaluate the box proposals in predictions.
        Fill self._results with the metrics for "box_proposals" task.
        """
        if self._output_dir:
            # Saving generated box proposals to file.
            # Predicted box_proposals are in XYXY_ABS mode.
            bbox_mode = BoxMode.XYXY_ABS.value
            ids, boxes, scores = [], [], []
            for prediction in predictions:
                ids.append(prediction["image_id"])
                boxes.append(prediction["proposals"].pred_boxes.tensor.numpy())
                scores.append(prediction["proposals"].scores.numpy())

            proposal_data = {
                "boxes": boxes,
                "scores": scores,
                "ids": ids,
                "bbox_mode": bbox_mode,
            }
            
            with PathManager.open(os.path.join(self._output_dir, "box_proposals.pkl"), "wb") as f:
                pickle.dump(proposal_data, f)

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return
        
        self._logger.info("Evaluating bbox proposals ...")
        res = {}
        res_iou50 = {}
        areas = {"all": "", "small": "s", "medium": "m", "large": "l"}
        # areas = {"medium": "m"}
        for limit in [10, 20, 50]:
            for area, suffix in areas.items():
                stats = _evaluate_box_proposals(predictions, self._coco_api, area=area, limit=limit)
                key = "AR{}@{:d}".format(suffix, limit)
                res[key] = float(stats["ar"].item() * 100)
                key = "R{}@{:d}".format(suffix, limit)
                res_iou50[key] = float(stats["recalls"][stats["thresholds"] == 0.5]) * 100.

        self._logger.info("Proposal metrics: \n" + create_small_table(res))
        self._logger.info("Proposal metrics @ IoU50: \n" + create_small_table(res_iou50))
        self._results["box_proposals"] = res

    def _evaluate_hico_on_matlab(self, dets_file, anno_file, bbox_file):
        import subprocess
        self._logger.info('-----------------------------------------------------')
        self._logger.info('Computing results with the official MATLAB eval code.')
        self._logger.info('-----------------------------------------------------')
        cmd = 'cd {} && '.format(self._hico_official_matlab_path)
        cmd += '{:s} -nodisplay -nodesktop '.format(self._matlab)
        cmd += '-r "dbstop if error; '
        cmd += 'hico_eval_wrapper(\'{:s}\', \'{:s}\', \'{:s}\'); quit;"'.format(
            dets_file, anno_file, bbox_file
        )
        self._logger.info('Running:\n{}'.format(cmd))
        subprocess.call(cmd, shell=True)

    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.

        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.

        Returns:
            a dict of {metric name: score}
        """

        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "interacting_bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        }[iou_type]

        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # the standard metrics
        results = {
            metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }
        self._logger.info(
            "Evaluation results for {}: \n".format(iou_type) + create_small_table(results)
        )
        if not np.isfinite(sum(results.values())):
            self._logger.info("Note that some metrics cannot be computed.")

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))

        # tabulate it
        N_COLS = min(8, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP: \n".format(iou_type) + table)

        results.update({"AP-" + name: ap for name, ap in results_per_category})
        
        # Compute per-category AP50
        AP50_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[0, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            AP50_per_category.append(("{}".format(name), float(ap * 100)))

        # tabulate it
        N_COLS = min(8, len(AP50_per_category) * 2)
        results_flatten = list(itertools.chain(*AP50_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP50"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP50: \n".format(iou_type) + table)

        return results
    
    def _derive_hico_results(self, hico_ap, hico_rec, metadata, area="all", print_full_table=False):
        """
        Derive the desired score numbers from summarized COCOeval.

        Args:
            hico_ap (List): computed ap results per hoi.
            hico_rec (List): computed rec results per hoi.
        Returns:
            a dict of {metric name: score}
        """
        interaction_classes = metadata.interaction_classes
        # Dervie per-category hoi AP and REC
        AP_PER_HOI = []
        for hoi_id, name in enumerate(interaction_classes):
            AP_PER_HOI.append(("{}".format(name), float(hico_ap[hoi_id] * 100)))

        # tabulate it
        N_COLS = min(4, len(AP_PER_HOI) * 2)
        results_flatten = list(itertools.chain(*AP_PER_HOI))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["HOI", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        if print_full_table:
            self._logger.info("Per-HOI AP: \n" + table)

        # Full, rare, non-rare
        non_interaction_ids = metadata.non_interaction_ids
        rare_interaction_ids = metadata.rare_interaction_ids
        valid_interaction_ids = np.setdiff1d(np.arange(600), non_interaction_ids)
        nonrare_interaction_ids = np.setdiff1d(valid_interaction_ids, rare_interaction_ids)
        rare_interaction_ids = np.intersect1d(rare_interaction_ids, valid_interaction_ids)

        full_map_withnon = np.mean(hico_ap)
        full_map = np.mean(hico_ap[valid_interaction_ids])
        rare_map = np.mean(hico_ap[rare_interaction_ids])
        non_rare_map = np.mean(hico_ap[nonrare_interaction_ids])
        
        self._logger.info("Default {}: full-wnon {:.2f}, full {:.2f}, rare {:.2f}, non-rare {:.2f}".format(
            area, full_map_withnon * 100., full_map * 100., rare_map * 100., non_rare_map * 100.))


def instances_to_coco_json(instances, img_id, max_dets=100):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.

    Args:
        instances (Instances):
        img_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    if len(instances) == 0:
        return []
    
    classes = torch.cat([instances.object_classes, instances.person_classes], dim=0)
    object_scores = instances.object_scores * instances.person_scores
    scores = torch.cat([object_scores, instances.person_scores], dim=0)
    boxes = torch.cat([instances.object_boxes.tensor, instances.person_boxes.tensor], dim=0)
    
    # Before NMS, remove duplicate boxes
    unique_dets, keep = set(), []
    for i in range(len(boxes)):
        det = tuple(boxes[i].tolist() + [float(classes[i])])
        if det not in unique_dets:
            unique_dets.add(det)
            keep.append(i)
    classes, scores, boxes = classes[keep], scores[keep], boxes[keep]

    # NMS
    keep = batched_nms(boxes, scores, classes, 0.5)[:max_dets]
    classes = classes[keep].tolist()
    scores = scores[keep].tolist()
    boxes = box_ops.box_xyxy_to_xywh(boxes[keep]).tolist()
    
    results = [{"image_id": img_id, "category_id": classes[i], "bbox": boxes[i], "score": scores[i]}
               for i in range(len(boxes))]
    return results


def _evaluate_box_proposals(dataset_predictions, coco_api, thresholds=None, area="all", limit=None):
    """
    Evaluate detection proposal recall metrics. This function is a much
    faster alternative to the official COCO API recall evaluation code. However,
    it produces slightly different results.
    """
    # Record max overlap value for each gt box
    # Return vector of overlap values
    areas = {
        "all": 0,
        "small": 1,
        "medium": 2,
        "large": 3,
        "96-128": 4,
        "128-256": 5,
        "256-512": 6,
        "512-inf": 7,
    }
    area_ranges = [
        [0 ** 2, 1e5 ** 2],  # all
        [0 ** 2, 32 ** 2],  # small
        [32 ** 2, 96 ** 2],  # medium
        [96 ** 2, 1e5 ** 2],  # large
        [96 ** 2, 128 ** 2],  # 96-128
        [128 ** 2, 256 ** 2],  # 128-256
        [256 ** 2, 512 ** 2],  # 256-512
        [512 ** 2, 1e5 ** 2],
    ]  # 512-inf
    assert area in areas, "Unknown area range: {}".format(area)
    area_range = area_ranges[areas[area]]
    gt_overlaps = []
    num_pos = 0

    for prediction_dict in dataset_predictions:
        predictions = prediction_dict["proposals"]

        # sort predictions in descending order
        # TODO maybe remove this and make it explicit in the documentation
        inds = predictions.scores.sort(descending=True)[1]
        predictions = predictions[inds]

        ann_ids = coco_api.getAnnIds(imgIds=prediction_dict["image_id"])
        anno = coco_api.loadAnns(ann_ids)
        anno = [obj for obj in anno if obj["category_id"] > 1] # exclude person boxes
        gt_boxes = [
            BoxMode.convert(obj["bbox"], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
            for obj in anno if obj["iscrowd"] == 0
        ]
        gt_boxes = torch.as_tensor(gt_boxes).reshape(-1, 4)  # guard against no boxes
        gt_boxes = Boxes(gt_boxes)
        gt_areas = torch.as_tensor([obj["area"] for obj in anno if obj["iscrowd"] == 0])

        if len(gt_boxes) == 0 or len(predictions) == 0:
            continue

        valid_gt_inds = (gt_areas >= area_range[0]) & (gt_areas <= area_range[1])
        gt_boxes = gt_boxes[valid_gt_inds]

        num_pos += len(gt_boxes)

        if len(gt_boxes) == 0:
            continue

        if limit is not None and len(predictions) > limit:
            predictions = predictions[:limit]

        overlaps = pairwise_iou(predictions.pred_boxes, gt_boxes)

        _gt_overlaps = torch.zeros(len(gt_boxes))
        for j in range(min(len(predictions), len(gt_boxes))):
            # find which proposal box maximally covers each gt box
            # and get the iou amount of coverage for each gt box
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)

            # find which gt box is 'best' covered (i.e. 'best' = most iou)
            gt_ovr, gt_ind = max_overlaps.max(dim=0)
            assert gt_ovr >= 0
            # find the proposal box that covers the best covered gt box
            box_ind = argmax_overlaps[gt_ind]
            # record the iou coverage of this gt box
            _gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert _gt_overlaps[j] == gt_ovr
            # mark the proposal box and the gt box as used
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1

        # to delete
        # if not all(_gt_overlaps > 0.5):
        #     print("image_id {}, {}".format(prediction_dict["image_id"], _gt_overlaps))

        # append recorded iou coverage level
        gt_overlaps.append(_gt_overlaps)
    gt_overlaps = (
        torch.cat(gt_overlaps, dim=0) if len(gt_overlaps) else torch.zeros(0, dtype=torch.float32)
    )
    gt_overlaps, _ = torch.sort(gt_overlaps)

    if thresholds is None:
        step = 0.05
        thresholds = torch.arange(0.5, 0.95 + 1e-5, step, dtype=torch.float32)
    recalls = torch.zeros_like(thresholds)
    # compute recall for each iou threshold
    for i, t in enumerate(thresholds):
        recalls[i] = (gt_overlaps >= t).float().sum() / float(num_pos)
    # ar = 2 * np.trapz(recalls, thresholds)
    ar = recalls.mean()
    return {
        "ar": ar,
        "recalls": recalls,
        "thresholds": thresholds,
        "gt_overlaps": gt_overlaps,
        "num_pos": num_pos,
    }


def _evaluate_predictions_on_coco(coco_gt, coco_results, iou_type, use_fast_impl=True):
    """
    Evaluate the coco results using COCOEval API.
    """
    assert len(coco_results) > 0

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = (COCOeval_opt if use_fast_impl else COCOeval)(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval


def write_results_hico_format(images, results, file_path):
    """
    Write HICO detection results into .mat file.

    Args:
        images (List[int]): A list of image id
        results (List[List]): A list of detection results, which is saved in a list containing
            [interaction_id, person box, object box, score]
        file_path (String): savefile name
    """
    assert len(images) == len(results)
    # dets = [[] for _ in range(9658)]

    # with open("/raid1/suchen/repo/fshoitx/output/hico_test_image_map.pkl", "rb") as f:
    #     img_name_to_id_map = pickle.load(f)
    
    # for img_id, results_per_image in zip(images, results):
    #     dets[img_name_to_id_map[img_id]] = results_per_image

    dets = [[] for _ in range(40000)]
    for img_id, results_per_image in zip(images, results):
        dets[img_id] = results_per_image
    
    with open(file_path, "wb") as f:
        pickle.dump(dets, f)
    # sio.savemat(file_path, mdict={'dets': dets})


def prepare_hico_gts(dataset_name, output_file, allow_cached=True):
    """
    Converts dataset into COCO format and saves it to a pkl file.
    dataset_name must be registered in DatasetCatalog and in detectron2's standard format.

    Args:
        dataset_name:
            reference from the config file to the catalogs
            must be registered in DatasetCatalog and in detectron2's standard format
        output_file: path of json file that will be saved to
        allow_cached: if json file is already present then skip conversion
    """
    logger = logging.getLogger(__name__)
    PathManager.mkdirs(os.path.dirname(output_file))
    with file_lock(output_file):
        if PathManager.exists(output_file) and allow_cached:
            logger.warning(
                f"Using previously cached HICO format annotations at '{output_file}'. "
                "You need to clear the cache file if your dataset has been modified."
            )
        else:
            logger.info(f"Converting annotations of dataset '{dataset_name}' to COCO format ...)")
            hico_gts = convert_to_hico_dict(dataset_name)

            logger.info(f"Caching HICO format annotations at '{output_file}' ...")
            tmp_file = output_file + ".tmp"
            with PathManager.open(tmp_file, "wb") as f:
                pickle.dump(hico_gts, f)
            shutil.move(tmp_file, output_file)


def convert_to_hico_dict(dataset_name):
    """
    Convert an hoi detection dataset into hico original format.

    Args:
        dataset_name (str):
            name of the source dataset
            Must be registered in DatastCatalog and in detectron2's standard format.
            Must have corresponding metadata "thing_classes"
    Returns:
        coco_dict: serializable dict in COCO json format
    """    
    dataset_dicts = DatasetCatalog.get(dataset_name)
    # dataset_dicts = dataset_dicts[3000:6000]
    metadata = MetadataCatalog.get(dataset_name)
    
    reverse_id_mapping = {v: k for k, v in metadata.thing_dataset_id_to_contiguous_id.items()}
    reverse_id_mapper = lambda contiguous_id: reverse_id_mapping[contiguous_id]
    
    action_object_to_interaction_map = metadata.action_object_to_interaction_map

    gts = {}
    for key, hoi_id in action_object_to_interaction_map.items():
        gts[hoi_id] = defaultdict(list)

    for _, image_dict in enumerate(dataset_dicts):
        image_id = image_dict["image_id"]
        anns_per_image = image_dict.get("annotations", [])
        for annotation in anns_per_image:
            person_box = annotation["bbox"]
            bbox_mode = annotation["bbox_mode"]
            person_box = BoxMode.convert(person_box, bbox_mode, BoxMode.XYXY_ABS)
            person_box = [round(float(x), 3) for x in person_box]
            for hoi_annotation in annotation["interaction"]:
                object_box = hoi_annotation["bbox"]
                bbox_mode = hoi_annotation["bbox_mode"]
                object_box = BoxMode.convert(object_box, bbox_mode, BoxMode.XYXY_ABS)
                object_box = [round(float(x), 3) for x in object_box]
                
                object_id = int(reverse_id_mapper(hoi_annotation["category_id"]))
                for action_id in np.where(hoi_annotation["action_id"])[0]:
                    hoi_id = action_object_to_interaction_map[f"{action_id} {object_id}"]
                    gts[hoi_id][image_id].append(person_box + object_box)

    for hoi_id in gts:
        for img_id in gts[hoi_id]:
            gts[hoi_id][img_id] = np.array(gts[hoi_id][img_id])

    return gts


def convert_to_coco_json(dataset_name, output_file, allow_cached=True):
    """
    Converts dataset into COCO format and saves it to a json file.
    dataset_name must be registered in DatasetCatalog and in detectron2's standard format.

    Args:
        dataset_name:
            reference from the config file to the catalogs
            must be registered in DatasetCatalog and in detectron2's standard format
        output_file: path of json file that will be saved to
        allow_cached: if json file is already present then skip conversion
    """
    logger = logging.getLogger(__name__)

    # TODO: The dataset or the conversion script *may* change,
    # a checksum would be useful for validating the cached data

    PathManager.mkdirs(os.path.dirname(output_file))
    with file_lock(output_file):
        if PathManager.exists(output_file) and allow_cached:
            logger.warning(
                f"Using previously cached COCO format annotations at '{output_file}'. "
                "You need to clear the cache file if your dataset has been modified."
            )
        else:
            logger.info(f"Converting annotations of dataset '{dataset_name}' to COCO format ...)")
            coco_dict = convert_to_coco_dict(dataset_name)

            logger.info(f"Caching COCO format annotations at '{output_file}' ...")
            tmp_file = output_file + ".tmp"
            with PathManager.open(tmp_file, "w") as f:
                json.dump(coco_dict, f)
            shutil.move(tmp_file, output_file)


def convert_to_coco_dict(dataset_name):
    """
    Convert an hoi detection dataset into COCO json format.

    Args:
        dataset_name (str):
            name of the source dataset
            Must be registered in DatastCatalog and in detectron2's standard format.
            Must have corresponding metadata "thing_classes"
    Returns:
        coco_dict: serializable dict in COCO json format
    """
    logger = logging.getLogger(__name__)

    dataset_dicts = DatasetCatalog.get(dataset_name)
    # dataset_dicts = dataset_dicts[3000:6000]
    metadata = MetadataCatalog.get(dataset_name)

    # unmap the category mapping ids for COCO
    if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):
        reverse_id_mapping = {v: k for k, v in metadata.thing_dataset_id_to_contiguous_id.items()}
        reverse_id_mapper = lambda contiguous_id: reverse_id_mapping[contiguous_id]
    else:
        reverse_id_mapper = lambda contiguous_id: contiguous_id

    categories = [
        {"id": reverse_id_mapper(id), "name": name}
        for id, name in enumerate(metadata.thing_classes)
    ]

    logger.info("Converting dataset dicts into COCO format")
    coco_images = []
    coco_annotations = []

    for image_id, image_dict in enumerate(dataset_dicts):
        coco_image = {
            "id": image_dict.get("image_id", image_id),
            "width": int(image_dict["width"]),
            "height": int(image_dict["height"]),
            "file_name": str(image_dict["file_name"]),
        }
        coco_images.append(coco_image)

        anns_per_image = image_dict.get("annotations", [])
        boxes_per_image = dict()
        for annotation in anns_per_image:
            # Append person instances
            coco_anno = _instance_to_coco_dict(annotation, reverse_id_mapper)
            box_key = tuple(coco_anno["bbox"] + [coco_anno["category_id"]])
            if box_key not in boxes_per_image:
                boxes_per_image[box_key] = coco_anno
            # Append object instances
            for hoi_annotation in annotation["interaction"]:
                coco_anno = _instance_to_coco_dict(hoi_annotation, reverse_id_mapper)
                box_key = tuple(coco_anno["bbox"] + [coco_anno["category_id"]])
                if box_key not in boxes_per_image:
                    boxes_per_image[box_key] = coco_anno
        
        for _, coco_anno in boxes_per_image.items():
            coco_anno["id"] = len(coco_annotations) + 1
            coco_anno["image_id"] = coco_image["id"]
            coco_annotations.append(coco_anno)

    logger.info(
        "Conversion finished, "
        f"#images: {len(coco_images)}, #annotations: {len(coco_annotations)}"
    )

    info = {
        "date_created": str(datetime.datetime.now()),
        "description": "Automatically generated COCO json file for Detectron2.",
    }
    coco_dict = {"info": info, "images": coco_images, "categories": categories, "licenses": None}
    if len(coco_annotations) > 0:
        coco_dict["annotations"] = coco_annotations
    return coco_dict


def _instance_to_coco_dict(annotation, reverse_id_mapper):
    # create a new dict with only COCO fields
    coco_annotation = {}
    # COCO requirement: XYWH box format
    bbox = annotation["bbox"]
    bbox_mode = annotation["bbox_mode"]
    bbox = BoxMode.convert(bbox, bbox_mode, BoxMode.XYWH_ABS)

    # Computing areas using bounding boxes
    bbox_xy = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
    area = Boxes([bbox_xy]).area()[0].item()
    bbox = [round(float(x), 3) for x in bbox]
    # COCO requirement:
    #   linking annotations to images
    #   "id" field must start with 1
    coco_annotation["bbox"] = bbox
    coco_annotation["area"] = float(area)
    coco_annotation["iscrowd"] = int(annotation.get("iscrowd", 0))
    coco_annotation["category_id"] = int(reverse_id_mapper(annotation["category_id"]))
    return coco_annotation


def instances_to_hico(instances, metadata, max_dets=4000):
    if len(instances) == 0:
        return []

    if len(instances) > max_dets:
        scores = instances.scores
        kept_scores = torch.topk(scores, k=max_dets)[0][-1]
        filter_mask = scores >= kept_scores
        instances = instances[filter_mask]

    action_object_to_interaction_map = metadata.action_object_to_interaction_map
    reverse_object_mapping = {
        v: k for k, v in metadata.thing_dataset_id_to_contiguous_id.items()
    }
    
    instances = instances[instances.scores.sort(descending=True)[1]]
    
    scores = instances.scores.tolist()
    person_boxes = instances.person_boxes.tensor.tolist()
    object_boxes = instances.object_boxes.tensor.tolist()
    object_classes = instances.object_classes.tolist()
    action_classes = instances.action_classes.tolist()
    offset_indices = instances.offset_indices.tolist()
    
    results = []
    for i in range(len(instances)):
        action_id = action_classes[i]
        object_id = reverse_object_mapping[object_classes[i]]
        hoi = "{} {}".format(action_id, object_id)
        if hoi not in action_object_to_interaction_map:
            continue
    
        person_box = person_boxes[i]
        object_box = object_boxes[i]
    
        results.append(
            [
                action_object_to_interaction_map[hoi],
                person_box[0], person_box[1], person_box[2], person_box[3],
                object_box[0], object_box[1], object_box[2], object_box[3],
                scores[i], offset_indices[i]
            ]
        )

    return results


def instances_to_hico_with_nms(instances, metadata, max_dets=4000):
    if len(instances) == 0:
        return []

    if len(instances) > max_dets:
        scores = instances.scores
        kept_scores = torch.topk(scores, k=max_dets)[0][-1]
        filter_mask = scores >= kept_scores
        instances = instances[filter_mask]

    action_object_to_interaction_map = metadata.action_object_to_interaction_map
    reverse_object_mapping = {
        v: k for k, v in metadata.thing_dataset_id_to_contiguous_id.items()
    }
    
    instances = instances[instances.scores.sort(descending=True)[1]]
    
    # Apply NMS on person boxes and object boxes separately
    action_classes = instances.action_classes
    object_classes = instances.object_classes
    device = action_classes.device
    hois = set([tuple([int(action_id), int(object_id)])
                for action_id, object_id in zip(action_classes, object_classes)])
    hois_to_contiguous_id = {hoi: i for i, hoi in enumerate(hois)}
    
    hoi_scores = instances.scores
    person_boxes = instances.person_boxes.tensor
    object_boxes = instances.object_boxes.tensor
    hoi_classes = torch.tensor([hois_to_contiguous_id[tuple([int(action_id), int(object_id)])]
                    for action_id, object_id in zip(action_classes, object_classes)]).to(device)
    person_filter_ids = batched_nms(person_boxes, hoi_scores, hoi_classes, 0.5)
    object_filter_ids = batched_nms(object_boxes, hoi_scores, hoi_classes, 0.5)

    person_filter_mask = torch.zeros_like(hoi_scores, dtype=torch.bool)
    object_filter_mask = torch.zeros_like(hoi_scores, dtype=torch.bool)
    person_filter_mask[person_filter_ids] = True
    object_filter_mask[object_filter_ids] = True
    filter_mask = torch.logical_or(person_filter_mask, object_filter_mask)
    
    instances = instances[filter_mask]
    
    scores = instances.scores.tolist()
    person_boxes = instances.person_boxes.tensor.tolist()
    object_boxes = instances.object_boxes.tensor.tolist()
    object_classes = instances.object_classes.tolist()
    action_classes = instances.action_classes.tolist()
    offset_indices = instances.offset_indices.tolist()
    
    results = []
    for i in range(len(instances)):
        action_id = action_classes[i]
        object_id = reverse_object_mapping[object_classes[i]]
        hoi = "{} {}".format(action_id, object_id)
        if hoi not in action_object_to_interaction_map:
            continue
    
        person_box = person_boxes[i]
        object_box = object_boxes[i]
    
        results.append(
            [
                action_object_to_interaction_map[hoi],
                person_box[0], person_box[1], person_box[2], person_box[3],
                object_box[0], object_box[1], object_box[2], object_box[3],
                scores[i], offset_indices[i],
            ]
        )

    return results


def calc_ap(scores, boxes, keys, gt_boxes):
    if len(keys) == 0:
        return 0, 0
    
    if isinstance(boxes, list):
        scores, boxes, key = np.array(scores), np.array(boxes), np.array(keys)
    
    hit = []
    idx = np.argsort(scores)[::-1]
    npos = 0
    used = {}
    
    for key in gt_boxes.keys():
        npos += gt_boxes[key].shape[0]
        used[key] = set()

    for i in range(min(len(idx), 19999)):
        pair_id = idx[i]
        box = boxes[pair_id, :]
        key = keys[pair_id]
        if key in gt_boxes:
            maxi = 0.0
            k    = -1
            for i in range(gt_boxes[key].shape[0]):
                tmp = calc_hit(box, gt_boxes[key][i, :])
                if maxi < tmp:
                    maxi = tmp
                    k    = i
            if k in used[key] or maxi < 0.5:
                hit.append(0)
            else:
                hit.append(1)
                used[key].add(k)
        else:
            hit.append(0)
    bottom = np.array(range(len(hit))) + 1
    hit    = np.cumsum(hit)
    rec    = hit / npos
    prec   = hit / bottom
    ap     = 0.0
    for i in range(11):
        mask = rec >= (i / 10.0)
        if np.sum(mask) > 0:
            ap += np.max(prec[mask]) / 11.0
    
    return ap, np.max(rec)


def calc_hit(det, gtbox):
    gtbox = gtbox.astype(np.float64)
    hiou = iou(det[:4], gtbox[:4])
    oiou = iou(det[4:], gtbox[4:])
    return min(hiou, oiou)


def iou(bb1, bb2, debug = False):
    x1 = bb1[2] - bb1[0]
    y1 = bb1[3] - bb1[1]
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    
    x2 = bb2[2] - bb2[0]
    y2 = bb2[3] - bb2[1]
    if x2 < 0:
        x2 = 0
    if y2 < 0:
        y2 = 0

    xiou = min(bb1[2], bb2[2]) - max(bb1[0], bb2[0])
    yiou = min(bb1[3], bb2[3]) - max(bb1[1], bb2[1])
    if xiou < 0:
        xiou = 0
    if yiou < 0:
        yiou = 0

    if debug:
        print(x1, y1, x2, y2, xiou, yiou)
        print(x1 * y1, x2 * y2, xiou * yiou)
    if xiou * yiou <= 0:
        return 0
    else:
        return xiou * yiou / (x1 * y1 + x2 * y2 - xiou * yiou)


def calc_hit(det, gtbox):
    gtbox = gtbox.astype(np.float64)
    hiou = iou(det[:4], gtbox[:4])
    oiou = iou(det[4:], gtbox[4:])
    return min(hiou, oiou)


def iou(bb1, bb2, debug = False):
    x1 = bb1[2] - bb1[0]
    y1 = bb1[3] - bb1[1]
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    
    x2 = bb2[2] - bb2[0]
    y2 = bb2[3] - bb2[1]
    if x2 < 0:
        x2 = 0
    if y2 < 0:
        y2 = 0

    xiou = min(bb1[2], bb2[2]) - max(bb1[0], bb2[0])
    yiou = min(bb1[3], bb2[3]) - max(bb1[1], bb2[1])
    if xiou < 0:
        xiou = 0
    if yiou < 0:
        yiou = 0

    if debug:
        print(x1, y1, x2, y2, xiou, yiou)
        print(x1 * y1, x2 * y2, xiou * yiou)
    if xiou * yiou <= 0:
        return 0
    else:
        return xiou * yiou / (x1 * y1 + x2 * y2 - xiou * yiou)