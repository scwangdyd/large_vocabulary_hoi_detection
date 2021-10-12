import os
import io
import copy
import json
import torch
import pickle
import shutil
import logging
import datetime
import itertools
import contextlib
import numpy as np
from tabulate import tabulate
from lvis import LVIS, LVISResults, LVISEval
from collections import OrderedDict, defaultdict
from fvcore.common.file_io import PathManager, file_lock

import choir.utils.comm as comm
import choir.utils.box_ops as box_ops
from choir.utils.logger import create_small_table
from choir.data import MetadataCatalog, DatasetCatalog
from choir.layers import batched_nms
from choir.structures import Boxes, BoxMode
from choir.evaluation.fast_eval_api import SWIGeval
from .evaluator import DatasetEvaluator


class SWIGEvaluator(DatasetEvaluator):
    """
    Evaluate object proposal, instance detection, HOI detection on SWiG dataset.
    """
    def __init__(self, dataset_name, cfg, distributed, output_dir):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            cfg (CfgNode): config instance
            distributed (True): if True, will collect results from all ranks and
                run evaluation in the main process. Otherwise, will evaluate the
                results in the current process.
            output_dir (str): Optional, an output directory to dump all results,
                including:
                * "instance_predictions.pth": a file in torch serialization
                    format that contains all raw predictions.
                * 
        """
        self._tasks = self._tasks_from_config(cfg)
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._metadata = MetadataCatalog.get(dataset_name)
        # Convert to LVIS's format such that we can use LVISapi to conduct
        # the object detection evaluation.
        cache_path = os.path.join(output_dir, f"{dataset_name}_lvis.json")
        self._metadata.lvis_file = cache_path
        if not os.path.isfile(cache_path):
            self._logger.info(f"'{cache_path}' is not found. "
                "Trying to convert it to LVIS's format ...")
            convert_to_lvis_json(dataset_name, cache_path)

        lvis_file = PathManager.get_local_path(self._metadata.lvis_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self.lvis_gt = LVIS(lvis_file)

        # Convert to HICO's format such that we can use HICO evaluation
        # metrics to investigate the HOI detection results.
        cache_path = os.path.join(output_dir, f"{dataset_name}_hico.pkl")
        self._metadata.hico_file = cache_path
        if not os.path.isfile(cache_path):
            self._logger.info(f"'{cache_path}' is not found. "
                "Trying to convert it to HICO's format")
            convert_to_hico_format(dataset_name, cache_path)

        hico_file = PathManager.get_local_path(cache_path)
        self._gts = pickle.load(open(hico_file, "rb"))

    def reset(self):
        self._predictions = []
    
    def _tasks_from_config(self, cfg):
        """
        Returns:
            tuple[str]: tasks that can be evaluated given configuration.
        """
        tasks = cfg.TEST.EVAL_TASKS
        return tasks
    
    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model. It is a list of dict.
                Each dict corresponds to an image and contains keys
                like "height", "width", "file_name", "image_id".
            outputs: the outputs of a model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            img_id = input["image_id"]
            res = {"image_id": img_id}

            if "results" not in output: continue
            results = output["results"].to(self._cpu_device)
            if "bbox" in self._tasks:
                res["box_instances"] = instances_to_lvis(results, img_id)
            if "hoi" in self._tasks:
                res["hoi_instances"] = instances_to_swig(results, self._metadata)

            self._predictions.append(res)
            
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
            self._logger.warning(
                "[SWIGEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)

        self._results = OrderedDict()
        if "bbox" in self._tasks and "box_instances" in predictions[0]:
            self._eval_box_predictions(copy.deepcopy(predictions))
        if "hoi" in self._tasks and "hoi_instances" in predictions[0]:
            self._eval_hoi_predictions(copy.deepcopy(predictions))
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_box_predictions(self, predictions, iou_type="bbox"):
        """
        Evaluate box predictions.
        Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for COCO format ...")
        box_predictions = [x["box_instances"] for x in predictions]
        lvis_results = list(itertools.chain(*box_predictions))

        # unmap the category ids for objects
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            id_mapping = self._metadata.thing_dataset_id_to_contiguous_id
            reverse_id_mapping = {v: k for k, v in id_mapping.items()}
            for result in lvis_results:
                category_id = result["category_id"]
                assert (category_id in reverse_id_mapping), \
                    f"A prediction has category_id={category_id}, " + \
                    "which is not available in the dataset."
                result["category_id"] = reverse_id_mapping[category_id]

        if self._output_dir:
            file_path = self._output_dir + "/lvis_instances_results.json"
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(lvis_results))
                f.flush()

        self._logger.info("Evaluating predictions with official LVIS API...")
        res = self._evaluate_predictions_on_lvis(lvis_results, iou_type)
        self._results[iou_type] = res

    def _evaluate_predictions_on_lvis(self, lvis_results, iou_type):
        """
        Evaluate the box  results using SWIGEval API.
        """          
        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl", "APr", "APc", "APf"],
        }[iou_type]
        
        if len(lvis_results) == 0:
            self._logger.warn("Empty predictions !")
            return {metric: float("nan") for metric in metrics}

        lvis_res = LVISResults(self.lvis_gt, lvis_results)
        # lvis_eval = SWIGeval(self.lvis_gt, lvis_res, iou_type) 
        lvis_eval = LVISEval(self.lvis_gt, lvis_res, iou_type)
        
        lvis_eval.run()
        lvis_eval.print_results()

        # Pull the standard metrics from the LVIS results
        results = lvis_eval.get_results()
        results = {metric: float(results[metric] * 100) for metric in metrics}
        self._logger.info(f"Evaluation results for {iou_type}: \n" + create_small_table(results))
        return results

    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        """
        Derive the desired score numbers from summarized SWIGeval.

        Args:
            coco_eval (None or SWIGEval):
            iou_type (str): "bbox"
            class_names (None or list[str]): use it to predict per-category AP.

        Returns:
            a dict of {metric name: score}
        """

        metrics = {"bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"]}[iou_type]

        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # Standard metrics
        results = {
            metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }
        self._logger.info(f"Evaluation results for {iou_type}: \n")
        self._logger.info(create_small_table(results))
        if not np.isfinite(sum(results.values())):
            self._logger.info("Note that some metrics cannot be computed.")

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AP
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
        results_2d = itertools.zip_longest(
            *[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info(f"Per-category {iou_type} AP: \n" + table)
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
        results_2d = itertools.zip_longest(
            *[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP50"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info(f"Per-category {iou_type} AP50: \n" + table)
        return results

    def _eval_hoi_predictions(self, predictions):
        """
        Evaluate the human-object interaction detections.
        Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Evaluate HOI results using HICO's metrics ...")
        images = [x["image_id"] for x in predictions]
        results = [x["hoi_instances"] for x in predictions]

        if self._output_dir:
            file_path = self._output_dir + "/swig_interaction_results.pkl"
            file_path = os.path.abspath(file_path)
            self._logger.info("Saving results to {}".format(file_path))
            write_hoi_results(images, results, file_path)
        else:
            file_path = "./output"
        
        eval_hois = np.asarray(self._metadata.interactions_for_eval)
        size = max(eval_hois) + 1
        scores = [[] for _ in range(size)]
        boxes = [[] for _ in range(size)]
        keys = [[] for _ in range(size)]

        for img_id, dets in zip(images, results):
            for det in dets:
                hoi_id = int(det[0])
                boxes[hoi_id].append(det[1:-2])
                scores[hoi_id].append(float(det[-2]))
                keys[hoi_id].append(img_id)

        swig_ap, swig_rec = np.zeros(size), np.zeros(size)
        for hoi_id in eval_hois:
            gts = self._gts[hoi_id]
            ap, rec = calc_ap(scores[hoi_id], boxes[hoi_id], keys[hoi_id], gts)
            swig_ap[hoi_id], swig_rec[hoi_id] = ap, rec
        # TODO: derive swig results
        res = self._derive_swig_results(swig_ap, swig_rec)
        self._results["hoi"] = res
        with open(os.path.join(self._output_dir, "swig_hoi_ap_rec.pkl"), "wb") as f:
            pickle.dump({"swig_ap": swig_ap, "swig_rec": swig_rec}, f)

    def _derive_swig_results(self, swig_ap, swig_rec):
        """
        Derive the desired score numbers from the results.
        Args:
            swig_ap (np.array): computed ap results per hoi.
            swig_rec (np.array): computed ap results per hoi.
        Returns:
            a dict of {metric name: score}
        """
        eval_hois = np.array(self._metadata.interactions_for_eval)
        rare_hois = np.array(self._metadata.rare_interaction_ids)
        novel_hois = np.array(self._metadata.novel_interaction_ids)
        common_hois = np.array(self._metadata.common_interaction_ids)
        
        eval_common_hois = np.intersect1d(eval_hois, common_hois)
        eval_rare_hois = np.intersect1d(eval_hois, rare_hois)
        eval_novel_hois = np.intersect1d(eval_hois, novel_hois)
        eval_known_hois = np.setdiff1d(eval_hois, novel_hois)
        
        full_map = np.mean(swig_ap[eval_known_hois]) * 100.
        rare_map = np.mean(swig_ap[eval_rare_hois]) * 100.
        non_rare_map = np.mean(swig_ap[eval_common_hois]) * 100.
        novel_map = np.mean(swig_ap[eval_novel_hois]) * 100.
        
        results = {
            "full": full_map,
            "rare": rare_map,
            "non_rare": non_rare_map,
            "novel": novel_map
        }
        self._logger.info(f"Evaluation results for HOI: \n" + create_small_table(results))
        return results


def instances_to_lvis(instances, img_id, max_dets=100):
    """
    Dump an "Instances" object to a LVIS-format json that's used for evaluation.
    
    Args:
        instances (Instances):
        img_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    if len(instances) == 0:
        return []
    
    object_classes = instances.object_classes
    person_classes = instances.person_classes
    object_scores = instances.object_scores
    person_scores = instances.person_scores
    object_boxes = instances.object_boxes.tensor
    person_boxes = instances.person_boxes.tensor
    
    classes = torch.cat([object_classes, person_classes], dim=0)
    scores = torch.cat([object_scores * person_scores, person_scores], dim=0)
    boxes = torch.cat([object_boxes, person_boxes], dim=0)
    
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
    
    results = [{"image_id": img_id,
                "category_id": classes[i],
                "bbox": boxes[i],
                "score": scores[i]} for i in range(len(boxes))]
    return results


def instances_to_swig(instances, metadata, max_dets=2000):
    """
    Dump an "Instances" object to a format that's used for evaluation.
    
    Args:
        instances (Instances): output given by the model

    Returns:
        results (list): list of prepared predictions:
            [hoi_id, person_box, object_box, score, offset_id]
    """
    if len(instances) == 0:
        return []
    
    if len(instances) > max_dets:
        scores = instances.scores
        kept_scores = torch.topk(scores, k=max_dets)[0][-1]
        filter_mask = scores >= kept_scores
        instances = instances[filter_mask]

    interactions_for_eval = metadata.get("interactions_for_eval", None)
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
        hoi = tuple([action_id, object_id])
        if hoi not in action_object_to_interaction_map:
            continue

        hoi_id = action_object_to_interaction_map[hoi]
        if interactions_for_eval and hoi_id not in interactions_for_eval:
            continue

        person_box = person_boxes[i]
        object_box = object_boxes[i]
    
        results.append(
            [
                hoi_id,
                person_box[0], person_box[1], person_box[2], person_box[3],
                object_box[0], object_box[1], object_box[2], object_box[3],
                scores[i], offset_indices[i]
            ]
        )
    return results


def write_hoi_results(images, results, file_path):
    """
    Write HOI detection results into .pkl file.

    Args:
        images (List[int]): A list of image id
        results (List[List]): A list of detection results, which is saved in a
            list containing [interaction_id, person box, object box, score]
        file_path (String): savefile name
    """
    assert len(images) == len(results)

    dets = [[] for _ in range(20000)]
    for img_id, results_per_image in zip(images, results):
        dets[img_id] = results_per_image
    
    with open(file_path, "wb") as f:
        pickle.dump(dets, f)


def convert_to_lvis_json(dataset_name, output_file, allow_cached=True):
    """
    Convert dataset into LVIS format and save it to a json file.
    dataset_name must be registered in DatasetCatalog.
    
    Args:
        dataset_name (str): reference from the config file to the catalogs
        output_file (str): path of json file that will be saved to
        allow_cached: if json file is already present then skip conversion
    """
    logger = logging.getLogger(__name__)
    PathManager.mkdirs(os.path.dirname(output_file))
    with file_lock(output_file):
        if PathManager.exists(output_file) and allow_cached:
            logger.warning(
                f"Using cached LVIS format annotation at '{output_file}'. "
                "You need to clear the cache file if your dataset has changed.")
        else:
            logger.info(
                f"Converting dataset '{dataset_name}' to LVIS format ...")
            coco_dict = _convert_to_lvis_dict(dataset_name)
            logger.info(f"Caching converted annotations at '{output_file}' ...")
            tmp_file = output_file + ".tmp"
            with PathManager.open(tmp_file, "w") as f:
                json.dump(coco_dict, f)
            shutil.move(tmp_file, output_file)

        
def _convert_to_lvis_dict(dataset_name, num_neg=500):
    """
    Convert an hoi detection dataset into LVIS json format.
    Prepare SWiG dataset as LVIS format by adding the following to image meta info
        - not_exhaustive_category_ids: List of category ids which don't have all of
            their instances marked exhaustively.
        - neg_category_ids: List of category ids which were verified as not present
            in the image.

    And the following to category meta info
        - image_count: Number of images in which the category is annotated.
        - instance_count: Number of annotated instances of the category.
        - frequency: We divide the categories into three buckets based on 
            image_count in the train set.

    Args:
        dataset_name (str): Name of the source dataset. Must be registered
            in DatastCatalog and in detectron2's standard format.
        num_neg (int): Number of negative images per category for evaluation.
    Returns:
        coco_dict: serializable dict in LVIS json format
    """
    logger = logging.getLogger(__name__)

    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)

    # unmap the category mapping ids for LVIS
    if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):
        id_mapping = metadata.thing_dataset_id_to_contiguous_id
        reverse_id_mapping = {v: int(k) for k, v in id_mapping.items()}
        reverse_id_mapper = lambda cat_id: reverse_id_mapping[cat_id]
    else:
        reverse_id_mapper = lambda cat_id: cat_id

    categories = [
        {"id": reverse_id_mapper(cat_id), "name": name}
        for cat_id, name in enumerate(metadata.thing_classes)
    ]

    coco_images = []
    coco_annotations = []

    for image_id, image_dict in enumerate(dataset_dicts):
        coco_image = {
            "id": image_dict.get("image_id", image_id),
            "width": int(image_dict["width"]),
            "height": int(image_dict["height"]),
            "file_name": str(image_dict["file_name"]),
            "neg_category_ids": [], # required by LVISapi
            "not_exhaustive_category_ids": [], # required by LVISapi
        }
        coco_images.append(coco_image)

        anns_per_image = image_dict.get("annotations", [])
        boxes_per_image = dict()
        for annotation in anns_per_image:
            # Append person instances
            coco_anno = _ann_to_coco_dict(annotation, reverse_id_mapper)
            box_key = tuple(coco_anno["bbox"] + [coco_anno["category_id"]])
            if box_key not in boxes_per_image:
                boxes_per_image[box_key] = coco_anno
            # Append object instances
            for hoi_annotation in annotation["interaction"]:
                coco_anno = _ann_to_coco_dict(hoi_annotation, reverse_id_mapper)
                box_key = tuple(coco_anno["bbox"] + [coco_anno["category_id"]])
                if box_key not in boxes_per_image:
                    boxes_per_image[box_key] = coco_anno
        
        for _, coco_anno in boxes_per_image.items():
            coco_anno["id"] = len(coco_annotations) + 1
            coco_anno["image_id"] = coco_image["id"]
            coco_annotations.append(coco_anno)

    # >>>>>> Start preparing LVIS required annotations >>>>>>
    exist_cats = {img_meta["id"]: set() for img_meta in coco_images}
    pos_per_cat = {cat["id"]: set() for cat in categories}
    neg_per_cat = {cat["id"]: set() for cat in categories}
    img_count = {cat["id"]: set() for cat in categories}
    ins_count = {cat["id"]: 0 for cat in categories}
    for anno in coco_annotations:
        img_id = anno["image_id"]
        cat_id = anno["category_id"]
        aux_cat_id = anno["aux_category_id"]
        
        exist_cats[img_id].add(cat_id)
        exist_cats[img_id].add(aux_cat_id)
        pos_per_cat[cat_id].add(img_id)
        img_count[cat_id].add(img_id)
        ins_count[cat_id] += 1
    # Random sampling n negative images per category to speed up the evaluation.
    for cat in categories:
        cat_id = cat["id"]
        pos_img_ids = pos_per_cat[cat_id]
        neg_img_ids = np.setdiff1d(list(exist_cats.keys()), list(pos_img_ids))
        n = min(len(neg_img_ids), num_neg)
        neg_per_cat[cat_id] = list(np.random.choice(neg_img_ids, n, replace=False))

    img_neg_ids = {img_meta["id"]: [] for img_meta in coco_images}
    for cat_id, neg_imgs in neg_per_cat.items():
        for img_id in neg_imgs:
            img_neg_ids[img_id].append(cat_id)

    for img_meta in coco_images:
        img_id = img_meta["id"]
        img_meta["neg_category_ids"] = sorted(img_neg_ids[img_id])
    
    # Split categories to frequent, common and rare
    img_count = {cat_id: len(imgs) for cat_id, imgs in img_count.items()}
    for cat in categories:
        num_imgs = img_count[cat["id"]]
        cat["image_count"] = num_imgs
        cat["instance_count"] = ins_count[cat["id"]]
        if num_imgs <= 10: cat["frequency"] = "r"
        elif num_imgs <= 100: cat["frequency"] = "c"
        else: cat["frequency"] = "f"
    # <<<<<< End preparing LVIS required annotations <<<<<<

    logger.info(
        "Conversion finished, "
        f"#images: {len(coco_images)}, #annotations: {len(coco_annotations)}"
    )

    info = {
        "date_created": str(datetime.datetime.now()),
        "description": "Automatically generated COCO json file for Detectron2.",
    }
    coco_dict = {"info": info,
                 "images": coco_images,
                 "categories": categories,
                 "licenses": None}
    if len(coco_annotations) > 0:
        coco_dict["annotations"] = coco_annotations
    return coco_dict


def _ann_to_coco_dict(anno, cat_mapper):
    """
    Create a new dict using COCO's format.
    """
    coco_anno = {}
    # COCO requirement: XYWH box
    bbox = anno["bbox"]
    bbox_mode = anno["bbox_mode"]
    bbox = BoxMode.convert(bbox, bbox_mode, BoxMode.XYWH_ABS)

    # Computing areas using bounding boxes
    bbox_xy = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
    area = Boxes([bbox_xy]).area()[0].item()
    bbox = [round(float(x), 3) for x in bbox]
    # COCO requirement:
    #   linking annotations to images
    #   "id" field must start with 1
    coco_anno["bbox"] = bbox
    coco_anno["area"] = float(area)
    coco_anno["iscrowd"] = int(anno.get("iscrowd", 0))
    coco_anno["category_id"] = cat_mapper(anno["category_id"])
    if "aux_category_id" not in anno or anno["aux_category_id"] == -1:
        coco_anno["aux_category_id"] = -1
    else:
        coco_anno["aux_category_id"] = cat_mapper(anno["aux_category_id"])
    return coco_anno


def convert_to_hico_format(dataset_name, output_file):
    """
    Convert dataset into HICO's format and save it to a pkl file.
    dataset_name must be registered in DatasetCatalog.
    
    Args:
        dataset_name (str): reference from the config file to the catalogs
        output_file (str): path of json file that will be saved to
    """
    logger = logging.getLogger(__name__)
    PathManager.mkdirs(os.path.dirname(output_file))
    logger.info(f"Converting annotations of dataset '{dataset_name}' to HICO's format ...")
    
    gts = _convert_to_hico_dict(dataset_name)
    
    logger.info(f"Caching HICO's format annotations at '{output_file}' ...")
    tmp_file = output_file + ".tmp"
    with PathManager.open(tmp_file, "wb") as f:
        pickle.dump(gts, f)
    shutil.move(tmp_file, output_file)


def _convert_to_hico_dict(dataset_name):
    """
    Convert an hoi detection dataset into HICO's original format.

    Args:
        dataset_name (str): Reference from the config file to the catalogs.
                            Must be registered in DatastCatalog
    Returns:
        gts (dict): converted annotations with shape (# HOIs x # images). The
            entry [i][j] is a list including all i-th interactions
            at j-th image. The interaction is denoted as
                [person_x1, person_y1, person_x2, person_y2,
                 object_x1, object_y1, object_x2, object_y2].
    """
    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)

    action_object_to_interaction_map = metadata.action_object_to_interaction_map
    
    gts = {hoi_id: defaultdict(list) for _, hoi_id in action_object_to_interaction_map.items()}
    for _, anno_dict in enumerate(dataset_dicts):
        img_id = anno_dict["image_id"]
        for annotation in anno_dict.get("annotations", []):
            person_box = annotation["bbox"]
            bbox_mode = annotation["bbox_mode"]
            person_box = BoxMode.convert(person_box, bbox_mode, BoxMode.XYXY_ABS)
            person_box = [round(float(x), 3) for x in person_box]
            for hoi_annotation in annotation["interaction"]:
                object_box = hoi_annotation["bbox"]
                bbox_mode = hoi_annotation["bbox_mode"]
                object_box = BoxMode.convert(object_box, bbox_mode, BoxMode.XYXY_ABS)
                object_box = [round(float(x), 3) for x in object_box]

                object_id = int(hoi_annotation["category_id"])
                for action_id in np.where(hoi_annotation["action_id"])[0]:
                    hoi_id = action_object_to_interaction_map[tuple([action_id, object_id])]
                    gts[hoi_id][img_id].append(person_box + object_box)

    for hoi_id in gts:
        for img_id in gts[hoi_id]:
            gts[hoi_id][img_id] = np.array(gts[hoi_id][img_id])

    return gts


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