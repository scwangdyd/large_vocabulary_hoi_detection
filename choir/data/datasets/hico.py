from collections import defaultdict
import json
import logging
import numpy as np
import os
from fvcore.common.file_io import PathManager
from fvcore.common.timer import Timer
from choir.data import DatasetCatalog, MetadataCatalog
from choir.structures import BoxMode

__all__ = ["register_hico_instances"]

logger = logging.getLogger(__name__)


def load_hico_json(json_file: str, image_root: str, dataset_name: str = None):
    """
    Load a json file with HOI's instances annotation.

    Args:
        json_file (str): full path to the json file in HOI instances annotation format.
        image_root (str or path-like): the directory where the images in this json file exists.
        dataset_name (str): the name of the dataset (e.g., `hico-det_train`).
            If provided, this function will also put "thing_classes" into
            the metadata associated with this dataset.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "iscrowd", "bbox", "category_id"). The values
            for these keys will be returned as-is. For example, the densepose annotations are
            loaded in this way.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard dataset dicts format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    imgs_anns = json.load(open(json_file, "r"))
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        # The categories in a custom json file may not be sorted.
        thing_classes = meta.thing_classes
        action_classes = meta.action_classes
        id_map = meta.thing_dataset_id_to_contiguous_idd

    dataset_dicts = []
    images_without_valid_annotations = []
    for anno_dict in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, anno_dict["file_name"])
        record["height"] = anno_dict["height"]
        record["width"] = anno_dict["width"]
        record["image_id"] = anno_dict["img_id"]
        
        if len(anno_dict["box_annotations"]) == 0 or len(anno_dict["hoi_annotations"]) == 0:
            images_without_valid_annotations.append(anno_dict)
            dataset_dicts.append(record)
            continue

        boxes = convert_xyxy_to_xywh([obj["bbox"] for obj in anno_dict["box_annotations"]])
        classes = [id_map[obj["category_id"]] for obj in anno_dict["box_annotations"]]
        
        hoi_anno_dicts = dict()
        for hoi in anno_dict["hoi_annotations"]:
            person_id = hoi['subject_id']
            object_id = hoi['target_id']
            action_id = hoi["action_id"]
            assert classes[person_id] == 0
            if person_id not in hoi_anno_dicts:
                hoi_anno_dicts[person_id] = defaultdict(list)
            hoi_anno_dicts[person_id][object_id].append(action_id)

        objs = []
        for person_id, hoi_dict in hoi_anno_dicts.items():
            obj = {
                "bbox": boxes[person_id],
                "category_id": 0,
                "iscrowd": 0,
                "bbox_mode": BoxMode.XYWH_ABS,
                "interaction": [],
            }
            for object_id, action_ids in hoi_dict.items():
                actions = np.zeros(len(action_classes), dtype=np.float32)
                actions[action_ids] = 1.
                obj["interaction"].append(
                    {
                        "bbox": boxes[object_id],
                        "category_id": classes[object_id],
                        "action_id": actions,
                        "bbox_mode": BoxMode.XYWH_ABS,
                    }
                )
            assert len(obj["interaction"]) > 0, "no valid interactions"
            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


def convert_xyxy_to_xywh(boxes):
    if isinstance(boxes, list):
        boxes = np.array(boxes)
    boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
    boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
    return boxes.tolist()


def register_hico_instances(name, metadata, json_file, image_root):
    """
    Register a hico-det dataset in COCO's json annotation format for human-object
    interaction detection (i.e., `instances_hico_*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "hico-det".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_hico_json(json_file, image_root, name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(image_root=image_root, evaluator_type="hico", **metadata)