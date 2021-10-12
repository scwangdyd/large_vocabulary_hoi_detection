import argparse
import pickle
import numpy as np
import os
from collections import defaultdict
import cv2
import tqdm
from fvcore.common.file_io import PathManager

from choir.structures.boxes import BoxMode
from choir.data import DatasetCatalog, MetadataCatalog
from choir.utils.visualizer import InteractionVisualizer


def convert_predictions(predictions, metadata, thresh):
    """
    Convert predicted interactions to
        * pred_boxes (np.array): a concatenation of person and object boxes with shape (n x 8).
        * pred_labels (List[str]): a list of string "{predicted interaction} {score}".
    """
    if len(predictions) == 0:
        return [], []
    
    if isinstance(predictions, list):
        predictions = np.array(predictions)
    
    interaction_classes = metadata.interaction_classes

    # [hoi_id, person boxes (4D), object boxes (4D), score, offset_id]
    pred_scores = predictions[:, -2]
    filter_mask = pred_scores > thresh
    
    pred_hois = predictions[filter_mask, 0].astype(int)
    pred_boxes = predictions[filter_mask, 1:-2].astype(int)
    pred_scores = predictions[filter_mask, -2]
    pred_labels = [interaction_classes[hoi_id] + " " + f"{score:.3f}"
                   for hoi_id, score in zip(pred_hois, pred_scores)]
    return pred_boxes, pred_labels


def convert_ground_truth(dataset_dict, metadata):
    """
    Convert ground truth annotations.
        * gt_boxes (np.array): a concatenation of person and object boxes with shape (n x 8).
        * gt_labels (List[str]): a list of string of gt interaction.
    """
    thing_classes = metadata.thing_classes
    action_classes = metadata.action_classes
    
    annotations = dataset_dict["annotations"]
    gt_boxes, gt_labels = [], []
    for anno in annotations:
        gt_person_box = BoxMode.convert(anno["bbox"], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
        
        for hoi_anno in anno["interaction"]:
            gt_object_box = BoxMode.convert(hoi_anno["bbox"], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
            gt_object_name = thing_classes[hoi_anno["category_id"]]
            for action_id in np.where(hoi_anno["action_id"])[0]:
                gt_action_name = action_classes[action_id]
                gt_boxes.append(gt_person_box + gt_object_box)
                gt_labels.append(f"{gt_action_name} {gt_object_name}")

    return np.array(gt_boxes), gt_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script that visualizes the json predictions from COCO or LVIS dataset."
    )
    parser.add_argument("--input", required=True, help="pkl result file produced by the model")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--dataset", help="name of the dataset", default="coco_2017_val")
    parser.add_argument("--conf-threshold", default=0.5, type=float, help="confidence threshold")
    args = parser.parse_args()

    with PathManager.open(args.input, "rb") as f:
        predictions = pickle.load(f)

    pred_by_image = defaultdict(list)
    for img_id, p in enumerate(predictions):
        pred_by_image[img_id].extend(p)

    dicts = list(DatasetCatalog.get(args.dataset))
    metadata = MetadataCatalog.get(args.dataset)
    if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):
        def dataset_id_map(ds_id):
            return metadata.thing_dataset_id_to_contiguous_id[ds_id]
    else:
        raise ValueError("Unsupported dataset: {}".format(args.dataset))

    os.makedirs(args.output, exist_ok=True)

    for dic in tqdm.tqdm(dicts[::50]):
        img = cv2.imread(dic["file_name"], cv2.IMREAD_COLOR)[:, :, ::-1]
        basename = os.path.basename(dic["file_name"])
        
        pred_boxes, pred_labels = convert_predictions(
            pred_by_image[dic["image_id"]], metadata, args.conf_threshold)
        vis = InteractionVisualizer(img)
        vis_pred = vis.draw_interaction_predictions(pred_boxes, pred_labels).get_image()

        gt_boxes, gt_labels = convert_ground_truth(dic, metadata)
        vis = InteractionVisualizer(img, metadata)
        vis_gt = vis.draw_interaction_predictions(gt_boxes, gt_labels).get_image()

        concat = np.concatenate((vis_gt, vis_pred), axis=1)
        cv2.imwrite(os.path.join(args.output, basename), concat[:, :, ::-1])
