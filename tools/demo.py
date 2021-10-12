import argparse
import glob
import numpy as np
import multiprocessing as mp
import torch
import os
import time
import tqdm

from choir.config import get_cfg
from choir.data.detection_utils import read_image
from choir.utils.logger import setup_logger
from choir.data import MetadataCatalog
from choir.engine.defaults import DefaultPredictor
from choir.utils.visualizer import ColorMode
from choir.utils.visualizer import InteractionVisualizer


class VisualizationDemo(object):
    def __init__(self, cfg, args, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode
        self.thresh = args.confidence_threshold
        self.parallel = parallel
        self.predictor = DefaultPredictor(cfg)
        
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TRAIN[0] if len(cfg.DATASETS.TRAIN) else "__unused")

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = InteractionVisualizer(image)

        instances = predictions["results"].to(self.cpu_device)
        pred_boxes, pred_labels = self.convert_predictions(instances)
        vis_output = visualizer.draw_interaction_predictions(pred_boxes, pred_labels)
        return predictions, vis_output
                
    def convert_predictions(self, predictions):
        """
        Convert predicted interactions to
            * pred_boxes (np.array): a concatenation of person and object boxes with shape (n x 8).
            * pred_labels (List[str]): a list of string "{predicted interaction} {score}".
        """
        if len(predictions) == 0:
            return [], []
        
        if isinstance(predictions, list):
            predictions = np.array(predictions)
        
        interaction_classes = self.metadata.interaction_classes
        action_object_to_interaction_map = self.metadata.action_object_to_interaction_map

        predictions = predictions[predictions.scores.sort(descending=True)[1]]
    
        scores = predictions.scores.tolist()
        person_boxes = predictions.person_boxes.tensor.tolist()
        object_boxes = predictions.object_boxes.tensor.tolist()
        object_classes = predictions.object_classes.tolist()
        action_classes = predictions.action_classes.tolist()
        
        pred_boxes = [] # [person boxes (4D), object boxes (4D)]
        pred_labels = []
        for i in range(len(predictions)):
            score = scores[i]
            if score < self.thresh:
                continue
            
            action_id = action_classes[i]
            object_id = object_classes[i]
            hoi = tuple([action_id, object_id])
            if hoi not in action_object_to_interaction_map:
                continue

            hoi_id = action_object_to_interaction_map[hoi]

            person_box = person_boxes[i]
            object_box = object_boxes[i]
        
            pred_boxes.append(person_box + object_box)
            pred_labels.append(interaction_classes[hoi_id] + " " + f"{score:.3f}")

        return np.asarray(pred_boxes).astype(int), pred_labels


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs/Base-HOIRCNN-FPN.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
        default='.'
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.2,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg, args)

    if len(args.input) == 1:
        args.input = glob.glob(os.path.expanduser(args.input[0]))
        assert args.input, "The input path(s) was not found"
    for path in args.input:
        # use PIL, to be consistent with evaluation
        img = read_image(path, format="BGR")
        start_time = time.time()
        predictions, visualized_output = demo.run_on_image(img)

        if os.path.isdir(args.output):
            assert os.path.isdir(args.output), args.output
            out_filename = os.path.join(args.output, "hoi_" + os.path.basename(path))
        else:
            assert len(args.input) == 1, "Please specify a directory with args.output"
            out_filename = args.output
        visualized_output.save(out_filename)
