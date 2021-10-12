# -*- coding: utf-8 -*-
import numpy as np
import os
import xml.etree.ElementTree as ET
from fvcore.common.file_io import PathManager
from tqdm import tqdm

from choir.data import DatasetCatalog, MetadataCatalog
from choir.structures import BoxMode

__all__ = ["load_doh_voc", "register_doh_instances"]

HAND_NAMES = (
    "left-hand",
    "right-hand"
)

CONTACT_NAMES = (
    "no-contact",
    "self-contact",
    "another-person",
    "portable-object",
    "stationary-object"
)


def load_doh_voc(dirname: str, split: str, dataset_name: str = None):
    """
    Load DOH detection annotations in VOC format to Detectron2 format.
    
    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
        class_names: list or tuple of class names
    """
    contact_state_id_maps = {
        0: 4, # no-contact to the last
        1: 0, # self-contact
        2: 1, # another-person
        3: 2, # portale-object
        4: 3, # stationary-object
    }
    
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        meta.contact_state_id_maps = contact_state_id_maps
    
    with PathManager.open(os.path.join(dirname, "ImageSets", "Main", split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=np.str)
        
    # Needs to read many small annotation files. Makes sense at local
    annotation_dirname = PathManager.get_local_path(os.path.join(dirname, "Annotations/"))
    dicts = []
    for fileid in tqdm(fileids): #fileids: 
        anno_file = os.path.join(annotation_dirname, fileid + ".xml")
        jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")

        with PathManager.open(anno_file) as f:
            tree = ET.parse(f)
            
        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
        instances = []
        
        for obj in tree.findall("object"):
            cls = obj.find("name").text
            if not cls == "hand":
                continue
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            # Original annotations are integers in the range [1, W or H]
            # Assuming they mean 1-based pixel indices (inclusive),
            # a box with annotation (xmin=1, xmax=W) covers the whole image.
            # In coordinate space this is represented by (xmin=0, xmax=W)
            bbox[0] = max(0, bbox[0] - 1.0)
            bbox[1] = max(0, bbox[1] - 1.0)
            handside = 0 if obj.find("handside").text is None \
                         else int(obj.find("handside").text)
            contact_state = 0 if obj.find("contactstate").text is None \
                              else int(obj.find("contactstate").text)
            contact_state = contact_state_id_maps[contact_state]
            
            obj_bbox = [obj.find(x).text for x in ["objxmin", "objymin", "objxmax", "objymax"]]
            obj_bbox = [0. if x == "None" else max(float(x)-1.0, 0) for x in obj_bbox]
            # Annotation bugs, contact_state > 2 (i.e., interacting with objects) but with no
            # object bounding boxes annotations
            if all([x == 0. for x in obj_bbox]):
                contact_state = 4
            
            instances.append(
                {
                    "category_id": handside,
                    "bbox": bbox,
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "interaction": [
                        {
                            "category_id": int(contact_state == 4),
                            "action_id": int(contact_state),
                            "bbox": obj_bbox,
                            "bbox_mode": BoxMode.XYXY_ABS,
                        }
                    ]
                }
            )
        
        if len(instances) == 0:
            continue

        r["annotations"] = instances
        dicts.append(r)
    return dicts
    

def register_doh_instances(split_name, anno_root, split):
    DatasetCatalog.register(split_name, lambda: load_doh_voc(anno_root, split, split_name))
    MetadataCatalog.get(split_name).set(
        hand_class_names=list(HAND_NAMES),
        contact_class_names=list(CONTACT_NAMES),
        dirname=anno_root,
        split=split,
        year=2007,
    )