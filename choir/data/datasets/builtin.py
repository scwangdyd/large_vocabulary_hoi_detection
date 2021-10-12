# -*- coding: utf-8 -*-
"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
"""

import os

from choir.data import MetadataCatalog

from .builtin_meta import _get_builtin_metadata
from .hico import register_hico_instances
from .swig import register_swig_instances
from .doh import register_doh_instances


# ==== Predefined splits for HICO-DET ===========

_PREDEFINED_SPLITS_HICO = {}
_PREDEFINED_SPLITS_HICO["hico-det"] = {
    "hico-det_train": (
        "hico_20160224_det/images/train2015",
        "hico_20160224_det/annotations/processed_hico_trainval.json",
    ),
    "hico-det_test": (
        "hico_20160224_det/images/test2015",
        "hico_20160224_det/annotations/ppdm_hico_test.json",
    ),
}


def register_all_hico(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_HICO.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            register_hico_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join("/raid1/suchen/repo/simplified/data", json_file),
                os.path.join(root, image_root),
            )


# ==== Predefined splits for 100days of hands (DOH) VOC ===========

_PREDEFINED_SPLITS_DOH = {}
_PREDEFINED_SPLITS_DOH["doh"] = {
    "doh_train": (
        "100DOH/original/pascal_voc_format/VOCdevkit2007_handobj_100K/VOC2007", "train"
    ),
    "doh_val": (
        "100DOH/original/pascal_voc_format/VOCdevkit2007_handobj_100K/VOC2007", "val"
    ),
    "doh_trainval": (
        "100DOH/original/pascal_voc_format/VOCdevkit2007_handobj_100K/VOC2007", "trainval"
    ),
    "doh_test": (
        "100DOH/original/pascal_voc_format/VOCdevkit2007_handobj_100K/VOC2007", "test"
    ),
}


def register_all_doh(root):
    for split_name, (anno_root, split) in _PREDEFINED_SPLITS_DOH["doh"].items():
            register_doh_instances(split_name, os.path.join(root, anno_root), split)
            MetadataCatalog.get(split_name).evaluator_type = "doh"


# ==== Predefined dataset and splits for SWiG ====================
_PREDEFINED_SPLITS_SWIG = {}
_PREDEFINED_SPLITS_SWIG["swig"] = {
    "swig_train": (
        "/raid1/suchen/dataset/swig/images_512",
        "/raid1/suchen/repo/simplified/data/swig_hoi/swig_train_1000.json"
    ),
    "swig_val": (
        "/raid1/suchen/dataset/swig/images_512",
        "/raid1/suchen/repo/simplified/data/swig_hoi/swig_test_1000.json",
    ),
    "swig_dev": (
        "/raid1/suchen/dataset/swig/images_512",
        "/raid1/suchen/repo/simplified/data/swig_hoi/swig_dev_1000.json",
    )
}


def register_all_swig(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_SWIG.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            register_swig_instances(
                key,
                _get_builtin_metadata(dataset_name),
                json_file,
                image_root
            )


# True for open source;
# Internally at fb, we register them elsewhere
if __name__.endswith(".builtin"):
    # Register them all under "./datasets"
    _root = os.getenv("HOI_DATASETS", "/raid1/suchen/dataset")
    register_all_hico(_root)
    register_all_swig(_root)
    register_all_doh(_root)
