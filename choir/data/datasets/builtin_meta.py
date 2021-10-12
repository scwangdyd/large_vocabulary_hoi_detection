# -*- coding: utf-8 -*-
from .hico_meta import (
    HICO_OBJECTS,
    HICO_ACTIONS,
    HICO_INTERACTIONS,
    RARE_INTERACTION_IDS,
    NON_INTERACTION_IDS,
)
from .swig_v1_meta import SWIG_CATEGORIES, SWIG_ACTIONS, SWIG_INTERACTIONS

__all__ = ["_get_builtin_metadata"]


def _get_hico_meta():
    """
    Returns metadata for HICO-DET dataset.
    """
    thing_ids = [k["id"] for k in HICO_OBJECTS if k["isthing"] == 1]
    thing_colors = {k["name"]: k["color"] for k in HICO_OBJECTS if k["isthing"] == 1}
    assert len(thing_ids) == 80, len(thing_ids)
    # Mapping from the noncontiguous category id to an id in [0, 79]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in HICO_OBJECTS if k["isthing"] == 1]
    # HICO-DET actions
    action_classes = [k["name"] for k in HICO_ACTIONS]
    action_priors  = [k["prior"] for k in HICO_ACTIONS]
    # Category id of `person`
    person_cls_id = [k["id"] for k in HICO_OBJECTS if k["name"] == 'person'][0]
    # Mapping interactions (action name + object name) to contiguous id 
    interaction_classes = [x["action"] + " " + x["object"] for x in HICO_INTERACTIONS]
    interaction_classes_to_contiguous_id = {
            x["action"] + " " + x["object"]: x["interaction_id"] for x in HICO_INTERACTIONS
        }
    
    action_name_to_id = {x["name"]: x["id"] for x in HICO_ACTIONS}
    object_name_to_id = {x["name"]: x["id"] for x in HICO_OBJECTS}
    action_object_to_interaction_map = {
        "{} {}".format(action_name_to_id[x["action"]],
                       object_name_to_id[x["object"]]
                       ): x["interaction_id"]
        for x in HICO_INTERACTIONS
    }

    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes":  thing_classes,
        "thing_colors":   thing_colors,
        "action_classes": action_classes,
        "action_priors":  action_priors,
        "person_cls_id":  person_cls_id,
        "interaction_classes": interaction_classes,
        "rare_interaction_ids": RARE_INTERACTION_IDS,
        "non_interaction_ids": NON_INTERACTION_IDS,
        "interaction_classes_to_contiguous_id": interaction_classes_to_contiguous_id,
        "action_object_to_interaction_map": action_object_to_interaction_map,
    }

    return ret


def _get_swig_meta():
    """
    Return metadata for SWiG dataset.
    """
    thing_ids = [k["id"] for k in SWIG_CATEGORIES]
    assert len(thing_ids) == 1000
    # Mapping from incontiguous category id to id in [0, 999]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in SWIG_CATEGORIES]
    action_classes = [k["name"] for k in SWIG_ACTIONS]
    action_object_to_interaction_map = {
        tuple([x["action_id"], x["object_id"]]): x["id"] for x in SWIG_INTERACTIONS
    }
    interaction_classes = [x["name"] for x in SWIG_INTERACTIONS]
    interactions_for_eval = [x["id"] for x in SWIG_INTERACTIONS if x["evaluation"] == 1]
    novel_interaction_ids = [x["id"] for x in SWIG_INTERACTIONS if x["frequency"] == 0]
    rare_interaction_ids = [x["id"] for x in SWIG_INTERACTIONS if x["frequency"] == 1]
    common_interaction_ids = [x["id"] for x in SWIG_INTERACTIONS if x["frequency"] == 2]
    
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "action_classes": action_classes,
        "person_cls_id": 0,
        "action_object_to_interaction_map": action_object_to_interaction_map,
        "interaction_classes": interaction_classes,
        "interactions_for_eval": interactions_for_eval,
        "novel_interaction_ids": novel_interaction_ids,
        "rare_interaction_ids": rare_interaction_ids,
        "common_interaction_ids": common_interaction_ids,
    }

    return ret


def _get_builtin_metadata(dataset_name):
    if dataset_name == "hico-det":
        return _get_hico_meta()
    elif dataset_name == "swig":
        return _get_swig_meta()
    # elif dataset_name == "doh":
    #     return _get_doh_meta()
    raise KeyError("No built-in metadata for dataset {}".format(dataset_name))