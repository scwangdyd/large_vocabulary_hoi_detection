
## SWiG-HOI
The assembled SWiG-HOI extracts human interactions with 1,000 object categories from the [SWiG](https://github.com/allenai/swig) dataset. After the filtering, ~400 verbs are kept. The processed annotations are stored in a JSON format for train, test and dev (the split is consistent with original SWiG). Below is an example annotation for one image.
```
[
    "file_name": "wrapping_196.jpg",
    "img_id": 745,
    "height": 512,
    "width": 681,
    "box_annotations": [
        {
            "bbox": [247, 134, 566, 512], # [x1, y1, x2, y2]
            "category_id": 28,
            "aux_category_id": [167, 323], # auxiliary categories given by multiple annotators
        },
        {
            "bbox": [282, 217, 441, 317],
            "category_id": 123,
            "aux_category_id": [] # if the auxiliary categories are empty, it usually means annotators give the consistent annotation. 
        },
        {
            "bbox": [0, 0, 438, 509],
            "category_id": 0,
            "aux_category_id": []
        }
    ],
    "hoi_annotations": [
        {"subject_id": 2, "object_id": 0, "action_id": 402},
        {"subject_id": 2, "object_id": 1, "action_id": 402},
    ],
]
```

This file [swig_v1_meta.py](https://github.com/scwangdyd/large_vocabulary_hoi_detection/blob/master/choir/data/datasets/swig_v1_meta.py) includes more meta information about the extracted SWiG-HOI. Below are some examples to show the metadata.

SWiG objects:
```
    {
        "gloss": ["bicycle", "bike", "wheel", "cycle"],
        "def": "a wheeled vehicle that has two wheels and is moved by foot pedals",
        "id": 7,
        "name": "bicycle",
        "noun_id": "n02834778",
    }
```
SWiG actions:

```
    {
        "id": 272,
        "name": "riding",
        "abstract": "an AGENT rides then VEHICLE at a PLACE",
        "def": "travel in or on (ex. a vehicle, animal, ect.)."
    }
```
SWiG interactions:
```
    {
        "id": 207,
        "name": "riding bicycle",
        "action_id": 272,
        "object_id": 7,
        "frequency": 2,  # 0 - novel (unseen), 1 - rare, 2 - common.
        "evaluation": 1, # 1 = this interaction exists in the test set and will be evaluated.
    }
```

## DOH-HOI
Coming soon.